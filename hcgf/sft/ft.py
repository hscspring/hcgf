from typing import Optional, List, Tuple, Dict, Any, Union
import copy

from pnlp import MagicDict
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload
)

from transformers import AutoTokenizer, AutoModel
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.generation.utils import (
    LogitsProcessorList,
    StoppingCriteriaList,
    StoppingCriteria
)
from transformers.generation.logits_process import LogitsProcessor

from .lora import LoraModel, LoraConfigLoader
from ..dataloader import GlmDataLoader
from ..data_model import Tensor
from ..trainer import Trainer
from ..trainer.fsdp import (
    setup, 
    cleanup,
    get_transformer_wrap_policy, 
    get_sharding_strategy,
    get_mp_policy, 
)
from ..utils import (
    print_layer_info, print_trainable_parameters, 
    create_token_tensor_list, get_model_name_from
)


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


class CustomStoppingCriteria(StoppingCriteria):

    def __init__(
        self,
        stop_tensor_list: List[Tensor["L", torch.LongTensor]],
        device: torch.device,
        encounters: int = 1
    ):
        super().__init__()
        self.stops = [stop.to(device) for stop in stop_tensor_list]
        self.encounters = encounters

    def __call__(
        self,
        input_ids: Tensor["B,L", torch.LongTensor],
        scores: torch.FloatTensor
    ):
        count = 0
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                count += 1
            if count >= self.encounters:
                return True
        return False


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


class GlmLora:

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        load_in_8bit: bool = False,
    ):
        world_size = torch.cuda.device_count()
        self.model_id = model_id
        self.model_name = get_model_name_from(model_id)

        self.load_in_8bit = False
        self.device = None
        
        if load_in_8bit:
            self.load_in_8bit = True
            import bitsandbytes as bnb
            self.mode = "8bit"
        elif device is not None:
            self.device = device
            self.mode = "single_gpu"
        elif device is None and world_size > 1:
            self.mode = "fsdp"
        else:
            msg = f"Invalid config: \n"
            msg += "  if load_in_8bit=True         ==> 8 bit without FSDP \n"
            msg += "  if device!=None              ==> use the specified single GPU \n"
            msg += "  if device=None & gpu_num > 1 ==> FSDP mode \n"
            raise ValueError(msg)
        
        print(f"You are in {self.mode} mode")
        print(f"Loading tokenizer {model_id}")
        self.tokenizer = self.load_tokenizer()
        self.lora_config = LoraConfigLoader(
            lora_r, lora_alpha, lora_dropout
        ).get_config(self.model_name)
        self.model_is_setup = False

        self.dataloader = None

        self.stop_tokens = ["问题："]
        self.builtin_stop_tensor_list = create_token_tensor_list(
            self.tokenizer, self.stop_tokens
        )
    
    def __cast_small_to(self, dtype: torch.Type):
        for name, param in self.model.named_parameters():
            if param.ndim == 1 and param.dtype != dtype:
                param.data = param.data.to(dtype)

    def __cast_lora_to(self, dtype: torch.Type):
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(dtype)
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        if self.model_name == "llama":
            from transformers import LlamaTokenizer
            tk = LlamaTokenizer.from_pretrained(self.model_id)
            tk.pad_token_id = (
                0  # unk. we want this to be different from the eos token
            )
            tk.padding_side = "left"
        else:
            tk = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        tk.max_model_input_len = self.max_input_length
        tk.model_name = self.model_name
        return tk
    
    def load_model(self, model_id: str) -> PreTrainedModel:
        if self.load_in_8bit:
            device_map = "auto"
        else:
            device_map = None
        if self.model_name == "llama":
            from transformers import LlamaForCausalLM
            ModelCls = LlamaForCausalLM
            trust_remote_code = False
        else:
            ModelCls = AutoModel
            trust_remote_code = True
        model = ModelCls.from_pretrained(
            model_id, 
            load_in_8bit=self.load_in_8bit,
            # 大模型half
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        # if not self.load_in_8bit:
        #     model.half()
        if self.device:
            model.to(self.device)
        return model

    def load_pretrained(self, pt_path: Optional[str]):
        if not self.model_is_setup:
            self.model = self.load_model(self.model_id)
            self.model = LoraModel(self.model, self.lora_config)
            self.model_is_setup = True
        if pt_path is not None:
            static = torch.load(pt_path)
            self.model.load_state_dict(static, strict=False)
        return self

    def load_data(self, data_path: str, max_seq_len: int = 512):
        print(f"Loading data with max_seq_len: {max_seq_len}")
        self.dataloader = GlmDataLoader(data_path, self.tokenizer, max_seq_len)
        return self
    
    def fsdp_tune(
        self, 
        rank: int, 
        world_size: int,
        params: Dict[str, Any]
    ) -> None:
        assert self.mode == "fsdp", "Only for FSDP"
        assert self.dataloader is not None, "Please `load_data` first"
        assert "batch_size" in params, "batch_size is a must parameter"
        def build_tune_params():
            default = {
                "strategy": "fsdp_zero3",
                "lr": 2e-4,
                "num_epochs": 3,
                "warmup_steps": None,
                "accumulate_steps": None,
                "out_dir": "./output/",
                "print_every": None,
                "pretrained_ckpt": None,
            }
            for key in default:
                if key not in params:
                    params[key] = default[key]

        build_tune_params()
        args = MagicDict(params)
        strategy = args.get("strategy", "fsdp_zero3")
        if not self.model_is_setup:
            self.model = self.load_model(self.model_id)
            # NOTE: ChatGLM use mixed float, FSDP needs all parameters in a shard to be the same
            setup(rank, world_size)
            train_loader, val_loader = self.dataloader.train_dev_split(args.batch_size, True, rank)
            auto_wrap_policy = get_transformer_wrap_policy(self.model, self.transformer_block)
            sharding_strategy = get_sharding_strategy(strategy)
            mp_policy = get_mp_policy()
            torch.cuda.set_device(rank)
            if rank == 0:
                print(f"Processing Lora Model in {self.mode} mode...")
                print(f"mp policy: {mp_policy}")
            # NOTE: There are still issues for cpuoffload: https://github.com/pytorch/pytorch/issues/91165
            self.model = FSDP(
                LoraModel(self.model, self.lora_config),
                auto_wrap_policy=auto_wrap_policy,
                sharding_strategy=sharding_strategy,
                cpu_offload=CPUOffload(offload_params=True),
                mixed_precision=mp_policy,
                device_id=torch.cuda.current_device(),
                # limit_all_gathers=True,
            )
            if args.pretrained_ckpt is not None:
                static = torch.load(args.pretrained_ckpt)
                self.model.load_state_dict(static, strict=False)
            self.model.config.use_cache = False
            if rank == 0:
                # print_layer_info(self.model)
                print_trainable_parameters(self.model)
            self.model_is_setup = True
        
        trainer = Trainer(
            lr=args.lr,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            accumulate_steps=args.accumulate_steps,
            out_dir=args.out_dir,
            device=None,
            print_every=args.print_every,
        )
        self.model.lm_head = CastOutputToFloat(self.model.lm_head)
        self.__cast_lora_to(torch.float32)
        trainer.train(self.model, train_loader, val_loader, True, rank)
        dist.barrier()
        cleanup()

    def tune(
        self,
        batch_size: int = 1,
        lr: float = 2e-4,
        num_epochs: int = 3,
        warmup_steps: Optional[int] = None,
        accumulate_steps: Optional[int] = 32,
        out_dir: str = "./output/",
        print_every: Optional[int] = None,
    ) -> None:
        """
        Note
        -----------
        if `warmup_steps` is None, will use 1/3 epoch to warmup by default
        if `accumulate_steps` is None, will use 1 as default
        """
        assert self.mode in ["8bit", "single_gpu"], "Not suit for FSDP, please specify `load_in_8bit` or `device`"
        assert self.dataloader is not None, "Please `load_data` first"

        if not self.model_is_setup:
            self.model = self.load_model(self.model_id)
            print(f"Processing Lora Model in {self.mode} mode...")
            self.model = LoraModel(self.model, self.lora_config)
            # print_layer_info(self.model)
            print_trainable_parameters(self.model)
            self.model_is_setup = True
        
        trainer = Trainer(
            lr,
            num_epochs,
            warmup_steps,
            accumulate_steps,
            out_dir,
            device=self.device,
            print_every=print_every,
        )
        print("Switch to training mode...")
        if self.load_in_8bit:
            self.__cast_small_to(torch.float32)
        self.model.lm_head = CastOutputToFloat(self.model.lm_head)
        self.__cast_lora_to(torch.float32)
        # turn on when infer
        self.model.config.use_cache = False
        if self.device is not None:
            self.model.to(self.device).train()
        else:
            self.model.train()
        
        train_dl, dev_dl = self.dataloader.train_dev_split(batch_size)
        print("Begining tunning")
        trainer.train(self.model, train_dl, dev_dl)

    def eval(self, quant_bit: Optional[int] = None):
        print("Switch to inference mode...")
        self.model.half()
        if self.load_in_8bit:
            self.__cast_lora_to(torch.float32)
        self.model.lora_config.inference_mode = True
        self.model.config.use_cache = True
        if quant_bit is not None and hasattr(self.model, "quantize"):
            self.model.quantize(quant_bit)
        # 8bit do not use device, device is None
        if self.device is not None:
            self.model.to(self.device).eval()
        else:
            self.model.eval()
    
    @property
    def max_input_length(self) -> int:
        return 2048
    
    @property
    def transformer_block(self) -> str:
        if self.model_name == "chatglm":
            return "GLMBlock"
        elif self.model_name == "llama":
            return "LlamaAttention"
    
    def get_generate_config(
        self,
        max_new_tokens: int,
        do_sample: bool,
        num_beams: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        logits_processor=None,
        stopping_criteria=None,
        **kwargs
    ) -> Dict[str, Any]:
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        
        gen_kwargs = {
            "max_length": max_new_tokens,
            "do_sample": do_sample,
            "num_beams": num_beams,
            "temperature": temperature,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            
            "logits_processor": logits_processor,
            "stopping_criteria": stopping_criteria,
            **kwargs
        }
        """
        # Glm GenerationConfig
        GenerationConfig {
            "_from_model_config": true,
            "bos_token_id": 130004,
            "eos_token_id": 130005,
            "pad_token_id": 3,
            "transformers_version": "4.28.1"
        }
        """
        return gen_kwargs
    
    @torch.no_grad()
    def generate(
        self,
        sents: Union[str, List[str]],
        max_new_tokens: int = 512,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 0.2,
        top_p: float = 0.7,
        repetition_penalty: float = 1.02,
        logits_processor=None,
        stopping_criteria=None,
        **kwargs
    ):
        prompt_len = self.max_input_length - max_new_tokens - 8
        if type(sents) == str:
            sents = [sents]
        sents = [v[-prompt_len:] for v in sents]
        inputs = tokenizer(sents, return_tensors="pt", padding=True)
        inputs = inputs.to(self.curr_device)
        gen_kwargs = self.get_generate_config(
            max_new_tokens, do_sample, num_beams, temperature, top_p, repetition_penalty,
            logits_processor, stopping_criteria, **kwargs,
        )
        outputs = self.model.generate(
            **inputs,
            **gen_kwargs,
        )
        batch_out_sents = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        res = []
        for i, out_s in enumerate(batch_out_sents):
            inp_s = sents[i]
            out_s = out_s.replace(inp_s, "")
            res.append(out_s)
        return res

    @torch.no_grad()
    def stream_chat(
        self,
        query: str,
        history: List[Tuple[str, str]] = None,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 0.95,
        top_p: float = 0.7,
        repetition_penalty: float = 1.02,
        logits_processor=None,
        stopping_criteria=None,
        **kwargs
    ):
        # remain some places for special tokens
        prompt_len = self.max_input_length - max_new_tokens - 8
        "From ChatGLM Model"
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n 问：{}\n 答：{}\n".format(
                    i, old_query, response)
            prompt += "[Round {}]\n 问：{}\n 答：".format(len(history), query)
        prompt = prompt[-prompt_len:]
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.curr_device)
        gen_kwargs = self.get_generate_config(
            max_new_tokens, do_sample, num_beams, temperature, top_p, repetition_penalty,
            logits_processor, stopping_criteria, **kwargs,
        )
        for outputs in self.model.stream_generate(
            **inputs, **gen_kwargs
        ):
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            response = self.tokenizer.decode(outputs)
            response = self.model.process_response(response)
            new_history = history + [(query, response)]
            yield response, new_history

    def chat(
        self,
        inp: str,
        history: List[Tuple[str, str]] = None,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 0.95,
        top_p: float = 0.7,
        repetition_penalty: float = 1.02,
        stop: List[str] = []
    ):
        if not history:
            history = []

        if stop:
            stop_tokens = [v for v in stop if v not in self.stop_tokens]
            custom_stop_tensor_list = create_token_tensor_list(
                self.tokenizer, stop_tokens)
            stop_tensor_list = self.builtin_stop_tensor_list + custom_stop_tensor_list
        else:
            stop_tensor_list = self.builtin_stop_tensor_list

        custom_stop_list = [
            CustomStoppingCriteria(
                stop_tensor_list,
                self.curr_device)]

        for response, history in self.stream_chat(
            inp, history, max_new_tokens,
            top_p=top_p, temperature=temperature,
            stopping_criteria=StoppingCriteriaList(custom_stop_list),
        ):
            ...
        return response, history

    @property
    def curr_device(self) -> torch.device:
        if self.device is None:
            return self.model.device
        return self.device