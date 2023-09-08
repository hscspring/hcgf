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

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.generation.utils import (
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    StoppingCriteria
)
from transformers.generation.logits_process import LogitsProcessor

from .lora import LoraModel, LoraConfigLoader
from .ia3 import Ia3Model, Ia3ConfigLoader
from ..dataloader import GlmDataLoader
from ..data_model import Tensor
from ..trainer import Trainer
from ..trainer.fsdp import (
    setup, 
    cleanup,
    get_transformer_wrap_policy, 
    get_sharding_strategy,
    check_bf16_ready,
    get_mp_policy, 
    apply_fsdp_checkpointing,
)
from ..utils import (
    print_layer_info, print_trainable_parameters, 
    create_token_tensor_list, get_model_type_from
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




class GlmBase:

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        torch_dtype: Optional[torch.Type] = None,
    ):
        world_size = torch.cuda.device_count()
        self.model_id = model_id
        self.llm_type = get_model_type_from(model_id)

        if check_bf16_ready():
            self.torch_dtype = torch.bfloat16
        else:
            self.torch_dtype = torch.float16
        
        if torch_dtype is not None:
            self.torch_dtype = torch_dtype

        print("llm type: ", self.llm_type.value)
        print("llm torch dtype: ", self.torch_dtype)
        
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
        self.model_is_setup = False

        self.dataloader = None

        self.stop_tokens = []
        if self.llm_type.value == "chatglm":
            self.stop_tokens.append("问题：")
        self.builtin_stop_tensor_list = self._init_stop_tensor_list(self.stop_tokens)
    
    def _init_stop_tensor_list(
        self, token_list: List[str]
    ) -> List[Tensor["L", torch.LongTensor]]:
        res = []
        for tensor in create_token_tensor_list(
            self.tokenizer, token_list
        ):
            if self.llm_type.value == "chatglm":
                tensor = tensor[1:]
            res.append(tensor)
        return res
    
    def _cast_small_to(self, dtype: torch.Type) -> None:
        for name, param in self.model.named_parameters():
            if param.ndim == 1 and param.dtype != dtype:
                param.data = param.data.to(dtype)
    
    def _cast_ln_to(self, dtype: torch.Type) -> None:
        for name, param in self.model.named_parameters():
            if "bias" not in name and param.ndim == 1 and param.dtype != dtype:
                param.data = param.data.to(dtype)

    def _cast_x_to(self, x: str, dtype: torch.Type) -> None:
        for name, param in self.model.named_parameters():
            if x in name:
                param.data = param.data.to(dtype)
    
    def _cast_lmhead(self):
        if hasattr(self.model, "lm_head"):
            self.model.lm_head = CastOutputToFloat(self.model.lm_head)
        # chatglm
        elif hasattr(self.model, "output_layer"):
            self.model.output_layer = CastOutputToFloat(self.model.output_layer)
    
    def _load_pretrained_x(self, pt_path: str):
        static = torch.load(pt_path)
        self.model.load_state_dict(static, strict=False)
    
    def _load_auto_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
    
    def load_tokenizer(self) -> PreTrainedTokenizer:
        default_pad_id = 0
        if self.llm_type.value == "llama":
            from transformers import LlamaTokenizer
            tk = LlamaTokenizer.from_pretrained(self.model_id)
            tk.bos_token_id = 1
            tk.eos_token_id = 2
        elif self.llm_type.value == "gpt2":
            from transformers import GPT2Tokenizer
            tk = GPT2Tokenizer.from_pretrained(self.model_id)
        elif self.llm_type.value in ["chatglm", "bloom"]:
            tk = self._load_auto_tokenizer()
            default_pad_id = 3
        elif self.llm_type.value in ["chatglm2"]:
            tk = self._load_auto_tokenizer()
            tk.bos_token_id = 1
        elif self.llm_type.value == "pangu":
            tk = self._load_auto_tokenizer()
            default_pad_id = 6
            tk.unk_token_id = 0
            tk.bos_token_id = 9
            tk.eos_token_id = 9
        elif self.llm_type.value in ["baichuan"]:
            tk = self._load_auto_tokenizer()
        elif self.llm_type.value in ["qwen"]:
            tk = self._load_auto_tokenizer()
            default_pad_id = tk.eod_id = 151643
            tk.eos_token_id = tk.eod_id = 151643
            tk.bos_token_id = tk.eod_id = 151643
        else:
            raise NotImplemented
        
        pad_id = getattr(tk, "pad_token_id")
        if pad_id is None:
            tk.pad_token_id = default_pad_id
        
        # donot use in train, only for inference
        tk.padding_side = "left"
        tk.max_model_input_len = self.max_input_length
        tk.model_name = self.llm_type.value
        tk.model_alias = self.llm_type.alias
        return tk
    
    def load_model(self, model_id: Optional[str] = None) -> PreTrainedModel:
        if model_id is None:
            model_id = self.model_id
        
        if self.load_in_8bit:
            device_map = "auto"
        else:
            device_map = None
        
        trust_remote_code = False

        if self.llm_type.value == "llama":
            from transformers import LlamaForCausalLM
            ModelCls = LlamaForCausalLM
        elif self.llm_type.value == "gpt2":
            from transformers import GPT2LMHeadModel
            ModelCls = GPT2LMHeadModel
        elif self.llm_type.value in ["chatglm", "chatglm2"]:
            ModelCls = AutoModel
            trust_remote_code = True
        elif self.llm_type.value in ["pangu", "baichuan", "bloom", "qwen"]:
            ModelCls = AutoModelForCausalLM
            trust_remote_code = True
        else:
            raise NotImplemented
        
        model = ModelCls.from_pretrained(
            model_id, 
            load_in_8bit=self.load_in_8bit,
            torch_dtype=self.torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        
        model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # if not self.load_in_8bit:
        #     model.half()
        if self.device:
            model.to(self.device)
        print(f"Model loaded, config: {model.config}")
        return model

    def load_data(self, data_path: str, max_seq_len: int = 512) -> "Self":
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
        
        # TODO: support bfloat16?
        self.torch_dtype = torch.float16
        
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
            self.load_pretrained()
        
        self._cast_x_to(self.task_type, torch.float32)
        self._cast_lmhead()
        self.model.config.use_cache = False

        # NOTE: ChatGLM use mixed float, FSDP needs all parameters in a shard to be the same
        setup(rank, world_size)
        train_loader, val_loader = self.dataloader.train_dev_split(
            args.batch_size, True, rank, train_include_dev=False, shuffle_train=True, dev_size=0.1
        )
        auto_wrap_policy = get_transformer_wrap_policy(self.model, self.transformer_block)
        sharding_strategy = get_sharding_strategy(strategy)
        mp_policy = get_mp_policy()
        torch.cuda.set_device(rank)
        if rank == 0:
            print(f"mp policy: {mp_policy}")
        # NOTE: There are still issues for cpuoffload: https://github.com/pytorch/pytorch/issues/91165
        self.model = FSDP(
            self.model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=sharding_strategy,
            cpu_offload=CPUOffload(offload_params=False),
            mixed_precision=mp_policy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
        )
        # NOTE: ?There are still issues for ckpt+FSDP: https://github.com/pytorch/pytorch/issues/82203
        if args.task_type == "sft":
            apply_fsdp_checkpointing(self.model, self.transformer_block)

        if rank == 0:
            print_layer_info(self.model)
            print_trainable_parameters(self.model)
        
        trainer = Trainer(
            lr=args.lr,
            num_epochs=args.num_epochs,
            warmup_steps=args.warmup_steps,
            accumulate_steps=args.accumulate_steps,
            out_dir=args.out_dir,
            device=None,
            print_every=args.print_every,
            task_type=args.task_type,
            adam_betas=tuple(args.adam_betas),
            weight_decay=args.weight_decay,
            torch_dtype=self.torch_dtype,
        )
        if rank == 0:
            print("Begining tunning")
        trainer.train(self.model, train_loader, val_loader, True, rank)
        dist.barrier()
        cleanup()

    def tune(
        self,
        batch_size: int = 1,
        lr: float = 1e-4,
        num_epochs: int = 3,
        warmup_steps: Optional[int] = None,
        accumulate_steps: Optional[int] = 32,
        out_dir: str = "./output/",
        print_every: Optional[int] = None,
        task_type: str = "lora",
        adam_betas: tuple = (0.9, 0.95),
        weight_decay: float = 0.01,
    ) -> None:
        """
        Note
        -----------
        if `warmup_steps` is None, will use 1/3 epoch to warmup by default
        if `accumulate_steps` is None, will use 1 as default
        """
        assert self.mode in ["8bit", "single_gpu"], "Not suit for FSDP, please specify `load_in_8bit` or `device`"
        assert self.dataloader is not None, "Please `load_data` first"
        print(f"Switch to training mode, device: {self.device}")

        if not self.model_is_setup:
            self.load_pretrained()
        
        self._cast_x_to(self.task_type, torch.float32)
        if self.load_in_8bit:
            # 8bit时cast，float16不cast？
            # from https://colab.research.google.com/drive/1jCkpikz0J2o20FBQmYmAGdiKmJGOMo-o?usp=sharing
            self._cast_small_to(torch.float32)
        self._cast_lmhead()
        
        print_trainable_parameters(self.model)

        if task_type == "sft":
            self.model.gradient_checkpointing_enable()

        # turn on when infer
        self.model.config.use_cache = False
        if self.device is not None:
            self.model.to(self.device).train()
        else:
            self.model.train()
        
        train_dl, dev_dl = self.dataloader.train_dev_split(
            batch_size, train_include_dev=False, shuffle_train=True, dev_size=0.1
        )
        print("Begining tunning")
        trainer = Trainer(
            lr,
            num_epochs,
            warmup_steps,
            accumulate_steps,
            out_dir,
            device=self.device,
            print_every=print_every,
            task_type=task_type,
            adam_betas=tuple(adam_betas),
            weight_decay=weight_decay,
            torch_dtype=self.torch_dtype,
        )
        trainer.train(self.model, train_dl, dev_dl)

    def eval(self, quant_bit: Optional[int] = None):
        print("Switch to inference mode...")

        # use float16 for infer mode
        self.model.half()

        if self.load_in_8bit:
            self._cast_x_to(self.task_type, torch.float32)

        if hasattr(self.model, "lora_config"):
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
        if self.llm_type.value in ["chatglm", "chatglm2"]:
            return "GLMBlock"
        elif self.llm_type.value == "qwen":
            return "QWenBlock"
        elif self.llm_type.value == "llama":
            return "LlamaDecoderLayer"
        elif self.llm_type.value == "gpt2":
            return "GPT2Block"
        elif self.llm_type.value == "pangu":
            return "GPTPanguBlock"
        elif self.llm_type.value == "bloom":
            return "BloomBlock"
        elif self.llm_type.value == "baichuan":
            return "DecoderLayer"
        else:
            raise NotImplemented
    
    def get_generate_config(
        self,
        max_new_tokens: int,
        do_sample: bool,
        num_beams: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        logits_processor=None,
        stopping_criteria=None,
        **kwargs
    ) -> Dict[str, Any]:
        
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        
        gen_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
        )
        
        gen_kwargs = {
            "generation_config": gen_config,
            "logits_processor": logits_processor,
            "stopping_criteria": stopping_criteria,
            **kwargs
        }
        return gen_kwargs
    
    def generate(
        self,
        sents: Union[str, List[str]],
        max_new_tokens: int,
        do_sample: bool,
        num_beams: int,
        temperature: float,
        top_p: float,
        top_k: int,
        repetition_penalty: float,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        logits_processor=None,
        stopping_criteria=None,
        **kwargs
    ):
        if type(sents) == str:
            sents = [sents]
        inputs = self.tokenizer(
            sents, return_tensors="pt", padding=True
        )
        inputs = inputs.to(self.curr_device)
        gen_kwargs = self.get_generate_config(
            max_new_tokens, do_sample, num_beams, 
            temperature, top_p, top_k, repetition_penalty,
            pad_token_id, bos_token_id, eos_token_id,
            logits_processor, stopping_criteria, 
            **kwargs,
        )
        with torch.cuda.amp.autocast(dtype=self.torch_dtype):
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    **gen_kwargs,
                )
        batch_out_sents = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        res = []
        for i, out_s in enumerate(batch_out_sents):
            inp_s = sents[i]
            out_s = out_s.replace(inp_s, "")
            res.append(out_s)
        return res

    def stream_chat(
        self,
        query: str,
        history: List[Tuple[str, str]] = None,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 0.95,
        top_p: float = 0.7,
        top_k: int = 50,
        repetition_penalty: float = 1.02,
        logits_processor=None,
        stopping_criteria=None,
        **kwargs
    ):
        assert self.llm_type.value == "chatglm", "only support `chatglm`"
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
        pad_token_id = self.tokenizer.pad_token_id
        bos_token_id = self.tokenizer.bos_token_id
        eos_token_id = self.tokenizer.eos_token_id
        gen_kwargs = self.get_generate_config(
            max_new_tokens, do_sample, num_beams, 
            temperature, top_p, top_k, repetition_penalty,
            pad_token_id, bos_token_id, eos_token_id,
            logits_processor, stopping_criteria, **kwargs,
        )
        with torch.cuda.amp.autocast(dtype=self.torch_dtype):
            with torch.no_grad():
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
        assert self.llm_type.value == "chatglm", "only support `chatglm`"
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

        response = ""
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
    

class GlmLora(GlmBase):

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        load_in_8bit: bool = False,
        torch_dtype: torch.Type = None,
    ):
        super().__init__(model_id, device, load_in_8bit, torch_dtype)
        self.task_type = "lora"
        self.lora_config = LoraConfigLoader(
            lora_r, lora_alpha, lora_dropout
        ).get_config(self.llm_type.value)
    
    def load_pretrained(self, pt_path: Optional[str] = None) -> "Self":
        if not self.model_is_setup:
            self.model = self.load_model(self.model_id)
            self.model = LoraModel(self.model, self.lora_config)
            self.model_is_setup = True
            if pt_path is not None:
                self._load_pretrained_x(pt_path)
        return self


class GlmIa3(GlmBase):

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        torch_dtype: torch.Type = None,
    ):
        super().__init__(model_id, device, load_in_8bit, torch_dtype)
        self.task_type = "ia3"
        self.ia3_config = Ia3ConfigLoader().get_config(self.llm_type.value)
    
    def load_pretrained(self, pt_path: Optional[str] = None) -> "Self":
        if not self.model_is_setup:
            self.model = self.load_model(self.model_id)
            self.model = Ia3Model(self.model, self.ia3_config)
            self.model_is_setup = True
            if pt_path is not None:
                self._load_pretrained_x(pt_path)
        return self


class GlmSft(GlmBase):

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        torch_dtype: torch.Type = None,
    ):
        super().__init__(model_id, device, load_in_8bit, torch_dtype)
        self.torch_dtype = torch.bfloat16
    
    def load_pretrained(self, pt_path: Optional[str] = None) -> "Self":
        if not self.model_is_setup:
            self.model = self.load_model(self.model_id)
            if pt_path is not None:
                static = torch.load(pt_path)
                self.model.load_state_dict(static)
            self.model_is_setup = True
        return self
