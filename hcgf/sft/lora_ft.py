from typing import Optional, List, Tuple
import copy

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import LogitsProcessorList, StoppingCriteriaList, StoppingCriteria
from peft import get_peft_model, LoraConfig, TaskType

from ..utils import print_trainable_parameters, create_token_tensor_list
from ..dataloader import GlmDataLoader
from ..data_model import Tensor
from ..trainer import Trainer


from .chatglm import ChatGLMForConditionalGeneration, ChatGLMTokenizer, InvalidScoreLogitsProcessor


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
        infer_mode: bool = False,
        cast_small_parameters: bool = True,
    ):
        print(f"Loading tokenizer and model of {model_id}")
        self.infer_mode = infer_mode
        self.load_in_8bit = load_in_8bit
        self.cast_small_parameters = cast_small_parameters
        if self.load_in_8bit:
            import bitsandbytes as bnb
            self.device = None
            model = self._load_8bit_glm(model_id)
        else:
            self.device = device or "cuda"
            model = self._load_glm(model_id)
        self.tokenizer = ChatGLMTokenizer.from_pretrained(model_id)
        self.config = LoraConfig(
            # peft config
            peft_type="LORA",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=infer_mode,
            # lora config
            target_modules=["query_key_value"],
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            enable_lora=[True, False, True],
            bias="none",
        )
        print("Processing peft model")
        self.model = get_peft_model(model, self.config)
        print_trainable_parameters(self.model)

        self.stop_tokens = ["问题："]
        self.builtin_stop_tensor_list = self.init_stop_tensor_list()

    def __cast_to(self, model: nn.Module, dtype: torch.Type) -> nn.Module:
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1 and param.dtype != dtype:
                # cast the small parameters (e.g. layernorm, bias) to fp32 for stability
                param.data = param.data.to(dtype)
        return model

    def _load_8bit_glm(self, model_id: str) -> PreTrainedModel:
        model = ChatGLMForConditionalGeneration.from_pretrained(
            model_id, load_in_8bit=True, device_map="auto")
        if self.cast_small_parameters:
            if self.infer_mode:
                dtype = torch.float16
                model = self.__cast_to(model, dtype)
            else:
                dtype = torch.float32
                model = self.__cast_to(model, dtype)
                model.lm_head = CastOutputToFloat(model.lm_head)
        return model

    def _load_glm(self, model_id: str) -> PreTrainedModel:
        model = ChatGLMForConditionalGeneration.from_pretrained(
            model_id).to(self.device)
        for param in model.parameters():
            param.requires_grad = False
        return model

    def load_pretrained(self, pt_path: str):
        static = torch.load(pt_path)
        self.model.load_state_dict(static, strict=False)
        return self

    def load_data(self, data_path: str, max_seq_len: int = 512):
        print("Loading data")
        self.dataloader = GlmDataLoader(data_path, self.tokenizer, max_seq_len)
        return self

    def tune(
        self,
        batch_size: int = 1,
        lr: float = 2e-4,
        num_epochs: int = 10,
        warmup_steps: Optional[int] = None,
        accumulate_steps: Optional[int] = 32,
        out_dir: str = "./output/",
        print_every: int = 10,
    ):
        """
        Note
        -----------
        if `warmup_steps` is None, will use one epoch to warmup by default
        if `accumulate_steps` is None, will use 1 as default
        """
        # turn on when infer
        self.model.config.use_cache = False
        if self.device is not None:
            self.model.to(self.device).train()
        else:
            self.model.train()
        trainer = Trainer(
            lr,
            num_epochs,
            warmup_steps,
            accumulate_steps,
            out_dir,
            device=self.device,
            print_every=print_every,
        )
        train_dl, dev_dl = self.dataloader.train_dev_split(batch_size)
        print("Begining tunning")
        trainer.train(self.model, train_dl, dev_dl)

    def eval(self, quant_bit: Optional[int] = None):
        self.model.config.use_cache = True

        # 8bit do not use half, just keep its type
        if not self.load_in_8bit:
            self.model.half()

        if quant_bit is not None:
            self.model.quantize(quant_bit)

        # 8bit do not use device, device is None
        if self.device is not None:
            self.model.to(self.device).eval()
        else:
            self.model.eval()
    
    @torch.no_grad()
    def stream_chat(
        self, 
        query: str, 
        history: List[Tuple[str, str]] = None, 
        max_length: int = 2048,
        do_sample=True, 
        top_p=0.7, 
        temperature=0.95, 
        logits_processor=None, 
        stopping_criteria=None,
        **kwargs
    ):
        "From ChatGLM Model"
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        if stopping_criteria is None:
            stopping_criteria = StoppingCriteriaList()
        gen_kwargs = {
            "max_length": max_length, 
            "do_sample": do_sample, 
            "top_p": top_p,
            "temperature": temperature, 
            "logits_processor": logits_processor,
            "stopping_criteria": stopping_criteria,
            **kwargs
        }
        if not history:
            prompt = query
        else:
            prompt = ""
            for i, (old_query, response) in enumerate(history):
                prompt += "[Round {}]\n 问：{}\n 答：{}\n".format(i, old_query, response)
            prompt += "[Round {}]\n 问：{}\n 答：".format(len(history), query)
        inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs = inputs.to(self.model.device)
        for outputs in self.model.stream_generate(
            inputs["input_ids"], **gen_kwargs
        ):
            outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):]
            response = self.tokenizer.decode(outputs)
            response = self.model.process_response(response)
            new_history = history + [(query, response)]
            yield response, new_history

    def chat(
        self, 
        inp: str, 
        history: List[str] = None, 
        max_len: int = 512, 
        stop: List[str] = []
    ):
        if not history:
            history = []
        
        if stop:
            stop_tokens = [v for v in stop if v not in self.stop_tokens]
            custom_stop_tensor_list = create_token_tensor_list(self.tokenizer, stop_tokens)
            stop_tensor_list = self.builtin_stop_tensor_list + custom_stop_tensor_list
        else:
            stop_tensor_list = self.builtin_stop_tensor_list

        custom_stop_list = [CustomStoppingCriteria(stop_tensor_list, self.model.device)]
        
        for response, history in self.stream_chat(
            inp, history, max_len,
            stopping_criteria=StoppingCriteriaList(custom_stop_list),
        ):
            ...
        return response, history
    
    def init_stop_tensor_list(self) -> List[Tensor["L", torch.LongTensor]]:
        init_tensor_list = [
            torch.LongTensor([20002]),  # eos
            torch.LongTensor([150005]), # eop
        ]
        new_add = create_token_tensor_list(self.tokenizer, self.stop_tokens)
        return init_tensor_list + new_add