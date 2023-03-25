from typing import Optional, List

import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from peft import get_peft_model, LoraConfig, TaskType

from ..utils import print_trainable_parameters
from ..dataloader import GlmDataLoader
from ..trainer import Trainer


from .chatglm import ChatGLMForConditionalGeneration, ChatGLMTokenizer


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
    ):
        print(f"Loading tokenizer and model of {model_id}")
        self.infer_mode = infer_mode
        self.load_in_8bit = load_in_8bit
        if self.load_in_8bit:
            import bitsandbytes as bnb
            self.device = None
            model = self._load_8bit_glm(model_id)
        else:
            self.device = device
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

    def __cast_to(self, model: nn.Module, dtype: torch.Type) -> nn.Module:
        for param in model.parameters():
            param.requires_grad = False
            if param.ndim == 1:
                # cast the small parameters (e.g. layernorm, bias) to fp32 for stability
                param.data = param.data.to(dtype)
        return model
    
    def _load_8bit_glm(self, model_id: str) -> PreTrainedModel:
        model = ChatGLMForConditionalGeneration.from_pretrained(
            model_id, load_in_8bit=True, device_map="auto")
        if self.infer_mode:
            dtype = torch.float16
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
        accumulate_steps: Optional[int] = 8,
        out_dir: str = "./output/",
        print_every: int = 10,
    ):
        """
        Note
        -----------
        if `warmup_steps` is None, will use one epoch to warmup by default
        if `accumulate_steps` is None, will use 1 as default
        """
        # turnoff when infer
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

        if self.device is not None:
            self.model.to(self.device).eval()
        else:
            self.model.eval()

    def chat(self, inp: str, history: List[str] = None, max_len: int = 128):
        if not history:
            history = []
        res = []
        for response, history in self.model.stream_chat(
            self.tokenizer, inp, history, max_len):
            query, response = history[-1]
            res.append(response)
        answer = "".join(response)
        return answer, history
