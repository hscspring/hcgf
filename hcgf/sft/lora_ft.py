from typing import Optional, List

import torch
from peft import get_peft_model, LoraConfig, TaskType

from ..utils import print_trainable_parameters
from ..dataloader import GlmDataLoader
from ..trainer import Trainer


from .chatglm import ChatGLMForConditionalGeneration, ChatGLMTokenizer


class GlmLora:

    def __init__(
        self,
        model_id: str,
        device: str,
        lora_r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        print(f"Loading tokenizer and model of {model_id}")
        self.tokenizer = ChatGLMTokenizer.from_pretrained(model_id)
        model = ChatGLMForConditionalGeneration.from_pretrained(
            model_id).to(device)
        self.config = LoraConfig(
            # peft config
            peft_type="LORA",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            # lora config
            target_modules=["query_key_value"],
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            enable_lora=[True, False, True],
            bias="none",
        )
        self.device = device
        print("Processing peft model")
        self.model = get_peft_model(model, self.config)
        print_trainable_parameters(self.model)

    def load_pretrained(self, pt_path: str):
        static = torch.load(pt_path)
        self.model.load_state_dict(static, strict=False)
        return self

    def load_data(self, data_path: str):
        print("Loading data")
        self.dataloader = GlmDataLoader(data_path, self.tokenizer)
        return self

    def tune(
        self,
        device: str,
        batch_size: int = 1,
        lr: float = 2e-4,
        num_epochs: int = 10,
        warmup_steps: Optional[int] = None,
        accumulate_steps: Optional[int] = 32,
        out_dir: str = "./output/",
    ):
        """
        Note
        -----------
        if `warmup_steps` is None, will use one epoch to warmup by default
        if `accumulate_steps` is None, will use 1 as default
        """
        self.model.to(self.device).train()
        trainer = Trainer(
            lr,
            num_epochs,
            warmup_steps,
            accumulate_steps,
            out_dir,
            device=self.device
        )
        train_dl, dev_dl = self.dataloader.train_dev_split(batch_size)
        print("Begining tunning")
        trainer.train(self.model, train_dl, dev_dl)

    def eval(self, quant_bit: Optional[int] = None):
        if quant_bit is not None:
            self.model.half().quantize(quant_bit).to(self.device).eval()
        else:
            self.model.half().to(self.device).eval()

    def chat(self, inp: str, history: List[str] = None):
        if not history:
            history = []
        response, history = self.model.chat(self.tokenizer, inp, history)
        return response, history
