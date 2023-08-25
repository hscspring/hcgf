from typing import List
from dataclasses import dataclass, field


@dataclass
class LoraConfig:
    target_modules: List[str] = field(default_factory=lambda: ["query_key_value"])
    r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    merge_weights: bool = False
    fan_in_fan_out: bool = False
    enable_lora: List[bool] = field(default_factory=lambda: [True, False, True])
    bias: str = "none"
    inference_mode: bool = False


class LoraConfigLoader:

    def __init__(self, lora_r: int, lora_alpha: int, lora_dropout: float):
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
    
    @property
    def chatglm(self):
        return LoraConfig(
            target_modules=["query_key_value"],
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            enable_lora=[True, False, True],
            bias="none"
        )

    @property
    def chatglm2(self):
        return self.chatglm
    
    @property
    def qwen(self):
        return LoraConfig(
            target_modules=["c_attn"],
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            enable_lora=[True, False, True],
            bias="none"
        )
    
    def get_base_gpt_config(self):
        return LoraConfig(
            target_modules=["q_proj", "v_proj"],
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            enable_lora=None,
            bias="none"
        )
    
    @property
    def gpt2(self):
        return LoraConfig(
            target_modules=["c_attn"],
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            enable_lora=[True, False, True],
            bias="none"
        )
    
    @property
    def pangu(self):
        return self.get_base_gpt_config()

    @property
    def llama(self):
        return self.get_base_gpt_config()
    
    @property
    def baichuan(self):
        return LoraConfig(
            target_modules=["W_pack"],
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            enable_lora=[True, False, True],
            bias="none"
        )

    @property
    def bloom(self):
        return self.chatglm

    def get_config(self, model_name: str) -> LoraConfig:
        return getattr(self, model_name)