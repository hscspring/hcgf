from typing import List, Optional
from dataclasses import dataclass


@dataclass
class Ia3Config:

    target_modules: List[str]
    feedforward_modules: List[str]
    enable_ia3: Optional[List[bool]] = None


class Ia3ConfigLoader:

    def __init__(self):
        ...
    
    @property
    def chatglm(self):
        return Ia3Config(
            ["query_key_value", "dense_4h_to_h"],
            ["dense_4h_to_h"],
            [False, True, True],
        )
    
    @property
    def chatglm2(self):
        return self.chatglm
    
    @property
    def qwen(self):
        return Ia3Config(
            ["c_attn", "mlp.c_proj"],
            ["mlp.c_proj"],
            [False, True, True],
        )

    @property
    def llama(self):
        return Ia3Config(
            ["k_proj", "v_proj", "down_proj"],
            ["down_proj"],
        )

    def get_config(self, model_name: str) -> Ia3Config:
        return getattr(self, model_name)