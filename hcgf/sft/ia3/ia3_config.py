from typing import List
from dataclasses import dataclass


@dataclass
class Ia3Config:

    target_modules: List[str]
    feedforward_modules: List[str]


class Ia3ConfigLoader:

    def __init__(self):
        ...
    
    @property
    def chatglm(self):
        return Ia3Config(
            [],
            ["dense_4h_to_h"],
        )
    
    @property
    def qwen(self):
        return Ia3Config(
            [],
            ["mlp.c_proj"],
        )

    @property
    def llama(self):
        return Ia3Config(
            ["k_proj", "v_proj", "down_proj"],
            ["down_proj"],
        )

    def get_config(self, model_name: str) -> Ia3Config:
        return getattr(self, model_name)