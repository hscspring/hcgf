"""
Modified From peft
"""
from typing import Optional

import torch.nn as nn
import torch.nn.functional as F

from .ia3_layer import Linear
from .ia3_config import Ia3Config
from ..base import BaseModel, BaseMixin


class Ia3Model(BaseModel, BaseMixin):
 
    def __init__(self, model: nn.Module, config: Ia3Config):
        super().__init__()
        self.model = model
        self.ia3_config = config
        self._find_and_replace()
        self.mark_only_x_as_trainable("ia3")
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        key_list = [key for key, _ in self.model.named_modules()]
        
        for key in key_list:
            target_module_found = any(
                self._is_valid_match(key, target_key) for target_key in self.ia3_config.target_modules
            )
            if not target_module_found:
                continue
            is_feedforward = any(
                key.endswith(target_key) for target_key in self.ia3_config.feedforward_modules
            )

            print(f"IA3 found module: ==> {key}, is_ffn: {is_feedforward}")
            parent, target, target_name = self._get_submodules(key)
            
            in_features, out_features = target.in_features, target.out_features
            bias = hasattr(target, "bias") and target.bias is not None

            if loaded_in_8bit:
                import bitsandbytes as bnb
                from .ia3_layer_8bit import Linear8bitLt
                eightbit_kwargs = {
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
                }
                new_module = Linear8bitLt(
                    in_features,
                    out_features,
                    is_feedforward,
                    bias,
                    self.ia3_config.enable_ia3,
                    **eightbit_kwargs,
                )
            else:
                new_module = Linear(
                    in_features, 
                    out_features, 
                    is_feedforward, 
                    bias, 
                    self.ia3_config.enable_ia3,
                )
            self._replace_module(parent, target_name, new_module, target)