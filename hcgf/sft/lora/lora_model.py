"""
Modified From peft v0.2.0
"""
from typing import Optional
import re

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from .lora_layer import Linear, MergedLinear
from .lora_config import LoraConfig
from ..base import BaseModel, BaseMixin


class LoraModel(BaseModel, BaseMixin):
    def __init__(self, model: nn.Module, config: LoraConfig):
        super().__init__()
        self.lora_config = config
        self.model = model
        self._find_and_replace()
        self.mark_only_x_as_trainable("lora_", self.lora_config.bias)
        self.forward = self.model.forward

    def _find_and_replace(self):
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)
        kwargs = {
            "r": self.lora_config.r,
            "lora_alpha": self.lora_config.lora_alpha,
            "lora_dropout": self.lora_config.lora_dropout,
            "fan_in_fan_out": self.lora_config.fan_in_fan_out,
            "merge_weights": self.lora_config.merge_weights or self.lora_config.inference_mode,
        }
        key_list = [key for key, _ in self.model.named_modules()]
        for _j, key in enumerate(key_list):
            if isinstance(self.lora_config.target_modules, str):
                target_module_found = re.fullmatch(self.lora_config.target_modules, key)
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.lora_config.target_modules)

            if not target_module_found:
                continue
            print(f"Lora found module: ==> {key}")
            
            parent, target, target_name = self._get_submodules(key)
            bias = target.bias is not None
            
            if loaded_in_8bit:
                import bitsandbytes as bnb
                from .lora_layer_8bit import Linear8bitLt, MergedLinear8bitLt
                if isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                if self.lora_config.enable_lora is None:
                    new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                else:
                    kwargs.update({"enable_lora": self.lora_config.enable_lora})
                    new_module = MergedLinear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
            
            elif isinstance(target, nn.Linear) and self.lora_config.enable_lora is None:
                new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)
            
            elif self.lora_config.enable_lora is not None:
                kwargs.update({"enable_lora": self.lora_config.enable_lora})
                if isinstance(target, Conv1D):
                    in_features, out_features = (
                        target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                    )
                else:
                    in_features, out_features = target.in_features, target.out_features
                    if kwargs["fan_in_fan_out"]:
                        kwargs["fan_in_fan_out"] = False
                new_module = MergedLinear(in_features, out_features, bias=bias, **kwargs)
            
            self._replace_module(parent, target_name, new_module, target)