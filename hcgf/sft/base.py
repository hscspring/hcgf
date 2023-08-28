from typing import Tuple

import torch.nn as nn


class BaseModel(nn.Module):

    def __init__(self):
        super().__init__()

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class BaseMixin:

    def mark_only_x_as_trainable(self, x: str, bias: str = "none") -> None:
        for n, p in self.model.named_parameters():
            if x not in n:
                p.requires_grad = False
    
    def _get_submodules(self, key: str) -> Tuple[nn.Module, nn.Module, str]:
        tmp = key.split(".")
        parent = self.model.get_submodule(".".join(tmp[:-1]))
        target_name = tmp[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(
        self, 
        parent_module: nn.Module, 
        child_name: str, 
        new_module: nn.Module,
        old_module: nn.Module,
    ):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

    def _is_valid_match(self, key: str, target_key: str):
        if key.endswith(target_key):
            if len(key) > len(target_key):
                return key.endswith("." + target_key)  # must be a sub module
            return True
        return False