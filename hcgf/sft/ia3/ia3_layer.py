"""
Modified From peft
"""
from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class Ia3Layer:

    def reset_parameters(self):
        if hasattr(self, "ia3"):
            nn.init.constant_(self.ia3.weight, 1.0)
    
    def get_ia3_ind_mask(self, out_features: int, enable_ia3: List[bool]) -> Tuple:
        assert out_features % len(enable_ia3) == 0
        ia3_ind = self.weight.new_zeros((out_features,)).view(len(enable_ia3), -1)
        ia3_ind[enable_ia3, :] = 1
        ia3_ind = ia3_ind.view(-1)
        ia3_mask = 1 - ia3_ind
        return ia3_ind, ia3_mask


class Linear(nn.Linear, Ia3Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_feedforward: bool,
        bias: bool,
        enable_ia3: Optional[List[bool]] = None
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)

        self.enable_ia3 = enable_ia3
        self.is_feedforward = is_feedforward
        if self.is_feedforward:
            self.ia3 = nn.Linear(self.in_features, 1, bias=False)
        else:
            self.ia3 = nn.Linear(1, self.out_features, bias=False)
            if self.enable_ia3 is not None:
                self.ia3_ind, self.ia3_mask = self.get_ia3_ind_mask(
                    out_features, enable_ia3
                )
        
        self.weight.requires_grad = False
        nn.Linear.reset_parameters(self)
        self.reset_parameters()
        self.to(self.weight.device)

    def forward(self, x: torch.Tensor):
        previous_dtype = x.dtype
        ia3_scaling = self.ia3.weight.flatten()
        if self.is_feedforward:
            x = x.to(ia3_scaling.dtype)
            interm = x * ia3_scaling
            result = F.linear(
                interm.to(self.weight.dtype),
                self.weight,
                bias=self.bias,
            )
        else:
            print(1, ia3_scaling)
            if self.enable_ia3 is not None:
                ia3_scaling = (ia3_scaling * self.ia3_ind) + self.ia3_mask
            print(2, ia3_scaling)
            result = F.linear(x, self.weight, bias=self.bias)
            result = result.to(ia3_scaling.dtype) * ia3_scaling
        result = result.to(previous_dtype)
        return result