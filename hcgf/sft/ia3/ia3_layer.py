"""
Modified From peft
"""
import torch
import torch.nn as nn


class Ia3Layer:

    def reset_parameters(self):
        if hasattr(self, "ia3"):
            nn.init.constant_(self.ia3.weight, 1.0)


class Linear(nn.Linear, Ia3Layer):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        is_feedforward: bool,
        bias: bool,
        **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, bias=bias, **kwargs)
        
        self.is_feedforward = is_feedforward
        if self.is_feedforward:
            self.ia3 = nn.Linear(self.in_features, 1, bias=False)
        else:
            self.ia3 = nn.Linear(1, self.out_features, bias=False)
        
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
            result = F.linear(x, self.weight, bias=self.bias)
            result = result.to(ia3_scaling.dtype) * ia3_scaling
        result = result.to(previous_dtype)
        return result