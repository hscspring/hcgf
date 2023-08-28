"""
Modified From peft
"""
import torch
import bitsandbytes as bnb


from .ia3_layer import Ia3Layer


class Linear8bitLt(bnb.nn.Linear8bitLt, Ia3Layer):
        def __init__(
            self,
            in_features: int,
            out_features: int,
            is_feedforward: bool,
            bias: bool,
            **kwargs,
        ) -> None:
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=bias,
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
            )

            self.is_feedforward = is_feedforward
            if self.is_feedforward:
                self.ia3 = bnb.nn.Linear8bitLt(self.in_features, 1, bias=False)
            else:
                self.ia3 = bnb.nn.Linear8bitLt(1, self.out_features, bias=False)
            
            self.weight.requires_grad = False
            self.reset_parameters()
            self.to(self.weight.device)


        def forward(self, x: torch.Tensor) -> torch.Tensor:
            ia3_scaling = self.ia3.weight.flatten()
            if self.is_feedforward:
                result = super().forward(x * ia3_scaling)
            else:
                result = super().forward(x)
                result = result * ia3_scaling
            return result