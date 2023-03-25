from typing import Dict

import torch
import torch.nn as nn


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    msg = f"trainable params: {trainable_params} || all params: {all_param} "
    msg += f"\ntrainable%: {100 * trainable_params / all_param}"
    print(msg)


def print_layer_info(model: nn.Module) -> None:
    for key, val in model.named_parameters():
        print(key, val.dtype, val.device)


def get_lora_state_dict(
        model: nn.Module, bias: str = "none") -> Dict[str, torch.Tensor]:
    """
    From https://github.com/microsoft/LoRA/
    """
    my_state_dict = model.state_dict()
    if bias == "none":
        return {k: my_state_dict[k] for k in my_state_dict if "lora_" in k}
    elif bias == "all":
        return {k: my_state_dict[k]
                for k in my_state_dict if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        for k in my_state_dict:
            if "lora_" in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split("lora_")[0] + "bias"
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
