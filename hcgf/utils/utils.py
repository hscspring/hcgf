from typing import Dict, List
from datetime import datetime
from numbers import Number

import torch
import torch.nn as nn
from transformers.tokenization_utils import PreTrainedTokenizer


from ..data_model import Tensor, LlmType


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
    msg = f"trainable params: {trainable_params} || all params: {all_param} || "
    msg += f"trainable%: {100 * trainable_params / all_param}"
    print(msg)


def print_layer_info(model: nn.Module) -> None:
    for key, val in model.named_parameters():
        msg = "\t".join(map(str, (key, val.shape, val.dtype, val.device, val.requires_grad)))
        print(msg)


def get_x_state_dict(
    model: nn.Module, 
    x: str,
) -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    return {k: my_state_dict[k] for k in my_state_dict if x in k}

def create_token_tensor_list(
    tokenizer: PreTrainedTokenizer, 
    tokens: List[str]
) -> List[Tensor["L", torch.LongTensor]]:
    tensor_list = []
    for token in tokens:
        tids = tokenizer(
            token, add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        tensor_list.append(tids)
    return tensor_list


def get_module_class_from_name(module: nn.Module, name: str) -> nn.Module:
    # From HuggingFace
    """
    Gets a class from a module by its name.
    Args:
        module (`torch.nn.Module`): The module to get the class from.
        name (`str`): The name of the class.
    """
    modules_children = list(module.children())
    if module.__class__.__name__ == name:
        return module.__class__
    elif len(modules_children) == 0:
        return
    else:
        for child_module in modules_children:
            module_class = get_module_class_from_name(child_module, name)
            if module_class is not None:
                return module_class


def get_date_of_run():
    # From FSDP_adavnced_tutorial
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    print(f"\n--> current date and time of run = {date_of_run}")
    return date_of_run


def format_metrics_to_gb(item):
    # From FSDP_adavnced_tutorial
    """quick function to format numbers to gigabyte and round to 4 digit precision"""
    g_gigabyte = 1024**3
    metric_num = item / g_gigabyte
    metric_num = round(metric_num, ndigits=4)
    return metric_num


def get_model_type_from(model_id: str) -> LlmType:
    model_id = model_id.lower()
    if "qwen" in model_id:
        mt = LlmType.qwen.val
    elif "chatglm2" in model_id:
        mt = LlmType.chatglm2.val
    elif "chatglm" in model_id:
        mt = LlmType.chatglm.val
    
    elif "llama" in model_id and "alpaca" in model_id:
        mt = LlmType.llama_alpaca.val
        mt.alias = "alpaca"
    elif "llama" in model_id and "ziya" in model_id:
        mt = LlmType.llama_ziya.val
        mt.alias = "ziya"
    elif "llama" in model_id and "belle" in model_id:
        mt = LlmType.llama_belle.val
        mt.alias = "belle"
    elif "chatflow" in model_id:
        mt = LlmType.llama_linly.val
        mt.alias = "linly"
    elif "llama" in model_id:
        mt = LlmType.llama_native.val
    
    elif "gpt2" in model_id:
        mt = LlmType.gpt2.val
    elif "pangu" in model_id:
        mt = LlmType.pangu.val
    
    # after llama
    elif "belle" in model_id:
        return LlmType.bloom.val
    elif "bloomz" in model_id:
        return LlmType.bloom.val
    elif "baichuan" in model_id:
        return LlmType.baichuan.val
    else:
        msg = f"Unsupported model: {model_id}, only support chatglm or llama. "
        msg += "Your input must contain either of them"
        raise ValueError(msg)
    return mt



def get_optim_parameters(
    model: nn.Module,
    weight_decay: float,
    no_decay_name_list=["bias", "layer_norm", "layernorm", "ln_"]
):
    ps = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (p.ndim != 1 and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (p.ndim == 1 and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return ps


def format_mem_to(num: Number, level: str = "g"):
    if level == "g":
        unit = 3
    else:
        unit == 1
    byte_unit = 1024 ** unit
    return round(num / byte_unit, ndigits=4)