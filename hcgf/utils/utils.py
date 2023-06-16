from typing import Dict, List
from datetime import datetime

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


def get_lora_state_dict(
    model: nn.Module, 
    bias: str = "none"
) -> Dict[str, torch.Tensor]:
    # From loralib
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
    if "chatglm" in model_id:
        return LlmType.chatglm.val
    elif "llama" in model_id and "alpaca" in model_id:
        return LlmType.llama_alpaca.val
    elif "llama" in model_id and "ziya" in model_id:
        return LlmType.llama_ziya.val
    elif "llama" in model_id and "belle" in model_id:
        return LlmType.llama_belle.val
    elif "llama" in model_id:
        return LlmType.llama_native.val
    elif "gpt2" in model_id:
        return LlmType.gpt2.val
    elif "pangu" in model_id:
        return LlmType.pangu.val
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