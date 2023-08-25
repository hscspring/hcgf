from dataclasses import dataclass, asdict
from enum import Enum
from typing import NewType, Tuple, TypedDict, Optional, List, TypeVar, Generic, Any
from typing_extensions import Annotated

import torch



Shape = TypeVar("Shape")
DType = TypeVar("DType", torch.IntType, torch.FloatType)

class Tensor(torch.Tensor, Generic[Shape, DType]):
    ...


Batch = NewType("Batch", int)
SeqLen = NewType("SeqLen", int)
Hidden = NewType("Hidden", int)

ND = NewType("ND", int)

GlmBatchInput = TypedDict(
    "GlmBatchInput",
    input_ids=Annotated[torch.LongTensor, Tuple[Batch, SeqLen]],
    position_ids=Optional[Annotated[torch.LongTensor, Tuple[Batch, ND, SeqLen]]],
    attention_mask=Optional[Annotated[torch.BoolTensor, Tuple[Batch, ND, SeqLen, SeqLen]]],
    labels=Annotated[torch.LongTensor, Tuple[Batch, SeqLen]],
)

LlamaBatchInput = TypedDict(
    "LlamaBatchInput",
    input_ids=Annotated[torch.LongTensor, Tuple[Batch, SeqLen]],
    attention_mask=Optional[Annotated[torch.BoolTensor, Tuple[Batch, SeqLen]]],
    labels=Annotated[torch.LongTensor, Tuple[Batch, SeqLen]],
)



@dataclass
class DataItem:

    input_ids: List[int]
    cxt_len: List[int]

    def to_dict(self):
        return asdict(self)


class LlmValue:

    def __init__(
        self, 
        value: str,
    ):
        self.value = value
        self._alias = None
    
    @property
    def alias(self):
        if self._alias is None:
            return self.value
        return self._alias
    
    @alias.setter
    def alias(self, alias):
        self._alias = alias


class LlmType(Enum):
    
    qwen = LlmValue("qwen")
    chatglm = LlmValue("chatglm")
    chatglm2 = LlmValue("chatglm2")
    llama_native = LlmValue("llama")
    llama_alpaca = LlmValue("llama")
    llama_ziya = LlmValue("llama")
    llama_belle = LlmValue("llama")
    llama_linly = LlmValue("llama")
    gpt2 = LlmValue("gpt2")
    pangu = LlmValue("pangu")
    bloom = LlmValue("bloom")
    baichuan = LlmValue("baichuan")

    @property
    def val(self):
        return self.value