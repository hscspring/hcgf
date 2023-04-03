from dataclasses import dataclass, asdict

from typing import NewType, Tuple, TypedDict, Optional, List, TypeVar, Generic
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


@dataclass
class DataItem:

    input_ids: List[int]
    cxt_len: List[int]

    def to_dict(self):
        return asdict(self)
