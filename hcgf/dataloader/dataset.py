from typing import List, Dict
from transformers.tokenization_utils import PreTrainedTokenizer

from .data_model import DataItem


class GlmMapStyleDataset:

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 512
    ):
        self.max_seq_len = max_seq_len
        self.data = data
        assert hasattr(
            tokenizer, "eos_token_id"), "tokenizer should have a `eos_token_id` attribute"
        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> DataItem:
        item = self.data[index]
        assert "prompt" in item, "one key must be `prompt`"
        assert "completion" in item, "one key must be `completion`"
        src = item["prompt"]
        tgt = item["completion"]
        src_ids = self.tokenizer.encode(
            src,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=True)
        tgt_ids = self.tokenizer.encode(
            tgt,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=False)
        input_ids = src_ids + tgt_ids + [self.tokenizer.eos_token_id]
        cxt_len = len(src_ids)
        return DataItem(input_ids, cxt_len)

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        for i in range(len(self.data)):
            yield self[i]
