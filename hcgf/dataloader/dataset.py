from typing import List, Dict
from transformers.tokenization_utils import PreTrainedTokenizer

from ..data_model import DataItem
from .preprocessor import Prompter


class GlmMapStyleDataset:

    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 512,
        remain_len: int = 8,
    ):
        self.max_seq_len = max_seq_len
        self.data = data
        self.tokenizer = tokenizer
        assert hasattr(
            tokenizer, "eos_token_id"), "tokenizer should have a `eos_token_id` attribute"
        assert hasattr(
            tokenizer, "max_model_input_len"), "tokenizer should have a `max_model_input_len` attribute"
        assert hasattr(
            tokenizer, "model_name"), "tokenizer should have a `model_name` attribute"
        assert self.max_seq_len <= self.tokenizer.max_model_input_len // 2, "max_seq_len should be less than max_model_input_len/2"
        self.prompter = Prompter(remain_len)
        self._check_and_print_data_info()

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        for i in range(len(self.data)):
            yield self[i]
    
    def _check_and_print_data_info(self):
        item = self.data[0]
        assert type(item) == dict, "data item should be a dict contains at least `prompt` and `completion` keys"
        assert "prompt" in item, "one key must be `prompt`"
        assert "completion" in item, "one key must be `completion`"
        instruction = item.get("instruction")
        if instruction is None:
            print("The given data without `instruction` key, we are in normal (prompt + completion) fine-tune mode")
        else:
            msg = f"The given data with `instruction` key, the instruction is: {instruction}"
            msg += "we are in instruction (instruction + prompt + completion) fine-tune mode"
            print(msg)

    def __getitem__(self, index: int) -> DataItem:
        item = self.data[index]
        instruction = item.get("instruction")
        prompt = self.prompter.process_prompt(
            self.tokenizer, instruction, item["prompt"], self.max_seq_len)

        if instruction is None:
            # without `instruction` key, normal ft
            src = prompt
        else:
            # only `llama` might got instructions now
            # instruction="" or "some instruction"
            src = self.prompter.build_llama_instruction(instruction, prompt)
        tgt = item["completion"]
        
        src_ids = self.tokenizer.encode(
            src,
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=True)
        tgt_ids = self.tokenizer.encode(
            tgt,
            max_length=self.max_seq_len - 1,
            truncation=True,
            add_special_tokens=False)
        if self.tokenizer.model_name == "llama":
            # remove the first blank ``, 29871
            tgt_ids = tgt_ids[1:]
        # ChatGLM use eop_token_id as eos_token_id.....
        input_ids = src_ids + tgt_ids + [self.tokenizer.eos_token_id]
        cxt_len = len(src_ids)
        return DataItem(input_ids, cxt_len)