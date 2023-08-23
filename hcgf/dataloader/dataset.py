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
        history = []
        prompt = item["prompt"]
        src = self.prompter.build_input(instruction, history, prompt, self.tokenizer.model_alias)
        print("样本 prompt ==> \n", src)

    def __getitem__(self, index: int) -> DataItem:
        item = self.data[index]
        prompt = item.get("prompt")
        instruction = item.get("instruction")
        history = item.get("history", [])
        completion = item["completion"]

        # cutoff those tokens more than `max_seq_len`
        prompt, history = self.prompter.process_prompt(
            self.tokenizer, instruction, history, prompt, completion, self.max_seq_len)

        src = self.prompter.build_input(
            instruction, history, prompt, self.tokenizer.model_alias
        )
        src_ids = self.tokenizer.encode(
            src, 
            max_length=self.max_seq_len,
            truncation=True,
            add_special_tokens=True
        )
        
        if (
            self.tokenizer.model_name == "pangu" and 
            src_ids[0] == self.tokenizer.bos_token_id
        ):
            # pangu might need modify => re_han_default = re.compile("([\u4E00-\u9FD5a-zA-Z0-9+#&\._%\-<>]+)", re.U)
            # remove bos and eos
            src_ids = src_ids[1:-1]
        
        tgt_ids = self.tokenizer.encode(
            completion, 
            max_length=self.max_seq_len - 1,
            truncation=True,
            add_special_tokens=False
        )
        
        if self.tokenizer.model_name == "llama":
            # remove the first blank ``, 29871
            tgt_ids = tgt_ids[1:]
        # ChatGLM use eop_token_id as eos_token_id.....
        # for pangu, whose eos/pad tokens are string....
        if isinstance(self.tokenizer.eos_token_id, str):
            input_ids = src_ids + tgt_ids
        else:
            input_ids = src_ids + tgt_ids + [self.tokenizer.eos_token_id]
        cxt_len = len(src_ids)
        return DataItem(input_ids, cxt_len)