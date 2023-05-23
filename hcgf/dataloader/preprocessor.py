from typing import Tuple, Optional

from transformers.tokenization_utils import PreTrainedTokenizer



class Prompter:


    def __init__(self, remain_len: int = 8):
        self.remain_len = remain_len
    
    def process_prompt(
        self, 
        tokenizer: PreTrainedTokenizer, 
        instruction: Optional[str], 
        prompt: str, 
        max_seq_len: int
    ) -> str:
        if instruction is None:
            keep = self.remain_len
        elif instruction == "":
            keep = 35 + self.remain_len
        else:
            keep = 49 + self.remain_len
        
        assert max_seq_len > keep, f"max_seq_len: {max_seq_len} must be greater than keep: {keep}"
        max_seq_len -= keep
        if instruction:
            iids = tokenizer.encode(instruction, add_special_tokens=False)
        else:
            iids = []
        pids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(iids + pids) > max_seq_len:
            pmax_len = max_seq_len - len(iids)
            prompt = tokenizer.decode(pids[-pmax_len:])
        return prompt
    
    def build_llama_instruction(self, instruction: str, prompt: str) -> str:
        if not instruction:
            # tokens=35
            inp = f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
        
### Instruction:
{prompt}

### Response:
"""
        else:
            # tokens=49
            inp = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{prompt}

### Response:
"""
        return inp
