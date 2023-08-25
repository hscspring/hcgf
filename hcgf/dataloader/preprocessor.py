from typing import Tuple, Optional, Union, Tuple, List

from transformers.tokenization_utils import PreTrainedTokenizer


class Prompter:


    def __init__(self, remain_len: int = 8):
        self.remain_len = remain_len
    
    def process_prompt(
        self, 
        tokenizer: PreTrainedTokenizer, 
        instruction: Optional[str], 
        history: List[Tuple[str, str]],
        prompt: str, 
        completion: str,
        max_seq_len: int
    ) -> Tuple[str, List]:
        keep = self.remain_len
        assert max_seq_len > keep, f"max_seq_len: {max_seq_len} must be greater than keep: {keep}"
        cids = tokenizer.encode(completion, add_special_tokens=False)

        max_seq_len -= (keep + len(cids))

        if instruction:
            iids = tokenizer.encode(instruction, add_special_tokens=False)
        else:
            iids = []
        
        pids = tokenizer.encode(prompt, add_special_tokens=False)
        
        if history:
            val = history[0]
            assert type(val) == tuple and len(val) == 2
            cxt_len = len(iids) + len(pids)
            truncated = []
            for q, r in reversed(history):
                qids = tokenizer.encode(q, add_special_tokens=False)
                rids = tokenizer.encode(r, add_special_tokens=False)
                cxt_len += (len(qids) + len(rids) + keep)
                if cxt_len > max_seq_len:
                    break
                truncated.append((q, r))
            history = list(reversed(truncated))
        else:
            if len(iids + pids) > max_seq_len:
                pmax_len = max_seq_len - len(iids)
                # Drop first tokens
                prompt = tokenizer.decode(pids[-pmax_len:])
        return prompt, history
    
    def build_qwen_input(
        self, 
        instruction: Optional[str], 
        history: List[Tuple[str, str]],
        prompt: str, 
    ) -> str:
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        inp = ""
        if instruction:
            inp = f"{im_start}system\n{instruction}{im_end}\n"

        if history:
            for i, (q, r) in enumerate(history):
                inp += f"""{im_start}user
{q}{im_end}
{im_start}assistant
{r}{im_end}
"""
        inp += f"""{im_start}user
{prompt}{im_end}
{im_start}assistant
"""
        return inp
    
    def build_linly_input(
        self, 
        instruction: Optional[str], 
        history: List[Tuple[str, str]],
        prompt: str, 
    ) -> str:
        inp = ""
        if instruction:
            inp = f"System: {instruction}\n"

        if history:
            for i, (q, r) in enumerate(history):
                inp += f"""User: {q}
Bot: {r}
"""
        inp += f"""User: {prompt}
Bot: """
        return inp
    
    def build_belle_input(
        self, 
        instruction: Optional[str], 
        history: List[Tuple[str, str]],
        prompt: str, 
    ) -> str:
        inp = ""
        if instruction:
            inp = f"System: \n{instruction}\n\n"

        if history:
            for i, (q, r) in enumerate(history):
                inp += f"""Human: 
{q}

Assistant: 
{r}

"""
        inp += f"""Human: 
{prompt}

Assistant: 
"""
        return inp
    
    def build_instructgpt_input(
        self, 
        instruction: str, 
        history: List[Tuple[str, str]], 
        prompt: str,
    ) -> str:
        inp = ""
        if instruction:
            inp = f"{instruction}\n\n"

        if history:
            for i, (q, r) in enumerate(history):
                inp += f"""Human: {q}
AI: {r}
"""
        inp += f"""Human: {prompt}
AI: """
        return inp
    
    def build_default_input(self, instruction: str, prompt: str) -> str:
        if instruction is not None:
            inp = f"""{instruction}

{prompt}
"""
        else:
            inp = f"{prompt}"
        return inp
    
    def build_chatglm2_input(
        self, 
        instruction: str, 
        history: List[Tuple[str, str]], 
        prompt: str,
    ):
        inp = ""
        for i, (old_query, response) in enumerate(history):
            inp += "[Round {}]\n\n 问：{}\n\n 答：{}\n\n".format(i + 1, old_query, response)
        inp += "[Round {}]\n\n 问：{}\n\n 答：".format(len(history) + 1, prompt)
        return inp
    
    def build_input(
        self, 
        instruction: str, 
        history: List[Tuple[str, str]], 
        prompt: str,
        alias: str,
    ) -> str:
        if alias == "qwen":
            return self.build_qwen_input(instruction, history, prompt)
        elif alias == "linly":
            return self.build_linly_input(instruction, history, prompt)
        elif alias == "belle":
            return self.build_belle_input(instruction, history, prompt)
        elif alias == "chatglm2":
            return self.build_chatglm2_input(instruction, history, prompt)
        else:
            return self.build_default_input(instruction, prompt)
    
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