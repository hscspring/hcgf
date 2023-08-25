from typing import Tuple
from torch.utils.data import DataLoader
import pytest
import numpy as np
import pnlp

from hcgf.dataloader.dataset import GlmMapStyleDataset
from hcgf.data_model import DataItem
from hcgf.dataloader.data_collector import ChatglmDataCollector, LlamaDataCollector
from hcgf.dataloader.preprocessor import Prompter


def test_dataset(glm_data_file, glm_tokenizer):
    data = pnlp.read_file_to_list_dict(glm_data_file)
    ds = GlmMapStyleDataset(data, glm_tokenizer)
    assert isinstance(ds[0], DataItem)
    for v in ds:
        assert isinstance(v, DataItem)
        break
    assert len(ds) == len(data)


@pytest.mark.parametrize("max_len,expected", [
    (5, 8),
    (128, 10),
])
def test_dataset_max_len(glm_tokenizer, max_len, expected):
    data = [
        {"prompt": "爱爱爱爱", "completion": "谁"}
    ]
    ds = GlmMapStyleDataset(data, glm_tokenizer, max_len, remain_len=0)
    v = ds[0]
    # 爱爱爱爱 encode后长度为7，超过后变为5，加上completion的三个：空白符、谁、EOS，共8个token
    assert len(v.input_ids) == expected


arr_dtype = np.int64


# NOTE: follow ChatGLM GitHub P-Tuning, padding to right, ChatGLM Tokenizer actually uses `padding_left`
# Here, when we are training, it's not an issue.
input_ids = np.array(
    [
        [5, 74874, 74874, 74874, 130001, 130004, 5, 68443, 130005],
        [5, 74874, 130001, 130004, 5, 64480, 130005, 3, 3],
    ],
    dtype=arr_dtype
)

labels = np.array(
    [
        [-100, -100, -100, -100, -100, 130004, 5, 68443, 130005],
        [-100, -100, -100, 130004, 5, 64480, 130005, -100, -100],
    ],
    dtype=arr_dtype
)

position_ids = np.array(
    [
        [[0, 1, 2, 3, 4, 4, 4, 4, 4],
         [0, 0, 0, 0, 0, 1, 2, 3, 4]],
        [[0, 1, 2, 2, 2, 2, 2, 2, 2],
         [0, 0, 0, 1, 2, 3, 4, 5, 6]],
    ],
    dtype=arr_dtype
)

attention_mask = np.array(
    [
        [[[False, False, False, False, False, True, True, True, True],
          [False, False, False, False, False, True, True, True, True],
          [False, False, False, False, False, True, True, True, True],
          [False, False, False, False, False, True, True, True, True],
          [False, False, False, False, False, True, True, True, True],
          [False, False, False, False, False, False, True, True, True],
          [False, False, False, False, False, False, False, True, True],
          [False, False, False, False, False, False, False, False, True],
          [False, False, False, False, False, False, False, False, False]]],
        [[[False, False, False, True, True, True, True, True, True],
          [False, False, False, True, True, True, True, True, True],
          [False, False, False, True, True, True, True, True, True],
          [False, False, False, False, True, True, True, True, True],
          [False, False, False, False, False, True, True, True, True],
          [False, False, False, False, False, False, True, True, True],
          [False, False, False, False, False, False, False, True, True],
          [False, False, False, False, False, False, False, False, True],
          [False, False, False, False, False, False, False, False, False]]],
    ],
    dtype=np.bool_
)


@pytest.mark.parametrize("inp_key,shape,expected", [
    ("input_ids", (2, 9), input_ids),
    ("position_ids", (2, 2, 9), position_ids),
    ("attention_mask", (2, 1, 9, 9), attention_mask),
    ("labels", (2, 9), labels),
])
def test_glm_data_collector(mocked_data, glm_tokenizer, inp_key, shape, expected):
    ds = GlmMapStyleDataset(mocked_data, glm_tokenizer, 32)
    binp = ChatglmDataCollector.collate_fn(ds)
    assert type(binp) == dict
    val = binp[inp_key]
    assert tuple(val.shape) == shape
    assert np.alltrue(val.numpy() == expected)


input_ids = np.array(
    [
        [   0, 0, 0, 1, 29871, 30919, 31076, 30919, 31076, 30919, 31076, 
            30392, 235, 179, 132, 2],
        [   0, 0, 0, 0, 0, 0, 0, 0, 1, 29871, 30919, 31076,
            235, 179, 132, 2]
    ],
    dtype=arr_dtype
)

# mask input
labels_mask_input = np.array(
    [
        [   -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 
            30392, 235, 179, 132, 2],
        [   -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
            235, 179, 132, 2]
    ],
    dtype=arr_dtype
)

# no mask input
labels = np.array(
    [
        [   -100, -100, -100, 1, 29871, 30919, 31076, 30919, 31076, 30919, 31076, 
            30392, 235, 179, 132, 2],
        [   -100, -100, -100, -100, -100, -100, -100, -100, 1, 29871, 30919, 31076,
            235, 179, 132, 2]
    ],
    dtype=arr_dtype
)

attention_mask = np.array(
    [
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
    ],
    dtype=np.bool_
)


@pytest.mark.parametrize("inp_key,shape,expected", [
    ("input_ids", (2, 16), input_ids),
    ("attention_mask", (2, 16), attention_mask),
    ("labels", (2, 16), labels_mask_input),
])
def test_llama_data_collector(mocked_data, llama_tokenizer, inp_key, shape, expected):
    ds = GlmMapStyleDataset(mocked_data, llama_tokenizer, 32)
    binp = LlamaDataCollector.collate_fn(ds)
    assert type(binp) == dict
    val = binp[inp_key]
    assert tuple(val.shape) == shape
    assert np.alltrue(val.numpy() == expected)



@pytest.mark.parametrize("func,expected", [
    ("train_dev_split", Tuple),
    ("load", DataLoader),
])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_data_loader(glm_dataloader, func, expected, batch_size):
    out = getattr(glm_dataloader, func)(batch_size=batch_size)
    assert isinstance(out, expected)
    if isinstance(out, DataLoader):
        dataloader = out
        data_len = len(glm_dataloader)
        num_batches = data_len // batch_size
        if data_len % batch_size != 0:
            num_batches += 1
        assert len(dataloader) == num_batches
    else:
        dataloader = out[0]
    for v in dataloader:
        input_ids = v["input_ids"]
        assert input_ids.shape[0] == batch_size
        break


@pytest.mark.parametrize("inp_key,shape", [
    ("input_ids", (1, 7)),
    ("position_ids", (1, 2, 7)),
    ("attention_mask", (1, 1, 7, 7)),
    ("labels", (1, 7)),
])
def test_dataset_collector(glm_tokenizer, inp_key, shape):
    data = [
        {"prompt": "你好", "completion": "谁"},
    ]
    ds = GlmMapStyleDataset(data, glm_tokenizer)
    binp = ChatglmDataCollector.collate_fn(ds)
    val = binp[inp_key]
    assert tuple(val.shape) == shape


@pytest.mark.parametrize("instruction", [None, "", "instruction"])
def test_prompter(instruction):
    prompter = Prompter()
    model = "llama"
    func = getattr(prompter, f"build_{model}_instruction")
    inp = func(instruction, "prompt")
    assert type(inp) == str


@pytest.mark.parametrize("instruction", [None, "", "请说话"])  # 2tokens
@pytest.mark.parametrize("max_seq_len", [11, 128])  # 最少(keep)8
def test_process_prompt(instruction, max_seq_len, glm_tokenizer):
    prompter = Prompter()
    prompt = "你叫什么名字" # 3tokens
    new, _ = prompter.process_prompt(glm_tokenizer, instruction, [], prompt, "", max_seq_len)
    assert type(new) == str
    if max_seq_len == 128 or not instruction:
        assert new == prompt
    else:
        assert new == "名字" # prompt只能保留1个tokens（8keep+2instruction+1prompt=11），所以为`名字`


@pytest.mark.parametrize("history, max_seq_len, expected", [
    ([("你好", "你叫什么名字")] * 2, 20, 0),
    ([("你好", "你叫什么名字")] * 2, 21, 1),  # 8+(2+3+8)*1 = 21
    ([("你好", "你叫什么名字")] * 2, 33, 1),  # 8+(2+3+8)*1 = 21
    ([("你好", "你叫什么名字")] * 2, 34, 2),  # 8+(2+3+8)*2 = 34
    ([("你好", "你叫什么名字")] * 2, 200, 2),
    ([("你好", "你叫什么名字"), ("好你", "你叫什么名字")], 22, 1),
])
def test_process_prompt_history(glm_tokenizer, history, max_seq_len, expected):
    prompter = Prompter()
    _, new = prompter.process_prompt(glm_tokenizer, None, history, "", "", max_seq_len)
    assert len(new) == expected
    if len(new) > 0:
        # NOTE: should from end to start
        assert new[0] == history[len(history) - expected]


inp1 = """<|im_start|>user
q1<|im_end|>
<|im_start|>assistant
r1<|im_end|>
<|im_start|>user
Prompt<|im_end|>
<|im_start|>assistant
"""
inp2 = """<|im_start|>system
Instruction<|im_end|>
<|im_start|>user
q1<|im_end|>
<|im_start|>assistant
r1<|im_end|>
<|im_start|>user
Prompt<|im_end|>
<|im_start|>assistant
"""
@pytest.mark.parametrize("history, instruction, expected", [
    ([("q1", "r1")], None, inp1),
    ([("q1", "r1")], "Instruction", inp2),
])
def test_qwen_prompt(history, instruction, expected):
    prompt = "Prompt"
    prompter = Prompter()
    inp = prompter.build_qwen_input(instruction, history, prompt)
    assert inp == expected


inp1_linly = """User: q1
Bot: r1
User: Prompt
Bot: """
inp2_linly = """System: Instruction
User: q1
Bot: r1
User: Prompt
Bot: """
@pytest.mark.parametrize("history, instruction, expected", [
    ([("q1", "r1")], None, inp1_linly),
    ([("q1", "r1")], "Instruction", inp2_linly),
])
def test_linly_prompt(history, instruction, expected):
    prompt = "Prompt"
    prompter = Prompter()
    inp = prompter.build_linly_input(instruction, history, prompt)
    assert inp == expected



inp1_belle = """Human: 
q1

Assistant: 
r1

Human: 
Prompt

Assistant: 
"""
inp2_belle = """System: 
Instruction

Human: 
q1

Assistant: 
r1

Human: 
Prompt

Assistant: 
"""
@pytest.mark.parametrize("history, instruction, expected", [
    ([("q1", "r1")], None, inp1_belle),
    ([("q1", "r1")], "Instruction", inp2_belle),
])
def test_belle_prompt(history, instruction, expected):
    prompt = "Prompt"
    prompter = Prompter()
    inp = prompter.build_belle_input(instruction, history, prompt)
    assert inp == expected