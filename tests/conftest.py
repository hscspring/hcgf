import pytest

import os

from hcgf.data_model import DataItem
from hcgf.dataloader.data_loader import GlmDataLoader

from transformers import AutoTokenizer, LlamaTokenizer


root = os.path.dirname(os.path.abspath(__file__))

_glm_tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
_llama_tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")


@pytest.fixture
def glm_data_file():
    return os.path.join(root, "test_data/test_data.json")


@pytest.fixture
def glm_tokenizer():
    _glm_tokenizer.max_model_input_len = 2048
    _glm_tokenizer.model_name = "chatglm"
    _glm_tokenizer.model_alias = "chatglm"
    return _glm_tokenizer


@pytest.fixture
def llama_tokenizer():
    _llama_tokenizer.max_model_input_len = 2048
    _llama_tokenizer.model_name = "llama"
    _llama_tokenizer.model_alias = "llama"
    # decapoda got some issue, need fix
    _llama_tokenizer.pad_token_id = 0
    _llama_tokenizer.bos_token_id = 1
    _llama_tokenizer.eos_token_id = 2
    _llama_tokenizer.padding_side="left"
    return _llama_tokenizer


@pytest.fixture
def glm_dataloader(glm_data_file):
    return GlmDataLoader(glm_data_file, _glm_tokenizer, 64)


@pytest.fixture
def glm_tune_param():
    params = {
        "lr": 2e-4,
        "num_epochs": 1, 
        "warmup_steps": 0, 
        "accumulate_steps": 1, 
        "print_every": 3, 
    }
    return params

@pytest.fixture
def mocked_data():
    data = [
        {"prompt": "你好你好你好", "completion": "是谁"},
        {"prompt": "你好", "completion": "谁"}
    ]
    return data
