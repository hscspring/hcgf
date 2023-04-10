import pytest

import os

from hcgf.data_model import DataItem
from hcgf.dataloader.data_loader import GlmDataLoader
from hcgf.sft.chatglm.tokenization_chatglm import ChatGLMTokenizer

root = os.path.dirname(os.path.abspath(__file__))

tokenizer = ChatGLMTokenizer.from_pretrained("THUDM/chatglm-6b")


@pytest.fixture
def glm_data_file():
    return os.path.join(root, "test_data/test_data.json")


@pytest.fixture
def glm_tokenizer():
    return tokenizer


@pytest.fixture
def glm_dataloader(glm_data_file):
    return GlmDataLoader(glm_data_file, tokenizer, 64)


@pytest.fixture
def glm_tune_param():
    params = {
        "lr": 2e-4,
        "num_epochs": 1, 
        "warmup_steps": 0, 
        "accumulate_steps": 1, 
        "print_every": 2, 
    }
    return params

@pytest.fixture
def mocked_dataset():
    """
    {"prompt": "你好你好你好", "completion": "是谁"},
    {"prompt": "你好", "completion": "谁"},
    """
    data = [
        DataItem([5, 94874, 94874, 94874, 130001, 130004, 5, 88443, 2], 6),
        DataItem([5, 94874, 130001, 130004, 5, 84480, 2], 4),
    ]
    return data
