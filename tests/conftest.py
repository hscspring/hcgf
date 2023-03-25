import pytest

import os

from hcgf.dataloader.data_model import DataItem
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
    return GlmDataLoader(glm_data_file, tokenizer)


@pytest.fixture
def mocked_dataset():
    """
    {"prompt": "你好你好你好", "completion": "是谁"},
    {"prompt": "你好", "completion": "谁"},
    """
    data = [
        DataItem([20005, 94874, 94874, 94874, 150001, 150004, 20005, 88443, 20002], 6),
        DataItem([20005, 94874, 150001, 150004, 20005, 84480, 20002], 4),
    ]
    return data
