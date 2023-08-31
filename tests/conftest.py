import pytest

import os
import shutil
from pathlib import Path

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


def run_ft(gl, glm_data_file: str, params: dict):
    p1 = "./test_output1/"
    p2 = "./test_output2/"
    params["out_dir"] = p1
    print("tuning...")
    (gl
    .load_data(glm_data_file, max_seq_len=32)
    .tune(**params))
    gl.eval()
    q = "你是谁？"
    response, history = gl.chat(q, temperature=0.1)
    print(q, response)
    params["out_dir"] = p2
    print(f"\n\ntuning again with params: {params}")
    gl.tune(**params)
    print("\n\ninference...")
    out_dir = Path(os.path.join(params["out_dir"], "ckpt"))
    best_ckpt = sorted(
        out_dir.glob("*best*"), 
        key=lambda x: int(x.stem.split("-")[-1])
    )[-1]
    gl.load_pretrained(best_ckpt).eval()
    response, history = gl.chat(q, temperature=0.1)
    print(q, response)
    assert 1, "should pass"

    for path in [p1, p2]:
        if os.path.exists(path):
            shutil.rmtree(path)


@pytest.fixture
def ft_runner():
    return run_ft