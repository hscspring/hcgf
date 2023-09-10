import os
import shutil
from pathlib import Path
import pytest
import torch

from hcgf.sft.ft import GlmIa3


@pytest.mark.slow
def test_ia3_signle_gpu_ft(ft_runner, glm_data_file, glm_tune_param):
    model_id = "THUDM/chatglm-6b"
    gl = GlmIa3(model_id, device="cuda:0", torch_dtype=torch.float16)
    glm_tune_param["lr"] = 6e-3
    ft_runner(gl, glm_data_file, glm_tune_param)


@pytest.mark.slow
def test_ia3_8bit_ft(ft_runner, glm_data_file, glm_tune_param):
    model_id = "THUDM/chatglm-6b"
    no_bnb = False
    try:
        import bitsandbytes as bnb
    except Exception:
        no_bnb = True
    if no_bnb:
        pass
    else:
        gl = GlmIa3(model_id, load_in_8bit=True, torch_dtype=torch.float16)
        glm_tune_param["lr"] = 6e-3
        ft_runner(gl, glm_data_file, glm_tune_param)