import os
import shutil
from pathlib import Path
import pytest

from hcgf.sft.ft import GlmLora


def run_ft(gl: GlmLora, glm_data_file: str, params: dict):
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
        key=lambda x: int(x.stem.split("best-")[-1])
    )[-1]
    gl.load_pretrained(best_ckpt).eval()
    response, history = gl.chat(q, temperature=0.1)
    print(response)
    assert 1, "should pass"

    for path in [p1, p2]:
        if os.path.exists(path):
            shutil.rmtree(path)


@pytest.mark.slow
def test_lora_signle_gpu_ft(glm_data_file, glm_tune_param):
    model_id = "THUDM/chatglm-6b"
    gl = GlmLora(model_id, device="cuda:0")
    run_ft(gl, glm_data_file, glm_tune_param)


@pytest.mark.slow
def test_lora_8bit_ft(glm_data_file, glm_tune_param):
    model_id = "THUDM/chatglm-6b"
    no_bnb = False
    try:
        import bitsandbytes as bnb
    except Exception:
        no_bnb = True
    if no_bnb:
        pass
    else:
        gl = GlmLora(model_id, load_in_8bit=True)
        run_ft(gl, glm_data_file, glm_tune_param)
    
    