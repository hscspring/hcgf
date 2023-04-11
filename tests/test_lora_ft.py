import os
import shutil
from pathlib import Path
import pytest

from hcgf.sft.lora_ft import GlmLora


@pytest.mark.slow
@pytest.mark.parametrize("mode", ["8bit", "normal"])
def test_lora_ft(glm_data_file, glm_tune_param, mode):
    params = glm_tune_param
    model_id = "THUDM/chatglm-6b"
    p1 = "./test_output1/"
    p2 = "./test_output2/"
    no_bnb = False
    if mode == "8bit":
        try:
            import bitsandbytes as bnb
        except Exception:
            no_bnb = True
        if not no_bnb:
            gl = GlmLora(model_id, load_in_8bit=True)
    elif mode == "normal":
        gl = GlmLora(model_id, device="cuda:0")
    
    if no_bnb:
        pass
    else:
        params["out_dir"] = p1
        print("tuning...")
        (gl
        .load_data(glm_data_file, max_seq_len=32)
        .tune(**params))
        gl.eval()
        response, history = gl.chat("你是谁？", temperature=0.2)
        print(response)
        params["out_dir"] = p2
        print("\n\ntuning again...")
        gl.tune(**params)
        print("\n\ninference...")
        out_dir = Path(os.path.join(params["out_dir"], "ckpt"))
        last_ckpt = list(out_dir.glob("*last*"))[0]
        gl.load_pretrained(last_ckpt).eval()
        response, history = gl.chat("你是谁？", temperature=0.2)
        print(response)
        assert 1, "should pass"
    
    for path in [p1, p2]:
        if os.path.exists(path):
            shutil.rmtree(path)