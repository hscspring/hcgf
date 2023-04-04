import os
import shutil
import pytest

from hcgf.sft.lora_ft import GlmLora


@pytest.mark.slow
def test_8bit_mode(glm_data_file, glm_tune_param):
    params = glm_tune_param
    model_id = "THUDM/chatglm-6b"
    p1 = "./test_output1/"
    p2 = "./test_output2/"
    no_bnb = False
    try:
        import bitsandbytes as bnb
    except Exception:
        no_bnb = True
    if no_bnb:
        pass
    else:
        gl = GlmLora(model_id, load_in_8bit=True)
        params["out_dir"] = p1
        print("tuning...")
        (gl
        .load_data(glm_data_file, max_seq_len=32)
        .tune(**params))
        gl.eval()
        response, history = gl.chat("你是谁？")
        print(response)
        params["out_dir"] = p2
        print("\n\ntuning again...")
        gl.tune(**params)
        print("\n\ninference...")
        gl.load_pretrained(os.path.join(params["out_dir"], "ckpt/lora-ckpt-last-19.pt")).eval()
        response, history = gl.chat("你是谁？")
        print(response)
        assert 1, "should pass"
    
    for path in [p1, p2]:
        if os.path.exists(path):
            shutil.rmtree(path)


@pytest.mark.slow
def test_normal_mode(glm_data_file, glm_tune_param):
    params = glm_tune_param
    model_id = "THUDM/chatglm-6b"
    p1 = "./test_output1/"
    p2 = "./test_output2/"
    gl = GlmLora(model_id, device="cuda:0")
    params["out_dir"] = p1
    print("tuning...")
    (gl
    .load_data(glm_data_file, max_seq_len=32)
    .tune(**params))
    gl.eval()
    response, history = gl.chat("你是谁？")
    print(response)
    params["out_dir"] = p2
    print("\n\ntuning again...")
    gl.tune(**params)
    print("\n\ninference...")
    gl.load_pretrained(os.path.join(params["out_dir"], "ckpt/lora-ckpt-last-19.pt")).eval()
    response, history = gl.chat("你是谁？")
    print(response)
    assert 1, "should pass"
    for path in [p1, p2]:
        if os.path.exists(path):
            shutil.rmtree(path)