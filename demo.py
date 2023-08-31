import torch
import hcgf


gl = hcgf.GlmIa3("THUDM/chatglm-6b", device="cuda:0", torch_dtype=torch.float16)


params = {
    "lr": 2e-4,
    "num_epochs": 2, 
    "warmup_steps": 0, 
    "accumulate_steps": 1, 
    "print_every": 2, 
}
gl.load_data("./tests/test_data/test_data.json", max_seq_len=32).tune(**params)