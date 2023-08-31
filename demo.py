import torch
import hcgf


gl = hcgf.GlmIa3("THUDM/chatglm-6b", device="cuda:0", torch_dtype=torch.float16)
params = {
    "lr": 5e-3,
    "num_epochs": 20,
    "warmup_steps": 0,
    "accumulate_steps": 1,
    "print_every": 3,
}
gl.load_data("./tests/test_data/test_data.json", max_seq_len=32).tune(**params)
# gl.load_pretrained("./output/ckpt/lora-2023-08-31-01:50:16_PM-ckpt-best-2.0425-320.pt")
gl.eval()
q = "你是谁？"
response, history = gl.chat(q, temperature=0.2)
print(q, response)