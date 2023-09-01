import torch
import hcgf

q1 = "你是谁？"
q2 = "用Python写一个快排。"
q3 = "中国首都是啥？"

gl = hcgf.GlmIa3("THUDM/chatglm-6b", device="cuda:0", torch_dtype=torch.float16)
params = {
    "lr": 5e-3,
    "num_epochs": 5,
    "warmup_steps": 0,
    "accumulate_steps": 1,
    "print_every": 3,
}
gl.load_data("./tests/test_data/test_data.json", max_seq_len=32).tune(**params)
gl.eval()
for q in [q1, q2, q3]:
    response, history = gl.chat(q, temperature=0.2)
    print(q)
    print(response)
    print("==" * 20)