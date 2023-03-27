from hcgf import GlmLora

data_path = "./benchmark_data.json"
model_id = "THUDM/chatglm-6b"
# for normal mode
gl = GlmLora(model_id, device="cuda:0", lora_r=8)

# for 8bit
# gl = GlmLora(model_id, load_in_8bit=True, lora_r=8)

gl.load_data(data_path, max_seq_len=512).tune(batch_size=1)