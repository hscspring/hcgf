from hcgf.sft import GlmLora

data_path = "./chatgpt_finetune_faq.json"
model_id = "THUDM/chatglm-6b"
device = "cuda:1"
model = GlmLora(model_id, device=device)
model.load_data(data_path).tune(device)