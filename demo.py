from transformers import LlamaTokenizer, LlamaForCausalLM

id = "../cache/models/BELLE-LLaMA-7B-2M/"

device = "cpu"

tokenizer = LlamaTokenizer.from_pretrained(id)
model = LlamaForCausalLM.from_pretrained(id).to(device)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
generate_ids = model.generate(inputs.input_ids, max_length=30)
output = tokenizer.batch_decode(
    generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]
print(output)