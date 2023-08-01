import pandas as pd
import json
from datetime import datetime
from typing import Union
from transformers import GenerationConfig
import torch
import pnlp
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
from transformers.models.llama import LlamaTokenizer, LlamaForCausalLM

import sys
sys.path.append("/data/hsc/hcgf/")

from hcgf.sft.ft import GlmLora

device = "cuda:2"
model_name = "/data/hsc/cache/huggingface/models--THUDM--chatglm-6b/snapshots/4d0fc39a58dcb747ab74ab2c4587bd66dcfa7e74/"
glm = GlmLora(model_name, device=device)
model = glm.load_model(model_name)

def chat(prompt, temperature, max_length):
    max_len = max_length+len(prompt)
    resp, his = model.chat(glm.tokenizer, prompt, None, temperature=temperature, max_length=max_len)
    return resp

def process(inp):
    return inp.replace("Human: ", "").replace("Assistant: ", "答案: ")


dt = pnlp.read_file_to_list_dict("llm_eva_l1_test.json")
df = pd.DataFrame(dt)
ti = dt[0]
resp = chat(process(ti["prompt"]), 0.2, 100)
print(resp)

i=0
res = []
max_length = 100
temperature = 0.2
for v in dt:
    prompt = v["prompt"]
    prompt = process(prompt)
    response = chat(prompt, temperature, max_length)
    res.append(response)
    if i % 10 == 0:
        print(i, len(dt), datetime.now())
    i+=1
print(len(res), datetime.now())

df["res"] = res

import json
def clean(x):
    x = x.replace(" ", "")
    x = x.replace("。", "")
    x = x.replace("！", "")
    return x
df["typ"] = df.meta.apply(lambda x: x["type"])
df["choice_num"] = df.meta.apply(lambda x: x["choice_num"])
df["is_right_include"] = df.apply(lambda x: int(x.completion in x.res), axis=1)
df["is_right_exact"] = df.apply(lambda x: int(x.completion == clean(x.res)), axis=1)

print(df.is_right_exact.value_counts(), df.is_right_include.value_counts())

for typ in ["d", "m"]:
    print(f"type: {typ}: ", df[df.typ==typ].is_right_include.value_counts())

for num in [2, 3, 4]:
    print(f"choice num {num}: ", df[df.choice_num==num].is_right_include.value_counts())

out_file_name = "output-chatglm-6.7b-0shot"
df.to_excel(f"{out_file_name}.xlsx", index=False)