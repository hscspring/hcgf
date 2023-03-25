
## 使用

先clone仓库或pip安装：

```bash
pip install hcgf
```

需要的依赖在`requirements.txt`中，通过下面命令安装：

```
pip install -r requirements.txt
```

注意：不支持PyTorch2.0，历史版本请参考下面链接安装：

https://pytorch.org/get-started/previous-versions/


### 准备数据

每一行一个json，必须包含`prompt`和`completion`两个字段。示例如下：

```bash
{"prompt": "问题：你是谁？\n", "completion": "不告诉你。"},
```

### 正常微调

至少需要一张20G以上显存的卡，建议32G。

```python
# 微调
from hcgf.sft import GlmLora
data_path = "/path/to/json_data/"
model_id = "THUDM/chatglm-6b"
device = "cuda:0"
model = GlmLora(model_id, device=device)
model.load_data(data_path).tune(device)

# 推理
from hcgf.sft import GlmLora
model_id = "THUDM/chatglm-6b"
device = "cuda:0"
model = GlmLora(model_id, device=device)
model.load_pretrained("/path/to/lora_pt").eval()
inp = "你是谁？"
model.chat(inp)
```
