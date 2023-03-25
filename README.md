
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

至少需要一张16G显存的卡。

```python
# 微调
import hcgf
gl = hcgf.GlmLora("THUDM/chatglm-6b", device="cuda:0")
gl.load_data("./data/chatgpt_finetune_faq.json").tune()

# 推理
import hcgf
gl = hcgf.GlmLora("THUDM/chatglm-6b", device="cuda:0", infer_mode=True)
gl.load_pretrained("/path/to/lora_pt").eval()
inp = "你是谁？"
gl.chat(inp)
```

### 8bit微调

至少需要一张12G显存的卡。不指定device。

```python
# 微调
import hcgf
gl = hcgf.GlmLora("THUDM/chatglm-6b", load_in_8bit=True)
gl.load_data("./data/chatgpt_finetune_faq.json").tune()

# 推理
gl = hcgf.GlmLora("THUDM/chatglm-6b", load_in_8bit=True, infer_mode=True)
gl.load_pretrained("/path/to/lora_pt").eval()
inp = "你是谁？"
gl.chat(inp)
```

### 配置

有几个影响显存的参数可以配置：`max_seq_len`，`batch_size`，`accumulate_steps`。


```python
(
gl
.load_data("./data/chatgpt_finetune_faq.json", max_seq_len=128)
.tune(batch_size=1, accumulate_steps=1)
)

```