先clone仓库或pip安装：

```bash
pip install hcgf
```

需要的依赖在`requirements.txt`中，通过下面命令安装：

```
pip install -r requirements.txt
```

建议使用PyTorch2.0


## 微调训练

### 准备数据

每一行一个dict的`.json`文件，必须包含`prompt`和`completion`两个字段。示例如下：

```python
{"prompt": "你是谁？", "completion": "不告诉你。"}
```

### 分布式微调

使用PyTorch的FSDP训练，支持Zero3、Zero2和DDP模式，使用方法请参考帮助文档：

```bash
hcgf_tune -h
```

至少要指定`model`和`data_path`参数，如下：

```bash
hcgf_tune --model THUDM/chatglm-6b --data_path path/to/train_data.json
```

首先要理解一下，模型训练时，除了模型（也就是参数）占用的空间外，还有


一共五种策略：

- fsdp_zero3：命令行模式默认策略，FULL_SHARD，参数、梯度、优化器状态SHARD，慢但是省显存，数据并行。
- fsdp_zero2：GRAD_OP_SHARD，梯度、优化器状态SHARD，比上面那个快一些，数据并行。
- mpdp(ddp)：NO_SHARD，类似DDP，就是把模型分别加载到每张卡上，比上面2个都快，数据并行。
- mpds(8bit)：8bit模式（下面的《8bit微调》），模型被分到多个卡（甚至CPU）上，没有数据并行，很慢。
- msds(single_gpu)：单卡模式（下面的《正常微调》），能跑起来的情况下比较快。

| 卡数 | 显存           | 训练数据 | 策略                  |
| ---- | -------------- | -------- | --------------------- |
| 多卡 | 单卡跑不起模型 | 数据很多 | fsdp_zero3/fsdp_zero2 |
|      | 单卡跑得起模型 | 数据很多 | mpdp                  |
|      | 单卡跑不起模型 | 数据很少 | mpds                  |
|      | 单卡跑得起模型 | 数据很少 | msds                  |
| 单卡 | 单卡跑不起模型 | -        | mpds                  |
|      | 单卡跑得起模型 | -        | msds                  |


注意事项：
- 这里显存是在训练模式下的，和推理模式占用不同，可参考下面的《配置》。推理只支持后两种模式。
- FSDP模式下可能还没有单卡快（单卡跑得起的时候），这是正常的，因为FSDP对数据分片了，而且为了更大限度地使用显存，还把一些数据倒腾到CPU了。



### 正常微调

至少需要一张16G显存的卡。如果不指定显卡，默认为`cuda`。

```python
#===== 微调 =====#
import hcgf
gl = hcgf.GlmLora("THUDM/chatglm-6b", device="cuda:0")
gl.load_data("/path/to/data.json").tune()

#===== 推理 =====#
gl = hcgf.GlmLora("THUDM/chatglm-6b", device="cuda:0")
gl.load_pretrained("/path/to/lora_pt").eval()
gl.chat("你是谁?")

#===== 切换模式 =====#
gl = hcgf.GlmLora("THUDM/chatglm-6b", device="cuda:0")
gl.load_data("/path/to/data.json").tune()
# 切换到推理模式
gl.eval()
gl.chat("你是谁？")
# 切换回微调模式，还是用原来的数据继续跑
gl.tune()
# 如果有新的数据集，参考上面的写法，先加载数据
gl.load_data("/path/to/new_data.json").tune()
# 如果在原来的基础上用新数据继续微调，先加载之前的pt文件，再加载数据微调
gl.load_pretrained("/path/to/lora_pt").load_data("/path/to/new_data.json").tune()
```

当然，也可以使用`hcgf_tune`:

```bash
hcgf_tune strategy msds --model THUDM/chatglm-6b --data_path path/to/train_data.json
```


### 8bit微调

至少需要一张12G显存的卡。不指定device。只需要初始化时改一下即可，其他操作和上面正常微调一样。

需要安装依赖: `bitsandbytes`

```python
gl = hcgf.GlmLora("THUDM/chatglm-6b", load_in_8bit=True)
```


当然，也可以使用`hcgf_tune`:

```bash
hcgf_tune strategy mpds --model THUDM/chatglm-6b --data_path path/to/train_data.json
```

### 继续微调

先加载之前的`pt`文件，然后加载数据微调。

```python
gl.load_pretrained("/path/to/lora_pt").load_data("/path/to/new_data.json").tune()
```

### 演示Demo/推理

请执行`hcgf_infer -h`查看帮助。


### 参数说明

主要有三个方法的参数，有值的表示默认值。

```python
load_data(
    data_path: str, 
    max_seq_len: int = 512, # 句子最大长度，超过会截断。注意，这里指Prompt或Completion的长度，应保证两者长度之和不大于模型最大长度。
)
tune(
    batch_size: int = 8,
    lr: float = 2e-4,
    num_epochs: int = 3,
    warmup_steps: Optional[int] = None,     # 为None时会用1/3个Epoch进行warmup
    accumulate_steps: Optional[int] = None, # 为None时等价于1
    out_dir: str = "./output/",
    print_every: Optional[int] = None,      # 为None时每1/10Epoch个Steps打印一次输出（Step、Loss、LearningRate）
)
chat(
    inp: str, 
    history: List[Tuple[str, str]] = None,  # (问，答)Pair对
    max_new_tokens: int = 512,              # 生成的文本最大长度，Prompt的长度=支持的最大长度-max_new_tokens，Prompt长度超过会被截断
    do_sample: bool = True,                 # 采样
    num_beams: int = 1,                     # Beam Search 的 beam 数量
    temperature: float = 0.95,              # 越小越确定，越大越随机，比如你微调后可以把它改成0.1
    top_p: float = 0.7,                     # 同上，两者不要同时调
    repetition_penalty: float = 1.02,       # 生成内容重复惩罚，越大越不容易重复
    stop: List[str] = []                    # 停止文本，可以是标点、特定词或句子等，输出不包含停止文本
)
```

Best Practice:

- `tune`: 如果内存不够可以调小batch_size，同时增加accumulate_steps，一般是batch_size的整数倍；
- `chat`: 一般只需调整`temerature`；


### 配置

有几个影响显存的参数可以配置：`max_seq_len`，`batch_size`。


```python
(
gl
.load_data("./data/chatgpt_finetune_faq.json", max_seq_len=128)
.tune(batch_size=1)
)

```

以下配置针对`ChatGLM-6B`。


不同配置 `8bit` 资源占用：

| max_seq_len | batch_size | memory |
| ----------- | ---------- | ------ |
| `64`        | 1          | 11G    |
| `128`       | 1          | 12G    |
| `512`       | 1          | 22G    |
| 128         | `2`        | 15G    |
| 128         | `4`        | 21G    |

不同配置正常资源占用：

| max_seq_len | batch_size | memory |
| ----------- | ---------- | ------ |
| `64`        | 1          | 15G    |
| `128`       | 1          | 16G    |
| `512`       | 1          | 30G    |
| 128         | `2`        | 19G    |
| 128         | `4`        | 25G    |


## RM

使用小模型（如BERT等）训练。

### 训练

### 准备数据

需要pair对数据，计算logits过程和普通预训练模型一样（一个Batch多个pair对）；计算loss时属于同一个pair对的logits放一块算。

推理时直接用logits就行。



## 测试

```bash
# 全部测试
python -m pytest
# 测试训练和推理，比较慢
python -m pytest -s -m slow
# 测试其他的
python -m pytest -m "not slow"
```


## 其他说明

如果遇到加载超时，可以直接load本地cache下的模型：

```Python
GlmLora("/path/to/huggingface/models--THUDM--chatglm-6b/snapshots/<id>/")
```


## 更新日志

- **v0.2.0** `20230513`
  - 支持分布式微调
  - 调整推理模式，支持Batch
- **v0.1.0** `20230412`
  - 支持ChatGLM新版Tokenizer
  - 使用官方调整后的MASK方式
- **v0.0.7** `20230405`