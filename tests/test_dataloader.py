from typing import Tuple
from torch.utils.data import DataLoader
import pytest
import numpy as np
import pnlp

from hcgf.dataloader.dataset import GlmMapStyleDataset
from hcgf.dataloader.data_model import DataItem
from hcgf.dataloader.data_collector import GlmDataCollector


def test_dataset(glm_data_file, glm_tokenizer):
    data = pnlp.read_file_to_list_dict(glm_data_file)
    ds = GlmMapStyleDataset(data, glm_tokenizer)
    assert isinstance(ds[0], DataItem)
    for v in ds:
        assert isinstance(v, DataItem)
        break
    assert len(ds) == len(data)


@pytest.mark.parametrize("max_len,expected",[
    (5, 8), 
    (128, 10),
])
def test_dataset_max_len(glm_tokenizer, max_len, expected):
    data = [
        {"prompt": "爱爱爱爱", "completion": "谁"}
    ]
    ds = GlmMapStyleDataset(data, glm_tokenizer, max_len)
    v = ds[0]
    # 爱爱爱爱 encode后长度为7，超过后变为5，加上completion的三个：空白符、谁、EOS，共8个token
    assert len(v.input_ids) == expected


arr_dtype = np.int64

input_ids = np.array(
    [
        [20005, 94874, 94874, 94874, 150001, 150004, 20005, 88443, 20002],
        [20005, 94874, 150001, 150004, 20005, 84480, 20002, 20002, 20002],
    ],
    dtype=arr_dtype
)

labels = np.array(
    [
        [-100, -100, -100, -100, -100, 150004, 20005, 88443, 20002],
        [-100, -100, -100, 150004, 20005, 84480, 20002, -100, -100],
    ],
    dtype=arr_dtype
)

position_ids = np.array(
    [
        [[0, 1, 2, 3, 4, 5, 6, 7, 8],
         [0, 0, 0, 0, 0, 1, 2, 3, 4]],
        [[0, 1, 2, 3, 4, 5, 6, 7, 8],
         [0, 0, 0, 1, 2, 3, 4, 5, 6]],
    ],
    dtype=arr_dtype
)

attention_mask = np.array(
    [
        [[[False, False, False, False, False, True, True, True, True],
          [False, False, False, False, False, True, True, True, True],
          [False, False, False, False, False, True, True, True, True],
          [False, False, False, False, False, True, True, True, True],
          [False, False, False, False, False, True, True, True, True],
          [False, False, False, False, False, False, True, True, True],
          [False, False, False, False, False, False, False, True, True],
          [False, False, False, False, False, False, False, False, True],
          [False, False, False, False, False, False, False, False, False]]],
        [[[False, False, False, True, True, True, True, True, True],
          [False, False, False, True, True, True, True, True, True],
          [False, False, False, True, True, True, True, True, True],
          [False, False, False, False, True, True, True, True, True],
          [False, False, False, False, False, True, True, True, True],
          [False, False, False, False, False, False, True, True, True],
          [False, False, False, False, False, False, False, True, True],
          [False, False, False, False, False, False, False, False, True],
          [False, False, False, False, False, False, False, False, False]]],
    ],
    dtype=np.bool_
)


@pytest.mark.parametrize("inp_key,shape,expected", [
    ("input_ids", (2, 9), input_ids),
    ("position_ids", (2, 2, 9), position_ids),
    ("attention_mask", (2, 1, 9, 9), attention_mask),
    ("labels", (2, 9), labels),
])
def test_data_collector(mocked_dataset, inp_key, shape, expected):
    binp = GlmDataCollector.collate_fn(mocked_dataset)
    assert type(binp) == dict
    val = binp[inp_key]
    assert tuple(val.shape) == shape
    assert np.alltrue(val.numpy() == expected)


@pytest.mark.parametrize("func,expected", [
    ("train_dev_split", Tuple),
    ("load", DataLoader),
])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
def test_data_loader(glm_dataloader, func, expected, batch_size):
    out = getattr(glm_dataloader, func)(batch_size=batch_size)
    assert isinstance(out, expected)
    if isinstance(out, DataLoader):
        dataloader = out
        num_batches = 50 // batch_size
        if 50 % batch_size != 0:
            num_batches += 1
        assert len(dataloader) == num_batches
    else:
        dataloader = out[0]
    for v in dataloader:
        input_ids = v["input_ids"]
        assert input_ids.shape[0] == batch_size
        break


@pytest.mark.parametrize("inp_key,shape", [
    ("input_ids", (1, 7)),
    ("position_ids", (1, 2, 7)),
    ("attention_mask", (1, 1, 7, 7)),
    ("labels", (1, 7)),
])
def test_dataset_collector(glm_tokenizer, inp_key, shape):
    data = [
        {"prompt": "你好", "completion": "谁"},
    ]
    ds = GlmMapStyleDataset(data, glm_tokenizer)
    binp = GlmDataCollector.collate_fn(ds)
    val = binp[inp_key]
    assert tuple(val.shape) == shape
