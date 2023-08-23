from typing import List, Dict, Tuple, Optional
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers.tokenization_utils import PreTrainedTokenizer
from pnlp import Reader

from .dataset import GlmMapStyleDataset
from ..base import Register



class GlmDataLoader:

    def __init__(
        self,
        data_path: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int,
        pattern: str = "*.json",
        input_dtype: torch.Type = torch.int64,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pattern = pattern
        self.input_dtype = input_dtype
        self.data = self._read_files(data_path)

    def _read_files(self, data_path: str) -> List[Dict]:
        data = []
        reader = Reader(self.pattern)
        for line in reader(data_path):
            js = json.loads(line.text.strip())
            data.append(js)
        return data

    def _split(
        self,
        data: List[Dict],
        test_size: float
    ) -> Tuple[List[Dict], List[Dict]]:
        total = len(data)
        test_num = int(total * test_size)
        assert test_num < total, f"{self}: test number: {test_num} must less than total number: {total}"
        picked = np.random.choice(total, size=test_num, replace=False)
        d1, d2 = [], []
        for i, v in enumerate(data):
            if i in picked:
                d2.append(v)
            else:
                d1.append(v)
        return d1, d2

    def train_dev_split(
        self,
        batch_size: int,
        is_distributed: bool = False,
        rank: Optional[int] = None,
        train_include_dev: bool = False,
        shuffle_train: bool = True,
        dev_size: float = 0.1,
    ) -> Tuple[DataLoader, DataLoader]:
        train, dev = self._split(self.data, test_size=dev_size)
        if train_include_dev:
            train = train + dev
        train_dataset = GlmMapStyleDataset(train, self.tokenizer, self.max_seq_len)
        dev_dataset = GlmMapStyleDataset(dev, self.tokenizer, self.max_seq_len)
        # shuffle
        tdl = self._build_dataloader(train_dataset, batch_size, shuffle_train, is_distributed, rank)
        # not shuffle
        ddl = self._build_dataloader(dev_dataset, batch_size, False, is_distributed, rank)
        return tdl, ddl

    def load(
        self,
        batch_size: int,
        shuffle: bool = True,
        is_distributed: bool = False,
        rank: Optional[int] = None
    ) -> DataLoader:
        ds = GlmMapStyleDataset(self.data, self.tokenizer, self.max_seq_len)
        dl = self._build_dataloader(ds, batch_size, shuffle, is_distributed, rank)
        return dl

    def _build_dataloader(
        self,
        dataset: GlmMapStyleDataset,
        batch_size: int,
        shuffle: bool, 
        is_distributed: bool = False,
        rank: Optional[int] = None,
    ) -> DataLoader:
        if is_distributed:
            assert rank is not None, f"rank should not be None under distribute setting"
            world_size = torch.cuda.device_count()
            assert world_size > 0, f"world size should be greater than 1, but got {world_size}"
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
            dl_shuffle = False
        else:
            sampler = None
            dl_shuffle = shuffle

        ins_name = self.tokenizer.model_name
        cls_ins = Register.get(ins_name, "DataCollector")
        if not cls_ins:
            msg = f"Unsupported data collector: {ins_name}"
            raise ValueError(msg)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=dl_shuffle,
            sampler=sampler,
            collate_fn=cls_ins.collate_fn,
            pin_memory=True
        )
        return dataloader
    
    def __len__(self) -> int:
        return len(self.data)
