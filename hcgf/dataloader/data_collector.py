from typing import List
import torch


from ..data_model import DataItem, GlmBatchInput, LlamaBatchInput
from ..base import Register


@Register.regist
class ChatglmDataCollector:

    """
    GLM special tokens:

    # old version
    150004 <sop>  # bos
    150005 <eop>  # eop
    20002  </s>    # eos
    20000  <unk>
    20003  <pad>
    150000 [MASK]
    150001 [gMASK]
    150002 [sMASK]

    # new version
    130000 [MASK]  # mask
    130001 [gMASK] # gmask
    130004 <sop>   # bos
    130005 <eop>   # eos
    3      <pad>   # pad
    0      <unk>   # unk
    """

    @classmethod
    def get_masks(cls, longest_seq_len: int, cxt_len: int, dtype=torch.int32):
        """
        Referenced from ChatGLMModel.get_masks
        """
        mask = torch.ones((longest_seq_len, longest_seq_len), dtype=dtype)
        mask.tril_()
        mask[..., :cxt_len - 1] = 1
        mask.unsqueeze_(0)
        mask = (mask < 0.5).bool()
        return mask

    @classmethod
    def get_position_ids(
        cls,
        longest_seq_len: int,
        cxt_len: int,
        mask_position: int,
        use_gmask: bool,
        position_encoding_2d: bool = True,
    ):
        """
        Referenced from ChatGLMMModel.get_position_ids
        """
        dtype=torch.int64
        position_ids = torch.arange(longest_seq_len, dtype=dtype)
        if position_encoding_2d:
            # NOTE: if position_encoding_2d, use_gmask is not used
            # https://github.com/THUDM/ChatGLM-6B/issues/498#event-8971243132
            position_ids[cxt_len:] = mask_position
            block_position_ids = torch.cat((
                torch.zeros(cxt_len, dtype=dtype),
                torch.arange(longest_seq_len - cxt_len, dtype=dtype) + 1
            ))
            position_ids = torch.stack(
                (position_ids, block_position_ids), dim=0)
        else:
            if not use_gmask:
                position_ids[longest_seq_len - 1:] = mask_position
        return position_ids

    @classmethod
    def collate_fn(
        cls,
        data_items: List[DataItem],
    ) -> GlmBatchInput:
        len_ids = [len(v.input_ids) for v in data_items]
        longest_seq_len = max(len_ids)
        id_list = []
        mask_list = []
        pid_list = []
        label_list = []
        # 长的在前
        for seq_len, item in sorted(
                zip(len_ids, data_items), key=lambda x: -x[0]):
            ids = item.input_ids
            cxt_len = item.cxt_len

            MASK, gMASK = 130000, 130001
            mask_token = MASK if MASK in ids else gMASK
            mask_position = ids.index(mask_token)
            use_gmask = False if MASK in ids else True

            _masks = cls.get_masks(longest_seq_len, cxt_len)
            # equal to cxt_len - 1
            cxt_idx = ids.index(130004)
            _pids = cls.get_position_ids(
                longest_seq_len, cxt_idx, mask_position, use_gmask,
                position_encoding_2d=True
            )

            padding_len = longest_seq_len - seq_len
            _labels = [-100] * (cxt_len - 1) + \
                ids[(cxt_len - 1):] + [-100] * padding_len
            # pad_id: 3
            # NOTE: follow ChatGLM GitHub P-Tuning, chatglm_tokenizer is `padding_left` by default
            _ids = ids + [3] * padding_len

            id_list.append(torch.LongTensor(_ids))
            pid_list.append(_pids)
            mask_list.append(_masks)
            label_list.append(torch.LongTensor(_labels))
        input_ids = torch.stack(id_list)
        position_ids = torch.stack(pid_list)
        attention_mask = torch.stack(mask_list)
        labels = torch.stack(label_list)
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

def general_collate_fn(
    pad_token_id: int, 
    data_items: List[DataItem],
    with_attention: bool = True,
) -> LlamaBatchInput:
    pad_to_multiple_of = 8
    len_ids = [len(v.input_ids) for v in data_items]
    longest_seq_len = max(len_ids)
    if longest_seq_len % pad_to_multiple_of != 0:
        longest_seq_len = ((longest_seq_len // pad_to_multiple_of) + 1) * pad_to_multiple_of
    id_list = []
    mask_list = []
    label_list = []
    for seq_len, item in sorted(
            zip(len_ids, data_items), key=lambda x: -x[0]):
        ids = item.input_ids
        cxt_len = item.cxt_len

        padding_len = longest_seq_len - seq_len
        special_label_id = -100
        _labels = [special_label_id] * (padding_len + cxt_len) + ids[cxt_len:]
        # _labels = [special_label_id] * padding_len + ids
        _ids = [pad_token_id] * padding_len + ids
        _masks = [0] * padding_len + [1] * len(ids)

        id_list.append(torch.LongTensor(_ids))
        mask_list.append(torch.BoolTensor(_masks))
        label_list.append(torch.LongTensor(_labels))
    input_ids = torch.stack(id_list)
    attention_mask = torch.stack(mask_list)
    labels = torch.stack(label_list)
    oup = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    if not with_attention:
        oup.pop("attention_mask")
    return oup



@Register.regist
class Chatglm2DataCollector:
    
    @classmethod
    def collate_fn(
        cls,
        data_items: List[DataItem],
    ) -> LlamaBatchInput:
        return general_collate_fn(0, data_items, False)


@Register.regist
class LlamaDataCollector:
    
    @classmethod
    def collate_fn(
        cls,
        data_items: List[DataItem],
    ) -> LlamaBatchInput:
        return general_collate_fn(0, data_items)


@Register.regist
class Gpt2DataCollector:
    
    @classmethod
    def collate_fn(
        cls,
        data_items: List[DataItem],
    ) -> LlamaBatchInput:
        return general_collate_fn(0, data_items)


@Register.regist
class PanguDataCollector:

    @classmethod
    def collate_fn(
        cls,
        data_items: List[DataItem],
    ) -> LlamaBatchInput:
        return general_collate_fn(6, data_items)


@Register.regist
class BaichuanDataCollector:

    @classmethod
    def collate_fn(
        cls,
        data_items: List[DataItem],
    ) -> LlamaBatchInput:
        return general_collate_fn(0, data_items)


@Register.regist
class BloomDataCollector:

    @classmethod
    def collate_fn(
        cls,
        data_items: List[DataItem],
    ) -> LlamaBatchInput:
        return general_collate_fn(3, data_items)


@Register.regist
class QwenDataCollector:

    @classmethod
    def collate_fn(
        cls,
        data_items: List[DataItem],
    ) -> LlamaBatchInput:
        return general_collate_fn(151643, data_items)