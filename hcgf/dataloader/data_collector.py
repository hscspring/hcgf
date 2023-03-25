from typing import List
import torch


from .data_model import DataItem, GlmBatchInput


class GlmDataCollector:

    """
    GLM special tokens:

    150000 [MASK]
    150001 [gMASK]
    150002 [sMASK]

    20000 <unk>
    20003 <pad>

    150004 <sop>  # bos
    20002 </s>    # eos
    150005 <eop>  # eop
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
        dtype=torch.int64,
    ):
        """
        Referenced from ChatGLMMModel.get_position_ids
        """
        position_ids = torch.arange(longest_seq_len, dtype=dtype)
        if position_encoding_2d:
            if not use_gmask:
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
        input_dtype: torch.Type = torch.int64
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

            MASK, gMASK = 150000, 150001
            mask_token = MASK if MASK in ids else gMASK
            mask_position = ids.index(mask_token)
            use_gmask = False if MASK in ids else True

            _masks = cls.get_masks(longest_seq_len, cxt_len)
            # 150004 == bos_token_id
            # equal to cxt_len - 1
            cxt_idx = ids.index(150004)
            _pids = cls.get_position_ids(
                longest_seq_len, cxt_idx, mask_position, use_gmask,
                position_encoding_2d=True, dtype=input_dtype
            )

            padding_len = longest_seq_len - seq_len
            _labels = ([-100] * (cxt_len - 1) +
                       ids[(cxt_len - 1):] + [-100] * padding_len)
            # 20002 == eos_token_id
            _ids = ids + [20002] * padding_len

            if input_dtype == torch.int32:
                TensorIns = torch.IntTensor
            else:
                TensorIns = torch.LongTensor

            id_list.append(TensorIns(_ids))
            pid_list.append(_pids)
            mask_list.append(_masks)
            label_list.append(TensorIns(_labels))
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
