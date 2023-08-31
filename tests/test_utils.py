import pytest

import torch

from hcgf.utils.utils import create_token_tensor_list


@pytest.mark.parametrize("tokens", [
    ["。"],
    ["。", "！"],
])
def test_create_token_tensor_list(glm_tokenizer, tokens):
    tensor_list = create_token_tensor_list(glm_tokenizer, tokens)
    assert type(tensor_list) == list
    assert len(tensor_list) == len(tokens)
    assert isinstance(tensor_list[0], torch.LongTensor)
    # [5, x], chatglm will add 5 at begin
    assert len(tensor_list[0]) == 2