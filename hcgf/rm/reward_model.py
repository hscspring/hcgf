"""
Referenced from `trl`
"""

import torch.nn as nn

from transfromers import AutoModel

from ..data_model import Hidden


class ValueHead(nn.Module):

    def __init__(self, config, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        hidden_size = config.hidden_size
        self.mlp = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_states: Hidden) -> Hidden:
        # BxLxH
        output = self.dropout(hidden_states)
        # BxLx1
        output = self.mlp(output)
        return output


class RewardModel(nn.Module):

    def __init__(self, model_id: str):
        super().__init__()
        self.pretrained_model = AutoModel.from_pretrained(
            model_id)
        self.value_head = ValueHead(self.pretrained_model.config)

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs
    ) -> Hidden:

        base_model_output = self.pretrained_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        # BxLxH
        last_hidden_state = base_model_output.hidden_states[-1]
        # BxLx1 => BxL
        value = self.value_head(last_hidden_state).squeeze(-1)
        return value
