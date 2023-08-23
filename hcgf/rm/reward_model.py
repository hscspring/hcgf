"""
Referenced from `trl` and DeepSpeedChat
"""
from typing import Dict, Union

import torch
import torch.nn as nn

from transfromers import AutoModel, AutoConfig

from ..data_model import Tensor


class ValueHead(nn.Module):

    def __init__(self, config: AutoConfig, **kwargs):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        hidden_size = config.hidden_size
        self.mlp = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, hidden_states: Tensor["B,L,H", torch.FloatTensor]
    ) -> Tensor["B,L", torch.FloatTensor]:
        # BxLxH
        output = self.dropout(hidden_states)
        # BxLx1
        output = self.mlp(output)
        output = output.squeeze(-1)
        return output


class RewardModel(nn.Module):

    def __init__(self, model_id: str, pad_token_id: int):
        super().__init__()
        self.llm = AutoModel.from_pretrained(
            model_id, trust_remote_code=True)
        self.value_head = ValueHead(self.llm.config)
        self.pad_id = pad_token_id

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        **kwargs
    ) -> Dict[str, Union[
        Tensor["", torch.FloatTensor],
        Tensor["B", torch.FloatTensor]
        ]
    ]:

        tf_output = self.llm(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        # BxLxH
        lhs: Tensor["2B,L,H", torch.FloatTensor] = tf_output.hidden_states[-1]
        # BxLx1 => BxL
        rewards: Tensor["2B,L", torch.FloatTensor] = self.value_head(lhs)

        chosen_scores = []
        rejected_scores = []

        assert len(input_ids.shape) == 2
        # The real batch size, bs == batch_size // 2
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]

        chosen_ids: Tensor["B,L", torch.LongTensor] = input_ids[:bs]
        rejected_ids = input_ids[bs:]
        chosen_rewards: Tensor["B,L", torch.FloatTensor] = rewards[:bs]
        rejected_rewards = rewards[bs:]

        loss = 0
        for i in range(bs):
            chosen_id: Tensor["L", torch.LongTensor] = chosen_ids[i]
            rejected_id = rejected_ids[i]
            chosen_reward: Tensor["L", torch.FloatTensor] = chosen_rewards[i]
            rejected_reward = rejected_rewards[i]

            # The frist pad token index (at the end)
            c_inds = (chosen_id == self.pad_id).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else seq_len
            # All different corresponding token_id indexes
            check_divergence = (chosen_id != rejected_id).nonzero()

            # They are all the same
            if len(check_divergence) == 0:
                r_ind = c_ind
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
            # Not same
            else:
                r_inds = (rejected_id == self.pad_id).nonzero()
                r_ind = r_inds[0].item() if len(r_inds) > 0 else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            # Choose only different token_ids to caculate loss
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            
            chosen_scores.append(chosen_reward[c_ind - 1])
            rejected_scores.append(rejected_reward[r_ind - 1])

            loss += -torch.nn.functional.logsigmoid(
                c_truncated_reward - r_truncated_reward
            ).mean()

        loss = loss / bs
        chosen_mean_score: Tensor["B", torch.FloatTensor] = torch.stack(chosen_scores)
        rejected_scores = torch.stack(rejected_scores)
        return {
            "loss": loss,
            "chosen_scores": chosen_scores,
            "rejected_scores": rejected_scores,
        }

    def forward_value(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        prompt_length=0,
        use_cache=False
    ) -> Dict[str, Union[
        Tensor["B,L", torch.FloatTensor],
        Tensor["B", torch.FloatTensor]
        ]
    ]:

        tf_output = self.rwtranrsformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            use_cache=use_cache,
        )
        lhs: Tensor["B,L,H", torch.FloatTensor] = tf_output.hidden_states[-1]
        values: Tensor["B,L", torch.FloatTensor] = self.value_head(lhs)

        bs = values.size(0)
        seq_len = input_ids.shape[1]
        chosen_scores = []
        for i in range(bs):
            input_id = input_ids[i]
            value = values[i]
            # Ignore prompt
            c_inds = (input_id[prompt_length:] == self.pad_id).nonzero()
            c_ind = c_inds[0].item() + prompt_length if len(c_inds) > 0 else seq_len
            chosen_scores.append(value[c_ind - 1])
        return {
            "values": values,
            "chosen_scores": torch.stack(chosen_scores),
        }