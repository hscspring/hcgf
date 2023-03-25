from dataclasses import dataclass
import pytest

import torch
import torch.nn as nn

from hcgf.trainer.trainer import Trainer


@dataclass
class Output:
    loss: torch.FloatTensor


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Linear(10, 2)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, labels):
        z = self.mlp(input_ids)
        z = self.dropout(z)
        z = self.softmax(z)
        loss = loss_fn(z, labels)
        return Output(loss=loss)


loss_fn = nn.NLLLoss()


def gen_inp():
    return torch.randn(2, 10)


def gen_oup():
    return torch.randint(0, 2, (2,))


train_loader = [
    {
        "input_ids": gen_inp(),
        "labels": gen_oup(),
    } for i in range(20)
]

model = Model()


@pytest.mark.parametrize("lr", [1e-1])
@pytest.mark.parametrize("num_epochs", [1, 2])
@pytest.mark.parametrize("warmup_steps", [None, 10])
@pytest.mark.parametrize("accumulate_steps", [1, 8])
def test_trainer(lr, num_epochs, warmup_steps, accumulate_steps):
    trainer = Trainer(
        lr,
        num_epochs,
        warmup_steps,
        accumulate_steps,
        "test_output/",
        "cpu",
        10,
    )
    trainer.train(model, train_loader, train_loader)
