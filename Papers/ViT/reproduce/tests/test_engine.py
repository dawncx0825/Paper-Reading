from dataclasses import replace

import torch
from torch import nn

from vit_repro.config import load_config
from vit_repro.engine import evaluate, learning_rate_at, make_scaler, optimizer_update


def test_learning_rate_warmup_and_cosine_end():
    cfg = replace(load_config("smoke"), total_updates=10, warmup_updates=2, base_lr=0.1)
    assert learning_rate_at(0, cfg) == 0.05
    assert learning_rate_at(1, cfg) == 0.1
    assert learning_rate_at(9, cfg) == 0.0


def test_accumulation_performs_one_optimizer_step():
    model = nn.Linear(3, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scaler = make_scaler(torch.device("cpu"), "fp32")
    batches = iter(
        [
            (torch.randn(2, 3), torch.tensor([0, 1])),
            (torch.randn(2, 3), torch.tensor([1, 0])),
        ]
    )
    steps = 0
    original_step = optimizer.step

    def counted_step(*args, **kwargs):
        nonlocal steps
        steps += 1
        return original_step(*args, **kwargs)

    optimizer.step = counted_step
    metrics = optimizer_update(
        model,
        optimizer,
        scaler,
        batches,
        accumulation_steps=2,
        device=torch.device("cpu"),
        precision="fp32",
        max_grad_norm=1.0,
    )
    assert steps == 1
    assert metrics["examples"] == 4


def test_evaluation_counts_last_partial_batch():
    class ConstantModel(nn.Module):
        def forward(self, images):
            logits = torch.zeros(images.shape[0], 2)
            logits[:, 0] = 1
            return logits

    batches = [
        (torch.randn(3, 3), torch.tensor([0, 1, 0]), torch.tensor([0, 1, 2])),
        (torch.randn(2, 3), torch.tensor([0, 1]), torch.tensor([3, 4])),
    ]
    result = evaluate(ConstantModel(), batches, torch.device("cpu"), "fp32")
    assert result["examples"] == 5
    assert result["correct"] == 3

