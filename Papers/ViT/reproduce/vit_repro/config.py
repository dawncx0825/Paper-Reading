from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    seed: int
    image_size: int
    num_classes: int
    train_subset_size: int | None
    train_split: str
    eval_split: str
    micro_batch_size: int
    effective_batch_size: int
    eval_batch_size: int
    num_workers: int
    total_updates: int
    warmup_updates: int
    base_lr: float
    momentum: float
    weight_decay: float
    max_grad_norm: float
    precision: str
    dropout: float
    attention_dropout: float
    activation_checkpointing: bool
    save_every: int
    log_every: int

    @property
    def accumulation_steps(self) -> int:
        if self.effective_batch_size % self.micro_batch_size:
            raise ValueError("effective_batch_size must be divisible by micro_batch_size")
        return self.effective_batch_size // self.micro_batch_size

    def validate(self) -> None:
        _ = self.accumulation_steps
        if self.image_size % 16:
            raise ValueError("image_size must be divisible by patch size 16")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Unsupported precision: {self.precision}")
        if self.train_split not in {"dev_train", "full"}:
            raise ValueError(f"Unsupported train_split: {self.train_split}")
        if self.eval_split not in {"dev", "test", "none"}:
            raise ValueError(f"Unsupported eval_split: {self.eval_split}")
        if not 0 <= self.warmup_updates <= self.total_updates:
            raise ValueError("warmup_updates must be within the training schedule")

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(name_or_path: str | Path) -> ExperimentConfig:
    candidate = Path(name_or_path)
    if not candidate.exists():
        candidate = PROJECT_ROOT / "configs" / f"{name_or_path}.json"
    data = json.loads(candidate.read_text(encoding="utf-8"))
    cfg = ExperimentConfig(**data)
    cfg.validate()
    return cfg

