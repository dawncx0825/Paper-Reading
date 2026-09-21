from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ExperimentConfig:
    profile: str
    seed: int
    model_name: str
    model_revision: str
    tldr_revision: str
    comparisons_revision: str
    prompt_max_tokens: int
    completion_max_tokens: int
    sequence_max_tokens: int
    sft_train_size: int
    rm_train_size: int
    rm_dev_size: int
    rm_test_size: int
    rm_center_size: int
    ppo_total_episodes: int
    generation_dev_size: int
    generation_test_size: int
    sft_max_steps: int
    rm_max_steps: int
    sft_eval_steps: int
    rm_eval_steps: int
    ppo_save_steps: int
    ppo_num_epochs: int
    ppo_rollout_batch_size: int
    ppo_forward_batch_size: int
    ppo_learning_rate: float
    ppo_kl_coef: float
    ppo_temperature: float
    judge_model: str
    judge_base_url: str
    judge_workers: int
    judge_max_tokens: int
    judge_bootstrap_samples: int

    @property
    def processed_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "processed" / self.profile

    @property
    def checkpoint_dir(self) -> Path:
        return PROJECT_ROOT / "checkpoints" / self.profile

    @property
    def result_dir(self) -> Path:
        return PROJECT_ROOT / "results" / self.profile

    @property
    def log_dir(self) -> Path:
        return PROJECT_ROOT / "logs" / self.profile

    @property
    def local_base_model_dir(self) -> Path:
        return PROJECT_ROOT / "models" / "Qwen2.5-0.5B"

    @property
    def base_model_source(self) -> str | Path:
        return self.local_base_model_dir if self.local_base_model_dir.exists() else self.model_name

    @property
    def base_model_revision(self) -> str | None:
        return None if self.local_base_model_dir.exists() else self.model_revision

    @property
    def sft_final_dir(self) -> Path:
        return self.checkpoint_dir / "sft" / "final"

    @property
    def rm_final_dir(self) -> Path:
        return self.checkpoint_dir / "rm" / "final"

    @property
    def ppo_actor_dir(self) -> Path:
        return self.checkpoint_dir / "ppo" / "actor"


def load_config(path_or_profile: str | Path) -> ExperimentConfig:
    path = Path(path_or_profile)
    if not path.exists():
        path = PROJECT_ROOT / "configs" / f"{path_or_profile}.json"
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return ExperimentConfig(**payload)


def ensure_output_dirs(cfg: ExperimentConfig) -> None:
    for path in (cfg.processed_dir, cfg.checkpoint_dir, cfg.result_dir, cfg.log_dir):
        path.mkdir(parents=True, exist_ok=True)
