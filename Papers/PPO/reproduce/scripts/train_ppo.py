#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from types import MethodType

import torch
from datasets import load_from_disk
from peft import LoraConfig, TaskType
from transformers import AutoModelForSequenceClassification
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from trl import PPOConfig, PPOTrainer

from ppo_repro.config import ensure_output_dirs, load_config
from ppo_repro.models import (
    align_model_and_tokenizer,
    apply_reward_center,
    load_causal_model,
    load_tokenizer,
)
from ppo_repro.utils import utc_now, write_json


class PPOTrainerWithCriticCheckpoints(PPOTrainer):
    """Keep the independent critic alongside TRL's policy/optimizer checkpoints."""

    def _save_checkpoint(self, model, trial):
        super()._save_checkpoint(model, trial)
        checkpoint_dir = Path(self.args.output_dir) / f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        critic_dir = checkpoint_dir / "value_model"
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.value_model.save_pretrained(str(critic_dir), safe_serialization=True)


def load_scalar_model(path: Path, tokenizer, train: bool):
    model = AutoModelForSequenceClassification.from_pretrained(
        str(path),
        num_labels=1,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    align_model_and_tokenizer(model, tokenizer)
    model.config.use_cache = False
    if train:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.requires_grad_(False)
        model.eval()
    return model


def tokenize_prompts(dataset, tokenizer):
    def tokenize(batch):
        return tokenizer(batch["prompt"], padding=False, truncation=False)

    return dataset.map(tokenize, batched=True, remove_columns=dataset.column_names, num_proc=1)


@torch.inference_mode()
def verify_adapter_reference(trainer, tokenized_dataset) -> float:
    policy = trainer.accelerator.unwrap_model(trainer.model).policy
    sample = torch.tensor([tokenized_dataset[0]["input_ids"]], device=trainer.accelerator.device)
    attention_mask = torch.ones_like(sample)
    policy.eval()
    enabled = policy(input_ids=sample, attention_mask=attention_mask, use_cache=False).logits.float()
    with policy.disable_adapter():
        disabled = policy(input_ids=sample, attention_mask=attention_mask, use_cache=False).logits.float()
    difference = float((enabled - disabled).abs().max().item())
    policy.train()
    if difference > 1e-5:
        raise RuntimeError(f"New LoRA actor does not initially match SFT reference: max_logit_diff={difference}")
    return difference


def optimizer_state_dtypes(optimizer) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                name = str(value.dtype)
                counts[name] = counts.get(name, 0) + value.numel()
    return counts


def add_checkpointing_api_to_wrapper(wrapper) -> None:
    """Bridge the gradient-checkpointing API omitted by TRL's PPO wrapper.

    TRL 0.24's generation context checks ``is_gradient_checkpointing`` on
    PolicyAndValueWrapper and then calls enable/disable methods that the
    wrapper itself does not define.  Forward both operations to the two
    wrapped trainable models so generation can temporarily disable
    checkpointing and restore the original training state.
    """

    def disable(self):
        self.policy.gradient_checkpointing_disable()
        self.value_model.gradient_checkpointing_disable()
        self.is_gradient_checkpointing = False

    def enable(self):
        self.policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.value_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        self.is_gradient_checkpointing = True

    if not hasattr(wrapper, "gradient_checkpointing_disable"):
        wrapper.gradient_checkpointing_disable = MethodType(disable, wrapper)
    if not hasattr(wrapper, "gradient_checkpointing_enable"):
        wrapper.gradient_checkpointing_enable = MethodType(enable, wrapper)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="smoke")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    center_path = cfg.checkpoint_dir / "rm" / "reward_center.json"
    if not cfg.sft_final_dir.exists() or not cfg.rm_final_dir.exists() or not center_path.exists():
        raise FileNotFoundError("SFT, RM, and reward_center.json must exist before PPO")
    with center_path.open("r", encoding="utf-8") as handle:
        reward_center = float(json.load(handle)["center"])

    output_root = cfg.checkpoint_dir / "ppo"
    output_root.mkdir(parents=True, exist_ok=True)
    tokenizer = load_tokenizer(cfg.sft_final_dir, padding_side="left")
    policy = load_causal_model(cfg.sft_final_dir, tokenizer, train=True)
    policy.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reward_model = load_scalar_model(cfg.rm_final_dir, tokenizer, train=False)
    value_model = load_scalar_model(cfg.rm_final_dir, tokenizer, train=True)
    reward_hook = apply_reward_center(reward_model, reward_center)
    value_hook = apply_reward_center(value_model, reward_center)

    raw_train = load_from_disk(str(cfg.processed_dir / "ppo_train"))
    raw_eval = load_from_disk(str(cfg.processed_dir / "ppo_eval"))
    train_dataset = tokenize_prompts(raw_train, tokenizer)
    eval_dataset = tokenize_prompts(raw_eval, tokenizer)
    if len(train_dataset) < cfg.ppo_rollout_batch_size:
        raise ValueError("PPO dataset is smaller than one rollout batch")
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    training_args = PPOConfig(
        output_dir=str(output_root),
        sft_model_path=str(cfg.sft_final_dir),
        reward_model_path=str(cfg.rm_final_dir),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=cfg.ppo_rollout_batch_size,
        learning_rate=cfg.ppo_learning_rate,
        lr_scheduler_type="constant",
        weight_decay=0.0,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        total_episodes=cfg.ppo_total_episodes,
        num_ppo_epochs=cfg.ppo_num_epochs,
        num_mini_batches=1,
        local_rollout_forward_batch_size=cfg.ppo_forward_batch_size,
        response_length=cfg.completion_max_tokens,
        stop_token="eos",
        temperature=cfg.ppo_temperature,
        missing_eos_penalty=None,
        whiten_rewards=False,
        kl_coef=cfg.ppo_kl_coef,
        kl_estimator="k1",
        cliprange=0.2,
        cliprange_value=0.2,
        vf_coef=0.1,
        gamma=1.0,
        lam=0.95,
        save_strategy="steps",
        save_steps=cfg.ppo_save_steps,
        save_total_limit=2,
        logging_steps=1,
        report_to=["tensorboard"],
        logging_dir=str(cfg.log_dir / "ppo_tensorboard"),
        seed=cfg.seed,
        data_seed=cfg.seed,
        optim="adamw_torch",
        num_sample_generations=0,
        dataset_num_proc=1,
    )
    trainer = PPOTrainerWithCriticCheckpoints(
        args=training_args,
        processing_class=tokenizer,
        model=policy,
        ref_model=None,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    add_checkpointing_api_to_wrapper(trainer.accelerator.unwrap_model(trainer.model))
    reference_difference = verify_adapter_reference(trainer, train_dataset)
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    roles = {
        "actor_trainable": sum(p.numel() for p in unwrapped.policy.parameters() if p.requires_grad),
        "actor_total": sum(p.numel() for p in unwrapped.policy.parameters()),
        "critic_trainable": sum(p.numel() for p in unwrapped.value_model.parameters() if p.requires_grad),
        "critic_total": sum(p.numel() for p in unwrapped.value_model.parameters()),
        "reward_trainable": sum(p.numel() for p in trainer.reward_model.parameters() if p.requires_grad),
        "reference_implementation": "SFT base obtained by disabling the newly initialized actor LoRA adapter",
        "initial_actor_reference_max_logit_difference": reference_difference,
    }
    if roles["reward_trainable"] != 0:
        raise RuntimeError("Reward model must be frozen")
    write_json(output_root / "model_roles.json", roles)

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    trainer.train()
    wall_seconds = time.time() - started
    cfg.ppo_actor_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(cfg.ppo_actor_dir))
    tokenizer.save_pretrained(str(cfg.ppo_actor_dir))
    critic_dir = output_root / "critic"
    unwrapped = trainer.accelerator.unwrap_model(trainer.model)
    unwrapped.value_model.save_pretrained(str(critic_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(critic_dir))
    shutil.copy2(center_path, output_root / "reward_center.json")
    trainer.save_state()
    metrics = {
        "finished_at": utc_now(),
        "wall_seconds": wall_seconds,
        "episodes": cfg.ppo_total_episodes,
        "rollout_batch_size": training_args.local_batch_size,
        "outer_batches": training_args.num_total_batches,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "optimizer_state_tensor_elements_by_dtype": optimizer_state_dtypes(trainer.optimizer),
        "model_roles": roles,
        "reward_center": reward_center,
    }
    write_json(output_root / "run_metrics.json", metrics)
    write_json(
        output_root / "actor_manifest.json",
        {
            "created_at": utc_now(),
            "base_sft_checkpoint": str(cfg.sft_final_dir),
            "reward_checkpoint": str(cfg.rm_final_dir),
            "critic_checkpoint": str(critic_dir),
            "reward_center": reward_center,
            "lora": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.0,
                "targets": sorted(peft_config.target_modules),
            },
        },
    )
    reward_hook.remove()
    value_hook.remove()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
