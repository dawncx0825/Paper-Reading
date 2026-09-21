#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainerCallback
from trl import RewardConfig, RewardTrainer

from ppo_repro.config import ensure_output_dirs, load_config
from ppo_repro.models import align_model_and_tokenizer, load_tokenizer
from ppo_repro.utils import utc_now, write_json


class MemoryCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="smoke")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    if not cfg.sft_final_dir.exists():
        raise FileNotFoundError(f"Missing SFT checkpoint: {cfg.sft_final_dir}")
    output_root = cfg.checkpoint_dir / "rm"
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg.sft_final_dir, padding_side="right")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(cfg.sft_final_dir),
        num_labels=1,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    align_model_and_tokenizer(model, tokenizer)
    model.config.use_cache = False
    train_dataset = load_from_disk(str(cfg.processed_dir / "rm_train"))
    eval_dataset = load_from_disk(str(cfg.processed_dir / "rm_dev"))
    steps_kwargs = {"max_steps": cfg.rm_max_steps} if cfg.rm_max_steps > 0 else {}
    training_args = RewardConfig(
        output_dir=str(output_root),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1.0,
        learning_rate=5e-6,
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=cfg.sequence_max_tokens,
        eval_strategy="steps",
        eval_steps=cfg.rm_eval_steps,
        save_strategy="steps",
        save_steps=cfg.rm_eval_steps,
        save_total_limit=2,
        logging_steps=1 if cfg.profile == "smoke" else 10,
        report_to=["tensorboard"],
        logging_dir=str(cfg.log_dir / "rm_tensorboard"),
        seed=cfg.seed,
        data_seed=cfg.seed,
        optim="adamw_torch",
        remove_unused_columns=True,
        dataset_num_proc=1,
        **steps_kwargs,
    )
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[MemoryCallback()],
    )
    started = time.time()
    result = trainer.train()
    metrics = dict(result.metrics)
    metrics.update(trainer.evaluate())
    metrics.update(
        {
            "finished_at": utc_now(),
            "wall_seconds": time.time() - started,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        }
    )
    cfg.rm_final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(cfg.rm_final_dir))
    tokenizer.save_pretrained(str(cfg.rm_final_dir))
    write_json(output_root / "run_metrics.json", metrics)
    write_json(
        output_root / "provenance.json",
        {
            "created_at": utc_now(),
            "profile": cfg.profile,
            "initialization": str(cfg.sft_final_dir),
            "data_audit": str(cfg.processed_dir / "audit.json"),
        },
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

