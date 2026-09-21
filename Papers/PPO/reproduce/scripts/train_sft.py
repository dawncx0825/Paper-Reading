#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import TrainerCallback
from trl import SFTConfig, SFTTrainer

from ppo_repro.config import ensure_output_dirs, load_config
from ppo_repro.models import load_causal_model, load_tokenizer
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
    output_root = cfg.checkpoint_dir / "sft"
    output_root.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg.base_model_source, cfg.base_model_revision, padding_side="right")
    model = load_causal_model(cfg.base_model_source, tokenizer, cfg.base_model_revision, train=True)
    train_dataset = load_from_disk(str(cfg.processed_dir / "sft_train"))
    eval_dataset = load_from_disk(str(cfg.processed_dir / "generation_dev"))
    steps_kwargs = {"max_steps": cfg.sft_max_steps} if cfg.sft_max_steps > 0 else {}
    training_args = SFTConfig(
        output_dir=str(output_root),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        num_train_epochs=1.0,
        learning_rate=1e-5,
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
        packing=False,
        completion_only_loss=True,
        eval_strategy="steps",
        eval_steps=cfg.sft_eval_steps,
        save_strategy="steps",
        save_steps=cfg.sft_eval_steps,
        save_total_limit=2,
        logging_steps=1 if cfg.profile == "smoke" else 10,
        report_to=["tensorboard"],
        logging_dir=str(cfg.log_dir / "sft_tensorboard"),
        seed=cfg.seed,
        data_seed=cfg.seed,
        optim="adamw_torch",
        remove_unused_columns=True,
        dataset_num_proc=1,
        **steps_kwargs,
    )
    trainer = SFTTrainer(
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
    cfg.sft_final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(cfg.sft_final_dir))
    tokenizer.save_pretrained(str(cfg.sft_final_dir))
    trainer.save_metrics("all", metrics)
    write_json(output_root / "run_metrics.json", metrics)
    write_json(
        output_root / "provenance.json",
        {
            "created_at": utc_now(),
            "profile": cfg.profile,
            "base_model": cfg.model_name,
            "base_revision": cfg.model_revision,
            "data_audit": str(cfg.processed_dir / "audit.json"),
            "config": str(Path(args.config).resolve()) if Path(args.config).exists() else args.config,
        },
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
