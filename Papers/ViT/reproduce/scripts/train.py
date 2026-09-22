#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from vit_repro.config import PROJECT_ROOT, load_config
from vit_repro.data import build_loaders, infinite_batches
from vit_repro.engine import evaluate, load_checkpoint, make_scaler, run_training
from vit_repro.model import parameter_count, vit_base_patch16
from vit_repro.utils import seed_everything, sha256_file, source_manifest, utc_now, write_json
from vit_repro.weights import load_official_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="smoke")
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--weights", default="checkpoints/pretrained/ViT-B_16.npz")
    parser.add_argument("--output-root", default="checkpoints/runs")
    parser.add_argument("--run-name")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--base-lr", type=float)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--total-updates", type=int)
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {}
    for argument, field in (
        (args.seed, "seed"),
        (args.base_lr, "base_lr"),
        (args.micro_batch_size, "micro_batch_size"),
        (args.total_updates, "total_updates"),
    ):
        if argument is not None:
            overrides[field] = argument
    cfg = replace(cfg, **overrides)
    cfg.validate()
    seed_everything(cfg.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    run_name = args.run_name or f"{cfg.name}-seed{cfg.seed}"
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", cfg.to_dict())

    model = vit_base_patch16(
        image_size=cfg.image_size,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout,
        activation_checkpointing=cfg.activation_checkpointing,
    )
    weight_report = load_official_npz(model, args.weights)
    weight_report["sha256"] = sha256_file(args.weights)
    write_json(run_dir / "weight_load_report.json", weight_report)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.base_lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
    )
    scaler = make_scaler(device, cfg.precision)
    train_loader, eval_loader, split, data_generator = build_loaders(cfg, args.data_root)
    start_update = 0
    samples_seen = 0
    if args.resume:
        checkpoint = load_checkpoint(args.resume, model, optimizer, scaler, data_generator)
        start_update = int(checkpoint["next_update"])
        samples_seen = int(checkpoint["samples_seen"])
    metadata = {
        "created_at": utc_now(),
        "run_name": run_name,
        "parameter_count": parameter_count(model),
        "train_examples": len(train_loader.dataset),
        "eval_examples": len(eval_loader.dataset) if eval_loader is not None else 0,
        "accumulation_steps": cfg.accumulation_steps,
        "effective_batch_size": cfg.effective_batch_size,
        "dev_split_seed": split["seed"],
        "pretrained_sha256": weight_report["sha256"],
        "source": source_manifest(PROJECT_ROOT),
    }
    write_json(run_dir / "metadata.json", metadata)
    print(json.dumps(metadata, indent=2), flush=True)

    summary = run_training(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        batches=iter(infinite_batches(train_loader)),
        cfg=cfg,
        device=device,
        run_dir=run_dir,
        data_generator=data_generator,
        start_update=start_update,
        samples_seen=samples_seen,
    )
    if eval_loader is not None:
        evaluation = evaluate(
            model,
            eval_loader,
            device,
            cfg.precision,
            predictions_path=run_dir / f"predictions_{cfg.eval_split}.jsonl",
        )
        evaluation["split"] = cfg.eval_split
        evaluation["evaluated_at"] = utc_now()
        write_json(run_dir / f"evaluation_{cfg.eval_split}.json", evaluation)
        summary["evaluation"] = evaluation
        write_json(run_dir / "train_summary.json", summary)
        print(json.dumps(evaluation, indent=2), flush=True)


if __name__ == "__main__":
    main()
