from __future__ import annotations

import contextlib
import math
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .config import ExperimentConfig
from .utils import append_jsonl, restore_rng_state, rng_state, utc_now, write_json


def learning_rate_at(update: int, cfg: ExperimentConfig) -> float:
    if update < 0 or update >= cfg.total_updates:
        raise ValueError(f"Update {update} is outside [0, {cfg.total_updates})")
    if cfg.warmup_updates and update < cfg.warmup_updates:
        return cfg.base_lr * float(update + 1) / float(cfg.warmup_updates)
    decay_updates = cfg.total_updates - cfg.warmup_updates
    if decay_updates <= 1:
        return cfg.base_lr
    progress = (update - cfg.warmup_updates) / float(decay_updates - 1)
    return cfg.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def amp_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_scaler(device: torch.device, precision: str):
    return torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and precision == "fp16")


def optimizer_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batches: Iterable,
    accumulation_steps: int,
    device: torch.device,
    precision: str,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    criterion = nn.CrossEntropyLoss()
    loss_sum = 0.0
    correct = 0
    examples = 0
    for _ in range(accumulation_steps):
        images, targets = next(batches)
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with amp_context(device, precision):
            logits = model(images)
            # Keep model compute in mixed precision but evaluate the softmax and
            # cross-entropy in FP32. This avoids BF16 quantization of losses near
            # log(100) during the zero-initialized-head warmup.
            loss = criterion(logits.float(), targets)
        scaler.scale(loss / accumulation_steps).backward()
        loss_sum += float(loss.detach()) * images.shape[0]
        correct += int((logits.detach().argmax(dim=1) == targets).sum())
        examples += images.shape[0]
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    return {
        "loss": loss_sum / examples,
        "accuracy": correct / examples,
        "grad_norm": float(grad_norm),
        "examples": examples,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader,
    device: torch.device,
    precision: str,
    predictions_path: str | Path | None = None,
) -> dict[str, float | int]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum = 0.0
    correct = 0
    examples = 0
    records = []
    for images, targets, indices in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with amp_context(device, precision):
            logits = model(images)
            loss = criterion(logits.float(), targets)
        predictions = logits.argmax(dim=1)
        loss_sum += float(loss)
        correct += int((predictions == targets).sum())
        examples += targets.shape[0]
        if predictions_path is not None:
            probabilities = logits.float().softmax(dim=1)
            confidence = probabilities.gather(1, predictions[:, None]).squeeze(1)
            for index, target, prediction, score in zip(
                indices.tolist(), targets.tolist(), predictions.tolist(), confidence.tolist()
            ):
                records.append(
                    {
                        "index": index,
                        "target": target,
                        "prediction": prediction,
                        "confidence": score,
                    }
                )
    if predictions_path is not None:
        target_path = Path(predictions_path)
        if target_path.exists():
            target_path.unlink()
        for record in records:
            append_jsonl(target_path, record)
    return {
        "loss": loss_sum / examples,
        "top1": correct / examples,
        "correct": correct,
        "examples": examples,
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    cfg: ExperimentConfig,
    next_update: int,
    samples_seen: int,
    data_generator: torch.Generator,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": cfg.to_dict(),
        "next_update": next_update,
        "samples_seen": samples_seen,
        "rng_state": rng_state(),
        "data_generator_state": data_generator.get_state(),
        "created_at": utc_now(),
        "resume_limitation": (
            "Model/optimizer/RNG state is restored, but worker prefetch and random image augmentation "
            "make resumed runs statistically reproducible rather than bitwise identical."
        ),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    data_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    restore_rng_state(checkpoint["rng_state"])
    if data_generator is not None:
        data_generator.set_state(checkpoint["data_generator_state"])
    return checkpoint


def run_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    batches: Iterable,
    cfg: ExperimentConfig,
    device: torch.device,
    run_dir: str | Path,
    data_generator: torch.Generator,
    start_update: int = 0,
    samples_seen: int = 0,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "train_metrics.jsonl"
    if start_update == 0 and metrics_path.exists():
        metrics_path.unlink()
    update_times: list[float] = []
    started = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for update in range(start_update, cfg.total_updates):
        learning_rate = learning_rate_at(update, cfg)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        metrics = optimizer_update(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            batches=batches,
            accumulation_steps=cfg.accumulation_steps,
            device=device,
            precision=cfg.precision,
            max_grad_norm=cfg.max_grad_norm,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - tick
        update_times.append(elapsed)
        samples_seen += int(metrics["examples"])
        record = {
            "timestamp": utc_now(),
            "update": update + 1,
            "learning_rate": learning_rate,
            "update_seconds": elapsed,
            "samples_seen": samples_seen,
            **metrics,
        }
        if device.type == "cuda":
            record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            record["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        append_jsonl(metrics_path, record)
        if (update + 1) % cfg.log_every == 0 or update == start_update:
            print(record, flush=True)
        if cfg.save_every and (update + 1) % cfg.save_every == 0 and (update + 1) < cfg.total_updates:
            save_checkpoint(
                run_dir / f"checkpoint-{update + 1}.pt",
                model,
                optimizer,
                scaler,
                cfg,
                next_update=update + 1,
                samples_seen=samples_seen,
                data_generator=data_generator,
            )
    save_checkpoint(
        run_dir / "final.pt",
        model,
        optimizer,
        scaler,
        cfg,
        next_update=cfg.total_updates,
        samples_seen=samples_seen,
        data_generator=data_generator,
    )
    sorted_times = sorted(update_times)
    summary = {
        "started_at_unix": started,
        "finished_at": utc_now(),
        "wall_seconds": time.time() - started,
        "updates_completed": cfg.total_updates - start_update,
        "samples_seen": samples_seen,
        "median_update_seconds": sorted_times[len(sorted_times) // 2],
        "mean_update_seconds": sum(update_times) / len(update_times),
    }
    if device.type == "cuda":
        summary["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        summary["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    write_json(run_dir / "train_summary.json", summary)
    return summary
