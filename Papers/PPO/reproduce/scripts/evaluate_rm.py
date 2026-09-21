#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification

from ppo_repro.config import ensure_output_dirs, load_config
from ppo_repro.models import align_model_and_tokenizer, load_tokenizer
from ppo_repro.utils import utc_now, write_json


@torch.inference_mode()
def score_texts(model, tokenizer, texts: list[str], batch_size: int = 4) -> list[float]:
    scores = []
    device = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)
        output = model(**batch).logits.squeeze(-1).float().cpu().tolist()
        scores.extend(output if isinstance(output, list) else [output])
    return scores


def grouped_bootstrap(accuracies, groups, samples, seed):
    by_group = defaultdict(list)
    for accuracy, group in zip(accuracies, groups):
        by_group[group].append(accuracy)
    keys = sorted(by_group)
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(samples):
        selected = rng.choice(keys, size=len(keys), replace=True)
        values = [value for key in selected for value in by_group[key]]
        estimates.append(float(np.mean(values)))
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="smoke")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    tokenizer = load_tokenizer(cfg.rm_final_dir, padding_side="right")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(cfg.rm_final_dir), torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )
    align_model_and_tokenizer(model, tokenizer)
    model.eval().cuda()
    eos = tokenizer.eos_token

    split_metrics = {}
    for split in ("rm_dev", "rm_test"):
        dataset = load_from_disk(str(cfg.processed_dir / split))
        chosen_texts = [row["prompt"] + row["chosen"] + eos for row in dataset]
        rejected_texts = [row["prompt"] + row["rejected"] + eos for row in dataset]
        chosen_scores = score_texts(model, tokenizer, chosen_texts, args.batch_size)
        rejected_scores = score_texts(model, tokenizer, rejected_texts, args.batch_size)
        accuracies = [float(chosen > rejected) for chosen, rejected in zip(chosen_scores, rejected_scores)]
        chosen_lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in chosen_texts]
        rejected_lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in rejected_texts]
        score_diff = np.asarray(chosen_scores) - np.asarray(rejected_scores)
        length_diff = np.asarray(chosen_lengths) - np.asarray(rejected_lengths)
        correlation = float(np.corrcoef(score_diff, length_diff)[0, 1]) if np.std(length_diff) else None
        groups = [row["content_hash"] for row in dataset]
        split_metrics[split] = {
            "pairs": len(dataset),
            "posts": len(set(groups)),
            "accuracy": float(np.mean(accuracies)),
            "post_grouped_bootstrap_95ci": grouped_bootstrap(
                accuracies, groups, cfg.judge_bootstrap_samples, cfg.seed
            ),
            "mean_margin": float(np.mean(score_diff)),
            "reward_length_difference_correlation": correlation,
            "chosen_reward_mean": float(np.mean(chosen_scores)),
            "rejected_reward_mean": float(np.mean(rejected_scores)),
        }

    center_dataset = load_from_disk(str(cfg.processed_dir / "rm_center"))
    center_texts = [row["prompt"] + row["completion"] + eos for row in center_dataset]
    center_scores = score_texts(model, tokenizer, center_texts, args.batch_size)
    center = float(np.mean(center_scores))
    center_payload = {
        "created_at": utc_now(),
        "definition": "Mean uncentered RM score on the fixed rm_center reference-summary set",
        "center": center,
        "samples": len(center_scores),
        "source": str(cfg.processed_dir / "rm_center"),
    }
    write_json(cfg.checkpoint_dir / "rm" / "reward_center.json", center_payload)
    payload = {"created_at": utc_now(), "splits": split_metrics, "reward_center": center_payload}
    write_json(cfg.result_dir / "rm_evaluation.json", payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

