#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import Counter

import numpy as np
from rouge_score import rouge_scorer

from ppo_repro.config import ensure_output_dirs, load_config
from ppo_repro.utils import utc_now, write_json


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def repetition_rate(text: str, n: int = 3) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(tokens[index : index + n]) for index in range(max(0, len(tokens) - n + 1))]
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values())
    return repeated / len(grams)


def summarize_model(records):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(row["reference"], row["summary"]) for row in records]
    return {
        "samples": len(records),
        "rouge1": float(np.mean([score["rouge1"].fmeasure for score in scores])),
        "rouge2": float(np.mean([score["rouge2"].fmeasure for score in scores])),
        "rougeL": float(np.mean([score["rougeL"].fmeasure for score in scores])),
        "word_count_mean": float(np.mean([len(row["summary"].split()) for row in records])),
        "generated_token_mean": float(np.mean([row["generated_tokens"] for row in records])),
        "repetition_3gram_mean": float(np.mean([repetition_rate(row["summary"]) for row in records])),
        "empty_rate": float(np.mean([not row["summary"].strip() for row in records])),
        "eos_rate": float(np.mean([row["hit_eos"] for row in records])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="smoke")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    all_records = {}
    for model in ("base", "sft", "ppo"):
        path = cfg.result_dir / f"generations_{args.split}_{model}.jsonl"
        all_records[model] = read_jsonl(path)
    hash_sets = {name: {row["content_hash"] for row in rows} for name, rows in all_records.items()}
    if len({frozenset(values) for values in hash_sets.values()}) != 1:
        raise RuntimeError(f"Generation files do not contain identical posts: { {k: len(v) for k,v in hash_sets.items()} }")
    payload = {
        "created_at": utc_now(),
        "split": args.split,
        "models": {name: summarize_model(rows) for name, rows in all_records.items()},
    }
    output = cfg.result_dir / f"automatic_metrics_{args.split}.json"
    write_json(output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

