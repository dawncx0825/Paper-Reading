#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from ppo_repro.config import ensure_output_dirs, load_config
from ppo_repro.utils import utc_now, write_json


def actual_winner(row):
    if row["winner"] == "tie":
        return "tie"
    return row["model_a"] if row["winner"] == "A" else row["model_b"]


def score_for(row, target):
    winner = actual_winner(row)
    if winner == "tie":
        return 0.5
    return 1.0 if winner == target else 0.0


def bootstrap(values, samples, seed):
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    estimates = [float(rng.choice(array, size=len(array), replace=True).mean()) for _ in range(samples)]
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="smoke")
    parser.add_argument("--split", choices=["dev", "test"], default="dev")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    path = cfg.result_dir / f"judge_{args.split}.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    grouped = defaultdict(list)
    for row in ok_rows:
        grouped[(row["pair"], row["content_hash"])].append(row)
    pair_results = {}
    for pair in sorted({row["pair"] for row in ok_rows}):
        target = pair.split("_")[0]
        per_post = []
        order_consistent = 0
        order_conflict = 0
        raw = Counter()
        factual_errors = Counter()
        pair_groups = [value for (name, _), value in grouped.items() if name == pair]
        complete_groups = [value for value in pair_groups if len({row["order"] for row in value}) == 2]
        for group in complete_groups:
            winners = [actual_winner(row) for row in group]
            raw.update(winners)
            order_consistent += winners[0] == winners[1]
            order_conflict += winners[0] != winners[1]
            per_post.append(float(np.mean([score_for(row, target) for row in group])))
            for row in group:
                if row["factual_error_A"]:
                    factual_errors[row["model_a"]] += 1
                if row["factual_error_B"]:
                    factual_errors[row["model_b"]] += 1
        pair_results[pair] = {
            "target_model": target,
            "complete_posts": len(per_post),
            "observed_post_groups": len(pair_groups),
            "target_preference_score": float(np.mean(per_post)) if per_post else None,
            "post_bootstrap_95ci": bootstrap(
                per_post, cfg.judge_bootstrap_samples, cfg.seed
            )
            if per_post
            else None,
            "order_consistent_posts": order_consistent,
            "order_conflict_posts": order_conflict,
            "order_consistency_rate": order_consistent / len(per_post) if per_post else None,
            "raw_actual_winners_across_calls": dict(raw),
            "factual_error_flags_across_calls": dict(factual_errors),
        }
    usage_prompt = sum((row.get("usage") or {}).get("prompt_tokens") or 0 for row in ok_rows)
    usage_completion = sum((row.get("usage") or {}).get("completion_tokens") or 0 for row in ok_rows)
    payload = {
        "created_at": utc_now(),
        "split": args.split,
        "calls": {"total_records": len(rows), "successful": len(ok_rows), "failed": len(rows) - len(ok_rows)},
        "usage": {"prompt_tokens": usage_prompt, "completion_tokens": usage_completion},
        "pairs": pair_results,
    }
    output = cfg.result_dir / f"judge_summary_{args.split}.json"
    write_json(output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

