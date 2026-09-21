#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import numpy as np


ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def load_records(path: Path) -> list[dict]:
    records = []
    text = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    for line in text.replace("\r", "\n").splitlines():
        if "{'eps':" not in line:
            continue
        start, end = line.find("{"), line.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            record = ast.literal_eval(line[start : end + 1])
        except (SyntaxError, ValueError):
            continue
        if isinstance(record, dict) and "episode" in record:
            records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--total-batches", type=int, default=320)
    args = parser.parse_args()
    records = load_records(args.log)
    if not records:
        raise SystemExit("No PPO metric records found")
    recent = records[-args.window :]
    keys = [
        "objective/kl",
        "objective/scores",
        "objective/rlhf_reward",
        "policy/approxkl_avg",
        "loss/value_avg",
        "policy/entropy_avg",
        "val/num_eos_tokens",
    ]
    completed = len(records)
    payload = {
        "completed_batches": completed,
        "total_batches": args.total_batches,
        "episodes": records[-1]["episode"],
        "rolling_window": len(recent),
        "rolling_mean": {
            key: float(np.mean([float(record[key]) for record in recent])) for key in keys
        },
        "latest": {key: records[-1][key] for key in keys},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
