#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from vit_repro.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output", default="results/formal_summary.json")
    args = parser.parse_args()
    rows = []
    for directory in map(Path, args.run_dirs):
        config = json.loads((directory / "config.json").read_text())
        result = json.loads((directory / "evaluation_test.json").read_text())
        train = json.loads((directory / "train_summary.json").read_text())
        rows.append(
            {
                "run": directory.name,
                "seed": config["seed"],
                "top1": result["top1"],
                "correct": result["correct"],
                "examples": result["examples"],
                "wall_seconds": train["wall_seconds"],
                "peak_allocated_bytes": train.get("peak_allocated_bytes"),
                "peak_reserved_bytes": train.get("peak_reserved_bytes"),
            }
        )
    values = [row["top1"] for row in rows]
    payload = {
        "paper_top1": 0.9167,
        "runs": rows,
        "mean_top1": statistics.mean(values),
        "sample_std_top1": statistics.stdev(values) if len(values) > 1 else None,
        "mean_minus_paper_percentage_points": (statistics.mean(values) - 0.9167) * 100,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

