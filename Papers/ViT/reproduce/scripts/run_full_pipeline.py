#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vit_repro.utils import utc_now, write_json


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "checkpoints" / "runs"
LOG_ROOT = ROOT / "logs" / "full"
RESULT_ROOT = ROOT / "results"
LEARNING_RATES = (0.001, 0.003, 0.01, 0.03)


def run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def set_status(stage: str, **details) -> None:
    write_json(LOG_ROOT / "pipeline.current.json", {"updated_at": utc_now(), "stage": stage, **details})


def train_if_needed(config: str, run_name: str, extra: list[str] | None = None) -> Path:
    directory = RUN_ROOT / run_name
    expected_split = "test" if config == "formal" else "dev"
    if (directory / f"evaluation_{expected_split}.json").exists():
        print(f"SKIP completed {run_name}", flush=True)
        return directory
    command = [sys.executable, "-u", "scripts/train.py", "--config", config, "--run-name", run_name]
    if extra:
        command.extend(extra)
    run(command)
    return directory


def main() -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    try:
        dev_runs = []
        for learning_rate in LEARNING_RATES:
            label = format(learning_rate, "g")
            run_name = f"dev-lr-{label}-seed42"
            set_status("lr_search", learning_rate=learning_rate, run_name=run_name)
            dev_runs.append(train_if_needed(f"configs/dev_lr_{label}.json", run_name))

        candidates = []
        for directory in dev_runs:
            config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
            evaluation = json.loads((directory / "evaluation_dev.json").read_text(encoding="utf-8"))
            candidates.append(
                {
                    "run": directory.name,
                    "learning_rate": config["base_lr"],
                    "dev_top1": evaluation["top1"],
                    "dev_correct": evaluation["correct"],
                    "dev_examples": evaluation["examples"],
                }
            )
        # Resolve an exact tie conservatively in favor of the smaller LR.
        selected = sorted(candidates, key=lambda row: (-row["dev_top1"], row["learning_rate"]))[0]
        selection = {
            "selected_at": utc_now(),
            "selection_rule": "highest final-step dev Top-1; exact ties choose smaller learning rate",
            "candidates": candidates,
            "selected": selected,
        }
        write_json(RESULT_ROOT / "selected_learning_rate.json", selection)

        formal_runs = []
        lr_label = format(selected["learning_rate"], "g")
        for seed in (42, 43, 44):
            run_name = f"formal-lr-{lr_label}-seed{seed}"
            set_status(
                "formal_training",
                learning_rate=selected["learning_rate"],
                seed=seed,
                run_name=run_name,
            )
            formal_runs.append(
                train_if_needed(
                    "formal",
                    run_name,
                    ["--base-lr", str(selected["learning_rate"]), "--seed", str(seed)],
                )
            )
        set_status("summarizing", learning_rate=selected["learning_rate"])
        run(
            [
                sys.executable,
                "scripts/summarize_results.py",
                *[str(directory) for directory in formal_runs],
                "--output",
                str(RESULT_ROOT / "formal_summary.json"),
            ]
        )
    except BaseException as error:
        write_json(LOG_ROOT / "pipeline.exit.json", {"exit_code": 1, "failed_at": utc_now(), "error": repr(error)})
        set_status("failed", error=repr(error))
        raise
    write_json(LOG_ROOT / "pipeline.exit.json", {"exit_code": 0, "completed_at": utc_now()})
    set_status("complete")


if __name__ == "__main__":
    main()

