#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from monitor_ppo import load_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs" / "full"
WATCH_LOG = LOG_DIR / "watch.jsonl"
ALERT = LOG_DIR / "watch.alert"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_state() -> dict[str, str] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        fields = subprocess.check_output(command, text=True, timeout=10).strip().split(", ")
    except (OSError, subprocess.SubprocessError):
        return None
    return dict(zip(("utilization_percent", "memory_mib", "temperature_c", "power_w"), fields))


def append(record: dict) -> None:
    with WATCH_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ALERT.unlink(missing_ok=True)
    while True:
        stage = (LOG_DIR / "pipeline.current").read_text(errors="replace").strip()
        record: dict = {"created_at": now(), "stage": stage, "gpu": gpu_state(), "alerts": []}
        if stage.endswith("ppo") and (LOG_DIR / "ppo.log").exists():
            metrics = load_records(LOG_DIR / "ppo.log")
            recent = metrics[-10:]
            if recent:
                keys = ["objective/kl", "objective/scores", "policy/approxkl_avg", "loss/value_avg"]
                rolling = {
                    key: sum(float(item[key]) for item in recent) / len(recent) for key in keys
                }
                rolling["eos_mean"] = sum(float(item["val/num_eos_tokens"]) for item in recent) / len(recent)
                record["ppo"] = {
                    "batches": len(metrics),
                    "episodes": metrics[-1]["episode"],
                    "rolling_window": len(recent),
                    **rolling,
                }
                if not all(math.isfinite(value) for value in rolling.values()):
                    record["alerts"].append("non-finite PPO metric")
                if rolling["objective/kl"] > 5.0:
                    record["alerts"].append("rolling sequence KL exceeds 5")
                if rolling["policy/approxkl_avg"] > 0.1:
                    record["alerts"].append("rolling policy approx-KL exceeds 0.1")
                if rolling["eos_mean"] < 16:
                    record["alerts"].append("fewer than half of rollouts end with EOS")
        exit_path = LOG_DIR / "pipeline.exit"
        if exit_path.exists():
            record["pipeline_exit"] = exit_path.read_text().strip()
        append(record)
        if record["alerts"]:
            ALERT.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if exit_path.exists():
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
