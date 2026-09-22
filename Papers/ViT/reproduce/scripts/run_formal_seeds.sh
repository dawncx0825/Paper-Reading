#!/usr/bin/env bash
set -euo pipefail

base_lr="${1:?usage: scripts/run_formal_seeds.sh SELECTED_LR}"
for seed in 42 43 44; do
  python -u scripts/train.py \
    --config formal \
    --base-lr "${base_lr}" \
    --seed "${seed}" \
    --run-name "formal-lr-${base_lr}-seed${seed}"
done
python scripts/summarize_results.py \
  "checkpoints/runs/formal-lr-${base_lr}-seed42" \
  "checkpoints/runs/formal-lr-${base_lr}-seed43" \
  "checkpoints/runs/formal-lr-${base_lr}-seed44"

