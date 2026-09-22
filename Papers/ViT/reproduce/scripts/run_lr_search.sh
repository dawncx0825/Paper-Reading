#!/usr/bin/env bash
set -euo pipefail

for lr in 0.001 0.003 0.01 0.03; do
  python -u scripts/train.py \
    --config "configs/dev_lr_${lr}.json" \
    --run-name "dev-lr-${lr}-seed42"
done

