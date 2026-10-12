#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export HF_HUB_DISABLE_TELEMETRY=1
PY="$PWD/.venv/bin/python"
profile="${1:?profile required}"
stage="${2:?stage required}"
shift 2
case "$stage" in
  env) "$PY" scripts/check_env.py ;;
  assets) "$PY" scripts/download_assets.py ;;
  data) "$PY" scripts/prepare_data.py "$profile" ;;
  sft) "$PY" -u scripts/train.py "$profile" sft --resume "$@" ;;
  dpo) "$PY" -u scripts/train.py "$profile" dpo --resume "$@" ;;
  preferred-ft) "$PY" -u scripts/train.py "$profile" preferred --resume "$@" ;;
  sample-preferences|score-preferences|cache-reference|evaluate-dev|evaluate-test)
    "$PY" -u scripts/stages.py "$profile" "$stage" "$@" ;;
  summarize) "$PY" scripts/summarize.py "$profile" ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
