#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv --without-pip .venv
fi
if ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  bootstrap_file="$(mktemp)"
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$bootstrap_file"
  .venv/bin/python "$bootstrap_file"
  rm "$bootstrap_file"
fi
.venv/bin/python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > requirements.lock.txt
