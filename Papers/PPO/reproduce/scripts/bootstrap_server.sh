#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${project_root}/.conda"

mkdir -p "${project_root}/.cache/huggingface" "${project_root}/tmp"
if [[ ! -x "${env_dir}/bin/python" ]]; then
  conda create -p "${env_dir}" python=3.11 pip -y
fi

"${env_dir}/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
"${env_dir}/bin/python" -m pip install -r "${project_root}/requirements.txt"
"${env_dir}/bin/python" -m pip install -e "${project_root}" --no-deps
"${env_dir}/bin/python" -m pip freeze > "${project_root}/requirements.lock.txt"

echo "Environment ready: ${env_dir}"

