#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
log_dir="${project_root}/logs/full"
current_file="${log_dir}/pipeline.current"
exit_file="${log_dir}/pipeline.exit"

mkdir -p "${log_dir}"
rm -f "${exit_file}"

finish() {
  local code=$?
  printf '%s\n' "${code}" > "${exit_file}"
  if [[ ${code} -eq 0 ]]; then
    printf 'complete\n' > "${current_file}"
  else
    printf 'failed exit=%s\n' "${code}" >> "${current_file}"
  fi
}
trap finish EXIT

mark() {
  printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" > "${current_file}"
}

mark "waiting-for-sft"
while [[ ! -f "${log_dir}/sft.exit" ]]; do
  sleep 30
done
if [[ "$(tr -d '[:space:]' < "${log_dir}/sft.exit")" != "0" ]]; then
  echo "SFT did not finish successfully; refusing to start downstream stages" >&2
  exit 1
fi

cd "${project_root}"
for stage in rm ppo generate-dev; do
  mark "${stage}"
  ./run_pipeline.sh full "${stage}"
done

key_file="${HOME}/.config/ppo/qwen.env"
if [[ ! -r "${key_file}" ]]; then
  echo "Missing readable DashScope credential file: ${key_file}" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "${key_file}"
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is empty after sourcing ${key_file}" >&2
  exit 1
fi

for stage in judge-dev generate-test judge-test; do
  mark "${stage}"
  ./run_pipeline.sh full "${stage}"
done
