#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="${1:-smoke}"
stage="${2:-train}"
python_bin="${project_root}/.conda/bin/python"
log_dir="${project_root}/logs/${profile}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Missing environment. Run scripts/bootstrap_server.sh first." >&2
  exit 1
fi

mkdir -p "${log_dir}" "${project_root}/.cache/huggingface" "${project_root}/tmp"
export HF_HOME="${project_root}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export TMPDIR="${project_root}/tmp"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TRL_EXPERIMENTAL_SILENCE=1
export HF_HUB_DISABLE_TELEMETRY=1
if [[ -d "${project_root}/models/Qwen2.5-0.5B" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

run_python() {
  local name="$1"
  shift
  "${python_bin}" -u "$@" 2>&1 | tee -a "${log_dir}/${name}.log"
}

case "${stage}" in
  env)
    run_python env "${project_root}/scripts/check_env.py" --config "${profile}" --load-tokenizer
    ;;
  data)
    run_python data "${project_root}/scripts/prepare_data.py" --config "${profile}"
    ;;
  sft)
    run_python sft "${project_root}/scripts/train_sft.py" --config "${profile}"
    ;;
  rm)
    run_python rm "${project_root}/scripts/train_rm.py" --config "${profile}"
    run_python rm_eval "${project_root}/scripts/evaluate_rm.py" --config "${profile}"
    ;;
  ppo)
    run_python ppo "${project_root}/scripts/train_ppo.py" --config "${profile}"
    ;;
  generate-dev)
    for model in base sft ppo; do
      run_python "generate_dev_${model}" "${project_root}/scripts/generate.py" --config "${profile}" --split dev --model "${model}"
    done
    run_python metrics_dev "${project_root}/scripts/evaluate_metrics.py" --config "${profile}" --split dev
    ;;
  generate-test)
    for model in base sft ppo; do
      run_python "generate_test_${model}" "${project_root}/scripts/generate.py" --config "${profile}" --split test --model "${model}"
    done
    run_python metrics_test "${project_root}/scripts/evaluate_metrics.py" --config "${profile}" --split test
    ;;
  judge-dev)
    run_python judge_dev "${project_root}/scripts/judge.py" --config "${profile}" --split dev --pairs ppo_sft --limit 50
    run_python judge_dev_summary "${project_root}/scripts/summarize_judge.py" --config "${profile}" --split dev
    ;;
  judge-test)
    run_python judge_test "${project_root}/scripts/judge.py" --config "${profile}" --split test
    run_python judge_test_summary "${project_root}/scripts/summarize_judge.py" --config "${profile}" --split test
    ;;
  train)
    "${project_root}/run_pipeline.sh" "${profile}" env
    "${project_root}/run_pipeline.sh" "${profile}" data
    "${project_root}/run_pipeline.sh" "${profile}" sft
    "${project_root}/run_pipeline.sh" "${profile}" rm
    "${project_root}/run_pipeline.sh" "${profile}" ppo
    "${project_root}/run_pipeline.sh" "${profile}" generate-dev
    ;;
  *)
    echo "Unknown stage: ${stage}" >&2
    exit 2
    ;;
esac
