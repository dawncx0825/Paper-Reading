# TL;DR RLHF reproduction on one RTX 3090

This directory implements the experiment in `report.md` as a staged, auditable pipeline:

```text
Qwen2.5-0.5B Base -> full-parameter SFT -> full-parameter RM
                                           |-> frozen RM
SFT -> LoRA Actor + adapter-disabled SFT Reference -> PPO
RM  -> independent full-parameter Critic -----------^
```

The implementation targets one 24GB RTX 3090. The original paper used much larger GPT-3-style models and substantially more data, so this is a method reproduction rather than an exact score reproduction.

## Reproducibility choices

- Model and both dataset snapshots are pinned by immutable Hugging Face revisions in `configs/*.json`.
- Raw Parquet files are retained and checksummed in `data/raw/manifest.json`.
- Splits are selected deterministically by content hash. Final generation-test posts are excluded from SFT, PPO, and RM training.
- RM comparisons use the row's `choice` field; array order is never assumed to encode preference.
- A dedicated `[PAD]` token is distinct from EOS. Prompt, padding, EOS, and post-EOS masks therefore remain distinguishable.
- SFT loss is completion-only. Long samples are filtered before sampling rather than silently truncating their source post.
- RM reward is centered using a fixed reference-summary subset. The scalar is saved in `reward_center.json` and applied to both the frozen RM and the Critic during PPO.
- The PPO Actor uses LoRA. Disabling the fresh adapter is the frozen SFT Reference; the script verifies equality before the first update. RM and Critic are independent model objects.
- PPO checkpoints include TRL's actor/optimizer state plus a separate Critic checkpoint. TRL 0.24.0 does not provide a fully verified mid-run resume path, so interrupted PPO recovery must be treated as unverified until explicitly tested.
- Final decoding is greedy for Base, SFT, and PPO. Qwen judging is blind, double-order, restartable, and reports order conflicts rather than discarding them.

## Server setup

On OpenBayes, keep the project and caches under `/openbayes/home`; other locations are ephemeral when the container stops.

```bash
cd /openbayes/home/ppo-reproduction
bash scripts/bootstrap_server.sh
```

The bootstrap creates a Python 3.11 environment in `.conda`, installs PyTorch 2.6.0 for CUDA 12.4, installs the pinned packages, and writes `requirements.lock.txt`.

## Execution

Always run a smoke profile before the full experiment:

```bash
./run_pipeline.sh smoke train
./run_pipeline.sh full train
```

`train` performs environment validation, data preparation, SFT, RM evaluation/centering, PPO, and development-set generation. Individual restartable stages are available:

```bash
./run_pipeline.sh full env
./run_pipeline.sh full data
./run_pipeline.sh full sft
./run_pipeline.sh full rm
./run_pipeline.sh full ppo
./run_pipeline.sh full generate-dev
./run_pipeline.sh full generate-test
```

Logs go to `logs/<profile>/`; checkpoints go to `checkpoints/<profile>/`; generated summaries and metrics go to `results/<profile>/`.

## Qwen judge

The Beijing DashScope-compatible endpoint is configured, but the key is read only from the process environment and is never printed or stored. In the same shell that starts evaluation:

```bash
export DASHSCOPE_API_KEY='your key'
./run_pipeline.sh full judge-dev
./run_pipeline.sh full judge-test
```

`judge-dev` makes 100 calls for 50 PPO-vs-SFT posts in both orders. Inspect JSON validity, usage, order consistency, and a small human sample before `judge-test`, which runs all three pairings over 500 posts (3,000 calls).

## Validation

```bash
.conda/bin/python -m pytest -q
```

Important audit files include:

- `data/processed/<profile>/audit.json`
- `logs/<profile>/environment.json`
- `checkpoints/<profile>/sft/run_metrics.json`
- `checkpoints/<profile>/rm/reward_center.json`
- `results/<profile>/rm_evaluation.json`
- `checkpoints/<profile>/ppo/model_roles.json`
- `checkpoints/<profile>/ppo/run_metrics.json`
- `results/<profile>/automatic_metrics_test.json`
- `results/<profile>/judge_summary_test.json`

Large data, caches, checkpoints, and result files are intentionally excluded from Git. The original paper PDF is never modified.

