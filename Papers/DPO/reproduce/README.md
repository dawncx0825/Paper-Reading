# DPO 单卡复现

本目录用于在 **1 × A100 40GB、192GB 内存、双路 Intel Xeon E5-2695 v3 @ 2.30GHz** 上复现 DPO。

当前状态：**2026-10-11 01:44（UTC+8）全队列完成。四档 β、主配置与 Preferred-FT 三 seed、十组独立测试均已完成并核验。主 DPO 测试奖励为 0.994014 ± 0.000401，对照为 0.639034 ± 0.003957（更新 seed 样本标准差）。** 详见 [最终复现报告](最终复现报告.md)、[轻量结果快照](results/final-2026-10-11/README.md)；规划和历史验证见 [复现规划与实验报告](report.md)。

## 第一版范围

```text
GPT-2 Large → IMDb SFT → 固定的 SFT reference
                  │              │
                  │              └→ 缓存偏好数据的 reference log-prob
                  ├→ 生成候选 → 情感分类器评分 → 偏好对
                  ├→ Preferred-FT
                  └→ DPO，扫描 β

SFT / Preferred-FT / DPO → 独立测试前缀 → reward–KL 与生成质量评测
```

已完成全参数 GPT-2 Large 的 IMDb 情感控制实验；PPO/PPO-GT 对照和 Pythia-2.8B 的 HH 对话实验仍是后续扩展。第一版不承诺复现论文全部数值，也不能在未运行 PPO 对照时声称验证了 DPO 优于 PPO。

## 执行顺序

1. 核验目标服务器，固定模型、数据和代码版本。
2. 编写并验证 token mask、DPO loss、reference 缓存和梯度累积。
3. 用 `smoke` 配置完成 SFT、偏好构造、DPO、生成及评测闭环。
4. 用 `benchmark` 配置测量真实训练、生成及评分成本。
5. 用 `pilot` 配置检查超参数和数据质量。
6. 冻结正式配置后运行 `full`，补齐主要配置的三个随机种子。
7. 汇总曲线、逐样本输出、统计结果和复现差异。

流程沿用本项目 [PPO](../../PPO/reproduce/README.md) 和 [ViT](../../ViT/reproduce/README.md) 的阶段组织与结果审计方式。

## 服务器与环境

目标服务器为 `ateam@10.15.20.29`，项目位于 `/home/ateam/dpo-reproduction`。使用独立 `.venv`，不把 SSH 密码写入任何文件。当前采用服务器已有的 Python 3.12，PyTorch 2.6.0 CUDA 12.4 和 Transformers 4.57.1，安装完成后保存完整锁文件。

本次队列曾使用 `dpo-train` 会话，编排 PID 为 `288778`，现已完成并退出。以下命令保留为部署与恢复参考，不需要重新运行已完成实验。最终状态与日志已同步至本地结果快照。

```bash
cd /home/ateam/dpo-reproduction
export PYTHONPATH="$PWD"
bash run_pipeline.sh smoke env
bash run_pipeline.sh smoke assets
.venv/bin/python -m pytest -q
bash run_pipeline.sh smoke data

# 主动中断一次，再通过 --resume 继续；仅用于首次 smoke 验证。
bash run_pipeline.sh smoke sft --stop-after 1
bash run_pipeline.sh smoke sft
.venv/bin/python scripts/validate_reload.py

# 完成小样本的全部对照与评测。
.venv/bin/python -u scripts/run_experiments.py smoke
.venv/bin/python -u scripts/benchmark_capacity.py

# 在独立 tmux 会话启动后续阶段，SSH 断开后继续运行。
tmux new-session -d -s dpo-train \
  '.venv/bin/python -u scripts/run_experiments.py benchmark pilot full > logs/orchestrator.log 2>&1'
```

`run_experiments.py` 持有单实例锁，任一阶段失败即停止并写入错误状态。`full` 依次运行共享 SFT、偏好生成及缓存、Preferred-FT 三 seed、DPO 四档 β 的 seed=42 及 β=0.1 的两个补充 seed，再执行预先固定的最终测试。这里的三 seed 仅覆盖偏好更新随机性。

pilot 结束后检查奖励范围、有效输出及退化信号；平均三元组重复率超过 0.5 或平均输出不足 8 tokens 时停止，留待检查样本后再决定是否运行 full。这是工程异常门槛，不是论文的成功标准，也不替代正式结果的人类抽查。

当前选定 SFT micro-batch=4、DPO/Preferred-FT micro-batch=8、生成 batch=32；训练有效 batch=64。完整实验默认主 β=0.1，其他 β=0.05/1/5 保留为独立对照。

## 状态与分阶段重启

```bash
cat logs/pipeline_status.json
tail -40 logs/orchestrator.log
tail -40 logs/pipeline_full.log
nvidia-smi

bash run_pipeline.sh full dpo --beta 0.1 --seed 42
bash run_pipeline.sh full evaluate-dev --run dpo_b0.1_s42
bash run_pipeline.sh full summarize
```

训练入口自动尝试 `latest` checkpoint 恢复，校验配置、代码和数据指纹；已完成且指纹一致的阶段可跳过。checkpoint 保存 RMSprop、更新数、RNG 与顺序数据位置；学习率按更新数重新计算。当前训练、生成与评分使用 FP32，关闭 TF32；DPO 使用 completion log-prob 总和。此次精度选择来自跨批次缓存验证，详见报告第 14 节。

生成中间文件为 `.partial.jsonl`；中断的生成阶段当前从头重做，不把部分文件当成完成结果。任意代码或数据变更后若触发指纹不匹配，应新建 run/profile 或审计后处理旧产物，禁止直接绕过检查。GPU 非确定性仍可能导致恢复轨迹的细小差异，不承诺逐位一致。

## 文件与产物

`configs/` 保存各 profile，`scripts/` 提供各阶段，`dpo_repro/` 包含 loss、mask 和生成逻辑，`tests/` 验证关键语义。服务器生成 `data/`、`checkpoints/`、`logs/` 和 `results/`。服务器主要结果为 `results/full/summary.json`、`seed_summary.json` 和 `reward_kl_*.png`，均已生成。本地完整轻量快照位于 `results/final-2026-10-11/`；[最终报告](最终复现报告.md)提供测试表格、β 与三 seed 对比及样本质量检查。

实现时排除环境、密钥、原始数据、模型权重和大体积 checkpoint；保留配置、版本清单和轻量结果以便审计。原论文 PDF 保持不变。
