# ViT-B/16 → CIFAR-100 单卡复现

> 当前为未完成的阶段性版本：已保存工程实现、短程验证记录与实验方案，完整学习率搜索及三 seed 正式结果尚未收录。进度与已有验证结果见 [复现报告](report.md)。

本目录实现 `report.md` 里的第一版实验：把 Google 官方 ImageNet-21k `ViT-B/16` JAX 权重转换到自写 PyTorch ViT，在 CIFAR-100 上以 384×384 输入做全参数微调，并与论文 Table 5 的 91.67% Top-1 比较。

## 实现边界

- 模型：12 层、768 hidden、12 heads、3072 MLP、16×16 patch、Pre-LN、近似 GELU。
- 权重：只接受官方 `ViT-B_16.npz`；分类头丢弃并零初始化，224→384 的 patch 位置编码使用二维 bicubic 插值，class token 不插值。
- 数据：训练随机裁剪面积范围 5%–100%、宽高比 3/4–4/3、随机水平翻转；评测直接 bicubic resize；像素归一化到 `[-1, 1]`。
- 调参：固定 seed=42 的分层 1,000 张开发集，每类 10 张；测试集只用于配置冻结后的正式结果。
- 优化：SGD momentum 0.9、有效 batch 512、500-step warmup、cosine decay、10,000 optimizer updates、global-norm clip 1.0、BF16。

## 服务器安装

```bash
cd /openbayes/home/vit-reproduction
conda create -p .conda python=3.11 pip -y
.conda/bin/pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 torchvision==0.21.0
.conda/bin/pip install numpy==2.1.3 pillow==11.0.0 pytest==8.3.5
export PYTHONPATH=.
```

## 执行顺序

```bash
PYTHONPATH=. .conda/bin/python scripts/check_env.py
PYTHONPATH=. .conda/bin/python scripts/download_assets.py
PYTHONPATH=. .conda/bin/python scripts/prepare_data.py
PYTHONPATH=. .conda/bin/python scripts/inspect_weights.py
PYTHONPATH=. .conda/bin/pytest

# 224 分辨率、2 updates 的端到端冒烟
PYTHONPATH=. .conda/bin/python -u scripts/train.py --config smoke

# 384 分辨率、有效 batch=512 的短测速；稳定后把配置的 total_updates 改为 100
PYTHONPATH=. .conda/bin/python -u scripts/train.py --config benchmark

# 四个论文候选学习率；每组从同一官方权重重新开始
PYTHONPATH=. .conda/bin/bash scripts/run_lr_search.sh

# 用开发集选出的学习率执行三个正式 seed，并汇总测试集结果
PYTHONPATH=. .conda/bin/bash scripts/run_formal_seeds.sh 0.01

# 或一次执行完整长流程：4 个学习率 → 自动选择 → 3 个正式 seed → 汇总
PYTHONPATH=. .conda/bin/python -u scripts/run_full_pipeline.py
```

所有输出按 run 保存到 `checkpoints/runs/`。每次运行包含配置、权重加载审计、逐 update JSONL、最终 checkpoint、逐样本预测、评测指标、耗时与峰值显存。

实现阶段还执行一次可选的跨框架检查：固定官方仓库 commit `64801f1b3b367b3611cc27a3d45cc22870a36fb3`，安装 `requirements-validation.txt` 后运行 `scripts/validate_jax_parity.py`，比较相同 FP32 输入下官方 Flax 与本实现的 224 分辨率 encoder 输出。正式训练不依赖 JAX/Flax。

## OOM 调整

先把 `micro_batch_size` 从 16 改为 8，保持 `effective_batch_size=512`；累积次数会自动从 32 变为 64。仍不足时启用 `activation_checkpointing`。这些操作不改变 optimizer update 的有效 batch 或学习率定义。

## 恢复限制

checkpoint 保存模型、优化器、AMP、Python/NumPy/PyTorch/CUDA RNG、数据生成器状态、update 和样本计数。由于多 worker 预取和在线随机图像增强，恢复后的运行在统计上可复现，但不能声称逐样本增强序列与中断前 bitwise 一致；正式独立结果优先不间断运行。
