# DPO 最终结果快照

采集日期：2026-10-11（Asia/Singapore）。服务器来源：`/home/ateam/dpo-reproduction`。

[最终复现报告](../../最终复现报告.md)汇总正式结果与限制。

- `configs/`、`requirements*.txt`、`scripts/`、`dpo_repro/`、`tests/`：运行时配置、依赖与源码副本。
- `logs/`：编排和 full 训练/验证记录；`data/full/` 仅包含轻量审计与元数据。
- `results/full/`：原始指标、统计汇总、协议和图；不包含完整逐条生成 JSONL。
- `derived/prompt_metrics/`：十组各 1,000 条前缀聚合记录，用于本地重算均值与 CI。
- `derived/samples/`：每组最多二十条生成样本；系统间隔十条，加最低奖励五条和最高重复五条，按索引去重。
- `source_manifest.json`：560 个同步源文件的 SHA-256；派生文件不在该源文件清单内。
- `derived/remote_audit.json`：服务器逐条数量、生成 ID 唯一性、数值与生成文件哈希核验。
- `derived/analysis.json`：本地独立核验与耗时统计。

原始数据、全量生成文本、模型和 checkpoint 仍留在服务器，没有被覆盖。该快照约 16 MB。报告中的来源哈希只覆盖清单文件，不能作为未下载权重的独立哈希复验。

可在复现目录运行 `python scripts/analyze_final_results.py`（需 NumPy 与 Matplotlib）重新核验并生成两张派生图。
