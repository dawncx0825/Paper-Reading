# DPO 单卡复现规划与实验报告

> 创建日期：2026-10-08（北京时间）。  
> 当前状态：2026-10-11 01:44（UTC+8）benchmark → pilot → full 全队列完成，十组正式测试及三 seed 汇总已核验；详见[最终复现报告](最终复现报告.md)和第 14.5 节。  
> 目标硬件：1 × NVIDIA A100 40GB；192GB 系统内存；双路 Intel Xeon E5-2695 v3 @ 2.30GHz。

第一版复现 GPT-2 Large 在 IMDb 上的偏好优化，完成 SFT、Preferred-FT、DPO 对照及 reward–KL 曲线。先验证实现与评测，再扩展 PPO 对照和 Pythia-2.8B 对话实验。结果无提升或出现退化也应完整报告，不把训练成功等同于论文结论得到验证。

## 1 目标与复现边界

论文为 *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*，NeurIPS 2023。使用工作区中的 [原论文](../2305.18290v3.pdf) 和 [论文总结 PDF](../论文总结.pdf)。

| 层次 | 内容 | 完成后可支持的结论 |
| --- | --- | --- |
| 实现验证 | 手写 DPO loss，与参考实现核对；验证数据和训练循环 | 实现符合所选 DPO 目标 |
| 第一版主实验 | GPT-2 Large + IMDb；SFT、Preferred-FT、DPO；β 扫描 | 当前设置下偏好优化的效果和 reward–KL 权衡 |
| 后续算法对照 | 同任务加入 PPO 和 PPO-GT | 在匹配的评测与 KL 范围内比较算法 |
| 后续对话实验 | Pythia-2.8B + Anthropic HH | 检验真实人类偏好数据上的迁移效果 |

第一版不包含 GPT-J TL;DR、Best of 128、完整人类研究或所有论文消融。没有实际 PPO 对照，不得声称复现了“DPO 优于 PPO”。现有 PPO 项目使用另一模型和摘要任务，其分数不可直接作为本实验对照。

数据子集、截断、优化器实现、采样长度及精度都有可能与历史实验不同。为每项差异保留依据与理由，将“沿用论文模型和目标”与“严格重现论文数值”区分开。

## 2 沿用现有项目的工作方式

参考 [PPO 方案](../../PPO/reproduce/report.md)、[PPO 入口](../../PPO/reproduce/README.md)、[ViT 方案](../../ViT/reproduce/report.md) 和 [ViT 入口](../../ViT/reproduce/README.md)，沿用以下约定：

- `README.md` 提供状态、环境和运行入口；`report.md` 保存规划并在执行后追加实测结果。
- 先小样本冒烟，再用正式长度测速，再进行开发集配置选择和独立测试。
- 模型、数据、代码固定到不可变 revision 或 commit；保留原始数据清单、校验和及划分索引。
- 每次运行保存完整配置、环境、seed、更新次数、逐步日志、耗时、显存和生成样本。
- 调参不读取测试结果。工程验收与效果验收分别报告。
- checkpoint 必须能保存、重载并核对输出；未验证的恢复能力不写成已支持。

本次不照搬旧项目的 RTX 3090 配置、软件锁文件或服务器目录。训练机器的操作系统、驱动、存储位置与可用磁盘在阶段 A 核验。

## 3 论文设置与本项目选择

原论文正文第 6 节、附录 B 和 C.1 给出了以下依据。未明确的设置不补写为论文原值。

| 项目 | 论文依据 | 本项目计划 |
| --- | --- | --- |
| 基座 | GPT-2 Large | 沿用，固定下载 revision |
| 任务 | IMDb 正面情感续写 | 沿用 |
| SFT | IMDb 子集上训练 1 epoch，见附录 C.1 | 训练划分上 1 epoch，记录具体子集 |
| 前缀长度 | 2–8 tokens | 沿用，以 GPT-2 tokenizer 计数 |
| 候选生成 | 25,000 个前缀，每个生成 4 条 | 先调试，再扩大到此目标规模 |
| 偏好构造 | 每个前缀形成 6 对 | 最多 150,000 对；同分或重复过滤后报告实际数 |
| 外部奖励 | `siebert/sentiment-roberta-large-english` | 沿用并固定 revision |
| DPO 默认配置 | β=0.1、batch=64、RMSprop、lr=1e-6、150-step warmup | 正式 DPO 优先沿用 |
| 情感实验 β 扫描 | 0.05、0.1、1、5 | 开发阶段扫描四档 |
| 序列长度及候选解码细节 | 实现前进一步核对原始代码 | 使用下文工程起点，差异显式记录 |

官方仓库当前 `config.yaml` 的学习率为 `5e-7`，与附录默认 `1e-6` 不同。正式配置必须显式填写参数，第一版以附录 DPO 学习率为起点，不能让库默认值悄悄决定实验。

## 4 硬件与软件方案

### 4.1 资源安排

| 资源 | 计划 |
| --- | --- |
| A100 40GB | 尽量独占；GPT-2 Large 全参数训练；优先 BF16 混合精度 |
| 192GB 内存 | 原始数据处理、token 缓存和 checkpoint 装载；必要时支持卸载 |
| 双路 E5-2695 v3 | 数据 worker 从 4–8 个起测；避免 tokenizer 与 BLAS 线程同时过量并行 |
| GPU 与 CPU 拓扑 | 记录 NUMA 和 GPU 所在 CPU 节点，只有发现供数或传输瓶颈时再调整绑定 |
| 磁盘 | 初期建议 100–200GB 可用 SSD；根据 checkpoint 保留策略更新预算 |
| 显存余量 | 压力测试覆盖最长序列、首次 optimizer step、保存和生成；尽量保留约 3GiB 余量 |

192GB 内存不能替代显存。CPU offload 增加传输及 CPU 运算开销，第一版不作为默认路径。单卡 FSDP 不会凭空增加总显存容量。

参数文件大小不是训练显存。以某些混合精度 Adam 实现约 16 字节/参数的训练状态估算，2.8B 约需 44.8GB，还未包含激活和 reference；这只是说明默认配方可能超限，不是本机实测。RMSprop 状态较少，但仍需记录实际参数、梯度和状态精度。

### 4.2 软件与版本

- Python 3.11 独立环境；选择与目标 GPU 驱动兼容的 PyTorch CUDA 构建。
- Transformers、Datasets、Accelerate、TRL 选择互相兼容的一组版本，冒烟通过后写入锁文件。
- 手写最小 DPO loss 及 completion log-prob 用于校验，正式训练优先复用经验证的 Trainer；RMSprop 通过明确配置或自定义 optimizer 接入。
- 固定原始 DPO 仓库 commit，核对 loss、偏好格式与超参数；不直接混用当前文档与历史 API。
- 优先使用已支持的注意力实现，初期不强制编译 FlashAttention，不引入多卡或量化依赖。
- 日志优先本地 JSON/JSONL/CSV；第一版训练、评分和评测不需要付费模型 API。

原始仓库的单卡 `BasicTrainer` 与多卡示例的混合精度、激活检查点支持存在区别。移植时核验实际能力，不能仅修改 trainer 名称就认为设置等价。

## 5 数据与模型流向

```text
IMDb 原始数据 → 按影评隔离训练 / 开发 / 测试
                     │
                     ├→ 训练影评 → GPT-2 Large SFT → 固定 SFT checkpoint
                     │                                  │
                     └→ 训练前缀 → 每个生成 4 条候选 ←───────┘
                                          │
                               固定情感分类器评分
                                          │
                                  chosen / rejected
                                          │
                           固定 reference log-prob 缓存
                                          │
                  相同 SFT 起点 ────┬───────┴─────┐
                                   ▼             ▼
                              Preferred-FT      DPO
                                   │             │
                     SFT ──────────┴──────┬──────┘
                                         ▼
                         独立前缀的生成和 reward–KL 评测
```

### 5.1 划分与审计

1. 保留官方 train/test 边界，从 train 按影评 ID 或规范化内容哈希留出约 1,000 条作为开发池；去重后记录实际规模。
2. 同一影评的所有前缀、候选和偏好对只属于一个划分，禁止按单个偏好对随机切分。
3. 从官方 test 预留固定 1,000 条影评作为最终生成测试池，第一版报告这是测试子集。测试池不参与 SFT、候选训练数据构造、β 或 checkpoint 选择。
4. 首轮不引入 IMDb 未标注 split。若后续扩大数据范围，作为配置变更记录。
5. 训练前缀目标为 25,000 个。若可用训练影评不足 25,000，可对训练影评构造不同长度的前缀；记录重复采样方式、唯一影评数和唯一前缀数，不从开发或测试池补数。
6. 完全相同前缀可能来自不同影评；额外检查跨集合前缀重复，报告处理规则及筛选数量。短前缀重合不等同于完整影评泄漏，但应避免重复评测输入影响结论。
7. 保存长度分位数、去重和过滤计数、标签来源、原始 split、样本 ID、内容哈希和来源 revision。

### 5.2 分阶段规模

| profile | SFT 数据 | 训练前缀与偏好对 | 评测 |
| --- | --- | --- | --- |
| smoke | 128 条影评，少量更新 | 32 个前缀，最多 192 对 | 16 个开发前缀 |
| benchmark | 已处理训练样本，正式长度 | 足够覆盖约 100 个稳定 micro-step 及多次完整更新 | 独立测生成和评分吞吐 |
| pilot | 约 5,000 条训练影评，1 epoch | 1,000 个前缀，最多 6,000 对 | 固定 200 个开发前缀 |
| full | 全部选定训练影评，1 epoch | 25,000 个前缀，最多 150,000 对 | 开发 500、最终测试 1,000 个前缀 |

数量是清洗后目标；不足时使用实际数量。开发偏好对由开发池独立生成，仅用于验证 loss 和排序能力。各 profile 使用独立输出路径，pilot 模型不直接作为 full 的初始化。

### 5.3 序列与奖励约定

- SFT 工程起点为影评的 causal LM 训练，长度上限 256，首轮不跨样本 packing；固定文本切分、截断及 EOS 规则。论文未充分说明的预处理细节作为复现差异记录。
- 偏好数据前缀 2–8 tokens，工程起点为最多生成 128 个新 tokens。记录自然 EOS 与长度上限停止的比例。
- 候选采样起点：temperature=1、top_p=1、关闭 top_k 截断，每个前缀 4 次随机生成。正式配置前核对原始实现；这些值不是已确认的论文超参数。
- 分类器输入为完整前缀与续写；按模型标签映射读取正面概率，不能直接把 pipeline 返回的“预测类别置信度”当正面概率。
- 情感分类器使用自己的 tokenizer，检查其最大长度；超长输入不得静默截断。记录候选实际被评分的文本及分数。
- 同一前缀按正面概率排序；重复 completion 和数值同分不强行指定偏好。保存过滤原因及实际对数。
- Preferred-FT 第一版在偏好对的 chosen 上继续训练，保留同一 completion 在不同 pair 中的出现次数，使其与 DPO 的 pair 暴露频次对应。
- DPO 仅对 completion 和实际生成的 EOS 求 token log-prob 之和；不对长度取平均，不把 prompt 或 padding 加入目标。
- 达到长度上限而未生成 EOS 的候选，不伪造“模型采样了 EOS”的记录。任何训练用 EOS 添加规则都要与生成记录区分并核对。
- 若 PAD 与 EOS 共用 token ID，依据位置和 attention mask 区分，禁止按 ID 把全部 EOS 屏蔽。

## 6 训练配置与正确性

### 6.1 各阶段起点

| 项目 | SFT | Preferred-FT | DPO |
| --- | --- | --- | --- |
| 初始化 | GPT-2 Large Base | 固定 SFT | 相同固定 SFT |
| 更新范围 | 全参数 | 全参数 | 全参数 |
| 训练信号 | 影评语言建模 | chosen 的 completion CE | 标准 sigmoid DPO |
| 优化器 | RMSprop，工程选择 | RMSprop | RMSprop，论文默认 |
| 学习率 | 1e-5，工程起点 | 1e-6，工程起点 | 1e-6，论文默认 |
| 训练量 | 1 epoch | 偏好对数据 1 epoch | 偏好对数据 1 epoch，工程起点 |
| 有效 batch | 64 条影评 | 64 个 pair 对应的 chosen | 64 个偏好对 |
| micro-batch | 从 2 起测 | 从 2 起测 | 从 1–2 个 pair 起测 |
| warmup | 总更新数的约 5%，工程选择 | 正式 150 updates，工程选择 | 正式 150 updates，论文默认 |

首轮学习率之后是否搜索，由 pilot 开发结果决定；不得用测试结果反向调整。正式 warmup 后保持常数学习率作为初始日程，所有日程、梯度裁剪、RMSprop 的 alpha/eps/momentum/weight_decay 显式记录并与固定版本源码核对。smoke/pilot 若更新很少，将 warmup 限制在计划更新数约 10% 内，单列为调试改动。

BF16 指混合精度计算，不意味着直接把所有参数、累积梯度和优化器状态转为 BF16。优先 FP32 参数与状态、BF16 autocast，log-softmax 与序列求和的精度通过数值检查确定。

一个 DPO pair 包含两条序列。micro-batch=2、累积=32 时，有效 batch=64 对，每次前向通常处理 4 条序列。调度、warmup、保存和评测间隔一律按 optimizer update 计数。尾批按实际 pair 数加权，避免短 batch 被不成比例地放大。

### 6.2 固定 reference 缓存

SFT 完成后，在 eval 模式预计算训练集和验证集的 chosen/rejected completion log-prob。缓存按样本 ID 保存两个分数，并绑定 SFT 权重哈希、tokenizer、token IDs、mask、截断规则和代码版本。核对后释放 reference 的 GPU 对象，再初始化正式训练进程。

TRL 的 `precompute_ref_log_probs` 可以作为实现路径，但必须验证所选版本的行为与内存释放情况。reference 固定，关闭同步更新；缓存只覆盖静态数据，生成评测的新样本仍需重新计算 reference log-prob，可按阶段顺序加载模型以节约显存。

缓存失配即失效，禁止跨不同 SFT、tokenizer 或长度配置复用。不同 β 和更新 seed 在起点及输入完全相同时可以共享缓存。

### 6.3 必要验证

- 手工小张量和官方 loss 对照：loss、符号和梯度方向正确；policy=reference 且关闭 dropout 时，初始 loss 接近 `log(2)`。
- 手工 token 例子覆盖 causal shift、prompt 边界、padding、真实 EOS、不同 completion 长度，确认使用 log-prob 总和。
- 在固定 FP32 小批次上比较缓存与在线 reference，预先设定误差容差；BF16 另设合理容差。
- 确认 reference 不参与梯度更新，训练前后其权重标识一致。
- 在关闭随机层的小模型上比较梯度累积与等效大 batch，覆盖最后不足一个完整 batch 的情况。
- 保存和重载后核对固定输入的 logits、loss 和生成结果；再验证恢复 optimizer、scheduler、RNG 和数据位置。
- 在完整优化器状态已创建后记录显存，覆盖长输入和评测切换，不能只验证能加载模型。

dropout 策略必须明确：reference 始终关闭；DPO policy 第一版关闭 dropout，记录为工程选择。正确性测试只承担关键语义验证，不用无关测试扩充范围。

## 7 实验顺序与选择规则

| 阶段 | 工作 | 通过条件 |
| --- | --- | --- |
| A 环境与来源 | 核验 GPU、BF16、存储；固定代码、模型、数据版本 | 离线加载正常，来源可追溯 |
| B 实现验证 | 完成第 6.3 节检查 | loss、mask、缓存、累积与保存语义正确 |
| C 冒烟与测速 | smoke 全流程；benchmark 正式长度 | 无 OOM/NaN，取得各阶段吞吐与峰值 |
| D 小规模试验 | pilot 比较 SFT、Preferred-FT、β=0.1 DPO | 数据与生成可解释，确定正式配置 |
| E 正式探索 | full 共享 SFT；β=0.05/0.1/1/5，seed=42 | 得到开发 reward–KL 曲线和资源成本 |
| F 重复验证 | 选定 β 的 DPO 与 Preferred-FT 各补 seed=43/44 | 三个更新 seed 的独立结果齐全 |
| G 最终评测 | 配置及 checkpoint 选择规则冻结后运行 test | 逐样本输出、置信区间和报告完整 |

首轮默认 β=0.1。β 扫描用于展示权衡，不直接按最高 reward 选冠军：参考开发 reward–KL 非支配曲线和重复率、长度质量检查。若正式结果需要单一推荐 β，应在测试前写明开发集决策理由；没有明确理由时保留 β=0.1 为预设主配置。

最终主结果使用各 run 的最后 checkpoint，避免按测试或单一训练 reward 挑最好的一步。开发阶段每约 100 optimizer updates 加训练结束时记录曲线；正式测试按预先固定的 checkpoint 清单生成点，不根据测试曲线再追加调参。

三个更新 seed 共享同一 SFT 和偏好数据，度量的是该固定起点下的训练随机性，不是整个流程的方差。若要对完整复现稳定性作结论，需额外重复 SFT 和候选生成。四档 β 全部三 seed 的完整扫描为后续预算允许时扩展，不能提前写成已完成。

## 8 评测与结果口径

### 8.1 主指标

| 指标 | 定义与用途 |
| --- | --- |
| 情感奖励 | 固定分类器对生成文本的正面概率，报告均值与分布 |
| KL 估计 | 当前 policy 生成分布相对固定 SFT 的序列 KL 蒙特卡洛估计 |
| 偏好准确率 | 验证偏好对上 implicit reward 是否正确排序 |
| 质量检查 | 生成长度、EOS/上限停止比例、重复率和盲抽文本 |
| 训练与资源 | loss、grad norm、log-prob、pair/s、token/s、GPU 小时和显存 |

DPO 的 implicit reward 与外部分类器奖励分别记录，不能互相替代。偏好准确率或 loss 变好，不足以证明生成文本质量提高。

### 8.2 reward–KL 曲线

在独立前缀上从当前 policy 采样 completion，估计：

```text
KL estimate = mean[sum over generated tokens(log π_policy(token | history)
                                            - log π_SFT(token | history))]
```

用于此估计的采样采用 temperature=1、top_p=1、无 top_k 截断，统一 EOS 与最大新 token 数，使采样分布和计算 log-prob 的分布一致。若更换温度或截断策略，必须重新定义评测分布，不能直接沿用上式声称测得原始 policy 的 KL。

报告这是有限长度生成过程的序列 KL 估计，单位为 nats/sequence；每 token 值可作辅助但不混用横轴。有限样本估计可能轻微为负，保留估计及误差而非强行截为零。离线 chosen/rejected 的平均 log-ratio 不能代替该指标。

开发使用固定前缀及采样 seed 列表。最终建议每个测试前缀采样 4 次，并按前缀聚类 bootstrap 计算 95% 置信区间；同一前缀的多个输出不能当成完全独立样本。不同方法共享测试前缀和采样次数。

画出各 β、各 checkpoint 的全部点，另标非支配前沿；没有匹配 KL 的重叠区间时，不强行宣称某方法全面优于另一方法。分类器奖励可能受到投机利用，至少固定随机抽查 50 个前缀下各方法的输出，记录重复、无意义和语义问题。

### 8.3 结果验收

工程验收要求来源、划分、训练、保存重载、评测和运行记录完整。结果验收要求如实说明 DPO 相对 SFT/Preferred-FT 的改善、无明显差异或退化，并报告训练 seed 差异和评测采样误差。不预设必须达到某个奖励值，不以一条最高分文本作为成功证据。

## 9 单卡适配与恢复

显存不足时依次处理：

1. 确认 reference 缓存完成且 GPU 副本释放；训练、候选生成、分类器评分串行运行。
2. 减少 micro-batch 并增加累积，保持有效 batch 和 optimizer update 定义。
3. 启用并验证梯度检查点，减少评测 batch；生成与训练正确切换 `use_cache`。
4. 查明 logits、长序列 padding、优化器临时张量和状态精度的占用。
5. 必要时考虑 CPU offload。修改长度、换小模型、LoRA 或量化均另列配置，不把实验目标变化隐藏为无影响优化。

checkpoint 保存模型、tokenizer、optimizer、scheduler、update、已处理样本计数、Python/NumPy/PyTorch/CUDA RNG、数据采样状态、配置及 reference 缓存标识。FP16 回退时额外保存 scaler。

恢复路径需要一次人为中断实验验证。多 worker 预取、随机采样或非确定性 kernel 若影响逐步一致性，应明确恢复限制；正式独立结果优先不中断运行。重启不得覆盖已有 run，失败日志保留。

## 10 时间与存储预算

### 10.1 以测量更新预算

以约两周作为开发与实验排期窗口，不作为 GPU 耗时承诺：环境与验证约 1–2 天，小数据闭环约 2–3 天，正式实验和分析约 5–8 天；实际取决于测速、失败修复和可用运行时段。

benchmark 同时记录 warmup 后稳定 micro-step 吞吐和完整 optimizer update 耗时，覆盖多个完整更新。分别测 SFT、DPO、候选生成、reference 打分、分类器评分和评测；GPU 计时同步或使用 CUDA events，不能只计 CPU 提交时间。

```text
单次 DPO 纯训练小时 = 实际偏好对数 × epoch / 实测偏好对每秒 / 3600
候选生成条数 = 训练前缀数 × 4
总预算 = SFT + 候选生成 + 分类器评分 + reference 缓存
       + 全部 DPO/Preferred-FT 运行 + 开发/测试生成 + 保存与分析
```

正式规模最多 150,000 对，batch=64、1 epoch、保留尾批时约 2,344 次 optimizer updates。25,000 个前缀需要 100,000 条候选生成，候选长度不同会明显改变耗时。先记录实际平均长度，再估算总成本。

至少预算四个 β 的探索、主配置两个补充 seed、Preferred-FT 三 seed 和共享 SFT。第一版无训练 API 费用；自有 GPU 记录时间及可获得的能耗，租用 GPU 按实际单价另算。

### 10.2 保留策略

原始数据和模型缓存复用；每个 run 保留必要恢复点、最后模型、完整轻量日志和预设评测点的输出。多个 β 可以复用相同 SFT 和 reference 缓存。根据实测 checkpoint 大小再确定保存频率，避免无限保留完整 optimizer 状态。

## 11 文件组织与交付

以下为规划结构；代码现已按此组织实现。实际运行数据按 `data/<profile>/` 保存，checkpoint 按 `checkpoints/<profile>/<run>/` 保存，日志与结果采用同样分层。目录中不存在的结果文件不表示已有实验结果：

```text
reproduce/
├── README.md                   # 状态、安装、阶段入口与恢复说明
├── report.md                   # 规划与后续实测结果
├── pyproject.toml              # 项目与依赖配置
├── requirements.lock.txt       # 冒烟通过后固定的软件环境
├── configs/                    # smoke / benchmark / pilot / full
├── dpo_repro/                  # 数据、log-prob、loss、训练与评测
├── scripts/                    # 下载、训练、生成、评分、缓存、汇总
├── tests/                      # mask、loss、缓存、累积及恢复的关键检查
├── data/
│   ├── raw/                    # 来源文件与下载 manifest
│   ├── splits/                 # 影评和前缀的固定划分
│   ├── processed/             # 各 profile 的 token 和偏好数据
│   └── ref_cache/              # 绑定 SFT 及输入版本的 log-prob
├── checkpoints/runs/           # run 独立的权重与恢复状态
├── logs/runs/                  # 环境、配置、JSONL 与资源记录
└── results/                    # 逐样本输出、统计汇总与图表
```

优先实现的阶段入口依次为 `env`、`data`、`sft`、`sample-preferences`、`score-preferences`、`cache-reference`、`preferred-ft`、`dpo`、`evaluate-dev`、`evaluate-test`、`summarize`。阶段接受 profile 与 run 标识，依赖缺失时清晰报错；输入版本不一致时不得悄悄复用旧产物。

最终交付包括可执行代码及锁定配置、数据审计清单、关键检查结果、每个 run 的指标和成本、reward–KL 图、抽样文本检查、复现差异表及结论。版本控制保留轻量审计结果，忽略密钥、环境、原始大数据、权重和大型 checkpoint；实施时添加对应 `.gitignore`。

本文件是复现文档，按项目规则不属于“总结 Markdown 只提交 PDF”的范围。论文总结仍沿用项目的 PDF 提交规则；训练与监测阶段未提交或推送文件；最终报告与轻量证据按用户要求归档至 Paper-Reading。

## 12 后续扩展

### 12.1 PPO 和 PPO-GT

在 IMDb 上从相同 SFT、训练前缀和生成评测规则出发。PPO 使用由同一训练偏好数据拟合的 RM；PPO-GT 直接使用固定情感分类器作为奖励。两者都报告训练采样量、调参机会和计算成本。

已有 PPO 项目的阶段组织、审计和恢复经验可复用，模型、任务和奖励接口需重新适配。正式比较 reward–KL 重叠区间，不将另一任务的 PPO 分数搬来对比。

### 12.2 Pythia-2.8B 和 HH

先在 Anthropic HH chosen completion 上做 SFT，再固定 reference 缓存并运行 DPO。保留历史对话，把最后一条 assistant 回复拆为 completion，先核对两侧 prompt 一致。

从总长度 512、micro-batch=1 开始测试全参数 RMSprop 路径，验证参考缓存、FP32 状态和激活的实际容量。显存不足时再决定卸载或 LoRA；LoRA 实验单独标记，不能冒充原始全参数设置。

独立对话测试计划 500–1,000 个 prompt。若采用大模型裁判，单独确定服务、预算、模型版本与提示词，采用盲评和双顺序检查。原论文使用 `gpt-4-0314`；替换裁判后不能将所得胜率视为同口径数值复现。

## 13 参考资料

- [DPO 原论文](../2305.18290v3.pdf)，正文第 6 节、Figure 2、附录 B 和 C.1。
- [DPO 官方实现](https://github.com/eric-mitchell/direct-preference-optimization)。
- [官方配置](https://github.com/eric-mitchell/direct-preference-optimization/blob/main/config/config.yaml)，实现时固定 commit。
- [TRL DPO Trainer](https://huggingface.co/docs/trl/dpo_trainer)，实现时改为与安装版本对应的文档。
- [情感分类器](https://huggingface.co/siebert/sentiment-roberta-large-english)。
- [Pythia-2.8B](https://huggingface.co/EleutherAI/pythia-2.8b)。
- [Anthropic HH 数据](https://huggingface.co/datasets/Anthropic/hh-rlhf)。

## 14 实测状态与结果

| 项目 | 状态 |
| --- | --- |
| 复现规划与目录入口 | 已建立 |
| 目标机器环境、驱动、磁盘 | SSH 已连接，硬件核验通过 |
| 依赖、代码、模型和数据版本锁定 | 已完成，见锁文件 |
| 训练与评测实现 | 已实现；FP32 全流程冒烟完成 |
| 正确性验证、smoke 和 benchmark | 六项测试、真实 checkpoint 恢复和容量测速历史验证通过；benchmark/pilot 已完成 |
| 正式对照、β 扫描和多 seed | 四档 β 的 seed42 完成，主 β=0.1 与 Preferred-FT 各三个更新 seed 完成 |
| reward–KL、生成质量和最终结论 | 十组测试及曲线齐全；定向样本检查完成，50 前缀盲评未完成；详见最终报告 |

第 14.1–14.4 节保留部署与启动历史，第 14.5 节记录完成后的实测结果；前文规划值与实际执行有差异时，以最终报告和锁定配置为准。

### 14.1 部署配置

2026-10-08 连接 `ateam@10.15.20.29`，部署目录 `/home/ateam/dpo-reproduction`。系统 Ubuntu 24.04.4 LTS，驱动 570.133.20；`nvidia-smi` 显示 NVIDIA A100-SXM4-40GB、40960MiB 显存，检查时无 GPU 计算进程。系统内存约 188GiB，CPU 为双路 E5-2695 v3，共 28 物理核、56 线程；本地文件系统约 928GB 可用。

实施阶段的明确选择：

- 使用服务器已有 Python 3.12.3 的独立虚拟环境，替代规划中的 Python 3.11，避免修改系统 Python。
- 使用固定版本 PyTorch 2.6.0 CUDA 12.4、Transformers 4.57.1。驱动报告的 CUDA 12.8 是支持能力，不等于必须安装相同版本 runtime。
- 使用独立 PyTorch RMSprop 训练循环，替代规划中的 Trainer 优先路径，以保持生成 token IDs、reference 缓存、精确尾批权重及保存状态可直接审计；不需要安装 TRL。与固定版本官方 DPO loss 做数值和梯度对照。
- 所有阶段关闭 dropout。SFT 使用逐序列 token 平均 CE，再按样本平均；Preferred-FT 沿用官方风格的 completion 负 log-prob 总和，DPO 使用标准 sigmoid 目标。二者尺度差异随配置记录，不把 SFT 学习率视为论文精确设置。
- checkpoint 的优化器状态和固定数据顺序足以实施按 update 恢复；生成阶段目前仅保留进度文件，失败后重新生成，不能声明采样支持精确续跑。
- 训练中每 100 updates 保存固定开发前缀的生成 token IDs 与当时 policy log-prob；训练结束后顺序加载 reference 和分类器完成曲线评分，避免多个大模型同时占用 GPU。

安装、下载、测试与训练各自输出到独立日志；密码不写入配置、脚本或日志。

### 14.2 精度检查与方案调整

BF16 工程冒烟完成了 SFT、偏好生成、分类器评分、reference 缓存、Preferred-FT、DPO 和开发评测。六项测试通过；SFT 主动在第 1 次更新后退出，再恢复完成第 2 次更新，checkpoint 重载 log-prob 与贪心输出核验通过。SFT 峰值 reserved 显存约 11.18GiB。此阶段只用于工程检查，不能作为正式训练结果。

检查缓存批次与训练批次的差异时，同一 reference 的序列 log-prob 出现可见偏差：

| 对照设置 | 观测到的最大绝对差 |
| --- | ---: |
| BF16，动态 padding，训练样本探针 | 0.447265625 |
| BF16，固定长度 136，批次改变探针 | 0.32232666015625 |
| FP32，固定长度 144，启用 TF32 | 0.00732421875 |
| FP32，固定长度 144，关闭 TF32 | 0.000091552734375 |

这些是具体探针结果，不是所有输入的误差上界。BF16 的累积偏差会被较大 β 放大，因此正式候选配置改为 FP32 计算并关闭 TF32，缓存一致性门槛从工程调试的 0.5 收紧至 0.002，随后重新验证实际流水线。生成与评分也采用同一 FP32 方案，避免 KL 采样与打分使用不同计算精度。

旧 BF16 数据、checkpoint、日志和结果分别归档到服务器各产物目录的 `smoke_bf16/`；对应源码快照保存在 `logs/bf16_engineering/source.tar.gz`。它们不混入后续正式数据。原规划的 BF16 优先策略保留作为历史方案，实际执行以本节和配置文件为准。

### 14.3 FP32 验证与容量测速

FP32 流程再次通过全部六项测试。真实 SFT 中断恢复与重载通过；训练和验证 reference 缓存探针最大偏差分别为 `0.000030517578125`、`0.00006103515625`，均低于收紧后的 `0.002` 门槛。DPO 首次更新前 loss 为 `0.6931475009769201`，符合 policy 与 reference 一致时的 `log(2)`；第二次更新的批次 loss 为 `0.6130196368321776`。这些仅验证实现与数值稳定性，不代表正式偏好提升。

SFT 自身生成的开发样本相对 SFT reference 的 KL 为 0，表明此基线上的生成、mask 和评分口径一致。小样本工程评测仅 16 个开发前缀，不用来对算法效果作统计结论。

下表的训练时间取三次完整 update 中后两次的均值，有效 batch=64；生成时间为 128 条、每条最多 128 个新 tokens 的完整生成及 policy log-prob 评分。峰值采用 PyTorch reserved 显存。

| 阶段 | 批次 | 实测时间或吞吐 | 峰值 reserved |
| --- | ---: | ---: | ---: |
| SFT | micro=2 | 7.48 秒/update | 9.94GiB |
| SFT | micro=4 | 7.19 秒/update | 10.35GiB |
| SFT | micro=8 | 7.26 秒/update | 11.53GiB |
| DPO | micro=2 | 8.74 秒/update | 10.02GiB |
| DPO | micro=4 | 8.35 秒/update | 10.28GiB |
| DPO | micro=8 | 8.27 秒/update | 12.33GiB |
| 生成与 policy 评分 | batch=16 | 3.20 条/秒 | 7.74GiB |
| 生成与 policy 评分 | batch=32 | 5.88 条/秒 | 13.41GiB |
| 生成与 policy 评分 | batch=64 | 10.84 条/秒 | 38.36GiB |
| 生成与 policy 评分 | batch=128 | 13.78 条/秒 | 38.99GiB |

据此选定 SFT micro=4、DPO/Preferred-FT micro=8、生成 batch=32。大生成批次虽未在探针中 OOM，但 reserved 显存接近上限，未用于后续长任务。benchmark profile 增加至最多 16 次 update、192 个训练前缀，以在所选 micro-batch 下覆盖约 100 个以上 micro-step；pilot 与 full 使用同一批次设置。

原始记录位于服务器 `logs/capacity_benchmark.json`、`data/smoke/cache_checks.json`、`logs/reload_checks.json` 和 `logs/smoke/`。本地同步轻量记录至相同相对目录，大型权重仍保留在服务器。

### 14.4 后台队列

2026-10-08 18:47（北京时间）在 `ateam@10.15.20.29` 启动 `tmux` 会话 `dpo-train`，编排进程初始 PID 为 `288778`，执行 `benchmark → pilot → full`。启动核验已观察到 benchmark SFT 多次真实参数更新；每次约 7 秒，loss 和梯度有限，无 OOM。

长任务运行目录为 `/home/ateam/dpo-reproduction`。主日志为 `logs/orchestrator.log`，阶段日志为 `logs/pipeline_<profile>.log`，当前阶段和失败原因写入 `logs/pipeline_status.json`。SSH 断开后会话继续运行；任何阶段错误会停止队列。pilot 另设明显重复或输出长度塌缩的工程门槛，通过后才进入 full。

启动时完整 β 扫描、多 seed 和最终测试尚未执行；该历史状态现已由第 14.5 节的完成记录更新。


### 14.5 最终实测结果（2026-10-11）

队列在 2026-10-11 01:44:04（UTC+8）完成，总墙钟耗时 54.94 小时，其中 full 为 53.76 小时。十组最终测试均为 1,000 个相同前缀 × 4 条生成，合计 40,000 条。主配置使用预设 β=0.1 与 final checkpoint，无测试后调参。

| 方法 | 测试奖励 | 序列 KL |
| --- | ---: | ---: |
| SFT | 0.520583 | 约 0 |
| Preferred-FT，三 seed | 0.639034 ± 0.003957 | 0.308569 ± 0.002713 |
| DPO β=0.1，三 seed | 0.994014 ± 0.000401 | 10.504534 ± 0.158106 |
| DPO β=0.05，seed42 | 0.997938 | 19.042186 |
| DPO β=1，seed42 | 0.880276 | 1.487764 |
| DPO β=5，seed42 | 0.779455 | 0.765373 |

± 为三个偏好更新 seed 的样本标准差，SFT 与偏好数据共享；不代表全流程方差。各组前缀聚类 bootstrap 95% CI、dev/test 分表及生成质量见[最终复现报告](最终复现报告.md)。

所有正式更新日志数值有限、编号连续，十组测试及最终汇总齐全。560 个本地源文件 SHA-256、测试生成指纹、每前缀四条输出、共同测试前缀集、CI 和三 seed 统计均通过核验。服务器只读检查确认影评 ID 和 token 前缀的跨划分交集为零。本次未重新加载全部 checkpoint 或重新计算大权重哈希，也未重跑历史六项 GPU 正确性测试。

DPO 改善了当前情感分类器奖励，同时带来更大 KL；样本中仍有高分重复与语义不连贯现象。仅完成定向样本检查，原规划的 50 前缀盲评未完成。没有 PPO 对照，不把本次结果写成完整论文数值复现或 DPO 优于 PPO。

轻量证据保存在[最终结果快照](results/final-2026-10-11/README.md)，离线复核入口为 `scripts/analyze_final_results.py`。原始数据、完整生成 JSONL 和模型权重仍留在服务器。最终报告与轻量证据按用户要求归档至 Paper-Reading。
