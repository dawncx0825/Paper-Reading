# Paper Reading

论文阅读资料与可复用的 AI 学习技能。`Skills/` 提供论文精读与闭卷复现式代码学习流程，`Papers/` 按论文保存原文、总结、问答和论文图片。

阅读记录保留结论，也保留形成理解的过程：问题从哪里来、哪些解释有帮助、哪些理解经过了纠正。重读时，可以结合原文和问答上下文回顾这些思路。

## 目录结构

```text
paper-reading/
├── README.md
├── Skills/
│   ├── paper-reading/
│   │   ├── SKILL.md                    # 通用论文精读流程
│   │   └── references/
│   │       └── glossary.md             # 初始术语表与维护规则
│   ├── paper-reading.skill             # 论文精读 ZIP 安装包
│   ├── learn-code-by-recall/
│   │   ├── SKILL.md                    # 闭卷复现式代码学习流程
│   │   ├── agents/openai.yaml          # Codex 展示信息与默认提示词
│   │   ├── README.md                   # 原仓库说明
│   │   └── LICENSE                     # 原仓库 MIT 许可证
│   └── learn-code-by-recall.skill       # 代码学习 ZIP 安装包
└── Papers/
    ├── instructGPT/
    │   ├── 2203.02155v1.pdf             # 论文原文
    │   ├── InstructGPT_论文总结.pdf      # 阅读总结
    │   ├── InstructGPT_论文问答记录.md   # 问题、回答与上下文
    │   └── figures/                    # 论文图片
    ├── ViT/
    │   ├── 2010.11929v2.pdf             # 论文原文
    │   ├── ViT论文总结.md               # 阅读总结
    │   ├── ViT论文问答记录.md           # 问题、回答与上下文
    │   └── figures/                    # 论文图片
    ├── V*/
    │   ├── 2312.14135v2.pdf             # 论文原文
    │   ├── Vstar论文总结.md             # 阅读总结
    │   └── figures/                    # 论文图片
    └── PPO/
        ├── 2009.01325v3.pdf             # Learning to Summarize from Human Feedback
        ├── 论文总结.md                  # 阅读总结
        ├── figures/                    # 论文图片
        └── reproduce/                  # 单卡 RLHF 复现
            ├── README.md               # 环境配置与运行说明
            ├── configs/                # 实验配置
            ├── ppo_repro/              # 数据与模型实现
            ├── scripts/                # 训练、生成与评测脚本
            ├── tests/                  # 复现代码测试
            ├── 最终复现报告.md           # 实验结果与局限性
            └── report-app/             # 交互式报告源码与构建产物
```

## 技能索引

| 技能 | 用途 | 安装包 |
| --- | --- | --- |
| [paper-reading](Skills/paper-reading/SKILL.md) | 论文总结、术语对照、概念与图表精读 | [下载](Skills/paper-reading.skill) |
| [learn-code-by-recall](Skills/learn-code-by-recall/SKILL.md) | 分段回忆代码、渐进提示、纠错与闭卷复现 | [下载](Skills/learn-code-by-recall.skill) |

## 使用论文精读 Skill

[paper-reading](Skills/paper-reading/SKILL.md) 适用于论文总结、精读，以及对概念、公式、图表和实验的追问。默认先生成中文总结，以术语对照表开头，再按问题逐步讲解；用户可以直接指定其他语言、深度和格式。

技能不依赖特定读者、个人历史、GitHub 仓库或固定输出路径。术语表提供机器学习领域的起始示例，其他领域按论文语境使用。技能本身无需安装 Python 依赖；PDF 读取和图片展示使用所在平台的能力，代码中的 PyMuPDF 仅是可选裁图示例。

### 安装

将整个 [Skills/paper-reading/](Skills/paper-reading/) 文件夹复制到所用 AI 工具的技能目录，保留 `SKILL.md` 与 `references/glossary.md` 的相对位置。

例如，在本地 Codex 中，默认用户技能目录为 `~/.codex/skills/`；若设置了 `CODEX_HOME`，则使用该目录下的 `skills/`。在克隆后的仓库根目录执行以下命令；目标目录已存在时，先自行合并已有自定义版本：

```bash
skill_dir="${CODEX_HOME:-$HOME/.codex}/skills"
mkdir -p "$skill_dir"
if [ ! -e "$skill_dir/paper-reading" ]; then
    cp -R Skills/paper-reading "$skill_dir/paper-reading"
else
    echo "paper-reading 已存在，请先合并已有版本。"
fi
```

若所用平台支持导入 `.skill` 文件，可直接导入 [Skills/paper-reading.skill](Skills/paper-reading.skill)。它是包含同一技能目录的 ZIP 包；不支持该格式的平台使用源文件夹即可。

### 开始阅读

安装后提供论文 PDF、可访问链接或标题，并提出需求。例如：

> 使用 paper-reading 技能总结这篇论文，先给术语对照表，再讲问题、方法和实验结果。

> 解释论文 Figure 2 中的数据流，先定义图里的符号。

> 继续解释刚才总结中的 KL 惩罚，这一轮先聚焦这个概念。

需要保存笔记时指定项目位置即可。默认归档建议为 `Papers/<论文简称>/`；跨论文积累的术语可另存为 `Papers/glossary.md`，无需改动已安装的技能。启用技能本身不会自动提交或上传文件。

## 使用代码学习 Skill

[learn-code-by-recall](Skills/learn-code-by-recall/SKILL.md) 帮助学习者独立理解并复现代码：先按语义和依赖拆分代码，再逐段尝试、接受最小必要提示并修正错误，最后通过闭卷重写、解释和迁移练习检验掌握程度。适用于论文实现、算法、Notebook 和日常编程学习。

将整个 [Skills/learn-code-by-recall/](Skills/learn-code-by-recall/) 文件夹复制到所用工具的技能目录；Codex 的目录规则与上文相同，目标子目录名使用 `learn-code-by-recall`。支持 `.skill` 导入的平台也可以使用 [安装包](Skills/learn-code-by-recall.skill)。

提供目标代码或文件，并使用以下提示词开始：

> 使用 $learn-code-by-recall 帮助逐步理解并独立写出这段代码，不要一开始给完整答案。

本技能来自 [dawncx0825/learn-code-by-recall](https://github.com/dawncx0825/learn-code-by-recall)，本次收录版本为 [bbeb40e](https://github.com/dawncx0825/learn-code-by-recall/commit/bbeb40e1107c65865696243bc4e9d9449f2bfc24)。保留原始技能、Codex 元数据、说明和 [MIT 许可证](Skills/learn-code-by-recall/LICENSE)。这里保存的是独立副本，原仓库更新后需手动同步。

## 已读论文

| 论文 | 主题 | 阅读资料 |
| --- | --- | --- |
| **InstructGPT** — Training Language Models to Follow Instructions with Human Feedback | 指令遵循、人类反馈、RLHF | [原文](Papers/instructGPT/2203.02155v1.pdf) · [总结](Papers/instructGPT/InstructGPT_论文总结.pdf) · [问答与上下文](Papers/instructGPT/InstructGPT_论文问答记录.md) · [图片](Papers/instructGPT/figures/) |
| **ViT** — An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale | 视觉 Transformer、图像分类、大规模预训练 | [原文](Papers/ViT/2010.11929v2.pdf) · [总结](Papers/ViT/ViT论文总结.md) · [问答与上下文](Papers/ViT/ViT论文问答记录.md) · [图片](Papers/ViT/figures/) |
| **V\*** — Guided Visual Search as a Core Mechanism in Multimodal LLMs | 多模态大语言模型、引导式视觉搜索、细粒度视觉定位 | [原文](Papers/V%2A/2312.14135v2.pdf) · [总结](Papers/V%2A/Vstar论文总结.md) · [图片](Papers/V%2A/figures/) |
| **PPO / RLHF 摘要** — Learning to Summarize from Human Feedback | 人类偏好、奖励模型、PPO、文本摘要 | [原文](Papers/PPO/2009.01325v3.pdf) · [总结](Papers/PPO/论文总结.md) · [图片](Papers/PPO/figures/) · [复现代码](Papers/PPO/reproduce/README.md) · [复现报告](Papers/PPO/reproduce/最终复现报告.md) |

需要回顾整体内容时先看总结；需要理解段落、公式或实验时，结合原文查阅问答记录。各论文的 `figures/` 文件夹保存论文图片，可配合总结中的图号查阅。可安装、可复用的通用流程以 `Skills/` 中的版本为准。

`Papers/PPO/` 收录的是使用 PPO 进行 RLHF 摘要训练的论文 *Learning to Summarize from Human Feedback*。其 `reproduce/` 目录提供基于 Qwen2.5-0.5B、面向单张 RTX 3090 的 SFT → 奖励模型 → PPO 复现流程，以及评测脚本、最终报告和交互式报告。复现侧重方法验证，实验结果与局限性详见复现报告；训练数据、模型、检查点和运行日志按该目录的 `.gitignore` 排除。

## 添加新论文

在 `Papers/` 下按论文简称新建文件夹，按需放入原文、总结和问答，论文图片统一放在该论文的 `figures/` 子目录，并在上方表格增加资料与图片索引。文件命名应能清楚区分原文、总结和后续补充。以后新增论文统一放在此目录。

## 维护技能安装包

修改技能时，以 `Skills/<技能名>/` 内的源文件为准，并同步重新生成对应的 `.skill` 安装包。在仓库根目录可执行以下命令；将 `skill_name` 改为需要打包的技能名：

```bash
python3 - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

skill_name = "paper-reading"  # 或 "learn-code-by-recall"
source = Path("Skills") / skill_name
with ZipFile(source.with_suffix(".skill"), "w", ZIP_DEFLATED) as archive:
    for path in sorted(source.rglob("*")):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(source).parts):
            archive.write(path, path.relative_to(source.parent).as_posix())
PY
```

## 关于这些记录

总结与问答是学习记录，其中包含 AI 辅助解释和讨论，可能仍有疏漏；涉及具体定义、公式和实验结论时，以论文原文为准。欢迎通过 Issue 交流和纠错。

论文原文的版权归原作者或相应权利人所有。
