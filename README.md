# Paper Reading

论文阅读资料与可复用的 AI 精读技能。`Skills/` 提供通用阅读流程，`Papers/` 按论文保存原文、总结、问答和相关材料。

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
│   └── paper-reading.skill             # 与上方源文件同步的 ZIP 安装包
└── Papers/
    └── instructGPT/
        ├── 2203.02155v1.pdf             # 论文原文
        ├── InstructGPT_论文总结.pdf      # 阅读总结
        ├── InstructGPT_论文问答记录.md   # 问题、回答与上下文
        └── 论文阅读项目_输出规范.md      # 当时阅读过程中的约定
```

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

## 已读论文

| 论文 | 主题 | 阅读资料 |
| --- | --- | --- |
| **InstructGPT** — Training Language Models to Follow Instructions with Human Feedback | 指令遵循、人类反馈、RLHF | [原文](Papers/instructGPT/2203.02155v1.pdf) · [总结](Papers/instructGPT/InstructGPT_论文总结.pdf) · [问答与上下文](Papers/instructGPT/InstructGPT_论文问答记录.md) |

需要回顾整体内容时先看总结；需要理解段落、公式或实验时，结合原文查阅问答记录。论文目录中的旧输出规范保留当时的阅读背景；可安装、可复用的通用流程以 `Skills/` 中的版本为准。

## 添加新论文

在 `Papers/` 下按论文简称新建文件夹，按需放入原文、总结、问答和图像，并在上方表格增加索引。文件命名应能清楚区分原文、总结和后续补充。以后新增论文统一放在此目录。

修改技能时，以 `Skills/paper-reading/` 内的源文件为准，并同步重新生成 `.skill` 安装包。在仓库根目录可执行：

```bash
python3 - <<'PY'
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

source = Path("Skills/paper-reading")
with ZipFile("Skills/paper-reading.skill", "w", ZIP_DEFLATED) as archive:
    for path in sorted(source.rglob("*")):
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(source).parts):
            archive.write(path, path.relative_to(source.parent).as_posix())
PY
```

## 关于这些记录

总结与问答是学习记录，其中包含 AI 辅助解释和讨论，可能仍有疏漏；涉及具体定义、公式和实验结论时，以论文原文为准。欢迎通过 Issue 交流和纠错。

论文原文的版权归原作者或相应权利人所有。
