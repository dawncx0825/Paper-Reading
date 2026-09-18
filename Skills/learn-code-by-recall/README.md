# Learn Code by Recall

An OpenAI Codex skill for learning code through active recall, incremental reconstruction, focused hints, and closed-book practice.

一个用于“闭卷复现式代码学习”的 Codex skill：让学习者先动手回忆和编写代码，再通过分段反馈、渐进提示与迁移练习真正掌握实现。

## What it does

- Breaks code into meaningful, dependency-aware blocks.
- Keeps the learner in an attempt–feedback loop instead of revealing the full solution immediately.
- Uses a progressive hint ladder, from behavioral clues to a complete block only when needed.
- Explains concepts at the point of confusion with concrete values, types, shapes, and state changes.
- Finishes with reconstruction, explanation, transfer, and delayed-recall checks.

## Installation

Clone this repository into your Codex skills directory:

```bash
git clone https://github.com/dawncx0825/learn-code-by-recall.git ~/.codex/skills/learn-code-by-recall
```

Restart Codex if the skill is not detected immediately.

## Usage

Invoke the skill explicitly in your prompt:

```text
Use $learn-code-by-recall to help me understand and independently reproduce this code. Do not show the full solution at first.
```

中文示例：

```text
使用 $learn-code-by-recall 帮我逐步理解并独立写出这段代码，不要一开始给完整答案。
```

Then provide the target code, notebook, algorithm, or implementation you want to learn.

## Repository structure

```text
.
├── SKILL.md
└── agents
    └── openai.yaml
```

`SKILL.md` contains the teaching workflow. `agents/openai.yaml` provides the display name, description, and default prompt shown by Codex.

## License

MIT License. See [LICENSE](LICENSE).
