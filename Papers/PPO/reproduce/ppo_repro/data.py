from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset

from .utils import normalize_text, sha256_text, stable_rank


TLDR_PROMPT_PATTERN = re.compile(
    r"^\s*SUBREDDIT:\s*r/(?P<subreddit>.*?)\s*\n+\s*TITLE:\s*(?P<title>.*?)\s*\n+\s*POST:\s*(?P<post>.*?)\s*\n+\s*TL;DR:\s*$",
    flags=re.DOTALL | re.IGNORECASE,
)


def format_prompt(subreddit: str, title: str, post: str) -> str:
    subreddit = (subreddit or "unknown").strip()
    title = (title or "").strip()
    post = (post or "").strip()
    return f"SUBREDDIT: r/{subreddit}\nTITLE: {title}\nPOST: {post}\nTL;DR:"


def parse_tldr_prompt(prompt: str) -> dict[str, str]:
    match = TLDR_PROMPT_PATTERN.match(prompt or "")
    if not match:
        raise ValueError("Unrecognized TL;DR prompt format")
    return {name: value.strip() for name, value in match.groupdict().items()}


def content_hash(title: str, post: str) -> str:
    return sha256_text(title, post)


def canonicalize_sft_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = parse_tldr_prompt(row["prompt"])
    prompt = format_prompt(fields["subreddit"], fields["title"], fields["post"])
    completion = " " + (row.get("completion") or "").strip()
    return {
        "post_id": content_hash(fields["title"], fields["post"]),
        "content_hash": content_hash(fields["title"], fields["post"]),
        "subreddit": fields["subreddit"],
        "title": fields["title"],
        "post": fields["post"],
        "prompt": prompt,
        "completion": completion,
    }


def canonicalize_comparison_row(row: dict[str, Any]) -> dict[str, Any] | None:
    info = row.get("info") or {}
    post = info.get("post")
    title = info.get("title")
    subreddit = info.get("subreddit")
    summaries = row.get("summaries") or []
    choice = row.get("choice")
    if not post or title is None or not subreddit or len(summaries) != 2 or choice not in (0, 1):
        return None
    chosen = summaries[choice]
    rejected = summaries[1 - choice]
    digest = content_hash(title, post)
    return {
        "post_id": str(info.get("id") or digest),
        "content_hash": digest,
        "prompt": format_prompt(subreddit, title, post),
        "chosen": " " + (chosen.get("text") or "").strip(),
        "rejected": " " + (rejected.get("text") or "").strip(),
        "chosen_policy": str(chosen.get("policy") or "unknown"),
        "rejected_policy": str(rejected.get("policy") or "unknown"),
        "source_split": str(row.get("split") or "unknown"),
        "source_batch": str(row.get("batch") or "unknown"),
    }


def add_token_lengths(row: dict[str, Any], tokenizer, eos_tokens: int = 1) -> dict[str, Any]:
    prompt_len = len(tokenizer(row["prompt"], add_special_tokens=False)["input_ids"])
    output = dict(row)
    output["prompt_tokens"] = prompt_len
    if "completion" in row:
        completion_len = len(tokenizer(row["completion"], add_special_tokens=False)["input_ids"]) + eos_tokens
        output["completion_tokens"] = completion_len
        output["sequence_tokens"] = prompt_len + completion_len
    if "chosen" in row:
        chosen_len = len(tokenizer(row["chosen"], add_special_tokens=False)["input_ids"]) + eos_tokens
        rejected_len = len(tokenizer(row["rejected"], add_special_tokens=False)["input_ids"]) + eos_tokens
        output["chosen_tokens"] = chosen_len
        output["rejected_tokens"] = rejected_len
        output["chosen_sequence_tokens"] = prompt_len + chosen_len
        output["rejected_sequence_tokens"] = prompt_len + rejected_len
    return output


def valid_sft_length(row: dict[str, Any], prompt_max: int, completion_max: int, sequence_max: int) -> bool:
    return (
        0 < row["prompt_tokens"] <= prompt_max
        and 1 < row["completion_tokens"] <= completion_max
        and row["sequence_tokens"] <= sequence_max
    )


def valid_rm_length(row: dict[str, Any], prompt_max: int, completion_max: int, sequence_max: int) -> bool:
    return (
        0 < row["prompt_tokens"] <= prompt_max
        and 1 < row["chosen_tokens"] <= completion_max
        and 1 < row["rejected_tokens"] <= completion_max
        and row["chosen_sequence_tokens"] <= sequence_max
        and row["rejected_sequence_tokens"] <= sequence_max
    )


def unique_by_hash(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    output = []
    for row in rows:
        digest = row["content_hash"]
        if digest not in seen:
            seen.add(digest)
            output.append(row)
    return output


def stable_select(rows: Iterable[dict[str, Any]], size: int, seed: int, key: str = "content_hash"):
    ranked = sorted(rows, key=lambda row: stable_rank(str(row[key]), seed))
    return ranked[: min(size, len(ranked))]


def stable_group_select(
    rows: Iterable[dict[str, Any]], size: int, seed: int, group_key: str = "content_hash"
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[group_key])].append(row)
    ordered_keys = sorted(groups, key=lambda value: stable_rank(value, seed))
    selected: list[dict[str, Any]] = []
    for value in ordered_keys:
        group = groups[value]
        if selected and len(selected) + len(group) > size:
            continue
        selected.extend(group)
        if len(selected) >= size:
            break
    return selected


def dataset_from_rows(rows: list[dict[str, Any]], columns: list[str]) -> Dataset:
    return Dataset.from_list([{column: row[column] for column in columns} for row in rows])


def save_dataset(dataset: Dataset, path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing processed dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(path))


def normalized_summary(text: str) -> str:
    return normalize_text(text)

