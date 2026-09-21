#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from collections import Counter
from pathlib import Path

from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download

from ppo_repro.config import PROJECT_ROOT, ensure_output_dirs, load_config
from ppo_repro.data import (
    add_token_lengths,
    canonicalize_comparison_row,
    canonicalize_sft_row,
    dataset_from_rows,
    save_dataset,
    stable_group_select,
    stable_select,
    unique_by_hash,
    valid_rm_length,
    valid_sft_length,
)
from ppo_repro.models import load_tokenizer
from ppo_repro.utils import quantiles, sha256_file, utc_now, write_json


TLDR_FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "validation": "data/validation-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
COMPARISON_FILES = {
    "train": "comparisons/train/0000.parquet",
    "validation": "comparisons/validation/0000.parquet",
}


def download_file(repo: str, revision: str, filename: str, target_root: Path) -> Path:
    local_path = target_root / filename
    if local_path.exists():
        return local_path
    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            revision=revision,
            filename=filename,
            local_dir=str(target_root),
        )
    )


def clean_sft_split(dataset, tokenizer, cfg):
    rows = []
    failures = 0
    for raw in dataset:
        try:
            row = add_token_lengths(canonicalize_sft_row(raw), tokenizer)
        except (KeyError, ValueError):
            failures += 1
            continue
        if valid_sft_length(row, cfg.prompt_max_tokens, cfg.completion_max_tokens, cfg.sequence_max_tokens):
            rows.append(row)
    return unique_by_hash(rows), failures


def stats_for(rows, fields):
    return {
        "count": len(rows),
        "unique_posts": len({row["content_hash"] for row in rows}),
        "token_lengths": {field: quantiles([row[field] for row in rows]) for field in fields},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="smoke")
    parser.add_argument("--force", action="store_true", help="Replace only the selected profile's processed data")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    if cfg.processed_dir.exists() and any(cfg.processed_dir.iterdir()):
        if not args.force:
            raise FileExistsError(f"{cfg.processed_dir} already contains data; pass --force to rebuild this profile")
        shutil.rmtree(cfg.processed_dir)
        cfg.processed_dir.mkdir(parents=True)

    raw_root = PROJECT_ROOT / "data" / "raw"
    tldr_root = raw_root / "tldr"
    comparisons_root = raw_root / "comparisons"
    files = []
    tldr_paths = {}
    for split, filename in TLDR_FILES.items():
        path = download_file("trl-lib/tldr", cfg.tldr_revision, filename, tldr_root)
        tldr_paths[split] = path
        files.append(("trl-lib/tldr", cfg.tldr_revision, filename, path))
    comparison_paths = {}
    for split, filename in COMPARISON_FILES.items():
        path = download_file(
            "openai/summarize_from_feedback", cfg.comparisons_revision, filename, comparisons_root
        )
        comparison_paths[split] = path
        files.append(("openai/summarize_from_feedback", cfg.comparisons_revision, filename, path))

    tokenizer = load_tokenizer(cfg.base_model_source, cfg.base_model_revision)
    tldr = load_dataset("parquet", data_files={key: str(value) for key, value in tldr_paths.items()})
    clean_train, train_failures = clean_sft_split(tldr["train"], tokenizer, cfg)
    clean_validation, validation_failures = clean_sft_split(tldr["validation"], tokenizer, cfg)
    clean_test, test_failures = clean_sft_split(tldr["test"], tokenizer, cfg)

    generation_dev = stable_select(clean_validation, cfg.generation_dev_size, cfg.seed + 11)
    generation_test = stable_select(clean_test, cfg.generation_test_size, cfg.seed + 12)
    reserved_hashes = {row["content_hash"] for row in generation_dev + generation_test}
    clean_train = [row for row in clean_train if row["content_hash"] not in reserved_hashes]
    sft_train = stable_select(clean_train, cfg.sft_train_size, cfg.seed + 21)
    ppo_train = stable_select(clean_train, cfg.ppo_total_episodes, cfg.seed + 22)
    rm_center = stable_select(sft_train, cfg.rm_center_size, cfg.seed + 23)

    comparison_sets = load_dataset(
        "parquet", data_files={key: str(value) for key, value in comparison_paths.items()}
    )
    all_comparisons = concatenate_datasets([comparison_sets[key] for key in comparison_sets])
    comparison_rows = []
    invalid_comparisons = 0
    for raw in all_comparisons:
        row = canonicalize_comparison_row(raw)
        if row is None:
            invalid_comparisons += 1
            continue
        row = add_token_lengths(row, tokenizer)
        if valid_rm_length(row, cfg.prompt_max_tokens, cfg.completion_max_tokens, cfg.sequence_max_tokens):
            comparison_rows.append(row)

    by_split = Counter(row["source_split"] for row in comparison_rows)
    rm_train_pool = [
        row
        for row in comparison_rows
        if row["source_split"] == "train" and row["content_hash"] not in reserved_hashes
    ]
    rm_dev_pool = [row for row in comparison_rows if row["source_split"] == "valid1"]
    rm_test_pool = [row for row in comparison_rows if row["source_split"] == "valid2"]
    rm_train = stable_group_select(rm_train_pool, cfg.rm_train_size, cfg.seed + 31)
    rm_dev = stable_group_select(rm_dev_pool, cfg.rm_dev_size, cfg.seed + 32)
    rm_test = stable_group_select(rm_test_pool, cfg.rm_test_size, cfg.seed + 33)

    sft_columns = ["post_id", "content_hash", "prompt", "completion"]
    prompt_columns = ["post_id", "content_hash", "prompt"]
    rm_columns = [
        "post_id",
        "content_hash",
        "prompt",
        "chosen",
        "rejected",
        "chosen_policy",
        "rejected_policy",
    ]
    save_dataset(dataset_from_rows(sft_train, sft_columns), cfg.processed_dir / "sft_train")
    save_dataset(dataset_from_rows(generation_dev, sft_columns), cfg.processed_dir / "generation_dev")
    save_dataset(dataset_from_rows(generation_test, sft_columns), cfg.processed_dir / "generation_test")
    save_dataset(dataset_from_rows(ppo_train, prompt_columns), cfg.processed_dir / "ppo_train")
    save_dataset(dataset_from_rows(generation_dev, prompt_columns), cfg.processed_dir / "ppo_eval")
    save_dataset(dataset_from_rows(rm_train, rm_columns), cfg.processed_dir / "rm_train")
    save_dataset(dataset_from_rows(rm_dev, rm_columns), cfg.processed_dir / "rm_dev")
    save_dataset(dataset_from_rows(rm_test, rm_columns), cfg.processed_dir / "rm_test")
    save_dataset(dataset_from_rows(rm_center, sft_columns), cfg.processed_dir / "rm_center")

    sets = {
        "sft_train": sft_train,
        "ppo_train": ppo_train,
        "generation_dev": generation_dev,
        "generation_test": generation_test,
        "rm_train": rm_train,
        "rm_dev": rm_dev,
        "rm_test": rm_test,
    }
    hashes = {name: {row["content_hash"] for row in rows} for name, rows in sets.items()}
    overlaps = {
        f"{left}__{right}": len(hashes[left] & hashes[right])
        for left in sets
        for right in sets
        if left < right
    }
    prohibited_pairs = [
        ("sft_train", "generation_test"),
        ("ppo_train", "generation_test"),
        ("rm_train", "generation_test"),
        ("generation_dev", "generation_test"),
    ]
    prohibited = ["__".join(sorted(pair)) for pair in prohibited_pairs]
    bad = {name: overlaps.get(name, 0) for name in prohibited if overlaps.get(name, 0)}
    if bad:
        raise RuntimeError(f"Prohibited train/test post overlap detected: {bad}")

    audit = {
        "created_at": utc_now(),
        "profile": cfg.profile,
        "filter_limits": {
            "prompt": cfg.prompt_max_tokens,
            "completion": cfg.completion_max_tokens,
            "sequence": cfg.sequence_max_tokens,
        },
        "parse_failures": {
            "tldr_train": train_failures,
            "tldr_validation": validation_failures,
            "tldr_test": test_failures,
            "comparisons": invalid_comparisons,
        },
        "comparison_internal_splits_after_filter": dict(by_split),
        "sets": {
            "sft_train": stats_for(sft_train, ["prompt_tokens", "completion_tokens", "sequence_tokens"]),
            "ppo_train": stats_for(ppo_train, ["prompt_tokens"]),
            "generation_dev": stats_for(
                generation_dev, ["prompt_tokens", "completion_tokens", "sequence_tokens"]
            ),
            "generation_test": stats_for(
                generation_test, ["prompt_tokens", "completion_tokens", "sequence_tokens"]
            ),
            "rm_train": stats_for(
                rm_train, ["prompt_tokens", "chosen_tokens", "rejected_tokens"]
            ),
            "rm_dev": stats_for(rm_dev, ["prompt_tokens", "chosen_tokens", "rejected_tokens"]),
            "rm_test": stats_for(rm_test, ["prompt_tokens", "chosen_tokens", "rejected_tokens"]),
        },
        "post_hash_overlaps": overlaps,
    }
    write_json(cfg.processed_dir / "audit.json", audit)
    manifest = {
        "created_at": utc_now(),
        "model": {"repo": cfg.model_name, "revision": cfg.model_revision},
        "files": [
            {
                "repo": repo,
                "revision": revision,
                "filename": filename,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "local_path": str(path.relative_to(PROJECT_ROOT)),
            }
            for repo, revision, filename, path in files
        ],
    }
    write_json(raw_root / "manifest.json", manifest)
    print(f"Prepared profile={cfg.profile} at {cfg.processed_dir}")
    print(audit["sets"])


if __name__ == "__main__":
    main()
