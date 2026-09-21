#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time

import torch
from datasets import load_from_disk
from peft import PeftModel

from ppo_repro.config import ensure_output_dirs, load_config
from ppo_repro.models import load_causal_model, load_tokenizer
from ppo_repro.utils import append_jsonl, utc_now


def load_model(kind, cfg, tokenizer):
    if kind == "base":
        return load_causal_model(cfg.base_model_source, tokenizer, cfg.base_model_revision, train=False)
    if kind == "sft":
        return load_causal_model(cfg.sft_final_dir, tokenizer, train=False)
    if kind == "ppo":
        base = load_causal_model(cfg.sft_final_dir, tokenizer, train=False)
        return PeftModel.from_pretrained(base, str(cfg.ppo_actor_dir))
    raise ValueError(kind)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="smoke")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--model", choices=["base", "sft", "ppo"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)
    source = cfg.processed_dir / f"generation_{args.split}"
    dataset = load_from_disk(str(source))
    tokenizer_source = cfg.sft_final_dir if cfg.sft_final_dir.exists() else cfg.base_model_source
    revision = None if cfg.sft_final_dir.exists() else cfg.base_model_revision
    tokenizer = load_tokenizer(tokenizer_source, revision=revision, padding_side="left")
    model = load_model(args.model, cfg, tokenizer).eval().cuda()
    output = cfg.result_dir / f"generations_{args.split}_{args.model}.jsonl"
    completed = set()
    if output.exists():
        with output.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    completed.add(json.loads(line)["content_hash"])

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    for start in range(0, len(dataset), args.batch_size):
        rows = [row for row in dataset.select(range(start, min(start + args.batch_size, len(dataset))))]
        rows = [row for row in rows if row["content_hash"] not in completed]
        if not rows:
            continue
        encoded = tokenizer(
            [row["prompt"] for row in rows], padding=True, truncation=False, return_tensors="pt"
        ).to(model.device)
        prompt_width = encoded.input_ids.shape[1]
        with torch.inference_mode():
            sequences = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=cfg.completion_max_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        generated = sequences[:, prompt_width:]
        records = []
        for row, token_ids in zip(rows, generated):
            ids = token_ids.tolist()
            hit_eos = tokenizer.eos_token_id in ids
            if hit_eos:
                ids = ids[: ids.index(tokenizer.eos_token_id)]
            text = tokenizer.decode(ids, skip_special_tokens=True).strip()
            records.append(
                {
                    "created_at": utc_now(),
                    "model": args.model,
                    "post_id": row["post_id"],
                    "content_hash": row["content_hash"],
                    "prompt": row["prompt"],
                    "reference": row["completion"].strip(),
                    "summary": text,
                    "generated_tokens": len(ids),
                    "hit_eos": hit_eos,
                }
            )
        append_jsonl(output, records)
        completed.update(record["content_hash"] for record in records)
        print(f"{args.model} {len(completed)}/{len(dataset)}", flush=True)
    elapsed = time.time() - started
    print(
        json.dumps(
            {
                "output": str(output),
                "records": len(completed),
                "new_wall_seconds": elapsed,
                "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
