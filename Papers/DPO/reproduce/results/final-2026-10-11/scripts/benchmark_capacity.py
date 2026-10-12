"""Measure real SFT/DPO optimizer updates before selecting production micro-batches."""
import gc
import time

import torch

from dpo_repro.core import ROOT, model, read_rows, setup, sft_path, tokenizer, write_json
from scripts.train import losses
from dpo_repro.generation import generate

tok = tokenizer()
records = []
for stage in ['sft', 'dpo']:
    source = ROOT / 'data/smoke' / ('sft.jsonl' if stage == 'sft' else 'cached_train.jsonl')
    rows = read_rows(source)[:64]
    for micro in [2, 4, 8]:
        setup(42)
        m = model(ROOT / 'models/base' if stage == 'sft' else sft_path('smoke'), train=True)
        opt = torch.optim.RMSprop(m.parameters(), lr=1e-6, foreach=False)
        torch.cuda.reset_peak_memory_stats()
        durations = []
        try:
            for _ in range(3):
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.monotonic()
                for i in range(0, len(rows), micro):
                    loss, _ = losses(m, rows[i:i+micro], stage, tok, .1)
                    (loss.sum() / len(rows)).backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 10, error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize()
                durations.append(time.monotonic() - start)
            record = dict(stage=stage, micro_batch=micro, effective_batch=len(rows),
                seconds=durations, steady_seconds=sum(durations[1:]) / 2,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, status='passed')
        except torch.cuda.OutOfMemoryError:
            record = dict(stage=stage, micro_batch=micro, status='oom')
        records.append(record)
        write_json(ROOT / 'logs/capacity_benchmark.json', records)
        print(record, flush=True)
        if 'loss' in locals():
            del loss
        del opt, m
        gc.collect()
        torch.cuda.empty_cache()

prompts = read_rows(ROOT / 'data/smoke/prompts_train.jsonl')[:32]
for generation_batch in [16, 32, 64, 128]:
    setup(42)
    m = model(sft_path('smoke'))
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    count = 0
    try:
        for chunk in generate(m, tok, prompts, 4, 128, generation_batch, 2026):
            count += len(chunk)
        torch.cuda.synchronize()
        seconds = time.monotonic() - start
        record = dict(stage='generation', batch_size=generation_batch, completions=count,
            seconds=seconds, completions_per_second=count/seconds,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30, status='passed')
    except torch.cuda.OutOfMemoryError:
        record = dict(stage='generation', batch_size=generation_batch, status='oom')
    records.append(record)
    write_json(ROOT / 'logs/capacity_benchmark.json', records)
    print(record, flush=True)
    del m
    gc.collect()
    torch.cuda.empty_cache()
