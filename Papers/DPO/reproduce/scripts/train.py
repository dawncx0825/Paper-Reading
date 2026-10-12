import argparse
import json
import math
import os
import shutil
import time

import torch

from dpo_repro.core import (ROOT, batch, cache_meta, cfg, digest, dpo_loss, forward_logps,
    model, pair_rows, read_json, read_rows, restore_rng, rng_state, setup, sft_path, sha,
    tokenizer, write_json, write_rows)
from dpo_repro.generation import generate


def losses(m, rows, stage, tok, beta):
    if stage == 'sft':
        lp, count = forward_logps(m, batch(rows, tok.pad_token_id))
        return -lp / count.clamp_min(1), {}
    b = batch(pair_rows(rows) if stage == 'dpo' else pair_rows(rows)[:len(rows)], tok.pad_token_id)
    lp, count = forward_logps(m, b)
    if stage == 'preferred':
        return -lp, {}
    rc = torch.tensor([r['ref_chosen'] for r in rows], device='cuda')
    rr = torch.tensor([r['ref_rejected'] for r in rows], device='cuda')
    loss, margin = dpo_loss(lp[:len(rows)], lp[len(rows):], rc, rr, beta)
    return loss, dict(margin=margin.detach().mean().item(), accuracy=(margin > 0).float().mean().item())


@torch.inference_mode()
def validate(m, rows, stage, tok, beta, micro):
    m.eval()
    total, n, acc, margin = 0., 0, 0., 0.
    for i in range(0, len(rows), micro):
        x = rows[i:i + micro]
        loss, metrics = losses(m, x, stage, tok, beta)
        total += loss.sum().item()
        n += len(x)
        acc += metrics.get('accuracy', 0) * len(x)
        margin += metrics.get('margin', 0) * len(x)
    m.train()
    return dict(loss=total / n, accuracy=acc / n, margin=margin / n, examples=n)


def save_checkpoint(dest, m, tok, opt, state, config):
    temp = dest.with_name(dest.name + '.writing')
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True)
    m.save_pretrained(temp, safe_serialization=True)
    tok.save_pretrained(temp)
    torch.save(dict(optimizer=opt.state_dict(), rng=rng_state(), **state), temp / 'training_state.pt')
    write_json(temp / 'run_config.json', config)
    backup = dest.with_name(dest.name + '.previous')
    if backup.exists():
        shutil.rmtree(backup)
    if dest.exists():
        dest.rename(backup)
    temp.rename(dest)
    if backup.exists():
        shutil.rmtree(backup)


def train(args):
    c, tok = cfg(args.profile), tokenizer()
    seed = args.seed
    setup(seed)
    run = 'sft' if args.stage == 'sft' else f'{args.stage}_b{args.beta:g}_s{seed}'
    dest = ROOT / 'checkpoints' / args.profile / run
    logs = ROOT / 'logs' / args.profile / run
    logs.mkdir(parents=True, exist_ok=True)
    source = ROOT / 'data' / args.profile
    if args.stage == 'sft':
        rows = read_rows(source / 'sft.jsonl')
        input_sha = sha(source / 'sft.jsonl')
        dev = []
    else:
        assert read_json(source / 'cache_meta.json') == cache_meta(args.profile), 'Reference cache mismatch'
        rows = read_rows(source / 'cached_train.jsonl')
        dev = read_rows(source / 'cached_dev.jsonl')[:c['validation_pairs']]
        input_sha = sha(source / 'cached_train.jsonl')
    total = math.ceil(len(rows) / c['effective_batch'])
    if c['max_updates']:
        total = min(total, c['max_updates'])
    assets = read_json(ROOT / 'configs/assets.lock.json')
    asset_versions = {k: assets[k]['revision'] for k in ['base', 'reward', 'imdb']}
    asset_versions['official_dpo_commit'] = assets['official_dpo_commit']
    initial = ROOT / 'models/base' if args.stage == 'sft' else sft_path(args.profile)
    run_config = dict(profile=c, stage=args.stage, seed=seed, beta=args.beta, input_sha=input_sha,
                      asset_versions=asset_versions, initial_model_sha=sha(initial / 'model.safetensors'),
                      code_sha=sha(__file__),
                      core_sha=sha(ROOT / 'dpo_repro/core.py'), total_updates=total)
    if (dest / 'final/run_metrics.json').exists():
        assert read_json(dest / 'final/run_config.json') == run_config
        print('Completed run already exists', run, flush=True)
        return
    latest = dest / 'latest'
    resume = args.resume and (latest / 'training_state.pt').exists()
    if latest.exists() and not resume:
        raise RuntimeError(f'Existing incomplete run: use --resume: {latest}')
    if resume:
        assert read_json(latest / 'run_config.json') == run_config, 'Resume config mismatch'
    m = model(latest if resume else initial, train=True)
    lr = c['sft_lr'] if args.stage == 'sft' else c['preference_lr']
    opt = torch.optim.RMSprop(m.parameters(), lr=lr, alpha=0.99, eps=1e-8,
                             weight_decay=0, momentum=0, centered=False, foreach=False)
    warmup = max(1, round(total * .05)) if args.stage == 'sft' else 150
    if args.profile != 'full':
        warmup = min(warmup, max(1, total // 10))
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    start, elapsed_before = 0, 0.
    if resume:
        state = torch.load(latest / 'training_state.pt', map_location='cpu', weights_only=False)
        opt.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])
        start, elapsed_before = state['update'], state['elapsed']
    torch.cuda.reset_peak_memory_stats()
    begin = time.monotonic()
    micro = c['sft_micro_batch'] if args.stage == 'sft' else c['pair_micro_batch']
    prompts = read_rows(source / 'prompts_dev.jsonl')[:c['curve_prompts']]
    last_metric = None
    for update in range(start, total):
        tick = time.monotonic()
        ix = order[update * c['effective_batch']:(update + 1) * c['effective_batch']]
        group = [rows[i] for i in ix]
        opt.zero_grad(set_to_none=True)
        for param_group in opt.param_groups:
            param_group['lr'] = lr * min(1., (update + 1) / warmup)
        total_loss, extra = 0., {}
        for offset in range(0, len(group), micro):
            part = group[offset:offset + micro]
            loss, metric = losses(m, part, args.stage, tok, args.beta)
            if not torch.isfinite(loss).all():
                raise FloatingPointError('Nonfinite loss')
            (loss.sum() / len(group)).backward()
            total_loss += loss.detach().sum().item()
            for k, v in metric.items():
                extra[k] = extra.get(k, 0.) + v * len(part) / len(group)
        norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 10., error_if_nonfinite=True)
        opt.step()
        torch.cuda.synchronize()
        last_metric = dict(update=update + 1, loss=total_loss / len(group), grad_norm=norm.item(),
            lr=opt.param_groups[0]['lr'], seconds=time.monotonic() - tick, examples=len(group),
            allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            reserved_gib=torch.cuda.max_memory_reserved() / 2**30, **extra)
        with open(logs / 'train.jsonl', 'a') as f:
            f.write(json.dumps(last_metric) + '\n')
        print(json.dumps(dict(run=run, **last_metric)), flush=True)
        at_eval = (update + 1) % c['eval_every'] == 0 or update + 1 == total
        if at_eval:
            # Evaluation must not change the subsequent training RNG stream.
            saved_rng = rng_state()
            if dev:
                v = validate(m, dev, args.stage, tok, args.beta, micro)
                write_json(logs / f'validation_{update + 1:06d}.json', v)
            curve = []
            for chunk in generate(m, tok, prompts, 1, c['max_new_tokens'], c['generation_batch'], 2026):
                curve.extend(chunk)
            write_rows(ROOT / 'results' / args.profile / run / f'curve_{update + 1:06d}.jsonl', curve)
            restore_rng(saved_rng)
            elapsed = elapsed_before + time.monotonic() - begin
            save_checkpoint(latest, m, tok, opt, dict(update=update + 1, elapsed=elapsed), run_config)
        if args.stop_after and update + 1 >= args.stop_after and update + 1 < total:
            if not at_eval:
                save_checkpoint(latest, m, tok, opt, dict(update=update + 1,
                    elapsed=elapsed_before + time.monotonic() - begin), run_config)
            return
    final = dest / 'final'
    latest.rename(final)
    metrics = dict(run=run, updates=total, training_examples=min(total * c['effective_batch'], len(rows)),
        elapsed_seconds=elapsed_before + time.monotonic() - begin, last=last_metric,
        model_sha256=sha(final / 'model.safetensors'),
        resume_used=resume, parameter_count=sum(p.numel() for p in m.parameters()))
    write_json(final / 'run_metrics.json', metrics)
    write_json(logs / 'run_metrics.json', metrics)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('profile')
    p.add_argument('stage', choices=['sft', 'preferred', 'dpo'])
    p.add_argument('--beta', type=float, default=.1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, default=0)
    train(p.parse_args())
