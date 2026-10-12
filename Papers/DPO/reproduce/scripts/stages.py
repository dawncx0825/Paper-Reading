import argparse
import gc
import itertools
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dpo_repro.core import (ROOT, batch, cache_meta, cfg, forward_logps, model, pair_rows,
    read_json, read_rows, setup, sft_path, sha, tokenizer, write_json, write_rows)
from dpo_repro.generation import generate


def release(m):
    m.to('cpu')
    del m
    gc.collect()
    torch.cuda.empty_cache()


def sample_file(profile, checkpoint, prompts_file, output, count):
    c, tok = cfg(profile), tokenizer()
    output = Path(output)
    meta = dict(model_sha=sha(Path(checkpoint) / 'model.safetensors'), prompts_sha=sha(prompts_file),
        count=count, max_new_tokens=c['max_new_tokens'], batch=c['generation_batch'], seed=2026,
        generation_code_sha=sha(ROOT / 'dpo_repro/generation.py'))
    if output.exists():
        assert read_json(output.with_suffix('.meta.json')) == meta, 'Sample fingerprint mismatch'
        return
    m = model(checkpoint)
    prompts = read_rows(prompts_file)
    rows = []
    start = time.monotonic()
    # Write a progress file; incomplete generations are never mistaken for complete data.
    partial = output.with_suffix('.partial.jsonl')
    partial.parent.mkdir(parents=True, exist_ok=True)
    with open(partial, 'w') as f:
        for chunk in generate(m, tok, prompts, count, c['max_new_tokens'], c['generation_batch'], 2026):
            rows.extend(chunk)
            for row in chunk:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
            f.flush()
            if len(rows) % 256 < len(chunk):
                print(json.dumps(dict(stage='generate', file=str(output), done=len(rows),
                    total=len(prompts) * count, seconds=time.monotonic() - start)), flush=True)
    assert len(rows) == len(prompts) * count
    write_json(output.with_suffix('.meta.json'), meta)
    partial.rename(output)
    write_json(output.with_suffix('.timing.json'), dict(seconds=time.monotonic() - start,
                                                      completions=len(rows)))
    release(m)


def score_reward(rows, batch_size):
    tok = AutoTokenizer.from_pretrained(ROOT / 'models/reward', local_files_only=True)
    m = AutoModelForSequenceClassification.from_pretrained(ROOT / 'models/reward',
                dtype=torch.float32, local_files_only=True).to('cuda').eval()
    labels = {int(i): s.upper() for i, s in m.config.id2label.items()}
    positive = [i for i, label in labels.items() if label == 'POSITIVE']
    assert len(positive) == 1, labels
    scores = []
    for i in range(0, len(rows), batch_size):
        part = rows[i:i + batch_size]
        encoded = tok([r['text'] for r in part], padding=True, truncation=False, return_tensors='pt')
        lengths = encoded['attention_mask'].sum(1)
        limit = min(tok.model_max_length, m.config.max_position_embeddings - 2)
        if lengths.max().item() > limit:
            raise ValueError(f'Reward input exceeds {limit}; no silent truncation allowed')
        with torch.inference_mode():
            logits = m(**{k: v.to('cuda') for k, v in encoded.items()}).logits
        probs = logits.float().softmax(-1)[:, positive[0]].tolist()
        scores.extend(probs)
        if i % 4096 == 0:
            print('Reward scored', i + len(part), '/', len(rows), flush=True)
    release(m)
    return scores


def make_pairs(profile):
    c = cfg(profile)
    data = ROOT / 'data' / profile
    for split in ['train', 'dev']:
        dest = data / f'pairs_{split}.jsonl'
        src = data / f'candidates_{split}.jsonl'
        provenance = dict(source=sha(src), reward_lock=sha(ROOT / 'configs/assets.lock.json'),
                          stages_sha=sha(__file__))
        if dest.exists():
            assert read_json(dest.with_suffix('.meta.json')) == provenance
            continue
        rows = read_rows(src)
        scores = score_reward(rows, c['score_batch'])
        groups = {}
        for r, score in zip(rows, scores):
            r['reward'] = score
            groups.setdefault(r['prompt_id'], []).append(r)
        out, duplicate, tied = [], 0, 0
        for values in groups.values():
            for a, b in itertools.combinations(values, 2):
                if a['completion_ids'] == b['completion_ids']:
                    duplicate += 1
                    continue
                if abs(a['reward'] - b['reward']) < 1e-8:
                    tied += 1
                    continue
                chosen, rejected = (a, b) if a['reward'] > b['reward'] else (b, a)
                out.append(dict(id=chosen['id'] + '>' + rejected['id'], prompt_id=chosen['prompt_id'],
                    review_id=chosen['review_id'], prompt_ids=chosen['prompt_ids'],
                    chosen_ids=chosen['completion_ids'], rejected_ids=rejected['completion_ids'],
                    chosen_reward=chosen['reward'], rejected_reward=rejected['reward']))
        assert out, 'No valid preferences'
        write_rows(data / f'scored_candidates_{split}.jsonl', rows)
        write_rows(dest, out)
        write_json(dest.with_suffix('.meta.json'), provenance)
        write_json(data / f'pairs_{split}_audit.json', dict(pairs=len(out), duplicate=duplicate,
                    tied=tied, prompts=len(groups), candidates=len(rows)))


@torch.inference_mode()
def cache_reference(profile):
    c, tok = cfg(profile), tokenizer()
    data = ROOT / 'data' / profile
    meta = cache_meta(profile)
    if (data / 'cache_meta.json').exists():
        assert read_json(data / 'cache_meta.json') == meta
        return
    m = model(sft_path(profile))
    checks = []
    for split in ['train', 'dev']:
        rows = read_rows(data / f'pairs_{split}.jsonl')
        for i in range(0, len(rows), c['cache_batch']):
            part = rows[i:i + c['cache_batch']]
            lp, _ = forward_logps(m, batch(pair_rows(part), tok.pad_token_id))
            for j, r in enumerate(part):
                r['ref_chosen'] = lp[j].item()
                r['ref_rejected'] = lp[j + len(part)].item()
            if i % 4096 == 0:
                print('Reference cached', split, i + len(part), '/', len(rows), flush=True)
        # Recompute with the training micro-batch to verify batch-shape stability.
        probe = rows[:c['pair_micro_batch']]
        lp, _ = forward_logps(m, batch(pair_rows(probe), tok.pad_token_id))
        old = torch.tensor([r['ref_chosen'] for r in probe] + [r['ref_rejected'] for r in probe], device='cuda')
        error = (lp - old).abs().max().item()
        assert error < 0.002, f'Reference FP32 batch-shape error too high: {error}'
        checks.append(dict(split=split, max_abs_logp_error=error, tolerance=0.002))
        write_rows(data / f'cached_{split}.jsonl', rows)
    write_json(data / 'cache_checks.json', checks)
    write_json(data / 'cache_meta.json', meta)
    release(m)


def reference_score(profile, rows):
    c, tok = cfg(profile), tokenizer()
    m = model(sft_path(profile))
    values = []
    for i in range(0, len(rows), c['score_batch']):
        part = rows[i:i + c['score_batch']]
        b = batch([dict(ids=r['prompt_ids'] + r['completion_ids'], prompt_len=len(r['prompt_ids']))
                   for r in part], tok.pad_token_id)
        with torch.inference_mode():
            lp, _ = forward_logps(m, b)
        values.extend(lp.tolist())
    release(m)
    return values


def evaluate_file(profile, path):
    path = Path(path)
    result = path.with_suffix('.metrics.json')
    fingerprint = dict(generation_sha=sha(path), reference_sha=sha(sft_path(profile) / 'model.safetensors'),
                       evaluator_sha=sha(__file__))
    if result.exists():
        assert read_json(result)['fingerprint'] == fingerprint
        return
    rows = read_rows(path)
    refs = reference_score(profile, rows)
    rewards = score_reward(rows, cfg(profile)['score_batch'])
    groups = {}
    for r, ref, reward in zip(rows, refs, rewards):
        r.update(ref_logp=ref, reward=reward, kl=r['policy_logp'] - ref)
        words = r['completion'].split()
        trigrams = list(zip(words, words[1:], words[2:]))
        r['repetition_3gram'] = 1 - len(set(trigrams)) / max(1, len(trigrams)) if trigrams else 0.
        groups.setdefault(r['prompt_id'], []).append([reward, r['kl']])
    group_means = np.array([np.mean(v, axis=0) for v in groups.values()])
    rng = np.random.default_rng(2026)
    boot = np.array([group_means[rng.integers(0, len(groups), len(groups))].mean(0) for _ in range(2000)])
    metrics = dict(fingerprint=fingerprint, prompts=len(groups), generations=len(rows),
        reward=float(group_means[:, 0].mean()), kl=float(group_means[:, 1].mean()),
        reward_ci95=np.percentile(boot[:, 0], [2.5, 97.5]).tolist(),
        kl_ci95=np.percentile(boot[:, 1], [2.5, 97.5]).tolist(),
        mean_length=float(np.mean([len(r['completion_ids']) for r in rows])),
        eos_fraction=float(np.mean([r['stopped_eos'] for r in rows])),
        repetition_3gram=float(np.mean([r['repetition_3gram'] for r in rows])))
    write_rows(path.with_suffix('.scored.jsonl'), rows)
    write_json(result, metrics)
    print(json.dumps(dict(file=str(path), **metrics)), flush=True)


def main(a):
    setup(42)
    c = cfg(a.profile)
    data = ROOT / 'data' / a.profile
    if a.stage == 'sample-preferences':
        for split in ['train', 'dev']:
            sample_file(a.profile, sft_path(a.profile), data / f'prompts_{split}.jsonl',
                        data / f'candidates_{split}.jsonl', 4)
    elif a.stage == 'score-preferences':
        make_pairs(a.profile)
    elif a.stage == 'cache-reference':
        cache_reference(a.profile)
    elif a.stage in ['evaluate-dev', 'evaluate-test']:
        split = a.stage.split('-')[1]
        checkpoint = ROOT / 'checkpoints' / a.profile / a.run / 'final'
        dest = ROOT / 'results' / a.profile / a.run
        out = dest / f'{split}.jsonl'
        sample_file(a.profile, checkpoint, data / f'prompts_{split}.jsonl', out,
                    c['test_samples'] if split == 'test' else 1)
        evaluate_file(a.profile, out)
        if split == 'dev':
            for path in sorted(dest.glob('curve_*.jsonl')):
                if '.scored.' not in path.name:
                    evaluate_file(a.profile, path)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('profile')
    p.add_argument('stage', choices=['sample-preferences', 'score-preferences', 'cache-reference',
                                   'evaluate-dev', 'evaluate-test'])
    p.add_argument('--run', default='sft')
    main(p.parse_args())
