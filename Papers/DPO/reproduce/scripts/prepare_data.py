import argparse
import hashlib
import re

import numpy as np
from datasets import load_dataset

from dpo_repro.core import ROOT, cfg, digest, read_json, sha, tokenizer, write_json, write_rows


def text_hash(s):
    return hashlib.sha256(re.sub(r'\s+', ' ', s).strip().encode()).hexdigest()


def prepare(profile):
    c = cfg(profile)
    dest = ROOT / 'data' / profile
    if (dest / 'audit.json').exists():
        old = read_json(dest / 'audit.json')
        assert old['config_digest'] == digest(c), 'Data config mismatch'
        print('Data already prepared', profile)
        return
    t = tokenizer()
    raw = ROOT / 'data/raw/imdb/plain_text'
    pools, seen = {}, set()
    # Test is assigned first, so any duplicated text is excluded from training.
    for split in ['test', 'train']:
        files = sorted(str(p) for p in raw.glob(f'{split}*.parquet'))
        assert files, f'Missing {split} parquet'
        ds = load_dataset('parquet', data_files=files, split='train')
        rows = []
        for x in ds:
            h = text_hash(x['text'])
            if h in seen:
                continue
            seen.add(h)
            rows.append(dict(id=h, text=x['text'], label=int(x['label']), original_split=split))
        pools[split] = sorted(rows, key=lambda x: x['id'])
    dev = pools['train'][:1000]
    train = pools['train'][1000:]
    splits = dict(train=train[:c['sft_rows']], dev=dev, test=pools['test'])
    sft = []
    for r in splits['train']:
        ids = t.encode(r['text'], add_special_tokens=False)
        truncated = len(ids) + 1 > c['sft_length']
        ids = ids[:c['sft_length']] if truncated else ids + [t.eos_token_id]
        if len(ids) > 1:
            sft.append(dict(id=r['id'], ids=ids, prompt_len=0, truncated=truncated))
    write_rows(dest / 'sft.jsonl', sft)
    # Assign prefixes to dev/test before train to exclude equal prefixes across splits.
    used = set()
    counts = {}
    for split, target in [('test', c['test_prompts']), ('dev', c['dev_prompts']),
                          ('train', c['train_prompts'])]:
        out = []
        encoded = [(r, t.encode(r['text'], add_special_tokens=False)) for r in splits[split]]
        for round_idx in range(7):
            for r, ids in encoded:
                n = 2 + (int(r['id'][:8], 16) + round_idx) % 7
                prefix = tuple(ids[:n])
                if len(ids) <= n or prefix in used:
                    continue
                used.add(prefix)
                out.append(dict(id=digest([r['id'], n]), review_id=r['id'], prompt_ids=list(prefix),
                                prompt=t.decode(prefix)))
                if len(out) == target:
                    break
            if len(out) == target:
                break
        assert out, split
        write_rows(dest / f'prompts_{split}.jsonl', out)
        counts[split] = dict(prefixes=len(out), unique_reviews=len({r['review_id'] for r in out}))
    split_ids = {k: [r['id'] for r in v] for k, v in splits.items()}
    assert not (set(split_ids['train']) & set(split_ids['dev']))
    assert not (set(split_ids['train']) & set(split_ids['test']))
    write_json(dest / 'splits.json', split_ids)
    write_json(dest / 'audit.json', dict(config_digest=digest(c), counts=counts, sft_rows=len(sft),
        review_pools={k: len(v) for k, v in splits.items()},
        label_counts={k: {str(label): sum(r['label'] == label for r in v) for label in [0, 1]}
                      for k, v in splits.items()},
        sft_truncated=sum(r['truncated'] for r in sft),
        sft_lengths=np.percentile([len(r['ids']) for r in sft], [0, 50, 95, 100]).tolist(),
        assets_lock_sha=sha(ROOT / 'configs/assets.lock.json'),
        files={p.name: sha(p) for p in dest.glob('*.jsonl')},
        split_policy='content-hash; official test excluded; 1000 training reviews held out for dev'))
    print(read_json(dest / 'audit.json'), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('profile')
    prepare(p.parse_args().profile)
