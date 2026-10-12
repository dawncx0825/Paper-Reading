import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def read_rows(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def cfg(profile):
    return read_json(ROOT / 'configs' / f'{profile}.json')


def setup(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'].cpu())
    torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


def tokenizer():
    t = AutoTokenizer.from_pretrained(ROOT / 'models/base', local_files_only=True)
    t.pad_token = t.eos_token
    return t


def model(path, train=False):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32,
                                           attn_implementation='sdpa', local_files_only=True)
    for module in m.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    m.to('cuda')
    m.config.use_cache = not train
    if train:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        m.train()
    else:
        m.eval()
        m.requires_grad_(False)
    return m


def batch(rows, pad_id, device='cuda'):
    """Rows contain exact token IDs and number of conditioning (prompt) tokens."""
    width = max(len(r['ids']) for r in rows)
    ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for i, r in enumerate(rows):
        n, p = len(r['ids']), r['prompt_len']
        if not 0 <= p < n:
            raise ValueError('Invalid prompt/completion boundary')
        ids[i, :n] = torch.tensor(r['ids'])
        mask[i, :n] = 1
        labels[i, p:n] = ids[i, p:n]
    return {k: v.to(device) for k, v in dict(input_ids=ids, attention_mask=mask, labels=labels).items()}


def logps(logits, labels):
    target = labels[:, 1:]
    # FP32 cross entropy avoids an additional full log-softmax tensor kept in Python.
    token_nll = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.size(-1)),
                               target.reshape(-1), ignore_index=-100, reduction='none')
    valid = target.ne(-100)
    return -token_nll.reshape_as(target).sum(-1), valid.sum(-1)


def forward_logps(m, b):
    out = m(input_ids=b['input_ids'], attention_mask=b['attention_mask'], use_cache=False)
    return logps(out.logits, b['labels'])


def dpo_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    margin = beta * ((chosen - ref_chosen) - (rejected - ref_rejected))
    return -F.logsigmoid(margin), margin


def pair_rows(pairs):
    return ([dict(ids=r['prompt_ids'] + r['chosen_ids'], prompt_len=len(r['prompt_ids'])) for r in pairs]
            + [dict(ids=r['prompt_ids'] + r['rejected_ids'], prompt_len=len(r['prompt_ids'])) for r in pairs])


def sft_path(profile):
    return ROOT / 'checkpoints' / profile / 'sft' / 'final'


def cache_meta(profile):
    p = ROOT / 'data' / profile
    return dict(sft=sha(sft_path(profile) / 'model.safetensors'),
                tokenizer=sha(ROOT / 'models/base/tokenizer.json'),
                train=sha(p / 'pairs_train.jsonl'), dev=sha(p / 'pairs_dev.jsonl'),
                core=sha(Path(__file__)), convention='exact_generated_ids_sum_completion_with_real_eos_v1')
