import argparse
import csv
import json
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dpo_repro.core import ROOT, read_json, write_json

p = argparse.ArgumentParser()
p.add_argument('profile')
a = p.parse_args()
root = ROOT / 'results' / a.profile
rows = []
for path in sorted(root.glob('*/*.metrics.json')):
    r = read_json(path)
    rows.append(dict(run=path.parent.name, evaluation=path.stem.replace('.metrics', ''), **r))
write_json(root / 'summary.json', rows)
seed_summary = []
for prefix in ['dpo_b0.1_s', 'preferred_b0.1_s']:
    selected = [r for r in rows if r['run'].startswith(prefix) and r['evaluation'] == 'test']
    if selected:
        seed_summary.append(dict(method=prefix, seed_runs=[r['run'] for r in selected],
            reward_mean=float(np.mean([r['reward'] for r in selected])),
            reward_sample_std=float(np.std([r['reward'] for r in selected], ddof=1)) if len(selected)>1 else None,
            kl_mean=float(np.mean([r['kl'] for r in selected])),
            kl_sample_std=float(np.std([r['kl'] for r in selected], ddof=1)) if len(selected)>1 else None,
            scope='update seeds conditional on shared SFT and preference data'))
write_json(root / 'seed_summary.json', seed_summary)
if rows:
    keys = ['run', 'evaluation', 'prompts', 'generations', 'reward', 'kl', 'mean_length', 'eos_fraction', 'repetition_3gram']
    with open(root / 'summary.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    for split in ['dev', 'test']:
        if not any(r['evaluation'] == split for r in rows):
            continue
        fig, ax = plt.subplots(figsize=(9, 6))
        for run in sorted({r['run'] for r in rows}):
            points = [r for r in rows if r['run'] == run and
                      (r['evaluation'] == split or (split == 'dev' and r['evaluation'].startswith('curve_')))]
            if points:
                ax.scatter([r['kl'] for r in points], [r['reward'] for r in points], label=run, s=24)
        ax.set(xlabel='Estimated sequence KL to SFT (nats)', ylabel='Positive sentiment probability',
               title=f'IMDb reward–KL ({split})')
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(root / f'reward_kl_{split}.png', dpi=160)
        plt.close(fig)
print(json.dumps(dict(profile=a.profile, evaluated_files=len(rows))), flush=True)
