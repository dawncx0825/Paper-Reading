"""Audit the downloaded final snapshot and build report tables/figures, offline.

Run with Python, NumPy and Matplotlib; no model weights or GPU are needed.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SNAP = ROOT / 'results/final-2026-10-11'
OUT = SNAP / 'derived'

def read(path):
    return json.loads(path.read_text())

def rows(path):
    with path.open() as f:
        return [json.loads(line) for line in f]

def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')

manifest = read(SNAP / 'source_manifest.json')
assert all(hashlib.sha256((SNAP / p).read_bytes()).hexdigest() == h for p, h in manifest.items())
protocol = read(SNAP / 'results/full/test_protocol.json')
expected = set(protocol['runs'])
assert len(expected) == 10
state = read(SNAP / 'logs/pipeline_status.json')
assert state['status'] == 'complete'
summary = read(SNAP / 'results/full/summary.json')
tests = {r['run']: r for r in summary if r['evaluation'] == 'test'}
dev = {r['run']: r for r in summary if r['evaluation'] == 'dev'}
assert set(tests) == set(dev) == expected
remote = read(OUT / 'remote_audit.json')['runs']
assert set(remote) == expected
reference_shas = {r['fingerprint']['reference_sha'] for r in tests.values()}
assert len(reference_shas) == 1
prompt_sets = []
for run, metric in tests.items():
    assert metric == dict(run=run, evaluation='test', **read(SNAP / 'results/full' / run / 'test.metrics.json'))
    a = remote[run]
    assert a['generation_sha_matches'] and a['numeric_finite']
    assert a['generations'] == a['unique_ids'] == metric['generations'] == 4000
    assert a['prompts'] == metric['prompts'] == 1000 and a['sample_counts'] == [4]
    points = read(OUT / 'prompt_metrics' / f'{run}.json')
    prompt_sets.append({p['prompt_id'] for p in points})
    assert all(p['samples'] == 4 for p in points)
    data = np.array([[p['reward'], p['kl']] for p in points])
    assert np.allclose(data.mean(0), [metric['reward'], metric['kl']], atol=1e-12, rtol=0)
    rng = np.random.default_rng(2026)
    boot = np.array([data[rng.integers(0, len(data), len(data))].mean(0) for _ in range(2000)])
    assert np.allclose(np.percentile(boot, [2.5, 97.5], axis=0).T,
                       [metric['reward_ci95'], metric['kl_ci95']], atol=1e-12, rtol=0)
assert all(s == prompt_sets[0] for s in prompt_sets)
seed_stats = read(SNAP / 'results/full/seed_summary.json')
assert len(seed_stats) == 2
for s in seed_stats:
    assert len(s['seed_runs']) == 3
    for k in ['reward', 'kl']:
        vals = [tests[r][k] for r in s['seed_runs']]
        assert abs(np.mean(vals) - s[k + '_mean']) < 1e-12
        assert abs(np.std(vals, ddof=1) - s[k + '_sample_std']) < 1e-12

train_audit = {}
for run in sorted(expected):
    log = rows(SNAP / 'logs/full' / run / 'train.jsonl')
    m = read(SNAP / 'logs/full' / run / 'run_metrics.json')
    assert [x['update'] for x in log] == list(range(1, m['updates'] + 1))
    assert m['updates'] == (372 if run == 'sft' else 2344)
    assert not m['resume_used']
    assert all(np.isfinite(v) for x in log for v in x.values() if isinstance(v, (int, float)))
    train_audit[run] = dict(updates=m['updates'], hours=m['elapsed_seconds']/3600,
        peak_reserved_gib=max(x['reserved_gib'] for x in log),
        last100_loss=float(np.mean([x['loss'] for x in log[-100:]])))
stages = rows(SNAP / 'logs/orchestrator.log')
assert all(s['status'] == 'complete' and s['returncode'] == 0 for s in stages)
assert set(s['args'][1] for s in stages if s['profile']=='full' and s['stage']=='evaluate-test') == expected
full = [s for s in stages if s['profile']=='full']
def span(a,b): return (datetime.fromisoformat(b)-datetime.fromisoformat(a)).total_seconds()/3600
durations = {}
for s in full:
    durations[s['stage']] = durations.get(s['stage'],0) + span(s['started_at'],s['finished_at'])
save('analysis.json', dict(verified_source_files=len(manifest), test_runs=10, test_generations=40000,
    identical_test_prompt_sets=True, test_bootstrap_recomputed=True, seed_statistics_recomputed=True,
    full_hours=span(full[0]['started_at'],state['finished_at']),
    queue_hours=span(stages[0]['started_at'],state['finished_at']),
    stage_hours=durations, training=train_audit))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors = {'sft':'#555555','preferred_b0.1_s42':'#d58728','dpo_b0.05_s42':'#c74050',
          'dpo_b0.1_s42':'#2866b2','dpo_b1_s42':'#25825b','dpo_b5_s42':'#885aaa'}
labels = {'sft':'SFT','preferred_b0.1_s42':'Preferred-FT', 'dpo_b0.05_s42':'DPO beta=0.05',
          'dpo_b0.1_s42':'DPO beta=0.1','dpo_b1_s42':'DPO beta=1','dpo_b5_s42':'DPO beta=5'}
fig, axes = plt.subplots(1,2,figsize=(11.5,4.5),gridspec_kw={'width_ratios':[1.6,1]})
for run in colors:
    m=tests[run]
    for ax in axes:
        ax.errorbar(m['kl'],m['reward'],xerr=[[m['kl']-m['kl_ci95'][0]],[m['kl_ci95'][1]-m['kl']]],
                    yerr=[[m['reward']-m['reward_ci95'][0]],[m['reward_ci95'][1]-m['reward']]],
                    fmt='o',capsize=3,color=colors[run],label=labels[run])
for ax in axes:
    ax.set(xlabel='Sequence KL to SFT (nats)',ylabel='Positive sentiment probability',ylim=(.47,1.02))
    ax.grid(alpha=.2)
axes[0].set_title('Final test: seed 42, 1,000 prompts x 4 samples')
axes[0].legend(fontsize=8,loc='lower right')
axes[1].set(xlim=(-.08,1.65),title='Low-KL detail')
fig.tight_layout();fig.savefig(OUT/'final_reward_kl.png',dpi=180);plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(11.5,4.3))
for run in list(colors)[2:]:
    curve=sorted([r for r in summary if r['run']==run and r['evaluation'].startswith('curve_')],key=lambda x:x['evaluation'])
    steps=[int(r['evaluation'].split('_')[1]) for r in curve]
    for ax,key in zip(axes,['reward','kl']):ax.plot(steps,[r[key] for r in curve],label=labels[run],color=colors[run])
for ax,key in zip(axes,['reward','kl']):
    ax.set(xlabel='Optimizer update',ylabel='Positive sentiment probability' if key=='reward' else 'Sequence KL (nats)')
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Development trajectories: 32 fixed prompts, seed 42 (not test)')
fig.tight_layout();fig.savefig(OUT/'dev_trajectories.png',dpi=180);plt.close(fig)

save('report_tables.json',dict(test=tests,dev=dev,seeds=seed_stats))
print(json.dumps(read(OUT/'analysis.json'),indent=2))
