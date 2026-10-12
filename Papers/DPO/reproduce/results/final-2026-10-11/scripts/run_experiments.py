"""Persistent sequential orchestrator. Each subprocess frees GPU memory on exit."""
import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone

from dpo_repro.core import ROOT, cfg, read_json, write_json


def stamp():
    return datetime.now(timezone.utc).isoformat()


def run(profile, stage, *extra):
    status = dict(profile=profile, stage=stage, args=list(extra), status='running',
                  started_at=stamp(), pid=os.getpid())
    write_json(ROOT / 'logs/pipeline_status.json', status)
    log = ROOT / 'logs' / f'pipeline_{profile}.log'
    with open(log, 'a', buffering=1) as f:
        f.write('\n' + json.dumps(status) + '\n')
        result = subprocess.run(['bash', 'run_pipeline.sh', profile, stage, *extra],
                                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    status.update(status='complete' if result.returncode == 0 else 'failed',
                  finished_at=stamp(), returncode=result.returncode)
    write_json(ROOT / 'logs/pipeline_status.json', status)
    print(json.dumps(status), flush=True)
    if result.returncode:
        raise RuntimeError(f'{profile}/{stage} failed; inspect {log}')


def one_profile(profile):
    for stage in ['data', 'sft', 'sample-preferences', 'score-preferences', 'cache-reference']:
        run(profile, stage)
    runs = ['sft']
    settings = [(stage, beta, 42) for stage, beta in [('preferred-ft', .1), ('dpo', .1)]]
    if profile == 'full':
        settings += [('dpo', b, 42) for b in [.05, 1., 5.]]
        settings += [(s, .1, seed) for seed in [43, 44] for s in ['preferred-ft', 'dpo']]
    for stage, beta, seed in settings:
        run(profile, stage, '--beta', str(beta), '--seed', str(seed))
        name = ('preferred' if stage == 'preferred-ft' else stage) + f'_b{beta:g}_s{seed}'
        runs.append(name)
        run(profile, 'evaluate-dev', '--run', name)
    run(profile, 'evaluate-dev', '--run', 'sft')
    run(profile, 'summarize')
    if profile == 'full':
        # The primary beta=0.1 and final checkpoints were fixed before test access.
        write_json(ROOT / 'results/full/test_protocol.json', dict(frozen_at=stamp(),
            primary_beta=.1, checkpoint='final', runs=runs, prompts=cfg(profile)['test_prompts'],
            samples_per_prompt=cfg(profile)['test_samples'], selection='prespecified; no test tuning'))
        for name in runs:
            run(profile, 'evaluate-test', '--run', name)
        run(profile, 'summarize')


def main(a):
    (ROOT / 'logs').mkdir(exist_ok=True)
    with open(ROOT / 'logs/pipeline.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for profile in a.profiles:
                one_profile(profile)
                # Numeric integrity gate; quality and deviations remain visible in outputs.
                if profile == 'pilot':
                    for name in ['sft', 'preferred_b0.1_s42', 'dpo_b0.1_s42']:
                        stats = read_json(ROOT / 'results/pilot' / name / 'dev.metrics.json')
                        if not 0 <= stats['reward'] <= 1 or stats['generations'] == 0:
                            raise RuntimeError('Pilot metrics failed integrity gate')
                        if stats['repetition_3gram'] > .5 or stats['mean_length'] < 8:
                            raise RuntimeError('Pilot shows severe repetition or length collapse; review samples before full')
            write_json(ROOT / 'logs/pipeline_status.json', dict(status='complete',
                profiles=a.profiles, finished_at=stamp(), pid=os.getpid()))
        except Exception as e:
            state = read_json(ROOT / 'logs/pipeline_status.json')
            state.update(status='failed', error=str(e), finished_at=stamp())
            write_json(ROOT / 'logs/pipeline_status.json', state)
            raise


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('profiles', nargs='+', choices=['smoke', 'benchmark', 'pilot', 'full'])
    main(p.parse_args())
