import json
import hashlib
import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(obj, indent=2) + '\n')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    api = HfApi()
    lock = ROOT / 'configs/assets.lock.json'
    specs = [('base', 'openai-community/gpt2-large', 'model'),
             ('reward', 'siebert/sentiment-roberta-large-english', 'model'),
             ('imdb', 'stanfordnlp/imdb', 'dataset')]
    if lock.exists():
        manifest = read_json(lock)
    else:
        manifest = {}
        for name, repo, kind in specs:
            info = api.repo_info(repo, repo_type=kind)
            manifest[name] = dict(repo=repo, kind=kind, revision=info.sha)
        remote = subprocess.check_output(['git', 'ls-remote',
                    'https://github.com/eric-mitchell/direct-preference-optimization.git', 'HEAD'], text=True)
        manifest['official_dpo_commit'] = remote.split()[0]
        write_json(lock, manifest)
    for name, repo, kind in specs:
        item = manifest[name]
        dest = ROOT / ('data/raw/imdb' if kind == 'dataset' else f'models/{name}')
        if kind == 'dataset':
            patterns = ['plain_text/train*', 'plain_text/test*', 'README.md', 'dataset_info.json']
        else:
            files = api.list_repo_files(repo, repo_type=kind, revision=item['revision'])
            weight = 'model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
            patterns = [weight, '*.json', '*.txt', '*.model', 'README.md']
        print('Downloading', name, item['revision'], flush=True)
        snapshot_download(repo, repo_type=kind, revision=item['revision'], local_dir=dest,
                          allow_patterns=patterns, max_workers=4)
        item['files'] = {str(p.relative_to(dest)): dict(bytes=p.stat().st_size, sha256=sha(p))
                         for p in dest.rglob('*') if p.is_file() and '.cache' not in p.parts}
        write_json(lock, manifest)
    target = ROOT / 'vendor/official_dpo'
    target.mkdir(parents=True, exist_ok=True)
    import urllib.request
    for file in ['trainers.py', 'config/config.yaml', 'config/loss/dpo.yaml']:
        out = target / file
        out.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve('https://raw.githubusercontent.com/eric-mitchell/'
            f'direct-preference-optimization/{manifest["official_dpo_commit"]}/{file}', out)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
