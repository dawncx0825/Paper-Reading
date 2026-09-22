#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from vit_repro.utils import sha256_file, utc_now, write_json


WEIGHT_URL = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "vit-reproduction/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="checkpoints/pretrained/ViT-B_16.npz")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    destination = Path(args.output)
    if args.force or not destination.exists():
        download(WEIGHT_URL, destination)
    payload = {
        "downloaded_at": utc_now(),
        "url": WEIGHT_URL,
        "path": str(destination.resolve()),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }
    write_json(destination.with_suffix(".manifest.json"), payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

