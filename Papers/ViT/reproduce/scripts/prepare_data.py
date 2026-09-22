#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from torchvision.datasets import CIFAR100

from vit_repro.data import prepare_dev_split
from vit_repro.utils import sha256_file, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument(
        "--archive-source",
        default="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    )
    args = parser.parse_args()
    root = Path(args.data_root)
    train = CIFAR100(root=root, train=True, download=True)
    test = CIFAR100(root=root, train=False, download=True)
    split = prepare_dev_split(root.parent, train.targets, seed=42)
    archive = root / "cifar-100-python.tar.gz"
    md5 = hashlib.md5(archive.read_bytes()).hexdigest()
    if md5 != "eb9058c3a382ffc7106e4002c42a8d85":
        raise RuntimeError(f"Unexpected CIFAR-100 archive MD5: {md5}")
    payload = {
        "created_at": utc_now(),
        "dataset": "CIFAR-100 Python version",
        "train_examples": len(train),
        "test_examples": len(test),
        "fine_classes": len(train.classes),
        "train_class_counts": dict(sorted(Counter(train.targets).items())),
        "test_class_counts": dict(sorted(Counter(test.targets).items())),
        "archive": {
            "source_used": args.archive_source,
            "canonical_source": "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
            "path": str(archive.resolve()),
            "bytes": archive.stat().st_size,
            "md5": md5,
            "sha256": sha256_file(archive),
        },
        "dev_split": split,
    }
    write_json(root.parent / "manifest.json", payload)
    print(json.dumps({key: value for key, value in payload.items() if key != "dev_split"}, indent=2))
    print(f"dev_examples={len(split['indices'])}")


if __name__ == "__main__":
    main()
