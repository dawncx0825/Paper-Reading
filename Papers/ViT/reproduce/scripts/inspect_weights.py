#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from vit_repro.model import parameter_count, vit_base_patch16
from vit_repro.utils import sha256_file, write_json
from vit_repro.weights import load_official_npz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="checkpoints/pretrained/ViT-B_16.npz")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--output", default="logs/weight_load_report.json")
    args = parser.parse_args()
    model = vit_base_patch16(image_size=args.image_size, num_classes=100)
    report = load_official_npz(model, args.weights)
    report.update(
        {
            "sha256": sha256_file(args.weights),
            "parameter_count": parameter_count(model),
            "head_weight_nonzero": int(torch.count_nonzero(model.head.weight)),
            "head_bias_nonzero": int(torch.count_nonzero(model.head.bias)),
        }
    )
    write_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

