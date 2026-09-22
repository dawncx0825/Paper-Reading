#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from vit_repro.config import load_config
from vit_repro.data import build_loaders
from vit_repro.engine import evaluate, load_checkpoint
from vit_repro.model import vit_base_patch16
from vit_repro.utils import utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="formal")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="data/raw")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    cfg = replace(load_config(args.config), eval_split=args.split)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vit_base_patch16(
        image_size=cfg.image_size,
        num_classes=cfg.num_classes,
        dropout=cfg.dropout,
        attention_dropout=cfg.attention_dropout,
        activation_checkpointing=False,
    )
    load_checkpoint(args.checkpoint, model)
    model.to(device)
    _, eval_loader, _, _ = build_loaders(cfg, args.data_root)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    result = evaluate(
        model,
        eval_loader,
        device,
        cfg.precision,
        predictions_path=output_dir / f"predictions_{args.split}.jsonl",
    )
    result.update({"split": args.split, "checkpoint": str(Path(args.checkpoint).resolve()), "evaluated_at": utc_now()})
    write_json(output_dir / f"evaluation_{args.split}.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

