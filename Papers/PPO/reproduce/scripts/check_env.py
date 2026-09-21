#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import torch
import transformers
import trl
from accelerate import __version__ as accelerate_version
from datasets import __version__ as datasets_version
from peft import __version__ as peft_version

from ppo_repro.config import ensure_output_dirs, load_config
from ppo_repro.models import load_tokenizer
from ppo_repro.utils import utc_now, write_json


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    return (result.stdout or result.stderr).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="smoke")
    parser.add_argument("--load-tokenizer", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_output_dirs(cfg)

    cuda_available = torch.cuda.is_available()
    payload = {
        "created_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "packages": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "trl": trl.__version__,
            "accelerate": accelerate_version,
            "datasets": datasets_version,
            "peft": peft_version,
        },
        "cuda": {
            "available": cuda_available,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if cuda_available else None,
            "bf16_supported": torch.cuda.is_bf16_supported() if cuda_available else False,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda_available else None,
        },
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "dashscope_key_visible": bool(os.environ.get("DASHSCOPE_API_KEY")),
    }
    if args.load_tokenizer:
        tokenizer = load_tokenizer(cfg.base_model_source, cfg.base_model_revision)
        payload["tokenizer"] = {
            "length": len(tokenizer),
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_is_distinct_from_eos": tokenizer.pad_token_id != tokenizer.eos_token_id,
        }
    if not cuda_available or not payload["cuda"]["bf16_supported"]:
        raise RuntimeError(f"CUDA BF16-capable GPU required: {json.dumps(payload['cuda'])}")
    output = cfg.log_dir / "environment.json"
    write_json(output, payload)
    print(json.dumps(payload, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
