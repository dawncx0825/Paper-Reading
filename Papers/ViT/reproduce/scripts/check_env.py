#!/usr/bin/env python3
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision

from vit_repro.config import PROJECT_ROOT
from vit_repro.utils import source_manifest, utc_now, write_json


def command_output(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return (result.stdout or result.stderr).strip()


def main() -> None:
    cuda = torch.cuda.is_available()
    payload = {
        "created_at": utc_now(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "numpy": np.__version__,
        "cuda_available": cuda,
        "torch_cuda": torch.version.cuda,
        "bf16_supported": torch.cuda.is_bf16_supported() if cuda else False,
        "device_name": torch.cuda.get_device_name(0) if cuda else None,
        "device_memory_bytes": torch.cuda.get_device_properties(0).total_memory if cuda else None,
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
                "--format=csv,noheader",
            ]
        ),
        "source": source_manifest(PROJECT_ROOT),
    }
    write_json(Path("logs") / "environment.json", payload)
    print(json.dumps(payload, indent=2))
    if not cuda or not payload["bf16_supported"]:
        raise RuntimeError("A CUDA GPU with BF16 support is required for the planned run")


if __name__ == "__main__":
    main()
