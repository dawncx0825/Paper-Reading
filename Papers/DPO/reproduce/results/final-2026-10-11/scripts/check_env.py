import platform
import subprocess
import sys

import torch
import transformers

from dpo_repro.core import ROOT, write_json

assert torch.cuda.is_available(), 'CUDA required'
assert torch.cuda.is_bf16_supported(), 'BF16 required'
info = dict(python=sys.version, platform=platform.platform(), torch=torch.__version__,
    transformers=transformers.__version__, cuda=torch.version.cuda,
    gpu=torch.cuda.get_device_name(), gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
    nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
    memory=subprocess.check_output(['free', '-h'], text=True),
    disk=subprocess.check_output(['df', '-h', str(ROOT)], text=True),
    cpu=subprocess.check_output(['lscpu'], text=True))
write_json(ROOT / 'logs/environment.json', info)
print(info, flush=True)
