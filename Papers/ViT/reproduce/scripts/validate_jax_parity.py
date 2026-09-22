#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from vit_repro.model import vit_base_patch16
from vit_repro.utils import write_json
from vit_repro.weights import load_official_npz


PINNED_OFFICIAL_COMMIT = "64801f1b3b367b3611cc27a3d45cc22870a36fb3"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="checkpoints/pretrained/ViT-B_16.npz")
    parser.add_argument("--official-repo", default="vendor/vision_transformer")
    parser.add_argument("--output", default="logs/jax_parity.json")
    args = parser.parse_args()

    official_repo = Path(args.official_repo).resolve()
    commit = subprocess.run(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_OFFICIAL_COMMIT:
        raise RuntimeError(f"Official repo must be at {PINNED_OFFICIAL_COMMIT}, found {commit}")
    sys.path.insert(0, str(official_repo))

    import jax

    jax.config.update("jax_platform_name", "cpu")
    import jax.numpy as jnp
    from flax import traverse_util
    import ml_collections
    from vit_jax import models_vit

    np.random.seed(123)
    images_nchw = np.random.uniform(-1.0, 1.0, size=(1, 3, 224, 224)).astype(np.float32)
    images_nhwc = np.transpose(images_nchw, (0, 2, 3, 1))

    archive = np.load(args.weights, allow_pickle=False)
    # The released 2020 checkpoint used Flax's former global auto-numbering
    # within a block. Modern Flax numbers each module class independently.
    # Rename only scopes; array values remain untouched.
    renamed = {}
    for key in archive.files:
        modern_key = key
        if "Transformer/encoderblock_" in key:
            modern_key = modern_key.replace("MultiHeadDotProductAttention_1", "MultiHeadDotProductAttention_0")
            modern_key = modern_key.replace("LayerNorm_2", "LayerNorm_1")
            modern_key = modern_key.replace("MlpBlock_3", "MlpBlock_0")
        renamed[tuple(modern_key.split("/"))] = archive[key]
    params = traverse_util.unflatten_dict(renamed)
    patches = ml_collections.ConfigDict({"size": (16, 16)})
    transformer = ml_collections.ConfigDict(
        {
            "mlp_dim": 3072,
            "num_heads": 12,
            "num_layers": 12,
            "attention_dropout_rate": 0.0,
            "dropout_rate": 0.0,
        }
    )
    flax_model = models_vit.VisionTransformer(
        num_classes=0,
        patches=patches,
        transformer=transformer,
        hidden_size=768,
        representation_size=None,
        classifier="token",
    )
    flax_features = np.asarray(
        flax_model.apply({"params": params}, jnp.asarray(images_nhwc), train=False)
    )

    torch_model = vit_base_patch16(image_size=224, num_classes=100)
    load_official_npz(torch_model, args.weights)
    torch_model.eval()
    with torch.inference_mode():
        torch_features = torch_model.forward_features(torch.from_numpy(images_nchw)).numpy()

    difference = np.abs(flax_features - torch_features)
    cosine = float(
        np.sum(flax_features * torch_features)
        / (np.linalg.norm(flax_features) * np.linalg.norm(torch_features))
    )
    report = {
        "official_repo_commit": commit,
        "input_shape_nchw": list(images_nchw.shape),
        "feature_shape": list(torch_features.shape),
        "max_absolute_error": float(difference.max()),
        "mean_absolute_error": float(difference.mean()),
        "cosine_similarity": cosine,
        "passed": bool(difference.max() < 5e-4 and cosine > 0.999999),
    }
    write_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("PyTorch encoder did not match the official Flax encoder")


if __name__ == "__main__":
    main()
