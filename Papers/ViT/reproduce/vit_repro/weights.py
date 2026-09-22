from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .model import VisionTransformer


def _tensor(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32))


def interpolate_position_embedding(
    source: torch.Tensor,
    target_grid_size: tuple[int, int],
) -> torch.Tensor:
    if source.ndim != 3 or source.shape[0] != 1:
        raise ValueError(f"Expected position embedding shape [1, tokens, hidden], got {tuple(source.shape)}")
    class_position = source[:, :1]
    patch_positions = source[:, 1:]
    source_grid = int(math.sqrt(patch_positions.shape[1]))
    if source_grid * source_grid != patch_positions.shape[1]:
        raise ValueError("Source patch positions do not form a square grid")
    if (source_grid, source_grid) == target_grid_size:
        return source
    patch_positions = patch_positions.reshape(1, source_grid, source_grid, source.shape[-1])
    patch_positions = patch_positions.permute(0, 3, 1, 2)
    patch_positions = F.interpolate(
        patch_positions,
        size=target_grid_size,
        mode="bicubic",
        align_corners=False,
    )
    patch_positions = patch_positions.permute(0, 2, 3, 1).reshape(
        1, target_grid_size[0] * target_grid_size[1], source.shape[-1]
    )
    return torch.cat((class_position, patch_positions), dim=1)


def _linear_weight(array: np.ndarray) -> torch.Tensor:
    return _tensor(array).reshape(array.shape[0], -1).t().contiguous()


def _attention_out_weight(array: np.ndarray) -> torch.Tensor:
    return _tensor(array).reshape(-1, array.shape[-1]).t().contiguous()


def _copy(parameter: torch.Tensor, value: torch.Tensor, source_key: str) -> None:
    if tuple(parameter.shape) != tuple(value.shape):
        raise ValueError(
            f"Shape mismatch for {source_key}: checkpoint {tuple(value.shape)} versus model {tuple(parameter.shape)}"
        )
    parameter.copy_(value)


def load_official_npz(model: VisionTransformer, checkpoint_path: str | Path) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_path)
    archive = np.load(checkpoint_path, allow_pickle=False)
    used: set[str] = set()

    def take(key: str) -> np.ndarray:
        if key not in archive.files:
            raise KeyError(f"Missing required checkpoint key: {key}")
        used.add(key)
        return archive[key]

    with torch.no_grad():
        _copy(
            model.patch_embedding.projection.weight,
            _tensor(take("embedding/kernel")).permute(3, 2, 0, 1).contiguous(),
            "embedding/kernel",
        )
        _copy(model.patch_embedding.projection.bias, _tensor(take("embedding/bias")), "embedding/bias")
        _copy(model.class_token, _tensor(take("cls")), "cls")

        source_position = _tensor(take("Transformer/posembed_input/pos_embedding"))
        target_grid = (
            model.spec.image_size // model.spec.patch_size,
            model.spec.image_size // model.spec.patch_size,
        )
        interpolated = interpolate_position_embedding(source_position, target_grid)
        _copy(model.position_embedding, interpolated, "Transformer/posembed_input/pos_embedding")

        for index, block in enumerate(model.blocks):
            prefix = f"Transformer/encoderblock_{index}/"
            _copy(block.norm1.weight, _tensor(take(prefix + "LayerNorm_0/scale")), prefix + "LayerNorm_0/scale")
            _copy(block.norm1.bias, _tensor(take(prefix + "LayerNorm_0/bias")), prefix + "LayerNorm_0/bias")

            attention = prefix + "MultiHeadDotProductAttention_1/"
            q_weight = _linear_weight(take(attention + "query/kernel"))
            k_weight = _linear_weight(take(attention + "key/kernel"))
            v_weight = _linear_weight(take(attention + "value/kernel"))
            q_bias = _tensor(take(attention + "query/bias")).reshape(-1)
            k_bias = _tensor(take(attention + "key/bias")).reshape(-1)
            v_bias = _tensor(take(attention + "value/bias")).reshape(-1)
            _copy(block.attention.qkv.weight, torch.cat((q_weight, k_weight, v_weight)), attention + "qkv/kernel")
            _copy(block.attention.qkv.bias, torch.cat((q_bias, k_bias, v_bias)), attention + "qkv/bias")
            _copy(
                block.attention.projection.weight,
                _attention_out_weight(take(attention + "out/kernel")),
                attention + "out/kernel",
            )
            _copy(
                block.attention.projection.bias,
                _tensor(take(attention + "out/bias")),
                attention + "out/bias",
            )

            _copy(block.norm2.weight, _tensor(take(prefix + "LayerNorm_2/scale")), prefix + "LayerNorm_2/scale")
            _copy(block.norm2.bias, _tensor(take(prefix + "LayerNorm_2/bias")), prefix + "LayerNorm_2/bias")
            mlp = prefix + "MlpBlock_3/"
            _copy(block.mlp.fc1.weight, _linear_weight(take(mlp + "Dense_0/kernel")), mlp + "Dense_0/kernel")
            _copy(block.mlp.fc1.bias, _tensor(take(mlp + "Dense_0/bias")), mlp + "Dense_0/bias")
            _copy(block.mlp.fc2.weight, _linear_weight(take(mlp + "Dense_1/kernel")), mlp + "Dense_1/kernel")
            _copy(block.mlp.fc2.bias, _tensor(take(mlp + "Dense_1/bias")), mlp + "Dense_1/bias")

        _copy(
            model.encoder_norm.weight,
            _tensor(take("Transformer/encoder_norm/scale")),
            "Transformer/encoder_norm/scale",
        )
        _copy(
            model.encoder_norm.bias,
            _tensor(take("Transformer/encoder_norm/bias")),
            "Transformer/encoder_norm/bias",
        )
        # The ImageNet-21k classification head and optional pre-logits layer are
        # intentionally discarded for downstream fine-tuning.
        model.head.weight.zero_()
        model.head.bias.zero_()

    ignored = sorted(set(archive.files) - used)
    expected_ignored_prefixes = ("head/", "pre_logits/")
    unexpected = [key for key in ignored if not key.startswith(expected_ignored_prefixes)]
    if unexpected:
        raise ValueError(f"Unexpected checkpoint keys were not mapped: {unexpected}")
    return {
        "checkpoint": str(checkpoint_path),
        "loaded_key_count": len(used),
        "loaded_keys": sorted(used),
        "ignored_keys": ignored,
        "unexpected_keys": unexpected,
        "source_position_tokens": int(source_position.shape[1]),
        "target_position_tokens": int(model.position_embedding.shape[1]),
        "position_interpolation": "bicubic_align_corners_false",
        "head_initialization": "zeros",
    }

