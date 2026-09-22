from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class ViTSpec:
    image_size: int = 384
    patch_size: int = 16
    in_channels: int = 3
    num_classes: int = 100
    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.0
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    activation_checkpointing: bool = False


class PatchEmbedding(nn.Module):
    def __init__(self, spec: ViTSpec) -> None:
        super().__init__()
        self.image_size = spec.image_size
        self.patch_size = spec.patch_size
        self.projection = nn.Conv2d(
            spec.in_channels,
            spec.hidden_size,
            kernel_size=spec.patch_size,
            stride=spec.patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(f"Expected BCHW images, received {tuple(images.shape)}")
        if images.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected {self.image_size}x{self.image_size}, received {tuple(images.shape[-2:])}"
            )
        patches = self.projection(images)
        return patches.flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, spec: ViTSpec) -> None:
        super().__init__()
        if spec.hidden_size % spec.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = spec.num_heads
        self.head_dim = spec.hidden_size // spec.num_heads
        self.qkv = nn.Linear(spec.hidden_size, 3 * spec.hidden_size)
        self.projection = nn.Linear(spec.hidden_size, spec.hidden_size)
        self.attention_dropout = spec.attention_dropout
        self.output_dropout = nn.Dropout(spec.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden = inputs.shape
        qkv = self.qkv(inputs).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        output = output.transpose(1, 2).reshape(batch, tokens, hidden)
        return self.output_dropout(self.projection(output))


class MLP(nn.Module):
    def __init__(self, spec: ViTSpec) -> None:
        super().__init__()
        self.fc1 = nn.Linear(spec.hidden_size, spec.mlp_dim)
        self.activation = nn.GELU(approximate="tanh")
        self.dropout1 = nn.Dropout(spec.dropout)
        self.fc2 = nn.Linear(spec.mlp_dim, spec.hidden_size)
        self.dropout2 = nn.Dropout(spec.dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.dropout2(self.fc2(self.dropout1(self.activation(self.fc1(inputs)))))


class EncoderBlock(nn.Module):
    def __init__(self, spec: ViTSpec) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(spec.hidden_size, eps=spec.layer_norm_eps)
        self.attention = Attention(spec)
        self.norm2 = nn.LayerNorm(spec.hidden_size, eps=spec.layer_norm_eps)
        self.mlp = MLP(spec)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attention(self.norm1(inputs))
        return hidden + self.mlp(self.norm2(hidden))


class VisionTransformer(nn.Module):
    def __init__(self, spec: ViTSpec) -> None:
        super().__init__()
        self.spec = spec
        self.patch_embedding = PatchEmbedding(spec)
        grid = spec.image_size // spec.patch_size
        self.class_token = nn.Parameter(torch.zeros(1, 1, spec.hidden_size))
        self.position_embedding = nn.Parameter(torch.zeros(1, grid * grid + 1, spec.hidden_size))
        self.embedding_dropout = nn.Dropout(spec.dropout)
        self.blocks = nn.ModuleList(EncoderBlock(spec) for _ in range(spec.depth))
        self.encoder_norm = nn.LayerNorm(spec.hidden_size, eps=spec.layer_norm_eps)
        self.head = nn.Linear(spec.hidden_size, spec.num_classes)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.class_token)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(images)
        class_token = self.class_token.expand(images.shape[0], -1, -1)
        hidden = torch.cat((class_token, patches), dim=1)
        hidden = self.embedding_dropout(hidden + self.position_embedding)
        for block in self.blocks:
            if self.spec.activation_checkpointing and self.training:
                hidden = checkpoint(block, hidden, use_reentrant=False)
            else:
                hidden = block(hidden)
        return self.encoder_norm(hidden)[:, 0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(images))


def vit_base_patch16(
    image_size: int = 384,
    num_classes: int = 100,
    dropout: float = 0.0,
    attention_dropout: float = 0.0,
    activation_checkpointing: bool = False,
) -> VisionTransformer:
    return VisionTransformer(
        ViTSpec(
            image_size=image_size,
            num_classes=num_classes,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_checkpointing=activation_checkpointing,
        )
    )


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())

