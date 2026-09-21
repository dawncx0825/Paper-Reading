from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase


PAD_TOKEN = "[PAD]"


def load_tokenizer(model_or_path: str | Path, revision: str | None = None, padding_side: str = "right"):
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_or_path),
        revision=revision,
        padding_side=padding_side,
        use_fast=True,
    )
    if tokenizer.pad_token is None or tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.padding_side = padding_side
    return tokenizer


def align_model_and_tokenizer(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> None:
    if model.get_input_embeddings().num_embeddings != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    model.config.pad_token_id = tokenizer.pad_token_id
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id


def load_causal_model(
    model_or_path: str | Path,
    tokenizer: PreTrainedTokenizerBase,
    revision: str | None = None,
    train: bool = False,
) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(
        str(model_or_path),
        revision=revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    align_model_and_tokenizer(model, tokenizer)
    model.config.use_cache = not train
    return model


def apply_reward_center(model: PreTrainedModel, center: float):
    """Subtract a fixed scalar from score outputs without changing checkpoint keys."""

    if not hasattr(model, "score"):
        raise TypeError(f"{type(model).__name__} has no scalar score head")

    def subtract_center(_module, _inputs, output):
        return output - center

    return model.score.register_forward_hook(subtract_center)


@contextmanager
def temporarily_disable_adapter(model) -> Iterator[None]:
    if not hasattr(model, "disable_adapter"):
        raise TypeError("Expected a PEFT model with disable_adapter()")
    with model.disable_adapter():
        yield
