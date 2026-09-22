from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from .config import ExperimentConfig, PROJECT_ROOT
from .utils import seed_worker, write_json


CIFAR100_MEAN = (0.5, 0.5, 0.5)
CIFAR100_STD = (0.5, 0.5, 0.5)


def train_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.05, 1.0),
                ratio=(0.75, 4.0 / 3.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def eval_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def stratified_dev_indices(targets: Sequence[int], seed: int = 42, per_class: int = 10) -> list[int]:
    generator = np.random.default_rng(seed)
    targets_array = np.asarray(targets)
    output: list[int] = []
    for class_id in sorted(np.unique(targets_array).tolist()):
        candidates = np.flatnonzero(targets_array == class_id)
        if len(candidates) < per_class:
            raise ValueError(f"Class {class_id} contains only {len(candidates)} examples")
        output.extend(generator.choice(candidates, size=per_class, replace=False).tolist())
    return sorted(output)


def prepare_dev_split(root: str | Path, targets: Sequence[int], seed: int = 42) -> dict:
    root = Path(root)
    split_path = root / "splits" / f"cifar100_dev_seed{seed}.json"
    expected = stratified_dev_indices(targets, seed=seed, per_class=10)
    if split_path.exists():
        import json

        stored = json.loads(split_path.read_text(encoding="utf-8"))
        if stored["indices"] != expected:
            raise RuntimeError(f"Existing split does not match deterministic split: {split_path}")
        return stored
    counts = Counter(int(targets[index]) for index in expected)
    payload = {
        "dataset": "CIFAR-100 train",
        "seed": seed,
        "strategy": "10 examples sampled without replacement from each fine-label class",
        "indices": expected,
        "class_counts": {str(key): value for key, value in sorted(counts.items())},
    }
    write_json(split_path, payload)
    return payload


class DatasetWithIndex(Dataset):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        source_index = self.dataset.indices[index] if isinstance(self.dataset, Subset) else index
        return image, target, source_index


def _subset_for_debug(dataset: Dataset, size: int | None, seed: int) -> Dataset:
    if size is None or size >= len(dataset):
        return dataset
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:size].tolist()
    return Subset(dataset, indices)


def build_datasets(cfg: ExperimentConfig, data_root: str | Path, download: bool = False):
    data_root = Path(data_root)
    base_train = datasets.CIFAR100(root=data_root, train=True, download=download)
    split = prepare_dev_split(PROJECT_ROOT / "data", base_train.targets, seed=42)
    dev_indices = set(split["indices"])
    train_indices = list(range(len(base_train)))
    if cfg.train_split == "dev_train":
        train_indices = [index for index in train_indices if index not in dev_indices]

    augmented_train = datasets.CIFAR100(
        root=data_root,
        train=True,
        transform=train_transform(cfg.image_size),
        download=False,
    )
    train_dataset: Dataset = Subset(augmented_train, train_indices)
    train_dataset = _subset_for_debug(train_dataset, cfg.train_subset_size, cfg.seed)

    if cfg.eval_split == "dev":
        deterministic_train = datasets.CIFAR100(
            root=data_root,
            train=True,
            transform=eval_transform(cfg.image_size),
            download=False,
        )
        eval_dataset: Dataset | None = Subset(deterministic_train, split["indices"])
    elif cfg.eval_split == "test":
        eval_dataset = datasets.CIFAR100(
            root=data_root,
            train=False,
            transform=eval_transform(cfg.image_size),
            download=False,
        )
    else:
        eval_dataset = None
    return train_dataset, eval_dataset, split


def build_loaders(cfg: ExperimentConfig, data_root: str | Path, download: bool = False):
    train_dataset, eval_dataset, split = build_datasets(cfg, data_root, download=download)
    generator = torch.Generator().manual_seed(cfg.seed)
    persistent = cfg.num_workers > 0
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.micro_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = DataLoader(
            DatasetWithIndex(eval_dataset),
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
            persistent_workers=persistent,
            worker_init_fn=seed_worker,
        )
    return train_loader, eval_loader, split, generator


def infinite_batches(loader: DataLoader):
    while True:
        yield from loader
