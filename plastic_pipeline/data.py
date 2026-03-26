from __future__ import annotations

import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .config import DataConfig, DatasetConfig
from .utils import normalize_label, prettify_label


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
SPLIT_ALIASES = {
    "train": "train",
    "training": "train",
    "tr": "train",
    "val": "val",
    "valid": "val",
    "validation": "val",
    "dev": "val",
    "test": "test",
    "testing": "test",
    "eval": "test",
}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    dataset_name: str
    label_key: str
    label_name: str
    split: str | None


class AddGaussianNoise:
    def __init__(self, mean: float = 0.0, std: float = 0.02, probability: float = 0.5) -> None:
        self.mean = mean
        self.std = std
        self.probability = probability

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.probability:
            return tensor
        noise = torch.randn_like(tensor).mul(self.std).add(self.mean)
        return torch.clamp(tensor + noise, 0.0, 1.0)


class MultiDatasetImageFolder(Dataset):
    def __init__(
        self,
        records: list[SampleRecord],
        label_to_index: dict[str, int],
        domain_to_index: dict[str, int],
        transform: transforms.Compose | None = None,
    ) -> None:
        self.records = records
        self.label_to_index = label_to_index
        self.domain_to_index = domain_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int]:
        record = self.records[index]
        try:
            image = Image.open(record.path).convert("RGB")
        except OSError as exc:
            raise RuntimeError(f"Failed to read image file: {record.path}") from exc
        if self.transform is not None:
            image = self.transform(image)
        return (
            image,
            self.label_to_index[record.label_key],
            self.domain_to_index[record.dataset_name],
        )


def build_transforms(data_config: DataConfig) -> tuple[transforms.Compose, transforms.Compose]:
    train_ops: list[object] = [
        transforms.RandomResizedCrop(
            data_config.image_size,
            scale=(data_config.train_crop_scale_min, data_config.train_crop_scale_max),
            ratio=(0.85, 1.15),
        ),
        transforms.RandomHorizontalFlip(p=data_config.horizontal_flip_probability),
    ]
    if data_config.use_randaugment:
        train_ops.append(
            transforms.RandAugment(
                num_ops=data_config.randaugment_num_ops,
                magnitude=data_config.randaugment_magnitude,
            )
        )
    train_ops.extend(
        [
            transforms.ColorJitter(
                brightness=data_config.color_jitter_brightness,
                contrast=data_config.color_jitter_contrast,
                saturation=data_config.color_jitter_saturation,
                hue=data_config.color_jitter_hue,
            ),
            transforms.ToTensor(),
            AddGaussianNoise(
                std=data_config.gaussian_noise_std,
                probability=data_config.gaussian_noise_probability,
            ),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(
                p=data_config.random_erasing_probability,
                scale=(0.02, 0.2),
                ratio=(0.3, 3.3),
                value="random",
            ),
        ]
    )
    train_transform = transforms.Compose(train_ops)
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(data_config.image_size * data_config.eval_resize_scale)),
            transforms.CenterCrop(data_config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def scan_dataset(dataset_config: DatasetConfig) -> list[SampleRecord]:
    if not dataset_config.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_config.root}")

    alias_lookup = {
        normalize_label(source): target for source, target in dataset_config.class_aliases.items()
    }
    records: list[SampleRecord] = []

    for dirpath, dirnames, filenames in os.walk(dataset_config.root):
        dirnames[:] = sorted(name for name in dirnames if not name.startswith("."))
        image_names = sorted(
            name
            for name in filenames
            if Path(name).suffix.lower() in IMAGE_SUFFIXES and not name.startswith(".")
        )
        if not image_names:
            continue

        directory = Path(dirpath)
        for filename in image_names:
            path = directory / filename
            relative_path = path.relative_to(dataset_config.root)
            split = infer_split(relative_path)
            raw_label = infer_label(relative_path, dataset_config.class_index)
            canonical_label = alias_lookup.get(normalize_label(raw_label), raw_label)
            label_name = prettify_label(canonical_label)
            label_key = normalize_label(label_name)
            records.append(
                SampleRecord(
                    path=path,
                    dataset_name=dataset_config.name,
                    label_key=label_key,
                    label_name=label_name,
                    split=split,
                )
            )

    if not records:
        raise ValueError(f"No images were found under dataset root: {dataset_config.root}")

    return records


def infer_split(relative_path: Path) -> str | None:
    split: str | None = None
    for part in relative_path.parts[:-1]:
        normalized = normalize_label(part)
        if normalized in SPLIT_ALIASES:
            split = SPLIT_ALIASES[normalized]
    return split


def infer_label(relative_path: Path, class_index: int | None) -> str:
    directory_parts = [
        part
        for part in relative_path.parts[:-1]
        if normalize_label(part) not in SPLIT_ALIASES
    ]
    if not directory_parts:
        directory_parts = list(relative_path.parts[:-1])
    if not directory_parts:
        raise ValueError(f"Unable to infer class label from path: {relative_path}")

    if class_index is None:
        return directory_parts[-1]

    try:
        return directory_parts[class_index]
    except IndexError as exc:
        raise ValueError(
            f"class_index={class_index} is invalid for path {relative_path}. "
            f"Resolved directory parts: {directory_parts}"
        ) from exc


def create_dataset_splits(
    records: list[SampleRecord],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[SampleRecord]]:
    explicit_train = [record for record in records if record.split == "train"]
    explicit_val = [record for record in records if record.split == "val"]
    explicit_test = [record for record in records if record.split == "test"]
    unsplit = [record for record in records if record.split is None]

    if explicit_val and explicit_test:
        train_records = explicit_train + unsplit
        if not train_records:
            raise ValueError("Training split is empty after parsing explicit validation/test folders.")
        return _validate_splits(train_records, explicit_val, explicit_test)

    if explicit_test and not explicit_val:
        train_pool = explicit_train + unsplit
        desired_val_count = _desired_holdout_count(len(records), val_ratio, reserve=len(explicit_test) + 1)
        train_records, val_records = _split_records(train_pool, desired_val_count, seed)
        return _validate_splits(train_records, val_records, explicit_test)

    if explicit_val and not explicit_test:
        train_pool = explicit_train + unsplit
        desired_test_count = _desired_holdout_count(len(records), test_ratio, reserve=len(explicit_val) + 1)
        train_records, test_records = _split_records(train_pool, desired_test_count, seed)
        return _validate_splits(train_records, explicit_val, test_records)

    pool = explicit_train + unsplit
    train_records, val_records, test_records = split_three_way(pool, val_ratio, test_ratio, seed)
    return _validate_splits(train_records, val_records, test_records)


def split_three_way(
    records: list[SampleRecord],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord], list[SampleRecord]]:
    total = len(records)
    if total < 3:
        raise ValueError(
            f"Need at least 3 images to create train/val/test splits, received {total}."
        )

    desired_test_count = _desired_holdout_count(total, test_ratio, reserve=2)
    train_pool, test_records = _split_records(records, desired_test_count, seed)

    desired_val_total = _desired_holdout_count(total, val_ratio, reserve=1)
    desired_val_count = min(desired_val_total, max(0, len(train_pool) - 1))
    train_records, val_records = _split_records(train_pool, desired_val_count, seed + 1)

    if not val_records:
        raise ValueError("Validation split is empty. Increase dataset size or val_ratio.")
    if not test_records:
        raise ValueError("Test split is empty. Increase dataset size or test_ratio.")

    return train_records, val_records, test_records


def _validate_splits(
    train_records: list[SampleRecord],
    val_records: list[SampleRecord],
    test_records: list[SampleRecord],
) -> dict[str, list[SampleRecord]]:
    if not train_records:
        raise ValueError("Training split is empty after split generation.")
    if not val_records:
        raise ValueError("Validation split is empty after split generation.")
    if not test_records:
        raise ValueError("Test split is empty after split generation.")
    return {"train": train_records, "val": val_records, "test": test_records}


def _desired_holdout_count(total: int, ratio: float, reserve: int) -> int:
    if ratio <= 0 or total <= reserve:
        return 0
    count = max(1, int(round(total * ratio)))
    return min(count, total - reserve)


def _split_records(
    records: list[SampleRecord],
    holdout_count: int,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    if holdout_count <= 0:
        return list(records), []
    if holdout_count >= len(records):
        raise ValueError("Holdout count must be smaller than the number of records.")

    labels = [record.label_key for record in records]
    stratify = labels if _can_stratify(labels, holdout_count) else None
    try:
        train_records, holdout_records = train_test_split(
            records,
            test_size=holdout_count,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError:
        train_records, holdout_records = train_test_split(
            records,
            test_size=holdout_count,
            random_state=seed,
            shuffle=True,
            stratify=None,
        )

    return list(train_records), list(holdout_records)


def _can_stratify(labels: list[str], holdout_count: int) -> bool:
    if len(set(labels)) < 2:
        return False
    counts = Counter(labels)
    train_count = len(labels) - holdout_count
    return (
        min(counts.values()) >= 2
        and holdout_count >= len(counts)
        and train_count >= len(counts)
    )


def build_label_mapping(
    dataset_splits: dict[str, dict[str, list[SampleRecord]]]
) -> tuple[dict[str, int], list[str]]:
    label_catalog: dict[str, str] = {}
    for split_map in dataset_splits.values():
        for split_records in split_map.values():
            for record in split_records:
                label_catalog[record.label_key] = record.label_name

    ordered_items = sorted(label_catalog.items(), key=lambda item: item[1])
    label_to_index = {label_key: index for index, (label_key, _) in enumerate(ordered_items)}
    class_names = [label_name for _, label_name in ordered_items]
    return label_to_index, class_names


def build_label_mapping_from_records(
    dataset_records: dict[str, list[SampleRecord]]
) -> tuple[dict[str, int], list[str]]:
    label_catalog: dict[str, str] = {}
    for records in dataset_records.values():
        for record in records:
            label_catalog[record.label_key] = record.label_name

    ordered_items = sorted(label_catalog.items(), key=lambda item: item[1])
    label_to_index = {label_key: index for index, (label_key, _) in enumerate(ordered_items)}
    class_names = [label_name for _, label_name in ordered_items]
    return label_to_index, class_names


def get_split_counts(records: list[SampleRecord]) -> dict[str, int]:
    counts = Counter(record.label_name for record in records)
    return dict(sorted(counts.items()))


def combine_splits(
    dataset_splits: dict[str, dict[str, list[SampleRecord]]],
    split_name: str,
) -> list[SampleRecord]:
    combined: list[SampleRecord] = []
    for split_map in dataset_splits.values():
        combined.extend(split_map[split_name])
    return combined


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loader(
    records: list[SampleRecord],
    label_to_index: dict[str, int],
    domain_to_index: dict[str, int],
    transform: transforms.Compose,
    data_config: DataConfig,
    seed: int,
    training: bool,
    device_type: str,
) -> DataLoader:
    dataset = MultiDatasetImageFolder(
        records=records,
        label_to_index=label_to_index,
        domain_to_index=domain_to_index,
        transform=transform,
    )
    pin_memory = data_config.pin_memory and device_type == "cuda"

    sampler = None
    shuffle = training
    if training and data_config.weighted_sampling:
        sampler = build_weighted_sampler(records)
        shuffle = False

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=data_config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=data_config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=data_config.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def build_weighted_sampler(records: list[SampleRecord]) -> WeightedRandomSampler:
    label_counts = Counter(record.label_key for record in records)
    weights = [1.0 / label_counts[record.label_key] for record in records]
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )
