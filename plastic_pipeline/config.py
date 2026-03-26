from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
    name: str
    root: Path
    class_aliases: dict[str, str] = field(default_factory=dict)
    class_index: int | None = None
    val_ratio: float | None = None
    test_ratio: float | None = None


@dataclass
class DataConfig:
    image_size: int = 224
    val_ratio: float = 0.1
    test_ratio: float = 0.2
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    weighted_sampling: bool = True
    use_randaugment: bool = True
    randaugment_num_ops: int = 2
    randaugment_magnitude: int = 9
    horizontal_flip_probability: float = 0.5
    train_crop_scale_min: float = 0.7
    train_crop_scale_max: float = 1.0
    color_jitter_brightness: float = 0.4
    color_jitter_contrast: float = 0.4
    color_jitter_saturation: float = 0.2
    color_jitter_hue: float = 0.05
    gaussian_noise_std: float = 0.03
    gaussian_noise_probability: float = 0.5
    random_erasing_probability: float = 0.25
    eval_resize_scale: float = 1.14


@dataclass
class ModelConfig:
    name: str = "efficientnet_b0"
    pretrained: bool = True
    dropout: float = 0.3
    freeze_backbone: bool = False
    use_mixstyle: bool = False
    mixstyle_probability: float = 0.5
    mixstyle_alpha: float = 0.1
    mixstyle_layers: int = 2


@dataclass
class TrainingConfig:
    epochs: int = 20
    optimizer: str = "adamw"
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    scheduler: str = "cosine"
    min_learning_rate: float = 1e-6
    early_stopping_patience: int = 5
    label_smoothing: float = 0.0
    amp: bool = True
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    cutmix_probability: float = 0.5
    use_coral: bool = False
    coral_weight: float = 0.1
    enable_tensorboard: bool = True
    use_torch_compile: bool = True
    gradient_clip_norm: float | None = 1.0
    gradient_accumulation_steps: int = 1
    track_loss_ratio: bool = False


@dataclass
class EvaluationConfig:
    run_pairwise_cross_dataset: bool = True
    run_combined_experiment: bool = True
    use_target_test_split: bool = True
    run_feature_baseline: bool = True
    save_confusion_matrices: bool = True
    run_ablations: bool = False
    disable_mixstyle: bool = False
    disable_mixup: bool = False
    disable_weighted_sampling: bool = False
    use_tta: bool = False
    tta_steps: int = 3


@dataclass
class ExperimentConfig:
    experiment_name: str
    output_dir: Path
    seed: int
    datasets: list[DatasetConfig]
    num_runs: int = 1
    seeds: list[int] = field(default_factory=list)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def load_experiment_config(config_path: str | Path) -> ExperimentConfig:
    config_path = Path(config_path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    base_dir = config_path.parent
    datasets = [
        DatasetConfig(
            name=item["name"],
            root=_resolve_path(item["root"], base_dir),
            class_aliases=item.get("class_aliases", {}),
            class_index=item.get("class_index"),
            val_ratio=item.get("val_ratio"),
            test_ratio=item.get("test_ratio"),
        )
        for item in payload["datasets"]
    ]

    if len(datasets) < 2:
        raise ValueError("At least two datasets are required for cross-dataset evaluation.")

    data_cfg = DataConfig(**payload.get("data", {}))
    model_cfg = ModelConfig(**payload.get("model", {}))
    training_cfg = TrainingConfig(**payload.get("training", {}))
    eval_cfg = EvaluationConfig(**payload.get("evaluation", {}))
    num_runs = int(payload.get("num_runs", 1))
    explicit_seeds = [int(seed) for seed in payload.get("seeds", [])]
    base_seed = int(payload.get("seed", 42))
    resolved_seeds = explicit_seeds or [base_seed + index for index in range(num_runs)]

    if not 0 < data_cfg.val_ratio < 1:
        raise ValueError("data.val_ratio must be between 0 and 1.")
    if not 0 < data_cfg.test_ratio < 1:
        raise ValueError("data.test_ratio must be between 0 and 1.")
    if data_cfg.val_ratio + data_cfg.test_ratio >= 1:
        raise ValueError("data.val_ratio + data.test_ratio must be less than 1.")

    for dataset in datasets:
        dataset_val_ratio = dataset.val_ratio if dataset.val_ratio is not None else data_cfg.val_ratio
        dataset_test_ratio = dataset.test_ratio if dataset.test_ratio is not None else data_cfg.test_ratio
        if not 0 < dataset_val_ratio < 1:
            raise ValueError(f"{dataset.name}.val_ratio must be between 0 and 1.")
        if not 0 < dataset_test_ratio < 1:
            raise ValueError(f"{dataset.name}.test_ratio must be between 0 and 1.")
        if dataset_val_ratio + dataset_test_ratio >= 1:
            raise ValueError(
                f"{dataset.name}.val_ratio + {dataset.name}.test_ratio must be less than 1."
            )
    if num_runs < 1:
        raise ValueError("num_runs must be at least 1.")
    if not resolved_seeds:
        raise ValueError("At least one seed must be provided.")
    if data_cfg.train_crop_scale_min <= 0 or data_cfg.train_crop_scale_max > 1:
        raise ValueError("Train crop scale bounds must satisfy 0 < min <= max <= 1.")
    if data_cfg.train_crop_scale_min > data_cfg.train_crop_scale_max:
        raise ValueError("train_crop_scale_min must be <= train_crop_scale_max.")
    if training_cfg.mixup_alpha < 0 or training_cfg.cutmix_alpha < 0:
        raise ValueError("MixUp and CutMix alpha values must be non-negative.")
    if training_cfg.cutmix_probability < 0 or training_cfg.cutmix_probability > 1:
        raise ValueError("cutmix_probability must be between 0 and 1.")
    if training_cfg.coral_weight < 0:
        raise ValueError("coral_weight must be non-negative.")
    if training_cfg.gradient_clip_norm is not None and training_cfg.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when provided.")
    if training_cfg.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")
    if model_cfg.mixstyle_layers < 0:
        raise ValueError("mixstyle_layers must be non-negative.")
    if model_cfg.mixstyle_probability < 0 or model_cfg.mixstyle_probability > 1:
        raise ValueError("mixstyle_probability must be between 0 and 1.")
    if data_cfg.horizontal_flip_probability < 0 or data_cfg.horizontal_flip_probability > 1:
        raise ValueError("horizontal_flip_probability must be between 0 and 1.")
    if data_cfg.gaussian_noise_probability < 0 or data_cfg.gaussian_noise_probability > 1:
        raise ValueError("gaussian_noise_probability must be between 0 and 1.")
    if data_cfg.random_erasing_probability < 0 or data_cfg.random_erasing_probability > 1:
        raise ValueError("random_erasing_probability must be between 0 and 1.")
    if eval_cfg.tta_steps < 1:
        raise ValueError("tta_steps must be at least 1.")

    return ExperimentConfig(
        experiment_name=payload.get("experiment_name", "plastic_domain_generalization"),
        output_dir=_resolve_path(payload.get("output_dir", "outputs"), base_dir),
        seed=base_seed,
        datasets=datasets,
        num_runs=num_runs,
        seeds=resolved_seeds,
        data=data_cfg,
        model=model_cfg,
        training=training_cfg,
        evaluation=eval_cfg,
    )


def clone_experiment_config(config: ExperimentConfig) -> ExperimentConfig:
    return copy.deepcopy(config)
