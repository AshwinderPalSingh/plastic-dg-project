from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


class MixStyle(nn.Module):
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6) -> None:
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0 or self.alpha <= 0 or inputs.size(0) < 2:
            return inputs
        if random.random() > self.p:
            return inputs

        if inputs.dim() == 4:
            reduce_dims = (2, 3)
            view_shape = (inputs.size(0), 1, 1, 1)
        elif inputs.dim() == 3:
            reduce_dims = (1,)
            view_shape = (inputs.size(0), 1, 1)
        else:
            return inputs

        mean = inputs.mean(dim=reduce_dims, keepdim=True)
        variance = inputs.var(dim=reduce_dims, keepdim=True, unbiased=False)
        std = (variance + self.eps).sqrt()

        normalized = (inputs - mean) / std
        permutation = torch.randperm(inputs.size(0), device=inputs.device)
        mix_mean = mean[permutation]
        mix_std = std[permutation]

        concentration = torch.full((inputs.size(0),), self.alpha, device=inputs.device)
        lam = torch.distributions.Beta(concentration, concentration).sample().view(view_shape)

        mixed_mean = lam * mean + (1.0 - lam) * mix_mean
        mixed_std = lam * std + (1.0 - lam) * mix_std
        return normalized * mixed_std + mixed_mean


@dataclass(frozen=True)
class MixAugmentationOutput:
    inputs: torch.Tensor
    soft_targets: torch.Tensor
    hard_targets: torch.Tensor
    method: str


def make_soft_targets(
    labels: torch.Tensor,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    soft_targets = F.one_hot(labels, num_classes=num_classes).float()
    if label_smoothing <= 0:
        return soft_targets

    smooth_value = label_smoothing / num_classes
    return soft_targets.mul(1.0 - label_smoothing).add(smooth_value)


def soft_target_cross_entropy(logits: torch.Tensor, soft_targets: torch.Tensor) -> torch.Tensor:
    return torch.sum(-soft_targets * F.log_softmax(logits, dim=1), dim=1).mean()


def apply_mix_augmentation(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    use_mixup: bool,
    mixup_alpha: float,
    use_cutmix: bool,
    cutmix_alpha: float,
    cutmix_probability: float,
    label_smoothing: float,
) -> MixAugmentationOutput:
    soft_targets = make_soft_targets(labels, num_classes, label_smoothing=label_smoothing)
    batch_size = inputs.size(0)
    if batch_size < 2:
        return MixAugmentationOutput(inputs=inputs, soft_targets=soft_targets, hard_targets=labels, method="none")

    if use_cutmix and cutmix_alpha > 0 and (not use_mixup or random.random() < cutmix_probability):
        return _apply_cutmix(
            inputs=inputs,
            soft_targets=soft_targets,
            hard_targets=labels,
            alpha=cutmix_alpha,
        )

    if use_mixup and mixup_alpha > 0:
        return _apply_mixup(
            inputs=inputs,
            soft_targets=soft_targets,
            hard_targets=labels,
            alpha=mixup_alpha,
        )

    return MixAugmentationOutput(inputs=inputs, soft_targets=soft_targets, hard_targets=labels, method="none")


def _apply_mixup(
    inputs: torch.Tensor,
    soft_targets: torch.Tensor,
    hard_targets: torch.Tensor,
    alpha: float,
) -> MixAugmentationOutput:
    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    lam = torch.distributions.Beta(alpha, alpha).sample().to(device=inputs.device, dtype=inputs.dtype)
    mixed_inputs = lam * inputs + (1.0 - lam) * inputs[permutation]
    mixed_targets = lam * soft_targets + (1.0 - lam) * soft_targets[permutation]
    return MixAugmentationOutput(
        inputs=mixed_inputs,
        soft_targets=mixed_targets,
        hard_targets=hard_targets,
        method="mixup",
    )


def _apply_cutmix(
    inputs: torch.Tensor,
    soft_targets: torch.Tensor,
    hard_targets: torch.Tensor,
    alpha: float,
) -> MixAugmentationOutput:
    if inputs.dim() != 4:
        return _apply_mixup(inputs, soft_targets, hard_targets, alpha)

    permutation = torch.randperm(inputs.size(0), device=inputs.device)
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    height, width = inputs.size(2), inputs.size(3)
    x1, y1, x2, y2 = _rand_bbox(height=height, width=width, lam=lam, device=inputs.device)

    mixed_inputs = inputs.clone()
    mixed_inputs[:, :, y1:y2, x1:x2] = inputs[permutation, :, y1:y2, x1:x2]

    patch_area = max(0, x2 - x1) * max(0, y2 - y1)
    adjusted_lam = 1.0 - patch_area / float(height * width)
    mixed_targets = adjusted_lam * soft_targets + (1.0 - adjusted_lam) * soft_targets[permutation]
    return MixAugmentationOutput(
        inputs=mixed_inputs,
        soft_targets=mixed_targets,
        hard_targets=hard_targets,
        method="cutmix",
    )


def _rand_bbox(height: int, width: int, lam: float, device: torch.device) -> tuple[int, int, int, int]:
    cut_ratio = (1.0 - lam) ** 0.5
    cut_width = int(width * cut_ratio)
    cut_height = int(height * cut_ratio)

    center_x = torch.randint(0, width, (1,), device=device).item()
    center_y = torch.randint(0, height, (1,), device=device).item()

    x1 = max(center_x - cut_width // 2, 0)
    y1 = max(center_y - cut_height // 2, 0)
    x2 = min(center_x + cut_width // 2, width)
    y2 = min(center_y + cut_height // 2, height)
    return x1, y1, x2, y2


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    if source_features.size(0) < 2 or target_features.size(0) < 2:
        return source_features.new_zeros(())

    eps = 1e-5
    max_loss = 10.0
    source_cov = _covariance(source_features, eps=eps)
    target_cov = _covariance(target_features, eps=eps)
    feature_dim = source_cov.size(0)
    loss = torch.sum((source_cov - target_cov) ** 2) / (4.0 * feature_dim * feature_dim)
    return torch.clamp(loss, min=0.0, max=max_loss)


def multi_domain_coral_loss(features: torch.Tensor, domain_labels: torch.Tensor) -> torch.Tensor:
    unique_domains = torch.unique(domain_labels)
    if unique_domains.numel() < 2:
        return features.new_zeros(())

    losses: list[torch.Tensor] = []
    for left_index in range(unique_domains.numel()):
        for right_index in range(left_index + 1, unique_domains.numel()):
            source_features = features[domain_labels == unique_domains[left_index]]
            target_features = features[domain_labels == unique_domains[right_index]]
            loss = coral_loss(source_features, target_features)
            if loss.numel() == 1:
                losses.append(loss)

    if not losses:
        return features.new_zeros(())
    return torch.stack(losses).mean()


def _covariance(features: torch.Tensor, eps: float) -> torch.Tensor:
    features = F.normalize(features.float(), p=2, dim=1, eps=eps)
    centered = features - features.mean(dim=0, keepdim=True)
    denominator = max(features.size(0) - 1, 1)
    covariance = centered.T @ centered / denominator
    diagonal = torch.eye(covariance.size(0), device=covariance.device, dtype=covariance.dtype)
    return covariance + diagonal * eps


def unpack_batch(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, dict):
        return batch["inputs"], batch["labels"], batch.get("domains")
    if isinstance(batch, (list, tuple)):
        if len(batch) == 3:
            inputs, labels, domains = batch
            return inputs, labels, domains
        if len(batch) == 2:
            inputs, labels = batch
            return inputs, labels, None
    raise TypeError(f"Unsupported batch format: {type(batch)!r}")
