from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ViT_B_16_Weights,
    convnext_tiny,
    efficientnet_b0,
    mobilenet_v2,
    vit_b_16,
)

from .config import ModelConfig
from .dg import MixStyle


class PlasticClassifier(nn.Module):
    def __init__(self, model_config: ModelConfig, num_classes: int) -> None:
        super().__init__()
        self.backbone_name = model_config.name

        if model_config.name == "efficientnet_b0":
            model = efficientnet_b0(
                weights=EfficientNet_B0_Weights.DEFAULT if model_config.pretrained else None
            )
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Identity()
            self.backbone = model
            self._attach_mixstyle(model.features, model_config)
        elif model_config.name == "mobilenet_v2":
            model = mobilenet_v2(
                weights=MobileNet_V2_Weights.DEFAULT if model_config.pretrained else None
            )
            in_features = model.classifier[-1].in_features
            model.classifier = nn.Identity()
            self.backbone = model
            self._attach_mixstyle(model.features, model_config)
        elif model_config.name == "convnext_tiny":
            model = convnext_tiny(
                weights=ConvNeXt_Tiny_Weights.DEFAULT if model_config.pretrained else None
            )
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Identity()
            self.backbone = model
            self._attach_mixstyle(model.features, model_config)
        elif model_config.name == "vit_b_16":
            model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT if model_config.pretrained else None)
            in_features = _get_vit_head(model).in_features
            _replace_vit_head_with_identity(model)
            self.backbone = model
            self._attach_mixstyle(model.encoder.layers, model_config)
        else:
            raise ValueError(
                "Unsupported model "
                f"'{model_config.name}'. Use 'efficientnet_b0', 'mobilenet_v2', 'convnext_tiny', or 'vit_b_16'."
            )

        self.classifier = nn.Sequential(
            nn.Dropout(p=model_config.dropout),
            nn.Linear(in_features, num_classes),
        )
        nn.init.normal_(self.classifier[-1].weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier[-1].bias)

    def _attach_mixstyle(self, stages: nn.Module, model_config: ModelConfig) -> None:
        if not model_config.use_mixstyle or model_config.mixstyle_layers <= 0:
            return
        if not isinstance(stages, nn.Sequential):
            return

        for index in range(min(model_config.mixstyle_layers, len(stages))):
            stages[index] = nn.Sequential(
                stages[index],
                MixStyle(
                    p=model_config.mixstyle_probability,
                    alpha=model_config.mixstyle_alpha,
                ),
            )

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        if features.dim() > 2:
            return torch.flatten(features, 1)
        return features

    def classify_features(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)

    def forward(self, inputs: torch.Tensor, return_features: bool = False):
        features = self.forward_features(inputs)
        logits = self.classify_features(features)
        if return_features:
            return logits, features
        return logits

    def freeze_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False


def build_model(
    model_config: ModelConfig,
    num_classes: int,
    logger: logging.Logger | None = None,
) -> PlasticClassifier:
    try:
        model = PlasticClassifier(
            model_config=model_config,
            num_classes=num_classes,
        )
    except Exception as exc:
        if model_config.pretrained:
            if logger is not None:
                logger.warning(
                    "Failed to load pretrained weights for %s. Falling back to random initialization. %s",
                    model_config.name,
                    exc,
                )
            fallback_config = ModelConfig(**{**model_config.__dict__, "pretrained": False})
            model = PlasticClassifier(
                model_config=fallback_config,
                num_classes=num_classes,
            )
        else:
            raise

    if model_config.freeze_backbone:
        model.freeze_backbone()

    return model


def _get_vit_head(model) -> nn.Module:
    if hasattr(model.heads, "head"):
        return model.heads.head
    if isinstance(model.heads, nn.Sequential):
        return model.heads[-1]
    raise ValueError("Unsupported ViT head structure.")


def _replace_vit_head_with_identity(model) -> None:
    if hasattr(model.heads, "head"):
        model.heads.head = nn.Identity()
        return
    if isinstance(model.heads, nn.Sequential):
        model.heads[-1] = nn.Identity()
        return
    raise ValueError("Unsupported ViT head structure.")
