from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from .dg import unpack_batch
from .evaluation import compute_metrics, save_evaluation_artifacts
from .utils import ensure_dir, unwrap_model


def extract_features(model: torch.nn.Module, data_loader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    feature_model = unwrap_model(model)
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels, _ = unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            features = feature_model.forward_features(inputs)
            feature_batches.append(features.cpu().numpy())
            label_batches.append(labels.cpu().numpy())

    return np.concatenate(feature_batches, axis=0), np.concatenate(label_batches, axis=0)


def run_random_forest_baseline(
    model: torch.nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
    class_names: list[str],
    output_dir: str | Path,
    random_state: int,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    train_features, train_labels = extract_features(model, train_loader, device)
    test_features, test_labels = extract_features(model, test_loader, device)

    classifier = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    metrics = compute_metrics(test_labels.tolist(), predictions.tolist(), class_names)
    metrics["samples"] = int(test_labels.shape[0])
    save_evaluation_artifacts(
        output_dir=output_dir,
        stem="random_forest_baseline",
        metrics=metrics,
        class_names=class_names,
        save_confusion_matrix=True,
    )
    return metrics
