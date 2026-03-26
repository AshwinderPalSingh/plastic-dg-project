from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .dg import unpack_batch
from .utils import ensure_dir, save_json


def evaluate_model(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    class_names: list[str],
    criterion: torch.nn.Module | None = None,
    use_tta: bool = False,
    tta_steps: int = 1,
) -> dict[str, Any]:
    model.eval()
    predictions: list[int] = []
    targets: list[int] = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in data_loader:
            inputs, labels, _ = unpack_batch(batch)
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = _predict_with_tta(
                model=model,
                inputs=inputs,
                use_tta=use_tta,
                tta_steps=tta_steps,
            )
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            predictions.extend(preds.cpu().tolist())
            targets.extend(labels.cpu().tolist())
            total_samples += labels.size(0)

    if not targets:
        raise ValueError("Evaluation loader is empty.")

    metrics = compute_metrics(targets, predictions, class_names)
    metrics["loss"] = total_loss / total_samples if criterion is not None else None
    metrics["samples"] = total_samples
    return metrics


def compute_metrics(
    targets: list[int],
    predictions: list[int],
    class_names: list[str],
) -> dict[str, Any]:
    label_indices = list(range(len(class_names)))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=label_indices,
        average="macro",
        zero_division=0,
    )
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=label_indices,
        average="weighted",
        zero_division=0,
    )
    report_text = classification_report(
        targets,
        predictions,
        labels=label_indices,
        target_names=class_names,
        zero_division=0,
    )
    report_dict = classification_report(
        targets,
        predictions,
        labels=label_indices,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(targets, predictions, labels=label_indices)
    per_class_accuracy = {}
    per_class_recall = {}
    row_sums = matrix.sum(axis=1)
    for index, class_name in enumerate(class_names):
        recall = float(matrix[index, index] / row_sums[index]) if row_sums[index] > 0 else 0.0
        per_class_accuracy[class_name] = recall
        per_class_recall[class_name] = recall
    balanced_accuracy = float(np.mean(list(per_class_recall.values()))) if per_class_recall else 0.0
    return {
        "accuracy": accuracy_score(targets, predictions),
        "balanced_accuracy": balanced_accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "per_class_accuracy": per_class_accuracy,
        "per_class_recall": per_class_recall,
        "classification_report": report_text,
        "classification_report_dict": report_dict,
        "confusion_matrix": matrix,
    }


def save_evaluation_artifacts(
    output_dir: str | Path,
    stem: str,
    metrics: dict[str, Any],
    class_names: list[str],
    save_confusion_matrix: bool = True,
    metrics_filename: str | None = None,
) -> None:
    output_dir = ensure_dir(output_dir)
    save_json(output_dir / (metrics_filename or f"{stem}_metrics.json"), metrics)
    report_path = output_dir / f"{stem}_classification_report.txt"
    report_path.write_text(metrics["classification_report"], encoding="utf-8")

    scalar_metrics = {
        "samples": metrics.get("samples"),
        "loss": metrics.get("loss"),
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "precision_macro": metrics.get("precision_macro"),
        "recall_macro": metrics.get("recall_macro"),
        "f1_macro": metrics.get("f1_macro"),
        "f1_weighted": metrics.get("f1_weighted"),
    }
    save_metrics_csv(output_dir / f"{stem}_metrics.csv", scalar_metrics)
    save_per_class_metric_csv(
        output_dir / f"{stem}_per_class_recall.csv",
        metrics.get("per_class_recall", {}),
        metric_name="recall",
    )

    matrix = np.asarray(metrics["confusion_matrix"])
    np.savetxt(output_dir / f"{stem}_confusion_matrix.csv", matrix, fmt="%d", delimiter=",")
    if save_confusion_matrix:
        save_confusion_matrix_plot(
            matrix=matrix,
            class_names=class_names,
            output_path=output_dir / f"{stem}_confusion_matrix.png",
            title=stem.replace("_", " ").title(),
        )


def save_metrics_csv(output_path: str | Path, metrics: dict[str, Any]) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    headers = list(metrics.keys())
    values = [metrics[key] for key in headers]
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(headers) + "\n")
        handle.write(",".join("" if value is None else str(value) for value in values) + "\n")


def save_per_class_metric_csv(
    output_path: str | Path,
    values: dict[str, float],
    metric_name: str,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"class,{metric_name}\n")
        for class_name, value in values.items():
            handle.write(f"{class_name},{value}\n")


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    axis.figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                format(matrix[row_index, column_index], "d"),
                ha="center",
                va="center",
                color="white" if matrix[row_index, column_index] > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _predict_with_tta(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    use_tta: bool,
    tta_steps: int,
) -> torch.Tensor:
    if not use_tta or tta_steps <= 1:
        return model(inputs)

    logits_sum = model(inputs)
    total_views = 1
    for variant_index in range(1, tta_steps):
        logits_sum = logits_sum + model(_build_tta_view(inputs, variant_index))
        total_views += 1
    return logits_sum / total_views


def _build_tta_view(inputs: torch.Tensor, variant_index: int) -> torch.Tensor:
    if variant_index == 1:
        return torch.flip(inputs, dims=(-1,))

    pad = max(1, min(inputs.size(-1), inputs.size(-2)) // 20)
    padded = F.pad(inputs, (pad, pad, pad, pad), mode="replicate")
    height, width = inputs.size(-2), inputs.size(-1)
    offsets = [
        (0, 0),
        (0, 2 * pad),
        (2 * pad, 0),
        (2 * pad, 2 * pad),
        (pad, pad),
    ]
    offset_y, offset_x = offsets[(variant_index - 2) % len(offsets)]
    return padded[:, :, offset_y : offset_y + height, offset_x : offset_x + width]


def plot_training_curves(history: dict[str, list[float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    epochs = list(range(1, len(history["train_loss"]) + 1))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss")
    if "train_ce_loss" in history:
        axes[0].plot(epochs, history["train_ce_loss"], label="train_ce_loss")
    if "train_coral_loss" in history:
        axes[0].plot(epochs, history["train_coral_loss"], label="train_coral_loss")
    axes[0].plot(epochs, history["val_loss"], label="val_loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_accuracy"], label="train_accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="val_accuracy")
    axes[1].plot(epochs, history["val_f1_macro"], label="val_f1_macro")
    axes[1].set_title("Accuracy / F1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    axes[2].plot(epochs, history["learning_rate"], label="learning_rate")
    axes[2].set_title("Learning Rate")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_training_curves_csv(history: dict[str, list[float]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    epochs = len(history.get("train_loss", []))
    column_map = [
        ("epoch", None),
        ("train_loss", "train_loss"),
        ("val_loss", "val_loss"),
        ("train_acc", "train_accuracy"),
        ("val_acc", "val_accuracy"),
        ("val_f1", "val_f1_macro"),
        ("train_ce_loss", "train_ce_loss"),
        ("train_coral_loss", "train_coral_loss"),
        ("coral_weight", "coral_weight"),
        ("coral_ce_ratio", "coral_ce_ratio"),
        ("learning_rate", "learning_rate"),
        ("skipped_batches", "skipped_batches"),
    ]
    fieldnames = [column_name for column_name, history_key in column_map if history_key is None or history_key in history]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for epoch_index in range(epochs):
            row = {"epoch": epoch_index + 1}
            for column_name, history_key in column_map:
                if history_key is None or history_key not in history:
                    continue
                row[column_name] = history[history_key][epoch_index]
            writer.writerow(row)
