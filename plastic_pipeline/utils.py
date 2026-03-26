from __future__ import annotations

import json
import logging
import math
import platform
import random
import re
import sys
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


LOGGER_NAME = "plastic_pipeline"


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def configure_logging(log_path: str | Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    log_path = Path(log_path)
    ensure_dir(log_path.parent)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_label(label: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", label.strip().lower())
    return normalized.strip("_")


def prettify_label(label: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", label).strip()
    compact = cleaned.replace(" ", "")
    if compact.isalpha() and len(compact) <= 6:
        return cleaned.upper()
    return cleaned.title()


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except TypeError:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def save_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(payload), handle, indent=2, sort_keys=False)


def unwrap_model(model):
    return getattr(model, "_orig_mod", model)


def collect_environment_metadata(device: Any, seeds: list[int] | None = None) -> dict[str, Any]:
    metadata = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "os": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "device": str(device),
        "seeds": list(seeds) if seeds is not None else None,
    }
    try:
        import torch
    except ImportError:
        metadata.update(
            {
                "torch_version": None,
                "cuda_available": False,
                "mps_available": False,
                "gpu_name": None,
            }
        )
        return metadata

    cuda_available = torch.cuda.is_available()
    gpu_name = None
    if cuda_available:
        index = device.index if getattr(device, "type", None) == "cuda" else 0
        gpu_name = torch.cuda.get_device_name(index or 0)

    metadata.update(
        {
            "torch_version": torch.__version__,
            "cuda_available": cuda_available,
            "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
            "mps_available": hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
            "gpu_name": gpu_name,
        }
    )
    return metadata


def aggregate_metric_rows(
    rows: list[dict[str, Any]],
    group_keys: list[str],
    metric_keys: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in group_keys)].append(row)

    mean_rows: list[dict[str, Any]] = []
    std_rows: list[dict[str, Any]] = []
    for group_values, group_rows in sorted(grouped.items()):
        shared = {key: value for key, value in zip(group_keys, group_values)}
        mean_row = dict(shared)
        std_row = dict(shared)
        mean_row["num_runs"] = len(group_rows)
        std_row["num_runs"] = len(group_rows)
        for metric_key in metric_keys:
            values = np.asarray([float(row.get(metric_key, 0.0)) for row in group_rows], dtype=float)
            mean_row[metric_key] = float(values.mean())
            std_row[metric_key] = float(values.std(ddof=0))
        mean_rows.append(mean_row)
        std_rows.append(std_row)

    return mean_rows, std_rows


def paired_t_test(left_values: list[float], right_values: list[float]) -> dict[str, Any]:
    left = np.asarray(left_values, dtype=float)
    right = np.asarray(right_values, dtype=float)
    if left.shape != right.shape:
        raise ValueError("Paired samples must have matching shapes.")

    differences = left - right
    sample_count = int(differences.size)
    mean_difference = float(differences.mean()) if sample_count else 0.0
    if sample_count < 2:
        return {
            "num_pairs": sample_count,
            "mean_difference": mean_difference,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "insufficient_samples",
        }

    std_difference = float(differences.std(ddof=1))
    if std_difference <= 1e-12:
        return {
            "num_pairs": sample_count,
            "mean_difference": mean_difference,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "zero_variance",
        }

    statistic = mean_difference / (std_difference / math.sqrt(sample_count))
    try:
        from scipy import stats

        p_value = float(2.0 * stats.t.sf(abs(statistic), df=sample_count - 1))
        method = "scipy"
    except ImportError:
        p_value = float(math.erfc(abs(statistic) / math.sqrt(2.0)))
        method = "normal_approximation"

    return {
        "num_pairs": sample_count,
        "mean_difference": mean_difference,
        "statistic": float(statistic),
        "p_value": p_value,
        "method": method,
    }


def format_results_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No results available."

    columns = [
        column
        for column in (
            "variant",
            "seed",
            "scenario",
            "source",
            "target",
            "samples",
            "accuracy",
            "balanced_acc",
            "precision",
            "recall",
            "f1",
            "f1_weighted",
        )
        if any(column in row for row in rows) or column in {"scenario", "source", "target", "samples", "accuracy", "precision", "recall", "f1"}
    ]
    formatted_rows = []
    for row in rows:
        formatted_rows.append(
            {
                "variant": str(row.get("variant", "")),
                "seed": str(row.get("seed", "")),
                "scenario": str(row.get("scenario", "")),
                "source": str(row.get("source", "")),
                "target": str(row.get("target", "")),
                "samples": str(row.get("samples", "")),
                "accuracy": f"{row.get('accuracy', 0.0):.4f}",
                "balanced_acc": f"{row.get('balanced_accuracy', 0.0):.4f}",
                "precision": f"{row.get('precision_macro', 0.0):.4f}",
                "recall": f"{row.get('recall_macro', 0.0):.4f}",
                "f1": f"{row.get('f1_macro', 0.0):.4f}",
                "f1_weighted": f"{row.get('f1_weighted', 0.0):.4f}",
            }
        )

    widths = {
        column: max(len(column), *(len(item[column]) for item in formatted_rows))
        for column in columns
    }
    header = " | ".join(column.ljust(widths[column]) for column in columns)
    separator = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(item[column].ljust(widths[column]) for column in columns)
        for item in formatted_rows
    ]
    return "\n".join([header, separator, *body])
