from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from plastic_pipeline.config import load_experiment_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Robust plastic classification with cross-dataset domain generalization."
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the experiment JSON config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override, for example 'cpu' or 'cuda:0'.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    dependency_map = {
        "torch": "torch",
        "torchvision": "torchvision",
        "scikit-learn": "sklearn",
        "matplotlib": "matplotlib",
        "Pillow": "PIL",
    }
    if config.training.enable_tensorboard:
        dependency_map["tensorboard"] = "tensorboard"
    missing = [package for package, module_name in dependency_map.items() if importlib.util.find_spec(module_name) is None]
    if missing:
        packages = ", ".join(missing)
        raise SystemExit(
            f"Missing dependencies: {packages}. Install them with `python3 -m pip install -r requirements.txt`."
        )

    from plastic_pipeline.experiment import run_experiment

    run_experiment(config=config, device_override=args.device)


if __name__ == "__main__":
    main()
