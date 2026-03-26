from __future__ import annotations

import csv
from dataclasses import asdict
from itertools import combinations, permutations
from pathlib import Path
import time
from typing import Any

import torch

from .baseline import run_random_forest_baseline
from .config import ExperimentConfig, clone_experiment_config
from .data import (
    build_label_mapping_from_records,
    build_transforms,
    combine_splits,
    create_data_loader,
    create_dataset_splits,
    get_split_counts,
    scan_dataset,
)
from .evaluation import (
    evaluate_model,
    plot_training_curves,
    save_evaluation_artifacts,
    save_training_curves_csv,
)
from .modeling import build_model
from .training import Trainer
from .utils import (
    aggregate_metric_rows,
    collect_environment_metadata,
    configure_logging,
    ensure_dir,
    format_results_table,
    paired_t_test,
    save_json,
    set_seed,
)


SUMMARY_METRICS = [
    "samples",
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
]
SIGNIFICANCE_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted",
]


def run_experiment(config: ExperimentConfig, device_override: str | None = None) -> list[dict[str, Any]]:
    experiment_start_time = time.perf_counter()
    output_root = ensure_dir(config.output_dir / config.experiment_name)
    logger = configure_logging(output_root / "experiment.log")
    device = resolve_device(device_override)
    logger.info("Using device: %s", device)
    logger.info("Experiment output directory: %s", output_root)
    save_json(output_root / "resolved_config.json", asdict(config))
    save_json(output_root / "environment.json", collect_environment_metadata(device=device, seeds=config.seeds))

    raw_dataset_records = {dataset.name: scan_dataset(dataset) for dataset in config.datasets}
    label_to_index, class_names = build_label_mapping_from_records(raw_dataset_records)
    domain_to_index = {dataset.name: index for index, dataset in enumerate(config.datasets)}
    logger.info("Unified label mapping: %s", {name: index for index, name in enumerate(class_names)})

    variants = _build_variants(config)
    all_rows: list[dict[str, Any]] = []

    for variant_name, variant_config in variants:
        logger.info("Running variant: %s", variant_name)
        variant_root = (
            output_root
            if len(variants) == 1 and len(variant_config.seeds) == 1
            else ensure_dir(output_root / variant_name)
        )
        if variant_root != output_root:
            save_json(variant_root / "variant_config.json", asdict(variant_config))

        for seed in variant_config.seeds:
            run_dir = (
                variant_root
                if len(variants) == 1 and len(variant_config.seeds) == 1
                else ensure_dir(variant_root / f"seed_{seed}")
            )
            rows = _run_single_seed(
                config=variant_config,
                seed=seed,
                raw_dataset_records=raw_dataset_records,
                label_to_index=label_to_index,
                class_names=class_names,
                domain_to_index=domain_to_index,
                output_dir=run_dir,
                device=device,
                logger=logger,
                variant_name=variant_name,
            )
            all_rows.extend(rows)

    _save_rows(output_root / "summary.json", output_root / "summary.csv", all_rows)
    mean_rows, std_rows = aggregate_metric_rows(
        rows=all_rows,
        group_keys=["variant", "scenario", "source", "target"],
        metric_keys=SUMMARY_METRICS,
    )
    _save_rows(output_root / "summary_mean.json", output_root / "summary_mean.csv", mean_rows)
    _save_rows(output_root / "summary_std.json", output_root / "summary_std.csv", std_rows)

    if len(variants) > 1:
        _save_rows(
            output_root / "ablation_comparison.json",
            output_root / "ablation_comparison.csv",
            mean_rows,
        )
        save_json(output_root / "statistical_tests.json", _run_statistical_tests(all_rows))

    total_experiment_time_seconds = time.perf_counter() - experiment_start_time
    environment_metadata = collect_environment_metadata(device=device, seeds=config.seeds)
    environment_metadata.update(
        {
            "experiment_name": config.experiment_name,
            "num_runs": config.num_runs,
            "variants": [variant_name for variant_name, _ in variants],
            "total_experiment_time_seconds": total_experiment_time_seconds,
        }
    )
    save_json(output_root / "environment.json", environment_metadata)

    logger.info("Run summary:\n%s", format_results_table(all_rows))
    logger.info("Mean summary:\n%s", format_results_table(mean_rows))
    logger.info("Total experiment time: %.2f seconds.", total_experiment_time_seconds)
    return all_rows


def _run_single_seed(
    config: ExperimentConfig,
    seed: int,
    raw_dataset_records: dict[str, list],
    label_to_index: dict[str, int],
    class_names: list[str],
    domain_to_index: dict[str, int],
    output_dir: Path,
    device: torch.device,
    logger,
    variant_name: str,
) -> list[dict[str, Any]]:
    run_start_time = time.perf_counter()
    output_dir = ensure_dir(output_dir)
    set_seed(seed)
    save_json(output_dir / "resolved_run_config.json", {**asdict(config), "active_seed": seed, "variant": variant_name})
    runtime_metadata = collect_environment_metadata(device=device)
    runtime_metadata.update({"seed": seed, "variant": variant_name})
    save_json(output_dir / "runtime_metadata.json", runtime_metadata)

    dataset_splits = {}
    for dataset in config.datasets:
        splits = create_dataset_splits(
            records=raw_dataset_records[dataset.name],
            val_ratio=dataset.val_ratio if dataset.val_ratio is not None else config.data.val_ratio,
            test_ratio=dataset.test_ratio if dataset.test_ratio is not None else config.data.test_ratio,
            seed=seed,
        )
        dataset_splits[dataset.name] = splits
        _log_dataset_summary(logger, dataset.name, splits, variant_name, seed)
        _save_dataset_summary(output_dir, dataset.name, splits, raw_dataset_records)

    integrity_report = _build_data_integrity_report(dataset_splits)
    save_json(output_dir / "data_integrity_check.json", integrity_report)
    if integrity_report["status"] != "passed":
        raise ValueError(
            "Data leakage detected across train/val/test partitions. "
            "See data_integrity_check.json for details."
        )

    train_transform, eval_transform = build_transforms(config.data)
    summary_rows: list[dict[str, Any]] = []
    total_training_time_seconds = 0.0

    if config.evaluation.run_pairwise_cross_dataset:
        pairwise_dir = ensure_dir(output_dir / "pairwise")
        for source_name, target_name in permutations(dataset_splits.keys(), 2):
            scenario_dir = ensure_dir(pairwise_dir / f"{source_name}_to_{target_name}")
            logger.info("[%s|seed=%d] Pairwise experiment: %s -> %s", variant_name, seed, source_name, target_name)
            source_splits = dataset_splits[source_name]
            target_splits = dataset_splits[target_name]

            evaluation_records = (
                target_splits["test"]
                if config.evaluation.use_target_test_split
                else target_splits["train"] + target_splits["val"] + target_splits["test"]
            )
            _warn_for_unseen_labels(
                logger=logger,
                source_records=source_splits["train"],
                target_records=evaluation_records,
                source_name=source_name,
                target_name=target_name,
            )

            train_loader = create_data_loader(
                records=source_splits["train"],
                label_to_index=label_to_index,
                domain_to_index=domain_to_index,
                transform=train_transform,
                data_config=config.data,
                seed=seed,
                training=True,
                device_type=device.type,
            )
            val_loader = create_data_loader(
                records=source_splits["val"],
                label_to_index=label_to_index,
                domain_to_index=domain_to_index,
                transform=eval_transform,
                data_config=config.data,
                seed=seed,
                training=False,
                device_type=device.type,
            )
            test_loader = create_data_loader(
                records=evaluation_records,
                label_to_index=label_to_index,
                domain_to_index=domain_to_index,
                transform=eval_transform,
                data_config=config.data,
                seed=seed,
                training=False,
                device_type=device.type,
            )

            model = build_model(config.model, num_classes=len(class_names), logger=logger)
            trainer = Trainer(
                model=model,
                class_names=class_names,
                training_config=config.training,
                device=device,
                output_dir=scenario_dir,
                logger=logger,
            )
            artifacts = trainer.fit(train_loader, val_loader)
            total_training_time_seconds += artifacts.training_time_seconds
            save_json(scenario_dir / "history.json", artifacts.history)
            plot_training_curves(artifacts.history, scenario_dir / "training_curves.png")
            save_training_curves_csv(artifacts.history, scenario_dir / "training_curves.csv")

            test_metrics = evaluate_model(
                model=trainer.model,
                data_loader=test_loader,
                device=device,
                class_names=class_names,
                use_tta=config.evaluation.use_tta,
                tta_steps=config.evaluation.tta_steps,
            )
            save_evaluation_artifacts(
                output_dir=scenario_dir,
                stem="test",
                metrics=test_metrics,
                class_names=class_names,
                save_confusion_matrix=config.evaluation.save_confusion_matrices,
            )
            summary_rows.append(
                _build_summary_row(
                    variant=variant_name,
                    seed=seed,
                    scenario="pairwise",
                    source=source_name,
                    target=target_name,
                    metrics=test_metrics,
                )
            )

    if config.evaluation.run_combined_experiment:
        combined_dir = ensure_dir(output_dir / "combined")
        logger.info("[%s|seed=%d] Combined multi-dataset experiment.", variant_name, seed)
        train_records = combine_splits(dataset_splits, "train")
        val_records = combine_splits(dataset_splits, "val")
        test_records = combine_splits(dataset_splits, "test")

        train_loader = create_data_loader(
            records=train_records,
            label_to_index=label_to_index,
            domain_to_index=domain_to_index,
            transform=train_transform,
            data_config=config.data,
            seed=seed,
            training=True,
            device_type=device.type,
        )
        val_loader = create_data_loader(
            records=val_records,
            label_to_index=label_to_index,
            domain_to_index=domain_to_index,
            transform=eval_transform,
            data_config=config.data,
            seed=seed,
            training=False,
            device_type=device.type,
        )
        test_loader = create_data_loader(
            records=test_records,
            label_to_index=label_to_index,
            domain_to_index=domain_to_index,
            transform=eval_transform,
            data_config=config.data,
            seed=seed,
            training=False,
            device_type=device.type,
        )

        model = build_model(config.model, num_classes=len(class_names), logger=logger)
        trainer = Trainer(
            model=model,
            class_names=class_names,
            training_config=config.training,
            device=device,
            output_dir=combined_dir,
            logger=logger,
        )
        artifacts = trainer.fit(train_loader, val_loader)
        total_training_time_seconds += artifacts.training_time_seconds
        save_json(combined_dir / "history.json", artifacts.history)
        plot_training_curves(artifacts.history, combined_dir / "training_curves.png")
        save_training_curves_csv(artifacts.history, combined_dir / "training_curves.csv")

        combined_metrics = evaluate_model(
            model=trainer.model,
            data_loader=test_loader,
            device=device,
            class_names=class_names,
            use_tta=config.evaluation.use_tta,
            tta_steps=config.evaluation.tta_steps,
        )
        save_evaluation_artifacts(
            output_dir=combined_dir,
            stem="combined_test",
            metrics=combined_metrics,
            class_names=class_names,
            save_confusion_matrix=config.evaluation.save_confusion_matrices,
        )
        summary_rows.append(
            _build_summary_row(
                variant=variant_name,
                seed=seed,
                scenario="combined",
                source="all",
                target="held_out_all",
                metrics=combined_metrics,
            )
        )

        for dataset_name, split_map in dataset_splits.items():
            logger.info("[%s|seed=%d] Evaluating combined model on dataset-specific test split: %s", variant_name, seed, dataset_name)
            dataset_test_loader = create_data_loader(
                records=split_map["test"],
                label_to_index=label_to_index,
                domain_to_index=domain_to_index,
                transform=eval_transform,
                data_config=config.data,
                seed=seed,
                training=False,
                device_type=device.type,
            )
            dataset_metrics = evaluate_model(
                model=trainer.model,
                data_loader=dataset_test_loader,
                device=device,
                class_names=class_names,
                use_tta=config.evaluation.use_tta,
                tta_steps=config.evaluation.tta_steps,
            )
            save_evaluation_artifacts(
                output_dir=combined_dir,
                stem=f"{dataset_name}_test",
                metrics=dataset_metrics,
                class_names=class_names,
                save_confusion_matrix=config.evaluation.save_confusion_matrices,
                metrics_filename=f"{dataset_name}_test.json",
            )
            logger.info(
                "[%s|seed=%d] Combined model on %s: acc=%.4f | balanced_acc=%.4f | f1_macro=%.4f",
                variant_name,
                seed,
                dataset_name,
                dataset_metrics["accuracy"],
                dataset_metrics["balanced_accuracy"],
                dataset_metrics["f1_macro"],
            )
            summary_rows.append(
                _build_summary_row(
                    variant=variant_name,
                    seed=seed,
                    scenario="combined_per_dataset",
                    source="all",
                    target=dataset_name,
                    metrics=dataset_metrics,
                )
            )

        if config.evaluation.run_feature_baseline:
            logger.info("[%s|seed=%d] RandomForest feature baseline.", variant_name, seed)
            baseline_train_loader = create_data_loader(
                records=train_records + val_records,
                label_to_index=label_to_index,
                domain_to_index=domain_to_index,
                transform=eval_transform,
                data_config=config.data,
                seed=seed,
                training=False,
                device_type=device.type,
            )
            baseline_metrics = run_random_forest_baseline(
                model=trainer.model,
                train_loader=baseline_train_loader,
                test_loader=test_loader,
                device=device,
                class_names=class_names,
                output_dir=combined_dir / "random_forest_baseline",
                random_state=seed,
            )
            summary_rows.append(
                _build_summary_row(
                    variant=variant_name,
                    seed=seed,
                    scenario="rf_baseline",
                    source="all_features",
                    target="held_out_all",
                    metrics=baseline_metrics,
                )
            )

    run_time_seconds = time.perf_counter() - run_start_time
    runtime_metadata.update(
        {
            "run_time_seconds": run_time_seconds,
            "total_training_time_seconds": total_training_time_seconds,
            "num_result_rows": len(summary_rows),
        }
    )
    save_json(output_dir / "runtime_metadata.json", runtime_metadata)
    logger.info(
        "[%s|seed=%d] Run finished in %.2f seconds (training %.2f seconds).",
        variant_name,
        seed,
        run_time_seconds,
        total_training_time_seconds,
    )
    return summary_rows


def resolve_device(device_override: str | None) -> torch.device:
    if device_override is not None:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_variants(config: ExperimentConfig) -> list[tuple[str, ExperimentConfig]]:
    variants = [("full", clone_experiment_config(config))]
    if not config.evaluation.run_ablations and not any(
        (
            config.evaluation.disable_mixstyle,
            config.evaluation.disable_mixup,
            config.evaluation.disable_weighted_sampling,
        )
    ):
        return variants

    if config.evaluation.disable_mixstyle:
        variant = clone_experiment_config(config)
        variant.model.use_mixstyle = False
        variants.append(("ablate_mixstyle", variant))

    if config.evaluation.disable_mixup:
        variant = clone_experiment_config(config)
        variant.training.use_mixup = False
        variant.training.use_cutmix = False
        variants.append(("ablate_mixup", variant))

    if config.evaluation.disable_weighted_sampling:
        variant = clone_experiment_config(config)
        variant.data.weighted_sampling = False
        variants.append(("ablate_weighted_sampling", variant))

    return variants


def _log_dataset_summary(
    logger,
    dataset_name: str,
    splits: dict[str, list],
    variant_name: str,
    seed: int,
) -> None:
    logger.info("[%s|seed=%d] Dataset '%s' summary:", variant_name, seed, dataset_name)
    for split_name in ("train", "val", "test"):
        counts = get_split_counts(splits[split_name])
        logger.info("  %s: %d samples | class distribution=%s", split_name, len(splits[split_name]), counts)


def _warn_for_unseen_labels(
    logger,
    source_records,
    target_records,
    source_name: str,
    target_name: str,
) -> None:
    source_labels = {record.label_name for record in source_records}
    target_labels = {record.label_name for record in target_records}
    unseen = sorted(target_labels - source_labels)
    if unseen:
        logger.warning(
            "Target dataset %s contains labels not present in source dataset %s training split: %s",
            target_name,
            source_name,
            unseen,
        )


def _build_summary_row(
    variant: str,
    seed: int,
    scenario: str,
    source: str,
    target: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "variant": variant,
        "seed": seed,
        "scenario": scenario,
        "source": source,
        "target": target,
        "samples": metrics["samples"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "f1_weighted": metrics["f1_weighted"],
    }


def _save_rows(json_path: Path, csv_path: Path, rows: list[dict[str, Any]]) -> None:
    save_json(json_path, rows)
    if not rows:
        return

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _save_dataset_summary(
    output_dir: Path,
    dataset_name: str,
    splits: dict[str, list],
    raw_dataset_records: dict[str, list],
) -> None:
    dataset_dir = ensure_dir(output_dir / "dataset_summaries" / dataset_name)
    split_sizes = {split_name: len(records) for split_name, records in splits.items()}
    class_distribution = {
        split_name: get_split_counts(records)
        for split_name, records in splits.items()
    }
    summary = {
        "dataset_name": dataset_name,
        "total_samples": sum(split_sizes.values()),
        "samples_per_split": split_sizes,
        "class_distribution": class_distribution,
        "domain_sizes": {
            name: len(records)
            for name, records in raw_dataset_records.items()
        },
    }
    save_json(dataset_dir / "dataset_summary.json", summary)


def _build_data_integrity_report(dataset_splits: dict[str, dict[str, list]]) -> dict[str, Any]:
    partition_sizes: dict[str, int] = {}
    path_sets: dict[str, set[str]] = {}
    duplicate_entries: list[dict[str, Any]] = []

    for dataset_name, split_map in dataset_splits.items():
        for split_name, records in split_map.items():
            partition_name = f"{dataset_name}:{split_name}"
            resolved_paths = [str(record.path.resolve()) for record in records]
            unique_paths = set(resolved_paths)
            partition_sizes[partition_name] = len(resolved_paths)
            path_sets[partition_name] = unique_paths
            if len(unique_paths) != len(resolved_paths):
                duplicate_entries.append(
                    {
                        "partition": partition_name,
                        "num_records": len(resolved_paths),
                        "num_unique_paths": len(unique_paths),
                    }
                )

    overlaps: list[dict[str, Any]] = []
    for left_name, right_name in combinations(sorted(path_sets), 2):
        shared_paths = path_sets[left_name].intersection(path_sets[right_name])
        if shared_paths:
            overlaps.append(
                {
                    "left_partition": left_name,
                    "right_partition": right_name,
                    "count": len(shared_paths),
                    "sample_paths": sorted(shared_paths)[:20],
                }
            )

    return {
        "status": "passed" if not duplicate_entries and not overlaps else "failed",
        "partition_sizes": partition_sizes,
        "duplicate_entries": duplicate_entries,
        "overlaps": overlaps,
        "total_overlap_pairs": len(overlaps),
    }


def _run_statistical_tests(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    full_rows = [row for row in rows if row.get("variant") == "full"]
    ablation_variants = sorted({row.get("variant") for row in rows if row.get("variant") not in {None, "full"}})
    if not full_rows or not ablation_variants:
        return []

    full_index = {
        (
            row["seed"],
            row["scenario"],
            row["source"],
            row["target"],
        ): row
        for row in full_rows
    }
    tests: list[dict[str, Any]] = []
    for variant_name in ablation_variants:
        variant_rows = [row for row in rows if row.get("variant") == variant_name]
        grouped_pairs: dict[tuple[str, str, str], list[tuple[dict[str, Any], dict[str, Any]]]] = {}
        for row in variant_rows:
            key = (row["scenario"], row["source"], row["target"])
            full_row = full_index.get((row["seed"], row["scenario"], row["source"], row["target"]))
            if full_row is None:
                continue
            grouped_pairs.setdefault(key, []).append((full_row, row))

        for (scenario, source, target), paired_rows in sorted(grouped_pairs.items()):
            for metric_name in SIGNIFICANCE_METRICS:
                full_values = [float(full_row[metric_name]) for full_row, _ in paired_rows]
                ablation_values = [float(ablation_row[metric_name]) for _, ablation_row in paired_rows]
                test_result = paired_t_test(full_values, ablation_values)
                tests.append(
                    {
                        "variant": variant_name,
                        "baseline_variant": "full",
                        "scenario": scenario,
                        "source": source,
                        "target": target,
                        "metric": metric_name,
                        **test_result,
                    }
                )
    return tests
