# Robust Plastic Classification Using Multi-Dataset Learning

This repository contains a publication-grade PyTorch pipeline for plastic type classification under domain shift. It supports multi-dataset learning, MixStyle, MixUp/CutMix, CORAL alignment, ConvNeXt/ViT backbones, multi-seed statistical evaluation, ablation studies, TensorBoard logging, metrics export, checkpointing, training-curve plots, and an optional RandomForest feature baseline.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Expected Dataset Layout

The scanner walks each dataset root recursively and supports both:

- `dataset/class_name/image.jpg`
- `dataset/train/class_name/image.jpg`, `dataset/val/class_name/image.jpg`, `dataset/test/class_name/image.jpg`

For more complex directory layouts, set `class_index` in the dataset config so the label is extracted from a different directory level.

## Configuration

Edit [configs/example_experiment.json](/Users/ashwinder./Desktop/mlproj/configs/example_experiment.json) and update the dataset roots. Paths in the config are resolved relative to the config file, so the example uses `../data/...` and `../outputs/...`. Per-dataset `class_aliases` let you map source-specific folder names to a shared taxonomy such as `PET`, `HDPE`, `PVC`, `LDPE`, `PP`, and `PS`.

## Run

```bash
python3 run_experiment.py --config configs/example_experiment.json
```

Optional device override:

```bash
python3 run_experiment.py --config configs/example_experiment.json --device cuda:0
```

## Outputs

The experiment writes results under `outputs/<experiment_name>/`:

- best checkpoints for each scenario
- raw, mean, and std JSON/CSV summaries across seeds
- ablation comparison tables when enabled
- classification reports
- confusion matrix CSV/PNG files
- training-curve plots
- TensorBoard event files
- a RandomForest baseline report for the combined experiment

## Domain Generalization Features

- MixStyle inserted into early backbone stages when `model.use_mixstyle=true`
- MixUp / CutMix applied in the training loop with soft-label cross-entropy
- CORAL covariance alignment for batches containing multiple source domains
- RandAugment + stronger color jitter + Gaussian noise + random erasing
- multi-seed evaluation via `num_runs` or explicit `seeds`
- config-driven ablations for MixStyle, MixUp, and weighted sampling
- backbone options: `efficientnet_b0`, `mobilenet_v2`, `convnext_tiny`, `vit_b_16`

## Research Protocol Implemented

The pipeline runs:

1. `Dataset A -> Dataset B`
2. `Dataset B -> Dataset A`
3. Combined training on all datasets -> combined held-out test
4. Combined model evaluation on each dataset-specific held-out set
