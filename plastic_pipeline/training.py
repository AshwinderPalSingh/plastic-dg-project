from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .config import TrainingConfig
from .dg import (
    apply_mix_augmentation,
    multi_domain_coral_loss,
    soft_target_cross_entropy,
    unpack_batch,
)
from .evaluation import evaluate_model
from .utils import ensure_dir, unwrap_model


@dataclass
class TrainingArtifacts:
    checkpoint_path: Path
    history: dict[str, list[float]]
    best_epoch: int
    best_score: float
    training_time_seconds: float


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        class_names: list[str],
        training_config: TrainingConfig,
        device: torch.device,
        output_dir: str | Path,
        logger: logging.Logger,
    ) -> None:
        self.model = model.to(device)
        self.class_names = class_names
        self.training_config = training_config
        self.device = device
        self.output_dir = ensure_dir(output_dir)
        self.logger = logger
        self.criterion = nn.CrossEntropyLoss(label_smoothing=training_config.label_smoothing)
        self.training_model = self._maybe_compile_model(self.model)
        self.trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.writer = self._build_tensorboard_writer()

    def fit(self, train_loader, val_loader) -> TrainingArtifacts:
        amp_enabled = self.training_config.amp and self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        checkpoint_path = self.output_dir / "best_model.pt"

        history = {
            "train_loss": [],
            "train_ce_loss": [],
            "train_coral_loss": [],
            "train_accuracy": [],
            "skipped_batches": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_f1_macro": [],
            "learning_rate": [],
            "coral_weight": [],
        }
        if self.training_config.track_loss_ratio:
            history["coral_ce_ratio"] = []

        best_score = float("-inf")
        best_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        training_start_time = time.perf_counter()

        for epoch in range(1, self.training_config.epochs + 1):
            train_stats = self._train_one_epoch(train_loader, scaler, amp_enabled)
            val_stats = evaluate_model(
                model=self.model,
                data_loader=val_loader,
                device=self.device,
                class_names=self.class_names,
                criterion=self.criterion,
            )

            current_lr = self.optimizer.param_groups[0]["lr"]
            history["train_loss"].append(train_stats["loss"])
            history["train_ce_loss"].append(train_stats["classification_loss"])
            history["train_coral_loss"].append(train_stats["coral_loss"])
            history["train_accuracy"].append(train_stats["accuracy"])
            history["skipped_batches"].append(train_stats["skipped_batches"])
            history["val_loss"].append(val_stats["loss"])
            history["val_accuracy"].append(val_stats["accuracy"])
            history["val_f1_macro"].append(val_stats["f1_macro"])
            history["learning_rate"].append(current_lr)
            history["coral_weight"].append(self.training_config.coral_weight)
            if self.training_config.track_loss_ratio:
                history["coral_ce_ratio"].append(train_stats["coral_ce_ratio"])
            self._log_tensorboard(epoch, train_stats, val_stats, current_lr)

            improved = (
                val_stats["f1_macro"] > best_score
                or (
                    abs(val_stats["f1_macro"] - best_score) < 1e-8
                    and val_stats["loss"] is not None
                    and val_stats["loss"] < best_loss
                )
            )
            if improved:
                best_score = val_stats["f1_macro"]
                best_loss = val_stats["loss"] if val_stats["loss"] is not None else best_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint(checkpoint_path, epoch, best_score)
            else:
                epochs_without_improvement += 1

            self._step_scheduler(val_stats["f1_macro"])
            log_message = (
                "Epoch %d/%d | train_loss=%.4f | ce=%.4f | coral=%.4f | coral_w=%.4f | "
                "train_acc=%.4f | skipped=%d | val_loss=%.4f | val_acc=%.4f | val_f1=%.4f | lr=%.6f"
            )
            log_args: list[float | int] = [
                epoch,
                self.training_config.epochs,
                train_stats["loss"],
                train_stats["classification_loss"],
                train_stats["coral_loss"],
                self.training_config.coral_weight,
                train_stats["accuracy"],
                train_stats["skipped_batches"],
                val_stats["loss"] if val_stats["loss"] is not None else float("nan"),
                val_stats["accuracy"],
                val_stats["f1_macro"],
                current_lr,
            ]
            if self.training_config.track_loss_ratio:
                log_message += " | coral/ce=%.4f"
                log_args.append(train_stats["coral_ce_ratio"])
            self.logger.info(log_message, *log_args)

            if epochs_without_improvement >= self.training_config.early_stopping_patience:
                self.logger.info(
                    "Early stopping triggered at epoch %d. Best validation macro-F1 %.4f at epoch %d.",
                    epoch,
                    best_score,
                    best_epoch,
                )
                break

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        unwrap_model(self.model).load_state_dict(checkpoint["model_state"])
        if self.writer is not None:
            self.writer.close()
        training_time_seconds = time.perf_counter() - training_start_time
        self.logger.info("Training finished in %.2f seconds.", training_time_seconds)
        return TrainingArtifacts(
            checkpoint_path=checkpoint_path,
            history=history,
            best_epoch=best_epoch,
            best_score=best_score,
            training_time_seconds=training_time_seconds,
        )

    def _train_one_epoch(self, train_loader, scaler, amp_enabled: bool) -> dict[str, float]:
        self.training_model.train()
        total_loss = 0.0
        total_classification_loss = 0.0
        total_coral_loss = 0.0
        total_correct = 0
        total_samples = 0
        skipped_batches = 0
        accumulation_steps = self.training_config.gradient_accumulation_steps
        self.optimizer.zero_grad(set_to_none=True)

        for step_index, batch in enumerate(train_loader, start=1):
            inputs, labels, domains = unpack_batch(batch)
            inputs = inputs.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            if domains is not None:
                domains = domains.to(self.device, non_blocking=True)

            autocast_context = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if amp_enabled
                else nullcontext()
            )
            with autocast_context:
                coral_penalty = inputs.new_zeros(())
                if self.training_config.use_mixup or self.training_config.use_cutmix:
                    mix_output = apply_mix_augmentation(
                        inputs=inputs,
                        labels=labels,
                        num_classes=len(self.class_names),
                        use_mixup=self.training_config.use_mixup,
                        mixup_alpha=self.training_config.mixup_alpha,
                        use_cutmix=self.training_config.use_cutmix,
                        cutmix_alpha=self.training_config.cutmix_alpha,
                        cutmix_probability=self.training_config.cutmix_probability,
                        label_smoothing=self.training_config.label_smoothing,
                    )
                    logits, _ = self.training_model(mix_output.inputs, return_features=True)
                    classification_loss = soft_target_cross_entropy(logits, mix_output.soft_targets)
                    metric_targets = mix_output.soft_targets.argmax(dim=1)
                    if self._should_apply_coral(domains):
                        _, clean_features = self.training_model(inputs, return_features=True)
                        coral_penalty = multi_domain_coral_loss(clean_features, domains)
                else:
                    logits, features = self.training_model(inputs, return_features=True)
                    classification_loss = self.criterion(logits, labels)
                    metric_targets = labels
                    if self._should_apply_coral(domains):
                        coral_penalty = multi_domain_coral_loss(features, domains)

                raw_loss = classification_loss + self.training_config.coral_weight * coral_penalty

            if not torch.isfinite(raw_loss):
                skipped_batches += 1
                self.logger.warning("Skipping unstable batch with non-finite loss.")
                self.optimizer.zero_grad(set_to_none=True)
                continue

            loss = raw_loss / accumulation_steps

            if amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            should_step = step_index % accumulation_steps == 0 or step_index == len(train_loader)
            if should_step:
                if self.training_config.gradient_clip_norm is not None:
                    if amp_enabled:
                        scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.trainable_parameters,
                        max_norm=self.training_config.gradient_clip_norm,
                    )

                if amp_enabled:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += raw_loss.item() * batch_size
            total_classification_loss += classification_loss.item() * batch_size
            total_coral_loss += float(coral_penalty.item()) * batch_size
            total_correct += (predictions == metric_targets).sum().item()
            total_samples += batch_size

        return {
            "loss": total_loss / total_samples if total_samples > 0 else float("inf"),
            "classification_loss": total_classification_loss / total_samples if total_samples > 0 else float("inf"),
            "coral_loss": total_coral_loss / total_samples if total_samples > 0 else 0.0,
            "coral_ce_ratio": (
                (total_coral_loss / total_samples)
                / max(total_classification_loss / total_samples, 1e-8)
                if total_samples > 0
                else 0.0
            ),
            "accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
            "skipped_batches": skipped_batches,
        }

    def _build_optimizer(self):
        optimizer_name = self.training_config.optimizer.lower()
        if optimizer_name == "adam":
            return torch.optim.Adam(
                self.trainable_parameters,
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
            )
        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                self.trainable_parameters,
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay,
            )
        raise ValueError(f"Unsupported optimizer '{self.training_config.optimizer}'.")

    def _build_scheduler(self):
        scheduler_name = self.training_config.scheduler.lower()
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.training_config.epochs),
                eta_min=self.training_config.min_learning_rate,
            )
        if scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=2,
                min_lr=self.training_config.min_learning_rate,
            )
        if scheduler_name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=max(1, self.training_config.epochs // 3),
                gamma=0.1,
            )
        raise ValueError(f"Unsupported scheduler '{self.training_config.scheduler}'.")

    def _step_scheduler(self, score: float) -> None:
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(score)
        else:
            self.scheduler.step()

    def _save_checkpoint(self, checkpoint_path: Path, epoch: int, score: float) -> None:
        model_to_save = unwrap_model(self.model)
        torch.save(
            {
                "epoch": epoch,
                "best_score": score,
                "model_state": model_to_save.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "class_names": self.class_names,
                "model_name": getattr(model_to_save, "backbone_name", "unknown"),
            },
            checkpoint_path,
        )

    def _should_apply_coral(self, domains: torch.Tensor | None) -> bool:
        return (
            self.training_config.use_coral
            and domains is not None
            and torch.unique(domains).numel() > 1
        )

    def _build_tensorboard_writer(self):
        if not self.training_config.enable_tensorboard:
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            self.logger.warning("TensorBoard is enabled but the tensorboard package is not installed.")
            return None
        return SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))

    def _log_tensorboard(
        self,
        epoch: int,
        train_stats: dict[str, float],
        val_stats: dict[str, object],
        learning_rate: float,
    ) -> None:
        if self.writer is None:
            return
        self.writer.add_scalar("train/loss", train_stats["loss"], epoch)
        self.writer.add_scalar("train/classification_loss", train_stats["classification_loss"], epoch)
        self.writer.add_scalar("train/coral_loss", train_stats["coral_loss"], epoch)
        self.writer.add_scalar("train/coral_weight", self.training_config.coral_weight, epoch)
        if self.training_config.track_loss_ratio:
            self.writer.add_scalar("train/coral_ce_ratio", train_stats["coral_ce_ratio"], epoch)
        self.writer.add_scalar("train/accuracy", train_stats["accuracy"], epoch)
        self.writer.add_scalar("train/skipped_batches", train_stats["skipped_batches"], epoch)
        self.writer.add_scalar("val/loss", float(val_stats["loss"] or 0.0), epoch)
        self.writer.add_scalar("val/accuracy", float(val_stats["accuracy"]), epoch)
        self.writer.add_scalar("val/f1_macro", float(val_stats["f1_macro"]), epoch)
        self.writer.add_scalar("val/f1_weighted", float(val_stats["f1_weighted"]), epoch)
        self.writer.add_scalar("val/balanced_accuracy", float(val_stats["balanced_accuracy"]), epoch)
        self.writer.add_scalar("optimizer/lr", learning_rate, epoch)
        for class_name, accuracy in val_stats.get("per_class_accuracy", {}).items():
            tag = class_name.lower().replace(" ", "_")
            self.writer.add_scalar(f"val/per_class_accuracy/{tag}", float(accuracy), epoch)

    def _maybe_compile_model(self, model: nn.Module) -> nn.Module:
        if not self.training_config.use_torch_compile or not hasattr(torch, "compile"):
            return model
        try:
            return torch.compile(model)
        except Exception as exc:
            self.logger.warning(
                "torch.compile is unavailable for this configuration. Continuing without it. %s",
                exc,
            )
            return model
