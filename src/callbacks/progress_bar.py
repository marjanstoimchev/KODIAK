# callbacks/progress_bar.py
"""
Rich Progress Bar for KODIAK Training.

Configurable progress bar for both pretraining and classification:
- Pretraining: Shows Loss, Mask, KoLeo, CLS_G (global), CLS_L (local), Utilization
- Classification: Shows Loss, Accuracy (top-1, top-5 if applicable)

Features:
- Proper epoch/step counting
- Iteration speed (it/s)
- ETA calculation
- DDP-safe (only rank 0 updates)
"""

import time
from typing import Any, Optional, Literal

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from rich.progress import (
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    SpinnerColumn,
)


class KodiakProgressBar(RichProgressBar):
    """
    Unified progress bar for KODIAK pretraining and classification.

    Args:
        mode: "pretraining" or "classification" - determines which metrics to show
        refresh_rate: How often to refresh the progress bar (default: 1)
        leave: Whether to leave the progress bar visible after completion

    Pretraining metrics:
        - Loss (total), Mask, KoLeo, CLS_G (global), CLS_L (local), Utilization

    Classification metrics:
        - Loss, Accuracy, Top-5 Accuracy (if >5 classes)
    """

    def __init__(
        self,
        mode: Literal["pretraining", "classification"] = "pretraining",
        refresh_rate: int = 1,
        leave: bool = True,
    ):
        theme = RichProgressBarTheme(
            description="white",
            progress_bar="#6206E0",
            progress_bar_finished="#6206E0",
            progress_bar_pulse="#6206E0",
            batch_progress="white",
            time="grey54",
            processing_speed="grey70",
            metrics="white",
        )
        super().__init__(refresh_rate=refresh_rate, leave=leave, theme=theme)

        self.mode = mode
        self.start_time: Optional[float] = None
        self.epoch_start_time: Optional[float] = None
        self.steps_per_epoch: int = 0
        self.cached_metrics: str = ""

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Initialize training progress tracking."""
        if trainer.global_rank != 0:
            return

        super().on_train_start(trainer, pl_module)
        self.start_time = time.time()

        # Calculate steps per epoch
        try:
            if hasattr(trainer, 'num_training_batches') and trainer.num_training_batches:
                self.steps_per_epoch = trainer.num_training_batches
            elif hasattr(trainer.datamodule, 'train_dataloader'):
                dataloader = trainer.datamodule.train_dataloader()
                self.steps_per_epoch = len(dataloader) if hasattr(dataloader, '__len__') else 100
            else:
                self.steps_per_epoch = 100
        except Exception:
            self.steps_per_epoch = 100

        self.steps_per_epoch = max(1, min(self.steps_per_epoch, 100000))

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Track epoch start time."""
        if trainer.global_rank != 0:
            return

        try:
            super().on_train_epoch_start(trainer, pl_module)
        except (AssertionError, AttributeError):
            pass

        self.epoch_start_time = time.time()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Update progress bar with metrics, speed, and ETA."""
        if trainer.global_rank != 0:
            return

        try:
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        except (AssertionError, AttributeError):
            pass

        if self.train_progress_bar_id is None:
            return

        # Build description
        epoch = trainer.current_epoch + 1
        max_epochs = trainer.max_epochs if trainer.max_epochs > 0 else 100
        step = batch_idx + 1

        # Calculate speed
        speed_str = self._calc_speed(step)

        # Calculate ETA
        eta_str = self._calc_eta(trainer, step)

        # Format metrics based on mode
        if self.mode == "pretraining":
            metrics = self._format_pretraining_metrics(trainer)
        else:
            metrics = self._format_classification_metrics(trainer)

        # Build description line
        description = (
            f"[bold blue]Epoch {epoch}/{max_epochs}[/bold blue] "
            f"[grey54]Step {step}/{self.steps_per_epoch}[/grey54] | "
            f"{metrics} | "
            f"[grey70]{speed_str}[/grey70] | "
            f"[grey54]ETA: {eta_str}[/grey54]"
        )

        self.progress.update(self.train_progress_bar_id, description=description)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Show epoch summary."""
        if trainer.global_rank != 0:
            return

        try:
            super().on_train_epoch_end(trainer, pl_module)
        except (AssertionError, AttributeError):
            pass

        # Print epoch summary (plain text, no Rich markup)
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            epoch = trainer.current_epoch + 1
            max_epochs = trainer.max_epochs if trainer.max_epochs > 0 else 100

            if self.mode == "pretraining":
                metrics = self._format_pretraining_metrics_plain(trainer)
            else:
                metrics = self._format_classification_metrics_plain(trainer)

            it_per_sec = self.steps_per_epoch / epoch_time if epoch_time > 0 else 0
            print(f"Epoch {epoch}/{max_epochs} completed in {epoch_time:.1f}s ({it_per_sec:.2f} it/s) | {metrics}")

    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update validation progress (classification only)."""
        if trainer.global_rank != 0:
            return

        try:
            super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
        except (AssertionError, AttributeError):
            pass

        if self.val_progress_bar_id is not None and self.mode == "classification":
            metrics = self._format_classification_metrics(trainer, phase="val")
            description = f"[bold cyan]Validating[/bold cyan] | {metrics}"
            self.progress.update(self.val_progress_bar_id, description=description)

    def _calc_speed(self, current_step: int) -> str:
        """Calculate iteration speed."""
        if self.epoch_start_time is None:
            return "-- it/s"

        elapsed = time.time() - self.epoch_start_time
        if elapsed > 0 and current_step > 0:
            it_per_sec = current_step / elapsed
            if it_per_sec >= 1:
                return f"{it_per_sec:.2f} it/s"
            else:
                return f"{1/it_per_sec:.2f} s/it"
        return "-- it/s"

    def _calc_eta(self, trainer: "pl.Trainer", current_step: int) -> str:
        """Calculate estimated time remaining."""
        if self.start_time is None:
            return "--"

        elapsed = time.time() - self.start_time
        global_step = trainer.global_step

        if global_step <= 0 or elapsed <= 0:
            return "--"

        # Total steps = steps_per_epoch * max_epochs
        max_epochs = trainer.max_epochs if trainer.max_epochs > 0 else 100
        total_steps = self.steps_per_epoch * max_epochs
        remaining_steps = max(0, total_steps - global_step)

        # ETA based on average speed
        speed = global_step / elapsed
        if speed > 0:
            eta_seconds = remaining_steps / speed
            return self._format_time(eta_seconds)

        return "--"

    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable string."""
        if seconds <= 0 or seconds > 999999:
            return "--"

        if seconds >= 3600:
            return f"{seconds/3600:.1f}h"
        elif seconds >= 60:
            return f"{seconds/60:.1f}m"
        else:
            return f"{seconds:.0f}s"

    def _format_pretraining_metrics(self, trainer: "pl.Trainer", for_summary: bool = False) -> str:
        """Format pretraining metrics: Loss, Mask, KoLeo, CLS_G, CLS_L, Util."""
        try:
            m = trainer.logged_metrics
            parts = []

            # Helper to get metric value (try with _step suffix first, then without)
            def get_metric(base_name: str):
                for key in [f"{base_name}_step", base_name, f"{base_name}_epoch"]:
                    if key in m:
                        return float(m[key])
                return None

            # Total loss
            val = get_metric('train_loss')
            if val is not None:
                parts.append(f"[#A23B72]Loss:{val:.4f}[/#A23B72]")

            # Mask loss
            val = get_metric('train_mask')
            if val is not None:
                parts.append(f"[#F18F01]Mask:{val:.4f}[/#F18F01]")

            # KoLeo loss
            val = get_metric('train_koleo')
            if val is not None:
                parts.append(f"[#C73E1D]KoLeo:{val:.4f}[/#C73E1D]")

            # Multi-Crop Proto CLS losses (global and local)
            val = get_metric('train_cls_global')
            if val is not None:
                parts.append(f"[#7209B7]CLS_G:{val:.4f}[/#7209B7]")
            val = get_metric('train_cls_local')
            if val is not None:
                parts.append(f"[#9D4EDD]CLS_L:{val:.4f}[/#9D4EDD]")

            # Prototype utilization
            val = get_metric('train_util')
            if val is not None:
                parts.append(f"[#00A8CC]Util:{val:.0f}[/#00A8CC]")

            if parts:
                self.cached_metrics = " | ".join(parts)
                return self.cached_metrics

            return self.cached_metrics if self.cached_metrics else "[grey54]...[/grey54]"
        except Exception:
            return self.cached_metrics if self.cached_metrics else "[grey54]...[/grey54]"

    def _format_classification_metrics(
        self,
        trainer: "pl.Trainer",
        phase: str = "train",
        for_summary: bool = False,
    ) -> str:
        """Format classification metrics: Loss, Acc, Top-5."""
        try:
            m = trainer.logged_metrics
            parts = []

            # Helper to get metric value (try with _step suffix first, then without)
            def get_metric(base_name: str):
                for key in [f"{base_name}_step", base_name, f"{base_name}_epoch"]:
                    if key in m:
                        return float(m[key])
                return None

            # Loss
            val = get_metric(f'{phase}_loss')
            if val is not None:
                parts.append(f"[#A23B72]Loss:{val:.4f}[/#A23B72]")

            # Accuracy (top-1)
            val = get_metric(f'{phase}_acc')
            if val is not None:
                # Display as percentage if < 1, otherwise as-is
                if val <= 1:
                    parts.append(f"[#F18F01]Acc:{val*100:.2f}%[/#F18F01]")
                else:
                    parts.append(f"[#F18F01]Acc:{val:.2f}%[/#F18F01]")

            # Top-5 accuracy (if available)
            val = get_metric(f'{phase}_acc_top5')
            if val is not None:
                if val <= 1:
                    parts.append(f"[#7209B7]Top5:{val*100:.2f}%[/#7209B7]")
                else:
                    parts.append(f"[#7209B7]Top5:{val:.2f}%[/#7209B7]")

            if parts:
                self.cached_metrics = " | ".join(parts)
                return self.cached_metrics

            return self.cached_metrics if self.cached_metrics else "[grey54]...[/grey54]"
        except Exception:
            return self.cached_metrics if self.cached_metrics else "[grey54]...[/grey54]"

    def _format_pretraining_metrics_plain(self, trainer: "pl.Trainer") -> str:
        """Format pretraining metrics as plain text (no Rich markup)."""
        try:
            m = trainer.logged_metrics
            parts = []

            def get_metric(base_name: str):
                for key in [f"{base_name}_step", base_name, f"{base_name}_epoch"]:
                    if key in m:
                        return float(m[key])
                return None

            val = get_metric('train_loss')
            if val is not None:
                parts.append(f"Loss:{val:.4f}")

            val = get_metric('train_mask')
            if val is not None:
                parts.append(f"Mask:{val:.4f}")

            val = get_metric('train_koleo')
            if val is not None:
                parts.append(f"KoLeo:{val:.4f}")

            val = get_metric('train_cls_global')
            if val is not None:
                parts.append(f"CLS_G:{val:.4f}")

            val = get_metric('train_cls_local')
            if val is not None:
                parts.append(f"CLS_L:{val:.4f}")

            val = get_metric('train_util')
            if val is not None:
                parts.append(f"Util:{val:.0f}")

            return " | ".join(parts) if parts else "..."
        except Exception:
            return "..."

    def _format_classification_metrics_plain(self, trainer: "pl.Trainer", phase: str = "train") -> str:
        """Format classification metrics as plain text (no Rich markup)."""
        try:
            m = trainer.logged_metrics
            parts = []

            def get_metric(base_name: str):
                for key in [f"{base_name}_step", base_name, f"{base_name}_epoch"]:
                    if key in m:
                        return float(m[key])
                return None

            val = get_metric(f'{phase}_loss')
            if val is not None:
                parts.append(f"Loss:{val:.4f}")

            val = get_metric(f'{phase}_acc')
            if val is not None:
                if val <= 1:
                    parts.append(f"Acc:{val*100:.2f}%")
                else:
                    parts.append(f"Acc:{val:.2f}%")

            val = get_metric(f'{phase}_acc_top5')
            if val is not None:
                if val <= 1:
                    parts.append(f"Top5:{val*100:.2f}%")
                else:
                    parts.append(f"Top5:{val:.2f}%")

            return " | ".join(parts) if parts else "..."
        except Exception:
            return "..."

    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> dict:
        """Override to hide default metrics display - we show them in description."""
        # Return empty dict to hide the default metrics on the right side
        # Our custom metrics are shown in the description (left side)
        return {}

    def configure_columns(self, trainer: "pl.Trainer") -> list:
        """Configure progress bar columns."""
        return [
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(complete_style="rgb(98,6,224)", finished_style="rgb(98,6,224)"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ]
