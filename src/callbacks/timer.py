"""Lightweight training timer callback for logging GPU hours and epoch times."""

import json
import time
from pathlib import Path
from typing import Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback


class TrainingTimer(Callback):
    """
    Tracks wall-clock time per epoch and total GPU hours.

    Logs to TensorBoard/CSV:
        - epoch_time_sec: wall-clock seconds per epoch
        - gpu_hours: cumulative GPU hours (num_gpus * wall_time)

    Saves training_summary.json at end of training with:
        - total_wall_time, total_gpu_hours, avg_epoch_time
        - num_gpus, num_epochs, throughput (samples/sec)
    """

    def __init__(self, save_dir: Optional[str] = None):
        super().__init__()
        self.save_dir = save_dir
        self.train_start: float = 0.0
        self.epoch_start: float = 0.0
        self.epoch_times: list = []
        self.num_gpus: int = 1
        self.total_train_samples: int = 0

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.train_start = time.time()
        # Count GPUs
        if hasattr(trainer, "num_devices"):
            self.num_gpus = trainer.num_devices
        elif isinstance(trainer.devices, list):
            self.num_gpus = len(trainer.devices)
        else:
            self.num_gpus = max(1, int(trainer.devices or 1))

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch_time = time.time() - self.epoch_start
        self.epoch_times.append(epoch_time)

        wall_elapsed = time.time() - self.train_start
        gpu_hours = (wall_elapsed * self.num_gpus) / 3600.0

        # Track samples processed
        if hasattr(trainer, "train_dataloader") and trainer.train_dataloader is not None:
            try:
                epoch_samples = len(trainer.train_dataloader.dataset)
            except (TypeError, AttributeError):
                epoch_samples = 0
        else:
            epoch_samples = 0
        self.total_train_samples += epoch_samples

        # Log scalars
        pl_module.log("epoch_time_sec", epoch_time, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
        pl_module.log("gpu_hours", gpu_hours, on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)

        # Print summary for this epoch
        if trainer.is_global_zero:
            avg_time = sum(self.epoch_times) / len(self.epoch_times)
            remaining_epochs = trainer.max_epochs - trainer.current_epoch - 1
            eta_sec = avg_time * remaining_epochs
            eta_str = self._format_time(eta_sec)
            print(
                f"  Epoch {trainer.current_epoch + 1}/{trainer.max_epochs} "
                f"| {epoch_time:.1f}s "
                f"| GPU-hours: {gpu_hours:.2f} "
                f"| ETA: {eta_str}"
            )

    def on_fit_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not trainer.is_global_zero or not self.epoch_times:
            return

        total_wall = time.time() - self.train_start
        total_gpu_hours = (total_wall * self.num_gpus) / 3600.0
        avg_epoch = sum(self.epoch_times) / len(self.epoch_times)
        throughput = self.total_train_samples / total_wall if total_wall > 0 else 0

        summary = {
            "total_wall_time_sec": round(total_wall, 1),
            "total_wall_time": self._format_time(total_wall),
            "total_gpu_hours": round(total_gpu_hours, 3),
            "num_gpus": self.num_gpus,
            "num_epochs": len(self.epoch_times),
            "avg_epoch_time_sec": round(avg_epoch, 1),
            "throughput_samples_per_sec": round(throughput, 1),
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }

        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING TIME SUMMARY")
        print("=" * 60)
        print(f"  Total wall time:   {summary['total_wall_time']}")
        print(f"  Total GPU hours:   {summary['total_gpu_hours']:.3f}")
        print(f"  Num GPUs:          {summary['num_gpus']} x {summary['gpu_model']}")
        print(f"  Num epochs:        {summary['num_epochs']}")
        print(f"  Avg epoch time:    {summary['avg_epoch_time_sec']:.1f}s")
        print(f"  Throughput:        {summary['throughput_samples_per_sec']:.1f} samples/sec")
        print("=" * 60)

        # Save to JSON
        save_dir = Path(self.save_dir) if self.save_dir else None
        if save_dir is None:
            # Try to get from logger
            if trainer.logger and hasattr(trainer.logger, "log_dir"):
                save_dir = Path(trainer.logger.log_dir)
            elif trainer.logger and hasattr(trainer.logger, "save_dir"):
                save_dir = Path(trainer.logger.save_dir)

        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            summary_path = save_dir / "training_summary.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"  Saved to: {summary_path}")

    @staticmethod
    def _format_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        return f"{s}s"
