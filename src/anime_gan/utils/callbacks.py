from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch

from anime_gan.utils.images import save_image_grid
from anime_gan.utils.metrics import compute_fid_is


class FidelityCallback(pl.Callback):
    def __init__(
        self,
        real_dir: Path,
        z_dim: int,
        sample_size: int = 256,
        batch_size: int = 64,
        every_n_epochs: int = 5,
        work_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.real_dir = real_dir
        self.z_dim = z_dim
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.every_n_epochs = max(1, every_n_epochs)
        self.work_dir = work_dir

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        work_dir = self.work_dir or Path(trainer.default_root_dir) / "metrics"
        work_dir.mkdir(parents=True, exist_ok=True)
        metrics = compute_fid_is(
            generator=pl_module.generator,
            real_dir=self.real_dir,
            z_dim=self.z_dim,
            sample_size=self.sample_size,
            batch_size=self.batch_size,
            device=pl_module.device,
            work_dir=work_dir,
        )
        for name, value in metrics.items():
            pl_module.log(f"metrics/{name}", value, prog_bar=False, on_epoch=True, sync_dist=False)


class SampleImageCallback(pl.Callback):
    def __init__(self, num_samples: int = 16, every_n_epochs: int = 1) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.every_n_epochs = max(1, every_n_epochs)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return
        noise = torch.randn(self.num_samples, pl_module.hparams.z_dim, device=pl_module.device)
        with torch.no_grad():
            samples = pl_module(noise)
        grid_path = pl_module.samples_dir / f"epoch_{epoch:04d}.png"
        save_image_grid(samples, grid_path, nrow=int(self.num_samples**0.5))
        if pl_module.logger is not None and hasattr(pl_module.logger, "experiment"):
            try:
                pl_module.logger.experiment.log(
                    {"samples/epoch": [pl_module.logger.experiment.Image(str(grid_path))]},
                    step=trainer.global_step,
                )
            except Exception:
                # Keep training even if logging fails
                pass
