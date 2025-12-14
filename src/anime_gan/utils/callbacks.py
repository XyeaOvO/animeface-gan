from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb

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
    def __init__(
        self,
        num_samples: int = 16,
        every_n_steps: int | None = None,
        every_n_epochs: int | None = 1,
    ) -> None:
        super().__init__()
        self.num_samples = num_samples
        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs

    def _log_samples(self, trainer: pl.Trainer, pl_module: pl.LightningModule, tag: str) -> None:
        noise = torch.randn(self.num_samples, pl_module.hparams.z_dim, device=pl_module.device)
        with torch.no_grad():
            samples = pl_module(noise)

        grid_path = pl_module.samples_dir / f"{tag}.png"
        save_image_grid(samples, grid_path, nrow=int(self.num_samples**0.5))

        if pl_module.logger is not None:
            experiment = getattr(pl_module.logger, "experiment", None)
            
            if hasattr(experiment, "log"):
                try:
                    experiment.log(
                        {
                            "samples": [wandb.Image(str(grid_path), caption=tag)]
                        },
                        step=trainer.global_step,
                    )
                except Exception as e:
                    print(f"Logging to WandB failed: {e}")

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: None | dict,
        batch: torch.Tensor,
        batch_idx: int,
    ) -> None:
        if self.every_n_steps is None:
            return
        if trainer.global_step > 0 and trainer.global_step % self.every_n_steps == 0:
            self._log_samples(trainer, pl_module, tag=f"step_{trainer.global_step:06d}")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.every_n_epochs is None:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % max(1, self.every_n_epochs) != 0:
            return
        self._log_samples(trainer, pl_module, tag=f"epoch_{epoch:04d}")
