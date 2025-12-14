from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from jaxtyping import Float
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from typeguard import check_type, typechecked

from anime_gan.data.datamodule import ImageBatch
from anime_gan.models.dcgan import Discriminator, Generator, NoiseBatch, init_weights
from anime_gan.utils.images import save_image_grid


@dataclass
class GANLossConfig:
    label_smoothing: float = 0.0


class DCGANModule(pl.LightningModule):
    def __init__(
        self,
        z_dim: int = 128,
        g_features: int = 128,
        d_features: int = 128,
        lr: float = 2e-4,
        weight_decay: float = 0.0,
        beta1: float = 0.5,
        beta2: float = 0.999,
        scheduler_t_max_epochs: int | None = None,
        scheduler_eta_min: float = 0.0,
        sample_every_n_steps: int = 200,
        sample_grid_size: int = 64,
        loss_cfg: GANLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.generator = Generator(z_dim=z_dim, feature_maps=g_features)
        self.discriminator = Discriminator(feature_maps=d_features)
        self.generator.apply(init_weights)
        self.discriminator.apply(init_weights)

        self.criterion = nn.BCEWithLogitsLoss()
        self.automatic_optimization = False
        self.example_input_array = torch.randn(1, z_dim)
        self.fixed_noise = torch.randn(sample_grid_size, z_dim)
        self.loss_cfg = loss_cfg or GANLossConfig()

        self.samples_dir = Path("samples")

    @typechecked
    def forward(self, noise: Tensor) -> Tensor:
        check_type(noise, Float[Tensor, "batch z_dim"])
        generated = self.generator(noise)
        check_type(generated, Float[Tensor, "batch 3 128 128"])
        return generated

    def configure_optimizers(self) -> tuple[list[Optimizer], list[Any]]:
        opt_g = AdamW(
            self.generator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        opt_d = AdamW(
            self.discriminator.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(self.hparams.beta1, self.hparams.beta2),
        )
        t_max = self.hparams.scheduler_t_max_epochs or self.trainer.max_epochs
        sched_d = CosineAnnealingLR(opt_d, T_max=t_max, eta_min=self.hparams.scheduler_eta_min)
        sched_g = CosineAnnealingLR(opt_g, T_max=t_max, eta_min=self.hparams.scheduler_eta_min)
        return [opt_d, opt_g], [sched_d, sched_g]

    def _smooth_labels(self, value: float, size: int) -> Tensor:
        if self.loss_cfg.label_smoothing <= 0:
            return torch.full((size, 1), value, device=self.device)
        delta = self.loss_cfg.label_smoothing
        if value == 1.0:
            return torch.empty(size, 1, device=self.device).uniform_(1 - delta, 1)
        return torch.empty(size, 1, device=self.device).uniform_(0, delta)

    def training_step(self, batch: ImageBatch, batch_idx: int) -> dict[str, float]:
        opt_d, opt_g = self.optimizers()
        real_images = batch.images
        batch_size = real_images.size(0)

        valid = self._smooth_labels(1.0, batch_size)
        fake = self._smooth_labels(0.0, batch_size)

        # --- Train Discriminator ---
        opt_d.zero_grad(set_to_none=True)
        real_pred = self.discriminator(real_images)
        noise_batch = NoiseBatch(z=torch.randn(batch_size, self.hparams.z_dim, device=self.device))
        fake_images = self(noise_batch.z).detach()
        fake_pred = self.discriminator(fake_images)
        d_loss_real = self.criterion(real_pred, valid)
        d_loss_fake = self.criterion(fake_pred, fake)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        self.manual_backward(d_loss)
        opt_d.step()

        # --- Train Generator ---
        opt_g.zero_grad(set_to_none=True)
        regenerated = self(noise_batch.z)
        gen_pred = self.discriminator(regenerated)
        g_loss = self.criterion(gen_pred, valid)
        self.manual_backward(g_loss)
        opt_g.step()

        self.log("loss_d", d_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("loss_g", g_loss, prog_bar=True, on_step=True, on_epoch=True)

        if batch_idx % self.hparams.sample_every_n_steps == 0:
            self._log_samples(global_step=self.global_step)

        return {"loss": d_loss + g_loss}

    def on_train_epoch_end(self) -> None:
        schedulers = self.lr_schedulers()
        for scheduler in schedulers:
            scheduler.step()

    @torch.no_grad()
    def _log_samples(self, global_step: int) -> None:
        self.generator.eval()
        noise = self.fixed_noise.to(self.device)
        samples = self(noise)
        grid_path = self.samples_dir / f"step_{global_step:06d}.png"
        save_image_grid(samples, grid_path, nrow=int(self.hparams.sample_grid_size**0.5))
        if self.logger is not None and hasattr(self.logger, "experiment"):
            try:
                self.logger.experiment.log(
                    {"samples": [self.logger.experiment.Image(str(grid_path))]}
                )
            except Exception:
                # Keep training even if logging fails
                pass
        self.generator.train()
