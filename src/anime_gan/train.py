from __future__ import annotations

import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from anime_gan.data.datamodule import AnimeFaceDataModule
from anime_gan.lit.dcgan_module import DCGANModule
from anime_gan.utils.callbacks import FidelityCallback, SampleImageCallback
from anime_gan.utils.paths import resolve_path
from anime_gan.utils.seed import seed_everything


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(cfg.seed)

    # wandb environment
    os.environ.setdefault("WANDB_MODE", cfg.logger.wandb.mode)
    if cfg.logger.wandb.entity:
        os.environ.setdefault("WANDB_ENTITY", cfg.logger.wandb.entity)
    os.environ.setdefault("WANDB_PROJECT", cfg.logger.wandb.project)
    os.environ.setdefault("WANDB_DIR", str(Path.cwd() / "wandb"))

    datamodule = AnimeFaceDataModule(**cfg.dataset)
    model = DCGANModule(**cfg.model)
    model.samples_dir = Path.cwd() / "samples"

    callbacks: list[pl.callbacks.Callback] = [
        ModelCheckpoint(
            dirpath=Path.cwd() / "checkpoints",
            filename="dcgan-{epoch:02d}-{step:06d}",
            save_top_k=3,
            monitor="loss_g",
            mode="min",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        SampleImageCallback(num_samples=cfg.model.sample_grid_size, every_n_epochs=1),
    ]

    if cfg.eval.enabled:
        real_dir = resolve_path(cfg.dataset.data_dir)
        callbacks.append(
            FidelityCallback(
                real_dir=real_dir,
                z_dim=cfg.model.z_dim,
                sample_size=cfg.eval.sample_size,
                batch_size=cfg.eval.batch_size,
                every_n_epochs=cfg.eval.every_n_epochs,
                work_dir=Path.cwd() / "metrics",
            )
        )

    logger = WandbLogger(
        project=cfg.logger.wandb.project,
        entity=cfg.logger.wandb.entity,
        mode=cfg.logger.wandb.mode,
        save_dir=str(Path.cwd()),
        log_model=cfg.logger.wandb.log_model,
    )

    trainer = pl.Trainer(
        **cfg.trainer, logger=logger, callbacks=callbacks, default_root_dir=str(Path.cwd())
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()
