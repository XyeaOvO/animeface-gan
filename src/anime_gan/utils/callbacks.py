from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from anime_gan.data.datamodule import AnimeFaceDataset 
from torchmetrics.image.fid import FrechetInceptionDistance

from anime_gan.data.datamodule import AnimeFaceDataset 
from anime_gan.utils.images import denormalize, save_image_grid

class FidelityCallback(pl.Callback):
    """
    一个用于在训练过程中计算 Frechet Inception Distance (FID) 的 Callback。

    该 Callback 经过优化，以高效处理显存，并兼容分布式数据并行 (DDP) 训练。

    工作流程:
    1. on_train_start: 预先计算整个真实数据集的 Inception 特征，然后将 FID 模型移回 CPU 以释放显存。
    2. on_train_epoch_end: 定期生成一批假图像，将 FID 模型临时移至 GPU，计算 FID 分数，
       然后再次将 FID 模型移回 CPU，确保不影响正常的训练流程。
    """

    def __init__(
        self,
        real_images_dir: str | Path,
        z_dim: int,
        num_samples: int = 10000,
        batch_size: int = 64,
        every_n_epochs: int = 5,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.real_images_dir = Path(real_images_dir)
        self.z_dim = z_dim
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.every_n_epochs = max(1, every_n_epochs)
        self.num_workers = num_workers

        # 初始化 FID 模块。
        # feature=2048: 使用 InceptionV3 的最终池化层，这是标准做法。
        # reset_real_features=False: 关键设置！确保预计算的真实特征不会在调用 .compute() 后被重置。
        # normalize=True: FID 模块期望输入是 [0, 255] 范围的 uint8 整数。
        self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=True)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """在训练开始前，计算并缓存真实图像的统计数据。"""
        if not trainer.is_global_zero:
            return  # 只有主进程需要打印日志

        print(f"[{self.__class__.__name__}] Computing FID statistics for real images...")

        # 1. 将 FID 模块移至 GPU 以加速特征提取
        self.fid.to(pl_module.device)

        # 2. 创建用于加载真实图像的 DataLoader
        dataset = AnimeFaceDataset(self.real_images_dir) # Dataset 应返回 PIL Image 或 [0,1] Tensor
        sampler: Sampler | None = (
            DistributedSampler(dataset, shuffle=False) if trainer.world_size > 1 else None
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True, # 加速 CPU -> GPU 数据传输
            drop_last=False,
        )

        # 3. 遍历数据并更新真实特征
        try:
            for batch in dataloader:
                # 假设 Dataset 返回的是 [0,1] 的 Tensor
                images = batch.to(pl_module.device)
                # FID 模块内部会将 [0,1] float 转为 [0,255] uint8
                images_uint8 = convert_image_dtype(images, dtype=torch.uint8)
                self.fid.update(images_uint8, real=True)
                del images, images_uint8
        finally:
            # 4. 关键：计算完成后，立刻将 FID 模块移回 CPU，并清理显存
            self.fid.cpu()
            self._clean_memory()

        print(f"[{self.__class__.__name__}] Real image statistics computed and FID module moved to CPU.")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """在每个 epoch 结束时，计算并记录 FID 分数。"""
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        # 确保在 DDP 环境下，所有进程生成的样本总数是你期望的
        num_samples_per_gpu = self.num_samples // trainer.world_size
        if num_samples_per_gpu == 0:
            if trainer.is_global_zero:
                print(f"[{self.__class__.__name__}] Warning: num_samples is too small for the number of GPUs.")
            return

        pl_module.eval()
        self._clean_memory() # 计算前先清理一次

        try:
            # 1. 将 FID 模块移至 GPU
            self.fid.to(pl_module.device)

            # 2. 生成假图像并更新假特征
            with torch.no_grad():
                generated_count = 0
                while generated_count < num_samples_per_gpu:
                    batch_n = min(self.batch_size, num_samples_per_gpu - generated_count)
                    noise = torch.randn(batch_n, self.z_dim, device=pl_module.device)

                    # 兼容不同的模块结构
                    if hasattr(pl_module, 'generator'):
                        fake_images = pl_module.generator(noise)
                    else:
                        fake_images = pl_module(noise)
                    
                    # 将 [-1, 1] 的 tanh 输出转换为 [0, 255] 的 uint8
                    fake_images_denorm = denormalize(fake_images)
                    fake_images_uint8 = convert_image_dtype(fake_images_denorm, torch.uint8)

                    self.fid.update(fake_images_uint8, real=False)
                    
                    generated_count += batch_n
                    del noise, fake_images, fake_images_denorm, fake_images_uint8

            # 3. 计算 FID 分数
            # .compute() 在 DDP 模式下会自动同步所有进程的特征，因此只在主进程记录即可
            fid_score = self.fid.compute()
            
            # log() 方法会自动处理 DDP 同步，这里 sync_dist=True 是安全的默认行为
            pl_module.log("metrics/fid", fid_score, on_step=False, on_epoch=True, sync_dist=True)

            # 4. **无需手动重置**
            # .compute() 后，非持久性状态 (fake_features) 会被自动清除。
            # real_features 因为 reset_real_features=False 而被保留。
            del fid_score

        except Exception as e:
            if trainer.is_global_zero:
                print(f"[{self.__class__.__name__}] Error computing FID on epoch {epoch}: {e}")
        finally:
            # 5. 关键：无论成功与否，都将 FID 移回 CPU 并恢复模型状态
            self.fid.cpu()
            pl_module.train()
            self._clean_memory()

    def _clean_memory(self) -> None:
        """执行一次激进的显存清理。"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        if trainer.global_rank == 0:
            noise = torch.randn(self.num_samples, pl_module.hparams.z_dim, device=pl_module.device)
            pl_module.eval()
            try:
                with torch.no_grad():
                    if hasattr(pl_module, 'generator'):
                        samples = pl_module.generator(noise)
                    else:
                        samples = pl_module(noise)
            finally:
                pl_module.train()

            grid_path = pl_module.samples_dir / f"{tag}.png"
            samples_denorm = denormalize(samples).detach().cpu()
            
            # 兼容处理 uint8 或 float
            if samples_denorm.dtype == torch.uint8:
                 samples_denorm = samples_denorm.float() / 255.0

            save_image_grid(samples, grid_path, nrow=int(self.num_samples**0.5))

            if pl_module.logger is not None:
                experiment = getattr(pl_module.logger, "experiment", None)
                if hasattr(experiment, "log"):
                    try:
                        grid_tensor = make_grid(samples_denorm, nrow=int(self.num_samples**0.5))
                        grid_np = (
                            (grid_tensor.permute(1, 2, 0).numpy() * 255)
                            .clip(0, 255)
                            .astype(np.uint8)
                        )
                        experiment.log(
                            {"samples": [wandb.Image(grid_np, caption=tag)]},
                            step=trainer.global_step,
                        )
                    except Exception:
                        pass

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
            trainer.strategy.barrier()
            self._log_samples(trainer, pl_module, tag=f"step_{trainer.global_step:06d}")
            trainer.strategy.barrier()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if self.every_n_epochs is None:
            return
        epoch = trainer.current_epoch
        if (epoch + 1) % max(1, self.every_n_epochs) != 0:
            return
        trainer.strategy.barrier()
        self._log_samples(trainer, pl_module, tag=f"epoch_{epoch:04d}")
        trainer.strategy.barrier()