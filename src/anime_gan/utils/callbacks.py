from __future__ import annotations

from pathlib import Path

import pytorch_lightning as pl
import torch
import wandb
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

# 导入你提供的模块 (假设保存为 anime_gan.data.datamodule)
from anime_gan.data.datamodule import AnimeFaceDataset 
from anime_gan.utils.images import denormalize, save_image_grid

import gc
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from anime_gan.data.datamodule import AnimeFaceDataset 
from anime_gan.utils.images import denormalize

class FidelityCallback(pl.Callback):
    def __init__(
        self,
        real_dir: Path,
        z_dim: int,
        sample_size: int = 256,
        batch_size: int = 64, 
        every_n_epochs: int = 5,
        image_size: int = 128,
        work_dir: Path | None = None,
    ) -> None:
        super().__init__()
        self.real_dir = real_dir
        self.z_dim = z_dim
        self.sample_size = sample_size
        self.batch_size = batch_size
        self.every_n_epochs = max(1, every_n_epochs)
        
        # 初始化 FID 模块 (默认在 CPU)
        # normalize=False 因为我们会手动处理成 uint8
        self.fid = FrechetInceptionDistance(feature=2048, reset_real_features=False, normalize=False)
        
        self.fid_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 255).byte())
        ])

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if trainer.is_global_zero:
            print(f"Computing FID statistics for real images...")
        
        # 1. 将 FID 移至 GPU 进行真实图片统计
        self.fid.to(pl_module.device)
        
        dataset = AnimeFaceDataset(self.real_dir, transform=self.fid_transform)
        # 确保 DDP 模式下采样器正常工作
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False) if trainer.world_size > 1 else None
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            num_workers=4, 
            sampler=sampler,
            drop_last=False,
            persistent_workers=False # 避免 Worker 进程占用内存
        )
        
        try:
            for batch in dataloader:
                real_imgs = batch.to(pl_module.device)
                self.fid.update(real_imgs, real=True)
                # 显式删除 batch 以释放引用
                del real_imgs
        finally:
            # 2. 统计完成后，立刻将 FID 移回 CPU 并清理显存
            self.fid.cpu()
            self._clean_memory()
            
        if trainer.is_global_zero:
            print("Real image statistics computed and moved to CPU.")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        # 确保每个 GPU 分配的任务量
        num_samples_per_gpu = self.sample_size // trainer.world_size
        if num_samples_per_gpu == 0: return
        
        # 3. 开始前先清理一次，确保有足够空间加载 Inception 模型
        self._clean_memory()

        pl_module.eval()
        # 暂时关闭自动梯度，防止生成图时构建计算图
        with torch.no_grad():
            try:
                # 4. 将 FID 搬回 GPU
                self.fid.to(pl_module.device)
                
                current_samples = 0
                while current_samples < num_samples_per_gpu:
                    batch_n = min(self.batch_size, num_samples_per_gpu - current_samples)
                    
                    noise = torch.randn(batch_n, self.z_dim, device=pl_module.device)
                    
                    if hasattr(pl_module, 'generator'):
                        fake_imgs = pl_module.generator(noise)
                    else:
                        fake_imgs = pl_module(noise)
                    
                    fake_imgs_denorm = denormalize(fake_imgs) 
                    
                    if fake_imgs_denorm.dtype != torch.uint8:
                        fake_imgs_uint8 = (fake_imgs_denorm * 255).clamp(0, 255).to(torch.uint8)
                    else:
                        fake_imgs_uint8 = fake_imgs_denorm

                    self.fid.update(fake_imgs_uint8, real=False)
                    
                    # 关键修改：显式删除中间变量
                    del noise, fake_imgs, fake_imgs_denorm, fake_imgs_uint8
                    current_samples += batch_n
                
                # 计算 FID
                fid_score = self.fid.compute()
                
                # Log 只有 rank 0 会写 wandb，但为了数据同步安全，这里建议 sync_dist=True 或者保持 False
                # 如果 sync_dist=False，每个 GPU 计算的是局部的 FID (因为采样也是局部的)
                # 如果你想算全局 FID，应该 sync_dist=True，但 torchmetrics 内部 compute() 应该已经处理了
                pl_module.log("metrics/fid", fid_score, sync_dist=False)
                
                # 重置 fake 统计数据 (保留 real)
                self.fid.reset()
                del fid_score # 删除结果张量
                
            except Exception as e:
                print(f"Error computing FID on rank {trainer.global_rank}: {e}")
            finally:
                # 5. 必须在 finally 块中清理，保证即使出错也能释放
                self.fid.cpu()
                pl_module.train()
                self._clean_memory()

    def _clean_memory(self):
        """激进的显存清理函数"""
        import gc
        gc.collect() # 强制 Python 垃圾回收，断开僵尸对象的引用
        torch.cuda.empty_cache() # 清理 PyTorch 缓存

class SampleImageCallback(pl.Callback):
    # 这个 Callback 保持不变，逻辑是正确的
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