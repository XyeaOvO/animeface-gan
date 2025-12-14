from __future__ import annotations

import shutil
from pathlib import Path

import torch
from torch import nn
from torch_fidelity import calculate_metrics
from torchvision.utils import save_image

from anime_gan.utils.images import denormalize


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def generate_images_to_dir(
    generator: nn.Module,
    z_dim: int,
    sample_size: int,
    batch_size: int,
    device: torch.device,
    dest: Path,
) -> None:
    _reset_dir(dest)
    generator.eval()
    total = 0
    while total < sample_size:
        current = min(batch_size, sample_size - total)
        noise = torch.randn(current, z_dim, device=device)
        with torch.no_grad():
            images = generator(noise)
        images = denormalize(images)
        for idx in range(current):
            save_image(images[idx], dest / f"{total + idx:06d}.png")
        total += current


def compute_fid_is(
    generator: nn.Module,
    real_dir: Path,
    z_dim: int,
    sample_size: int,
    batch_size: int,
    device: torch.device,
    work_dir: Path,
) -> dict[str, float]:
    fake_dir = work_dir / "generated_for_fid"
    generate_images_to_dir(generator, z_dim, sample_size, batch_size, device, fake_dir)

    metrics = calculate_metrics(
        input1=str(real_dir),
        input2=str(fake_dir),
        fid=True,
        isc=True,
        kid=False,
        verbose=False,
        samples_find_deep=True,
        sample_size=sample_size,
        batch_size=batch_size,
        device=str(device),
    )
    return {
        "fid": float(metrics.get("frechet_inception_distance", 0.0)),
        "is": float(metrics.get("inception_score_mean", 0.0)),
    }
