from __future__ import annotations

from pathlib import Path

from jaxtyping import Float
from torch import Tensor
from torchvision.utils import save_image


def denormalize(images: Float[Tensor, "*batch 3 128 128"]) -> Float[Tensor, "*batch 3 128 128"]:
    """Convert images from [-1, 1] to [0, 1]."""
    return images.mul(0.5).add(0.5).clamp(0.0, 1.0)


def save_image_grid(images: Float[Tensor, "batch 3 128 128"], path: Path, nrow: int = 8) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(denormalize(images), path, nrow=nrow)
