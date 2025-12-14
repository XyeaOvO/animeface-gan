from __future__ import annotations

from dataclasses import dataclass

from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch.nn.utils import spectral_norm
from typeguard import check_type, typechecked


@dataclass
class NoiseBatch:
    z: Float[Tensor, "batch z_dim"]


class Generator(nn.Module):
    def __init__(self, z_dim: int = 128, feature_maps: int = 128) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.feature_maps = feature_maps

        self.net = nn.Sequential(
            nn.ConvTranspose2d(
                z_dim, feature_maps * 16, kernel_size=4, stride=1, padding=0, bias=False
            ),
            nn.BatchNorm2d(feature_maps * 16),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                feature_maps * 16, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                feature_maps, 3, kernel_size=4, stride=2, padding=1, bias=False
            ),
            nn.Tanh(),
        )

    @typechecked
    def forward(self, noise: Tensor) -> Tensor:
        check_type(noise, Float[Tensor, "batch z_dim"])
        noise_bchw = rearrange(noise, "b c -> b c 1 1")
        generated = self.net(noise_bchw)
        check_type(generated, Float[Tensor, "batch 3 128 128"])
        return generated


class Discriminator(nn.Module):
    def __init__(self, feature_maps: int = 128) -> None:
        super().__init__()
        self.feature_maps = feature_maps
        self.net = nn.Sequential(
            spectral_norm(
                nn.Conv2d(3, feature_maps, kernel_size=4, stride=2, padding=1, bias=False)
            ),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False)),
        )

    @typechecked
    def forward(self, images: Tensor) -> Tensor:
        check_type(images, Float[Tensor, "batch 3 128 128"])
        logits_map = self.net(images)
        logits = logits_map.mean(dim=(1, 2, 3))
        check_type(logits, Float[Tensor, "batch"])
        return logits


def init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d | nn.ConvTranspose2d | nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None and module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
