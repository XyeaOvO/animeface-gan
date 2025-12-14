from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl
import torch
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from anime_gan.utils.paths import resolve_path


def _ensure_modelscope_patch() -> None:
    """Patch HuggingFace datasets for ModelScope compatibility."""

    import datasets
    from datasets.features import Sequence

    if not hasattr(datasets, "LargeList"):
        # ModelScope expects LargeList which was removed in newer datasets versions.
        class LargeList(Sequence):
            pass

        datasets.LargeList = LargeList  # type: ignore[attr-defined]
        try:
            import datasets.features as hf_features

            hf_features.LargeList = LargeList  # type: ignore[attr-defined]
        except Exception:
            # Optional best-effort patch
            pass


def _locate_cached_zip() -> Path | None:
    downloads_dir = Path.home() / ".cache" / "modelscope" / "hub" / "datasets" / "downloads"
    if not downloads_dir.exists():
        return None
    for meta_file in downloads_dir.glob("*.json"):
        try:
            info = json.loads(meta_file.read_text())
        except Exception:
            continue
        url = info.get("url", "")
        if "AnimeFace128.zip" in url:
            candidate = meta_file.with_suffix("")
            if candidate.exists():
                return candidate
    return None


def _download_animeface128(
    target_root: Path, subset_name: str = "default", split: str = "train"
) -> None:
    """Download AnimeFace128 images to the target directory if missing."""

    target_root.mkdir(parents=True, exist_ok=True)
    existing = list(target_root.glob("*.png")) + list(target_root.glob("*.jpg"))
    if len(existing) >= 50000:
        return
    if existing:
        shutil.rmtree(target_root)
        target_root.mkdir(parents=True, exist_ok=True)

    _ensure_modelscope_patch()
    from modelscope.msdatasets import MsDataset

    # Trigger download to cache if needed
    MsDataset.load("yanghaitao/AnimeFace128", subset_name=subset_name, split=split)
    zip_path = _locate_cached_zip()
    if zip_path is None:
        raise RuntimeError("AnimeFace128 archive not found in ModelScope cache.")

    shutil.unpack_archive(zip_path, target_root, format="zip")
    inner_dir = target_root / "AnimeFace128"
    if inner_dir.exists() and inner_dir.is_dir():
        for file in inner_dir.iterdir():
            shutil.move(str(file), target_root / file.name)
        inner_dir.rmdir()


class AnimeFaceDataset(Dataset[Float[Tensor, "3 128 128"]]):
    def __init__(
        self,
        root: Path,
        transform: Callable[[Image.Image], Float[Tensor, "3 128 128"]] | None = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.files = sorted(list(self.root.glob("*.png")) + list(self.root.glob("*.jpg")))
        if not self.files:
            raise RuntimeError(
                f"No images found in {self.root}. Please ensure the dataset is downloaded."
            )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Float[Tensor, "3 128 128"]:
        file_path = self.files[index]
        image = Image.open(file_path).convert("RGB")
        if self.transform:
            return self.transform(image)
        raise RuntimeError("Transform must be provided to AnimeFaceDataset")


@dataclass
class ImageBatch:
    images: Float[Tensor, "batch 3 128 128"]


def collate_image_batch(batch: list[Float[Tensor, "3 128 128"]]) -> ImageBatch:
    images_bchw = torch.stack(batch, dim=0)
    return ImageBatch(images=images_bchw)


class AnimeFaceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/animeface128",
        batch_size: int = 64,
        num_workers: int = 4,
        pin_memory: bool = True,
        subset_name: str = "default",
        split: str = "train",
        image_size: int = 128,
        eval_subset: int | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.data_dir = resolve_path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.subset_name = subset_name
        self.split = split
        self.image_size = image_size
        self.eval_subset = eval_subset
        self._dataset: AnimeFaceDataset | None = None

    def prepare_data(self) -> None:
        _download_animeface128(self.data_dir, subset_name=self.subset_name, split=self.split)

    def setup(self, stage: str | None = None) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = AnimeFaceDataset(self.data_dir, transform=transform)

        if self.eval_subset is not None:
            dataset_for_eval = Subset(dataset, range(min(self.eval_subset, len(dataset))))
        else:
            dataset_for_eval = dataset

        self._dataset = dataset
        self.train_set = dataset
        self.eval_set = dataset_for_eval

    def train_dataloader(self) -> DataLoader[ImageBatch]:
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_image_batch,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader[ImageBatch]:
        # Reuse eval_set for potential callbacks/validation
        return DataLoader(
            self.eval_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_image_batch,
            drop_last=False,
        )

    def build_eval_dataloader(
        self, sample_size: int, batch_size: int | None = None
    ) -> DataLoader[ImageBatch]:
        batch_size = batch_size or self.batch_size
        if self._dataset is None:
            raise RuntimeError("DataModule.setup must be called before build_eval_dataloader")
        subset = Subset(self._dataset, range(min(sample_size, len(self._dataset))))
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_image_batch,
            drop_last=False,
        )
