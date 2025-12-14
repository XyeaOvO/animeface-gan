#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from anime_gan.data.datamodule import AnimeFaceDataModule
from anime_gan.lit.dcgan_module import DCGANModule
from anime_gan.utils.metrics import compute_fid_is
from anime_gan.utils.paths import resolve_path
from anime_gan.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate DCGAN with FID/IS")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to Lightning checkpoint"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/eval",
        help="Directory to store evaluation artifacts",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/animeface128", help="AnimeFace128 data root"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50000,
        help="Number of generated samples for FID/IS",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for generation and fidelity computation",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for generation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    checkpoint_path = Path(args.checkpoint)
    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = resolve_path(args.data_dir)

    datamodule = AnimeFaceDataModule(
        data_dir=str(data_dir), batch_size=args.batch_size, num_workers=4, pin_memory=True
    )
    datamodule.prepare_data()

    device = torch.device(args.device)
    module = DCGANModule.load_from_checkpoint(str(checkpoint_path), map_location=device)
    module.eval()

    metrics = compute_fid_is(
        generator=module.generator.to(device),
        real_dir=data_dir,
        z_dim=module.hparams.z_dim,
        sample_size=args.num_samples,
        batch_size=args.batch_size,
        device=device,
        work_dir=output_dir,
    )

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"FID/IS computed and saved to {metrics_path}")
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
