#!/usr/bin/env bash
set -euo pipefail

# Minimal smoke test on CPU: 1 epoch, few batches
uv run python -m anime_gan.train \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2 \
  trainer.limit_val_batches=0 \
  model.sample_grid_size=4 \
  dataset.batch_size=8 \
  logger.wandb.mode=offline \
  eval.enabled=false
