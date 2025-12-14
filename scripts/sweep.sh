#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-conf/sweeps/dcgan.yaml}
PROJECT=${PROJECT:-animeface-gan}
ENTITY=${ENTITY:-}

export WANDB_PROJECT=${PROJECT}
if [[ -n "${ENTITY}" ]]; then
  export WANDB_ENTITY=${ENTITY}
fi

# Create sweep and immediately start an agent
SWEEP_ID=$(wandb sweep -q "${CONFIG}")
echo "Created sweep: ${SWEEP_ID}"
wandb agent "${SWEEP_ID}"
