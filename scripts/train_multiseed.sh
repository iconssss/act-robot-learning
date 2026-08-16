#!/usr/bin/env bash
# Run on robot-cloud: bash scripts/train_multiseed.sh {baseline|short} {1001|1002}
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 {baseline|short} {seed}" >&2
  exit 2
fi

CONDITION="$1"
SEED="$2"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
  echo "Seed must be a non-negative integer: $SEED" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$CONDITION" in
  baseline)
    export CONFIG_PATH="${PROJECT_ROOT}/configs/baseline/act_aloha_transfer_cube.yaml"
    ;;
  short)
    export CONFIG_PATH="${PROJECT_ROOT}/configs/chunk_ablation/act_aloha_chunk_50.yaml"
    ;;
  *)
    echo "Unknown condition: $CONDITION (expected baseline or short)" >&2
    exit 2
    ;;
esac

export RUN_DIR="/root/shared-nvme/act-robot-learning/experiments/multiseed/${CONDITION}_seed${SEED}"
export LOG_PATH="${RUN_DIR}.console.log"
if [[ -e "$RUN_DIR" ]]; then
  echo "Refusing to overwrite existing run: $RUN_DIR" >&2
  exit 2
fi

# Keep every non-seed field frozen in the original condition config. The CLI
# override creates a distinct run directory while preserving the config source.
exec bash "${PROJECT_ROOT}/scripts/train_baseline.sh" \
  --seed="$SEED" \
  --output_dir="$RUN_DIR" \
  --job_name="act_aloha_${CONDITION}_seed${SEED}"
