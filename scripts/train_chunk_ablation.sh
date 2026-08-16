#!/usr/bin/env bash
# Run on robot-cloud: bash scripts/train_chunk_ablation.sh {short|long}
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 {short|long}" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
case "$1" in
  short)
    export CONFIG_PATH="${PROJECT_ROOT}/configs/chunk_ablation/act_aloha_chunk_50.yaml"
    export RUN_DIR="/root/shared-nvme/act-robot-learning/experiments/chunk_ablation/act_aloha_chunk_50_seed1000"
    ;;
  long)
    export CONFIG_PATH="${PROJECT_ROOT}/configs/chunk_ablation/act_aloha_chunk_150.yaml"
    export RUN_DIR="/root/shared-nvme/act-robot-learning/experiments/chunk_ablation/act_aloha_chunk_150_seed1000"
    ;;
  *)
    echo "Unknown configuration: $1 (expected short or long)" >&2
    exit 2
    ;;
esac
export LOG_PATH="${RUN_DIR}.console.log"
exec bash "${PROJECT_ROOT}/scripts/train_baseline.sh"
