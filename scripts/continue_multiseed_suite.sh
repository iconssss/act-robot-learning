#!/usr/bin/env bash
# Continue after baseline seed 1001 has been launched. Run on robot-cloud only.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIRST_RUN="/root/shared-nvme/act-robot-learning/experiments/multiseed/baseline_seed1001"
FIRST_LOG="${FIRST_RUN}.console.log"

echo "Waiting for existing baseline seed 1001 run to finish..."
while ! grep -q "End of training" "$FIRST_LOG" 2>/dev/null; do
  if ! pgrep -f "lerobot-train.*baseline_seed1001" >/dev/null; then
    echo "baseline seed 1001 stopped before End of training; not starting later runs." >&2
    exit 1
  fi
  sleep 60
done
echo "baseline seed 1001 completed; starting remaining serial runs."

for job in "short 1001" "baseline 1002" "short 1002"; do
  read -r condition seed <<<"$job"
  echo "Starting ${condition} seed ${seed} at $(date -Is)"
  bash "${PROJECT_ROOT}/scripts/train_multiseed.sh" "$condition" "$seed"
  echo "Completed ${condition} seed ${seed} at $(date -Is)"
done

echo "All multiseed runs completed at $(date -Is)"
