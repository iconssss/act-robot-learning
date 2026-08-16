#!/usr/bin/env bash
# Run on robot-cloud only. The default is the fixed ACT baseline config.
# Extra CLI arguments override YAML fields, e.g. `--steps=100` for profiling.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PYTHON="${ENV_PYTHON:-/root/shared-nvme/conda-envs/lerobot-act/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/baseline/act_aloha_transfer_cube.yaml}"
RUN_DIR="${RUN_DIR:-/root/shared-nvme/act-robot-learning/experiments/baseline/act_aloha_transfer_cube_seed1000}"
LOG_PATH="${LOG_PATH:-${RUN_DIR}.console.log}"

if [[ ! -x "${ENV_PYTHON}" ]]; then
  echo "Missing cloud Python environment: ${ENV_PYTHON}" >&2
  exit 1
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing baseline config: ${CONFIG_PATH}" >&2
  exit 1
fi

# LeRobot intentionally refuses to overwrite a pre-existing output_dir.  Keep
# the streamed terminal log beside the run directory so the trainer remains
# the sole creator of its resolved config/checkpoints/metrics directory.
mkdir -p "$(dirname "${LOG_PATH}")" /root/shared-nvme/hf-cache /root/shared-nvme/torch-hub
export HF_HOME="${HF_HOME:-/root/shared-nvme/hf-cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TORCH_HOME="${TORCH_HOME:-/root/shared-nvme/torch-hub}"
# gym-aloha renders camera observations through dm_control during evaluation.
# This cloud container is headless, so explicitly select the GPU-backed EGL
# platform instead of the display-dependent GLFW default.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "Using config: ${CONFIG_PATH}"
echo "Writing outputs: ${RUN_DIR}"
echo "Writing console log: ${LOG_PATH}"
echo "MuJoCo render backend: ${MUJOCO_GL}"
echo "Command overrides: $*"

"${ENV_PYTHON%/python}/lerobot-train" --config_path "${CONFIG_PATH}" "$@" 2>&1 \
  | tee "${LOG_PATH}"
