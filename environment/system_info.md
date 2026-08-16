# Cloud environment record

Validated on `robot-cloud` for the ACT + ALOHA simulation project.

| Field | Value |
| --- | --- |
| Platform | Parallel Intelligence Cloud container |
| OS image | Ubuntu 24.04 / PyTorch-25.03-py3 |
| GPU | NVIDIA GeForce RTX 4090, 24,564 MiB |
| GPU driver | 580.105.08 |
| Driver CUDA capability | 13.0 |
| Environment path | `/root/shared-nvme/conda-envs/lerobot-act` |
| Python | 3.12.13 |
| LeRobot | 0.6.0 (PyPI stable release) |
| PyTorch | 2.11.0+cu130 |
| TorchVision | 0.26.0 |
| TorchCodec | 0.11.1+cpu |
| MuJoCo | 3.8.1 |
| FFmpeg | 8.1.2 (Conda Forge) |
| Matplotlib | 3.11.1 (episode visualization only) |

## Storage

- Persistent, writable: `/root/shared-nvme` (50 GB; ~40 GB free after environment installation).
- Read-only shared volume: `/shared-public`.
- Container root filesystem is transient and must not hold unique project artifacts.

Set caches and outputs under `/root/shared-nvme` before downloading datasets or training.

## Installation decision log

1. The preinstalled `mamba` command cannot import the platform's Conda API. This is an image-level `mamba`/`conda` mismatch, not a LeRobot issue. We use `/base/mambaforge/bin/conda` only and leave `base` unchanged.
2. `ffmpeg` initially resolved to 9.0.1. `torchcodec==0.11.1` supports FFmpeg 4--8 and failed to load with FFmpeg 9's `libavutil.so.61` ABI.
3. Pinning Conda Forge `ffmpeg=8` supplies `libavutil.so.60`; `import torchcodec` now succeeds.
4. `matplotlib==3.11.1` was installed only in `lerobot-act` for phase-3 plots. `imageio` and Pillow were already present through existing dependencies.

## Verified dataset and phase-3 artifacts

- Dataset: `lerobot/aloha_sim_transfer_cube_human` at revision `6a43d500f101255823a9d2b9dc244eeb01a2cd31`.
- Persistent dataset root: `/root/shared-nvme/datasets/aloha_sim_transfer_cube_human` (67 MB after download).
- Episode-0 visualizations: `/root/shared-nvme/results/phase03_episode0/`.
- The exported MP4 is 8.0 seconds at 50 FPS; the local repository retains only the two PNG figures under `results/figures/`.

## Activation

```bash
conda activate /root/shared-nvme/conda-envs/lerobot-act
```

When `conda` has not been initialized in a shell:

```bash
source /base/mambaforge/etc/profile.d/conda.sh
conda activate /root/shared-nvme/conda-envs/lerobot-act
```
