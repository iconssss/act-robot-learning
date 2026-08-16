# Environment and smoke-test record

## Development roles

| Location | Role | GPU expectation |
| --- | --- | --- |
| Windows + WSL2 Ubuntu 22.04 | Git, code review, CPU schema/debug smoke tests | `torch.cuda.is_available() == False` is correct |
| `robot-cloud` Ubuntu 24.04 | CUDA, simulation, ACT training, rollout, evaluation | RTX 4090 / 24 GB |

## Version pins

| Component | Local WSL target | `robot-cloud` validated |
| --- | --- | --- |
| Python | 3.12.13 | 3.12.13 |
| LeRobot | 0.6.0 | 0.6.0 |
| PyTorch | 2.11.0 (Conda Forge CPU build) | 2.11.0+cu130 |
| TorchVision | 0.26.0 (CPU build) | 0.26.0 |
| Dataset | `lerobot/aloha_sim_transfer_cube_human` @ `6a43d500f101255823a9d2b9dc244eeb01a2cd31` | same |

The WSL environment is intentionally isolated at `/home/xinyue/miniforge3/envs/lerobot-act-cpu`; it does not alter Ubuntu's system Python 3.10 or Windows. The cloud environment is at `/root/shared-nvme/conda-envs/lerobot-act`.

WSL runtime artifacts remain in the Linux filesystem: the local dataset is
`/home/xinyue/datasets/aloha_sim_transfer_cube_human` (67 MB after video download),
and the Miniforge package cache is `/home/xinyue/miniforge3/pkgs`. The isolated
environment is about 2.2 GB. They are local debugging aids rather than a second
training copy of the project.

## Network decision record

- Direct `huggingface.co` access times out from both WSL and `robot-cloud`; public dataset traffic uses a command-scoped `HF_ENDPOINT=https://hf-mirror.com` fallback when needed. With the current `huggingface_hub`, the same command also sets `HF_HUB_DISABLE_XET=1`: otherwise the Xet/CAS download path bypasses the mirror and returns HTTP 401. No system proxy has been added.
- Direct large GitHub release downloads were extremely slow in WSL/Windows. Miniforge was obtained via USTC's mirror for the official `conda-forge/miniforge` release. The file length matched the official release response. GitHub's API was rate limited while querying the release digest, so an upstream digest comparison was not available; this is recorded rather than hidden.
- Conda package installation uses USTC's conda-forge mirror per command, without persistent global channel edits.

## Phase-5 acceptance criteria

The bounded smoke test must show, for 1--3 batches:

1. The cloud check uses the official `chunk_size=100`, so the action window has shape `(B, 100, 14)` and `action_is_pad` has shape `(B, 100)`. The local CPU check deliberately uses `chunk_size=10` solely to keep the no-GPU debugging run bounded.
2. The DataLoader produces RGB `(B, 3, 480, 640)` and state `(B, 14)`.
3. ACT initializes and its training `forward()` returns a finite masked-L1/KL loss.
4. Inference returns an action chunk `(B, 100, 14)` and `select_action()` returns one `(B, 14)` action.

## Cloud smoke-test result

Completed on `robot-cloud` on 2026-08-16, batch size 1, one batch, no backward or optimizer step:

```text
ACT parameters:              51,613,582
RGB batch:                   (1, 3, 480, 640)
state batch:                 (1, 14)
future action target:        (1, 100, 14)
action padding mask:         (1, 100)
predicted action chunk:      (1, 100, 14)
select_action output:        (1, 14)
```

The random-initialized policy had a finite loss (`l1_loss=0.7818`, `kld_loss=7.1949`); this is a wiring check, not a quality metric. The JSON report is persistent at `/root/shared-nvme/results/phase05_act_smoke_gpu.json`.

## Local WSL CPU smoke-test result

Completed on 2026-08-16 using one batch, `batch_size=1`, `chunk_size=10`, no
backward or optimizer step:

```text
ACT parameters:              51,567,502
RGB batch:                   (1, 3, 480, 640)
state batch:                 (1, 14)
future action target:        (1, 10, 14)
action padding mask:         (1, 10)
predicted action chunk:      (1, 10, 14)
select_action output:        (1, 14)
torch.cuda.is_available():   False
```

The random-initialized policy returned finite `l1_loss=1.0987` and
`kld_loss=7.4409`. This verifies the local dataset schema, video decoding,
DataLoader, preprocessing, ACT training forward, and inference queue without
claiming any policy quality. The local report is
`/home/xinyue/datasets/phase05_act_smoke_cpu.json`; the cloud report remains
the reproducible reference for the official 100-step chunk.
