# ACT baseline training

## Baseline definition

This is a controlled behavior-cloning baseline, not a new architecture. It
uses the pinned ALOHA transfer-cube demonstrations and LeRobot 0.6.0 ACT
defaults: a single top RGB camera, 14-D proprioceptive state, 14-D action,
ResNet-18, CVAE latent dimension 32, and a 100-step action chunk at 50 Hz.

The frozen executable configuration is
`configs/baseline/act_aloha_transfer_cube.yaml`. It has seed 1000,
`batch_size=8`, 100,000 optimisation steps, checkpoint and in-training
simulation evaluation every 20,000 steps, and 50 evaluation episodes per
milestone. Checkpoints and logs are deliberately written to the cloud shared
volume, not the container filesystem.

## Why this is a baseline

Only the model and data needed for the supported benchmark are used. The first
run keeps the current ACT default `chunk_size=100` and `n_action_steps=100`,
with temporal ensembling disabled. This makes the later chunk ablation a
single-variable experiment: it will vary the chunk/execution choice while
holding data, seed, architecture, batch size, steps and evaluation protocol
constant.

## Execution protocol

Run on **robot-cloud**, from the persistent cloud checkout:

```bash
bash scripts/train_baseline.sh
```

For a bounded throughput calibration only (not a reported baseline result):

```bash
bash scripts/train_baseline.sh --steps=100 --env_eval_freq=0 --save_freq=100
```

The script fixes `HF_HOME` and `TORCH_HOME` under `/root/shared-nvme`, uses the
command-scoped Hugging Face mirror, disables Xet for that mirror, and tees the
console stream to `train.log`. It does not use WSL or the Windows Python
installation.

## Cost gate

Before the 100,000-step run, record the measured calibration wall time
`t_100`. Estimate training-only GPU time as `1000 * t_100`; add the measured
simulation-evaluation time separately. Multiply the result by the platform's
RTX 4090 hourly voucher/RMB rate. Do not launch the full run if that estimate
exceeds the 200 RMB project cap without an explicit budget decision.

## Calibration result

On 2026-08-16, `robot-cloud` ran 100 real optimisation steps with the frozen
configuration, batch size 8, seed 1000 and no simulation evaluation. The
official `ResNet18_Weights.IMAGENET1K_V1` file (44.7 MB) was downloaded once to
`/root/shared-nvme/torch-hub` before the loop. The training loop completed in
about 12 seconds; the checkpointed end-to-end calibration process took about
26 seconds including setup. The last trainer record was `loss=10.768`,
`updt_s=0.079`, `smp/s=79`, and GPU memory `2.09 GB`.

For budget safety, use the conservative end-to-end throughput of roughly
8.5 steps/s: 100,000 steps are estimated at 3.3 GPU-hours before simulation
evaluation. With five planned 50-episode milestones and checkpoint I/O, reserve
3.5--4 GPU-hours. The successful calibration checkpoint is persistent at
`/root/shared-nvme/act-robot-learning/experiments/calibration/act_aloha_transfer_cube_seed1000_attempt3/checkpoints/000100`;
it occupies about 592 MB and is a systems check, not a reported policy result.

Two earlier calibration attempts are retained only as short console logs. They
exposed and fixed: (1) the current package's CLI is `lerobot-train`, not a
`lerobot.scripts.train` module; (2) LeRobot must create a fresh `output_dir`;
and (3) `policy.push_to_hub` must be explicitly false for this local-only run.

## Headless simulation requirement

The 4090 container has no desktop display. At the first full-run evaluation
(step 20,000), `gym-aloha` asks `dm_control` to render the top camera and the
GLFW default fails with `mujoco.FatalError: an OpenGL platform library has not
been loaded`. This is a **simulation rendering** issue, not an ACT loss,
dataset, CUDA, or checkpoint problem. The completed `020000` checkpoint is
preserved on shared storage. `scripts/train_baseline.sh` therefore exports
`MUJOCO_GL=egl` and `PYOPENGL_PLATFORM=egl`, selecting the headless,
GPU-backed EGL renderer. A 64x64 `dm_control` render test passed on the RTX
4090 before resuming. This changes only rendering backend selection during
simulation; it does not change policy weights, demonstrations, optimizer, or
the experimental configuration.

## Required run record

Each completed run must preserve its resolved `train_config.json`, command,
Git commit, GPU/driver, package versions, dataset revision, random seed,
steps, batch size, action-chunk parameters, checkpoints, loss log and
evaluation output. A low training loss alone is never reported as task
success; closed-loop rollout evaluation is the acceptance criterion.
