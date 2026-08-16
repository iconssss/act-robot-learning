# ACT Robot Learning

An experimental robotics-learning project that studies visual imitation learning and closed-loop simulation control with LeRobot and ACT (Action Chunking with Transformers).

## Goal

Build and explain the complete learning loop:

`robot observation -> LeRobotDataset -> ACT action chunk -> simulation rollout -> quantitative evaluation -> ablation`

The first benchmark is planned as the official ALOHA simulated transfer-cube task:

- Dataset: `lerobot/aloha_sim_transfer_cube_human`
- Environment: `AlohaTransferCube-v0`
- Policy: ACT

The dataset/environment/policy combination is currently covered by LeRobot's source tests. The implementation is pinned to LeRobot 0.6.0; the environment record and exact dataset revision are in `environment/system_info.md` and `docs/00_environment.md`.

## Development topology

| Location | Responsibility | Persistent location |
| --- | --- | --- |
| Windows (`D:\\600-Robot\\300-Project\\100-Project01`) | Project coordination, Git working copy, documentation, Codex | local disk + GitHub once connected |
| Local WSL2 | Lightweight CPU inspection, package/schema debugging, reproducibility checks | `~/projects/act-robot-learning` when the Linux working copy is created |
| `robot-cloud` RTX 4090 | CUDA, MuJoCo simulation, training, rollout, evaluation | `/root/shared-nvme/act-robot-learning` |

Do not treat a container filesystem as the only copy of code or results. Git/GitHub is the code source of truth; cloud shared storage holds durable runtime artifacts.

## Cloud storage policy

- `/root/shared-nvme`: write project artifacts here (50 GB persistent shared volume).
- `/shared-public`: read-only; never write checkpoints or results here.
- Container `/`: temporary (30 GB); do not keep unique data or checkpoints here.

Initial cloud layout, to create only when GPU setup begins:

```text
/root/shared-nvme/act-robot-learning/  # clone of this repository
/root/shared-nvme/hf-cache/            # Hugging Face datasets/models
/root/shared-nvme/checkpoints/          # retained checkpoints
/root/shared-nvme/results/              # logs, tables, figures, rollout videos
```

## Current status

- [x] WSL2 and local resource audit
- [x] `robot-cloud` direct key-based SSH path and RTX 4090 CUDA validation
- [x] Cloud persistent-storage paths verified
- [x] LeRobot 0.6.0 environment and version pin
- [x] Dataset inspection (pinned ALOHA revision; one decoded sample)
- [x] Episode visualization (RGB, state/action trajectories, MP4 and GIF)
- [x] ACT source-level architecture and action-chunk data-flow analysis
- [x] Cloud ACT smoke test (1 batch; dataset → forward/loss → action chunk)
- [x] Local WSL CPU reproduction of the smoke test (1 bounded CPU batch)
- [x] GPU ACT training calibration (100 optimisation steps; throughput and storage measured)
- [x] Full GPU ACT baseline (100k steps) and 50-episode closed-loop evaluation
- [x] Baseline rollout videos and quantitative evaluation table
- [ ] Action-chunk ablation and failure analysis

## Baseline result

The 100k-step ACT baseline achieves **70.0% success (35/50 episodes)** in
closed-loop `AlohaTransferCube-v0` evaluation. Intermediate checkpoints were
evaluated under the same protocol; see `docs/04_rollout.md` and
`results/tables/baseline_metrics.csv`. Training loss is not used as the task
success claim: the reported metric comes from simulation rollouts.

## Security note

The cloud direct SSH endpoint is key-authenticated. Private keys remain only on the Windows host and are never committed to this repository.
