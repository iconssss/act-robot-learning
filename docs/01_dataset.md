# Dataset: ALOHA simulated transfer cube

## Selected benchmark

- Dataset: `lerobot/aloha_sim_transfer_cube_human`
- Dataset revision: `6a43d500f101255823a9d2b9dc244eeb01a2cd31`
- Simulation task: `AlohaTransferCube-v0`
- LeRobot version: `0.6.0`
- Environment package: `gym-aloha 0.1.4`

The dataset is a compact supervised imitation-learning benchmark: 50 human demonstration episodes, 20,000 frames in total, and a 50 Hz control rate. Each episode has 400 frames (8 seconds). The task is: **the right arm picks up a cube and transfers it to the left arm**.

The inspected pinned revision occupies 67 MB on the cloud shared volume: Parquet frame data, one AV1 MP4 top-camera file, and metadata under `meta/`. The metadata field `files_size_in_mb: 500` is retained by the dataset but ignored by LeRobot 0.6.0; it is a metadata-version compatibility warning, not a decoding failure. The dataset must not be stored only on the cloud container's transient system disk.

## What one supervised sample means

At a control time `t`, an observation contains what the policy can observe:

- `observation.images.top`: a 480 x 640 RGB top-camera frame;
- `observation.state`: the robot's 14-dimensional proprioceptive joint state.

The recorded `action` is the 14-dimensional target command issued at that timestep. During ACT training it is not treated as an unrelated image label: the policy receives the current observation (and any configured history) and is trained against a *future sequence* of such actions. The exact future horizon is an ACT configuration choice, covered in the architecture stage.

## State and action dimensions (verified from `gym_aloha` source)

Both vectors use this order:

| Indices | Meaning |
| --- | --- |
| 0-5 | left arm: waist, shoulder, elbow, forearm roll, wrist angle, wrist rotate joint positions |
| 6 | left gripper opening, normalized: 0 = closed, 1 = open |
| 7-12 | right arm: waist, shoulder, elbow, forearm roll, wrist angle, wrist rotate joint positions |
| 13 | right gripper opening, normalized: 0 = closed, 1 = open |

`observation.state` is the measured/current configuration (`qpos`). `action` has the same coordinate system but means the desired absolute joint positions for the controller to execute. Thus equal dimensionality does **not** mean they are the same semantic signal.

The simulator also internally exposes velocity and additional camera views, but this benchmark's selected LeRobot policy features determine what ACT is actually allowed to consume. `scripts/inspect_dataset.py` prints the authoritative feature schema after download rather than assuming it.

## Reproducible inspection command

Run this on **robot-cloud**, after Hugging Face connectivity has been enabled:

```bash
/root/shared-nvme/conda-envs/lerobot-act/bin/python scripts/inspect_dataset.py \
  --repo-id lerobot/aloha_sim_transfer_cube_human \
  --root /root/shared-nvme/datasets/aloha_sim_transfer_cube_human
```

The script prints episode/frame counts, FPS, every observation/action feature, task table, episode-0 metadata, stored dataset statistics, and the decoded sample shapes.

## Current external dependency

The `robot-cloud` container reaches PyPI but cannot open HTTPS connections to `huggingface.co` through either IPv4 or IPv6. This is a cloud-network policy issue, not a LeRobot installation problem. For this public, pinned dataset only, the one-shot download used `HF_ENDPOINT=https://hf-mirror.com`; this was a command-scoped transport setting and did not modify the system or persist credentials. Future model/dataset downloads should prefer the platform's official outbound-network solution; if a mirror is necessary, its use and the pinned upstream revision must be recorded.

## Verified inspection result (2026-08-15)

The full inspection script decoded dataset index 0 successfully on `robot-cloud`:

```text
observation.images.top: torch.float32, shape (3, 480, 640)
observation.state:      torch.float32, shape (14,)
action:                 torch.float32, shape (14,)
episode_index:          torch.int64 scalar
frame_index:            torch.int64 scalar
timestamp:              torch.float32 scalar
next.done:              torch.bool scalar
task_index:             torch.int64 scalar
```

The image is returned as channel-first (`C, H, W`) and normalized to float by the LeRobot reader. The persisted episode-0 record has `dataset_from_index=0`, `dataset_to_index=400`, and a video timestamp range of 0.0 to 8.0 seconds.

## Why this is not ordinary image classification

In image classification, an image has an independent class label. Here, a frame is one point in a dynamical control trajectory: an action changes the robot, the robot changes the next image/state, and a small action error can make later observations leave the demonstration distribution. The visualizer therefore shows image, measured state, and target action **over time**. In the ACT stage, the supervision target becomes a future action sequence rather than one unrelated label.
