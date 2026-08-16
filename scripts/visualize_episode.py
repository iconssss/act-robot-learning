#!/usr/bin/env python3
"""Visualize one complete LeRobot demonstration episode.

Run on robot-cloud::

    python scripts/visualize_episode.py \
      --root /root/shared-nvme/datasets/aloha_sim_transfer_cube_human \
      --revision 6a43d500f101255823a9d2b9dc244eeb01a2cd31 \
      --episode-index 0 \
      --output-dir /root/shared-nvme/results/phase03_episode0 \
      --export-gif
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # The cloud container has no interactive display.
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402


DEFAULT_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
DEFAULT_REVISION = "6a43d500f101255823a9d2b9dc244eeb01a2cd31"
DEFAULT_FFMPEG = "/root/shared-nvme/conda-envs/lerobot-act/bin/ffmpeg"


def to_hwc_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert LeRobot's normalized CHW float image to an HWC uint8 image."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected CHW RGB image, got shape {tuple(image.shape)}.")
    return (
        image.detach().cpu().clamp(0, 1).permute(1, 2, 0).mul(255).round().to(torch.uint8).numpy()
    )


def save_contact_sheet(dataset: LeRobotDataset, output_path: Path) -> list[dict[str, Any]]:
    """Save first/middle/last RGB frames and return their scalar metadata."""
    indices = [0, len(dataset) // 2, len(dataset) - 1]
    figure, axes = plt.subplots(1, len(indices), figsize=(15, 4.5), constrained_layout=True)
    frame_info: list[dict[str, Any]] = []
    for axis, local_index in zip(axes, indices, strict=True):
        sample = dataset[local_index]
        axis.imshow(to_hwc_uint8(sample["observation.images.top"]))
        axis.axis("off")
        axis.set_title(
            f"local frame {local_index}\nepisode={sample['episode_index'].item()}, "
            f"t={sample['timestamp'].item():.2f}s"
        )
        frame_info.append(
            {
                "local_frame_index": local_index,
                "episode_index": int(sample["episode_index"].item()),
                "frame_index": int(sample["frame_index"].item()),
                "timestamp_s": float(sample["timestamp"].item()),
            }
        )
    figure.suptitle("ALOHA transfer-cube demonstration: top RGB camera", fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return frame_info


def save_trajectory_plot(
    timestamps: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    motor_names: list[str],
    output_path: Path,
) -> None:
    """Render aligned measured-state and target-action curves for every motor."""
    if states.shape != actions.shape:
        raise ValueError(f"State/action shape mismatch: {states.shape} vs {actions.shape}.")
    if states.shape[1] != len(motor_names):
        raise ValueError("Feature motor names do not match state dimension.")
    figure, axes = plt.subplots(len(motor_names), 1, figsize=(13, 23), sharex=True, constrained_layout=True)
    for dim, (axis, motor_name) in enumerate(zip(axes, motor_names, strict=True)):
        axis.plot(timestamps, states[:, dim], label="observation.state", linewidth=1.2)
        axis.plot(timestamps, actions[:, dim], label="action target", linewidth=1.0, alpha=0.8)
        axis.set_ylabel(motor_name, fontsize=8)
        axis.grid(alpha=0.25)
        if dim == 0:
            axis.legend(loc="upper right", ncols=2)
    axes[-1].set_xlabel("episode time (s)")
    figure.suptitle("Measured robot state and commanded action for one demonstration", fontsize=14)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def export_episode_mp4(
    dataset_root: Path,
    episode_record: dict[str, Any],
    output_path: Path,
    ffmpeg_binary: Path,
) -> None:
    """Trim the shared source MP4 to this episode's timestamp interval."""
    video_key = "observation.images.top"
    chunk = int(episode_record[f"videos/{video_key}/chunk_index"])
    file_index = int(episode_record[f"videos/{video_key}/file_index"])
    source = dataset_root / "videos" / video_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
    if not source.is_file():
        raise FileNotFoundError(f"Expected local source video at {source}.")
    if not ffmpeg_binary.is_file():
        resolved = shutil.which("ffmpeg")
        if resolved is None:
            raise FileNotFoundError(f"ffmpeg was not found at {ffmpeg_binary} or on PATH.")
        ffmpeg_binary = Path(resolved)
    start = float(episode_record[f"videos/{video_key}/from_timestamp"])
    duration = float(episode_record[f"videos/{video_key}/to_timestamp"]) - start
    command = [
        str(ffmpeg_binary), "-y", "-ss", f"{start:.6f}", "-i", str(source), "-t", f"{duration:.6f}",
        "-an", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def export_gif(dataset: LeRobotDataset, output_path: Path, stride: int) -> None:
    """Create a compact GIF sampled from decoded RGB observations."""
    frames: list[np.ndarray] = []
    for local_index in range(0, len(dataset), stride):
        image = Image.fromarray(to_hwc_uint8(dataset[local_index]["observation.images.top"]))
        image.thumbnail((320, 240))
        frames.append(np.asarray(image))
    imageio.mimsave(output_path, frames, fps=max(1, round(dataset.fps / stride)), loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--episode-index", default=0, type=int)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ffmpeg", default=DEFAULT_FFMPEG, type=Path)
    parser.add_argument("--export-gif", action="store_true")
    parser.add_argument("--gif-stride", default=5, type=int, help="Keep every Nth frame for GIF export.")
    args = parser.parse_args()
    if args.episode_index < 0 or args.gif_stride <= 0:
        raise ValueError("episode-index must be non-negative and gif-stride must be positive.")

    dataset = LeRobotDataset(
        args.repo_id, root=args.root, episodes=[args.episode_index], revision=args.revision, download_videos=True
    )
    if dataset.num_episodes != 1:
        raise RuntimeError(f"Expected one selected episode, got {dataset.num_episodes}.")
    episode_record = dataset.meta.episodes[args.episode_index]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    states = np.asarray(dataset.hf_dataset["observation.state"], dtype=np.float32)
    actions = np.asarray(dataset.hf_dataset["action"], dtype=np.float32)
    timestamps = np.asarray(dataset.hf_dataset["timestamp"], dtype=np.float32)
    motor_names = dataset.features["observation.state"]["names"]["motors"]

    contact_sheet = args.output_dir / "rgb_contact_sheet.png"
    trajectories = args.output_dir / "state_action_trajectories.png"
    mp4_path = args.output_dir / "episode_top_camera.mp4"
    selected_frames = save_contact_sheet(dataset, contact_sheet)
    save_trajectory_plot(timestamps, states, actions, motor_names, trajectories)
    export_episode_mp4(args.root, episode_record, mp4_path, args.ffmpeg)

    gif_path: Path | None = None
    if args.export_gif:
        gif_path = args.output_dir / "episode_top_camera.gif"
        export_gif(dataset, gif_path, args.gif_stride)

    summary = {
        "repo_id": args.repo_id, "revision": args.revision, "episode_index": args.episode_index,
        "task": episode_record["tasks"], "frames": int(len(dataset)), "fps": int(dataset.fps),
        "duration_s": float(timestamps[-1] - timestamps[0] + 1 / dataset.fps),
        "state_shape": list(states.shape), "action_shape": list(actions.shape), "motor_names": motor_names,
        "selected_frames": selected_frames,
        "artifacts": {"contact_sheet": str(contact_sheet), "trajectories": str(trajectories),
                      "episode_mp4": str(mp4_path), "episode_gif": str(gif_path) if gif_path else None},
    }
    summary_path = args.output_dir / "episode_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"episode={args.episode_index}, task={episode_record['tasks']}")
    print(f"frames={len(dataset)}, fps={dataset.fps}, duration={summary['duration_s']:.2f}s")
    for name, artifact in summary["artifacts"].items():
        if artifact:
            print(f"{name}: {artifact}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
