#!/usr/bin/env python3
"""Inspect a LeRobot dataset without training a policy.

Example (run on robot-cloud after the dataset can be reached)::

    python scripts/inspect_dataset.py \
        --repo-id lerobot/aloha_sim_transfer_cube_human \
        --root /root/shared-nvme/datasets/aloha_sim_transfer_cube_human

The script deliberately reports metadata first, then reads only one frame.  It is
therefore useful both for checking a newly downloaded dataset and for explaining
what ACT receives at a single control timestep.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


DEFAULT_REPO_ID = "lerobot/aloha_sim_transfer_cube_human"


def shape_of(value: Any) -> str:
    """Return a compact shape/type description without assuming a tensor."""
    if isinstance(value, torch.Tensor):
        return f"shape = {tuple(value.shape)}, dtype = {value.dtype}"
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return f"shape = {tuple(shape)}, dtype = {dtype}"
    return f"type = {type(value).__name__}, value = {value!r}"


def tensor_summary(value: Any) -> str:
    """Summarize a stored statistic while keeping output readable."""
    if not isinstance(value, torch.Tensor):
        return repr(value)
    flat = value.detach().cpu().reshape(-1)
    if flat.numel() <= 8:
        return repr(flat.tolist())
    return f"tensor(shape={tuple(value.shape)}, min={flat.min().item():.4g}, max={flat.max().item():.4g})"


def print_feature_schema(features: Mapping[str, dict[str, Any]]) -> None:
    print("\nFeature schema")
    print("-" * 72)
    for key, spec in features.items():
        dtype = spec.get("dtype")
        shape = spec.get("shape")
        names = spec.get("names")
        print(f"{key}: dtype={dtype}, shape={shape}, names={names}")


def print_statistics(stats: Mapping[str, Mapping[str, Any]] | None) -> None:
    print("\nDataset statistics")
    print("-" * 72)
    if not stats:
        print("No aggregate statistics were found in metadata.")
        return
    for feature_name, values in stats.items():
        rendered = ", ".join(f"{name}={tensor_summary(value)}" for name, value in values.items())
        print(f"{feature_name}: {rendered}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Persistent local cache/dataset directory. Omit to use LeRobot's default cache.",
    )
    parser.add_argument("--revision", default=None, help="Optional immutable Hugging Face revision/commit hash.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Do not decode a sample frame. Useful for checking metadata and cache paths first.",
    )
    args = parser.parse_args()

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        revision=args.revision,
        download_videos=not args.metadata_only,
    )

    print("LeRobot dataset inspection")
    print("=" * 72)
    print(f"repo_id: {dataset.repo_id}")
    print(f"revision: {dataset.revision}")
    print(f"local root: {dataset.root}")
    print(f"episodes: {dataset.num_episodes}")
    print(f"frames: {dataset.num_frames}")
    print(f"fps: {dataset.fps}")
    print_feature_schema(dataset.features)

    print("\nTasks")
    print("-" * 72)
    if dataset.meta.tasks is None or len(dataset.meta.tasks) == 0:
        print("No task table found.")
    else:
        print(dataset.meta.tasks.to_string())

    print("\nEpisode 0 metadata")
    print("-" * 72)
    if dataset.meta.episodes is None or len(dataset.meta.episodes) == 0:
        print("No episode table found.")
    else:
        # In LeRobot 0.6.0 this is a Hugging Face ``datasets.Dataset`` rather
        # than a pandas DataFrame, so positional access is ``[0]`` (not
        # ``.iloc[0]``).
        episode_zero = dataset.meta.episodes[0]
        for key, value in episode_zero.items():
            print(f"{key}: {value}")

    print_statistics(dataset.meta.stats)

    if args.metadata_only:
        return

    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"--sample-index must be in [0, {len(dataset) - 1}], got {args.sample_index}.")

    sample = dataset[args.sample_index]
    print(f"\nSample at dataset index {args.sample_index}")
    print("-" * 72)
    for key, value in sample.items():
        print(f"{key}: {shape_of(value)}")

    for expected_key in ("observation.images.top", "observation.state", "action"):
        if expected_key in sample:
            print(f"\n{expected_key}:\n{shape_of(sample[expected_key])}")


if __name__ == "__main__":
    main()
