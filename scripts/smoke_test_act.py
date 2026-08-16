#!/usr/bin/env python3
"""Run a bounded ACT dataset/DataLoader/forward/loss smoke test.

This is deliberately not a training script: it executes only the requested
number of batches, creates no optimizer, and writes no checkpoint. Use GPU for
the currently available cloud validation, then rerun with ``--device cpu`` once
the local WSL Python 3.12 environment is installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.processor_act import make_act_pre_post_processors


REPO_ID = "lerobot/aloha_sim_transfer_cube_human"
REVISION = "6a43d500f101255823a9d2b9dc244eeb01a2cd31"
IMAGE_KEY = "observation.images.top"
STATE_KEY = "observation.state"


def make_config(device: str, chunk_size: int) -> ACTConfig:
    """Create the ALOHA-compatible ACT shape config without downloading weights."""
    return ACTConfig(
        device=device,
        input_features={
            IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        },
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(14,))},
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        # A random backbone is sufficient for a schema/forward smoke test and
        # prevents a hidden network download of ImageNet weights.
        pretrained_backbone_weights=None,
    )


def summarize_batch(batch: dict[str, object]) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            summary[key] = {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-batches", type=int, default=1, choices=(1, 2, 3))
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but torch.cuda.is_available() is False.")
    if args.batch_size <= 0 or args.chunk_size <= 0:
        raise ValueError("batch-size and chunk-size must be positive.")

    # ACTION at t ... t+(chunk_size-1), expressed in seconds at the data FPS.
    metadata_dataset = LeRobotDataset(REPO_ID, root=args.root, revision=REVISION, download_videos=False)
    action_offsets_s = [step / metadata_dataset.fps for step in range(args.chunk_size)]
    dataset = LeRobotDataset(
        REPO_ID,
        root=args.root,
        revision=REVISION,
        delta_timestamps={"action": action_offsets_s},
        return_uint8=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    config = make_config(args.device, args.chunk_size)
    policy = ACTPolicy(config)
    # The full training entrypoint delegates this placement to Accelerate. This
    # standalone smoke test owns device placement explicitly because its
    # preprocessor already moves batches to ``config.device``.
    policy.to(args.device)
    preprocessor, postprocessor = make_act_pre_post_processors(config, dataset.meta.stats)
    parameter_count = sum(parameter.numel() for parameter in policy.parameters())

    reports: list[dict[str, object]] = []
    for batch_index, raw_batch in enumerate(loader):
        if batch_index >= args.num_batches:
            break

        # This matches the current LeRobot training loop: dataset video frames
        # arrive as uint8 and are first scaled to [0, 1].
        for camera_key in dataset.meta.camera_keys:
            if camera_key in raw_batch and raw_batch[camera_key].dtype == torch.uint8:
                raw_batch[camera_key] = raw_batch[camera_key].to(torch.float32) / 255.0

        batch = preprocessor(raw_batch)
        policy.train()
        loss, loss_dict = policy(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss on batch {batch_index}: {loss.item()}")

        # Verify the inference-only path as well. It predicts an entire action
        # chunk, while select_action exposes only the next queued action.
        policy.reset()
        action_chunk = policy.predict_action_chunk(batch)
        selected_normalized_action = policy.select_action(batch)
        selected_action = postprocessor(selected_normalized_action)

        reports.append(
            {
                "batch_index": batch_index,
                "raw_batch": summarize_batch(raw_batch),
                "model_batch": summarize_batch(batch),
                "loss": float(loss.item()),
                "loss_dict": loss_dict,
                "action_chunk_shape": list(action_chunk.shape),
                "selected_normalized_action_shape": list(selected_normalized_action.shape),
                "selected_action_type": type(selected_action).__name__,
            }
        )

    report = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "device": args.device,
        "torch_cuda_available": torch.cuda.is_available(),
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "chunk_size": args.chunk_size,
        "policy_parameter_count": parameter_count,
        "reports": reports,
    }
    print(json.dumps(report, indent=2))
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report_written={args.report}")


if __name__ == "__main__":
    main()
