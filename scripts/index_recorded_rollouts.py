#!/usr/bin/env python3
"""Build a success/failure index for LeRobot's recorded evaluation videos."""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path


MARKER = "Suite per_task aggregated:"
EPISODE_PATTERN = re.compile(r"eval_episode_(\d+)\.mp4$")
FIELDS = ["condition", "step", "episode_index", "success", "sum_reward", "max_reward", "video_path"]


def parse_run(condition: str, log_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if MARKER not in line:
            continue
        try:
            task_records = ast.literal_eval(line.split(MARKER, maxsplit=1)[1].strip())
        except (SyntaxError, ValueError):
            continue
        for task_record in task_records:
            metrics = task_record["metrics"]
            for video_path in metrics.get("video_paths", []):
                match = EPISODE_PATTERN.search(video_path)
                if not match:
                    continue
                episode_index = int(match.group(1))
                rows.append(
                    {
                        "condition": condition,
                        "step": re.search(r"videos_step_(\d+)", video_path).group(1),
                        "episode_index": str(episode_index),
                        "success": str(bool(metrics["successes"][episode_index])).lower(),
                        "sum_reward": str(metrics["sum_rewards"][episode_index]),
                        "max_reward": str(metrics["max_rewards"][episode_index]),
                        "video_path": video_path,
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=LOG_PATH",
        help="Repeat for each run, e.g. baseline=/path/train.log",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for item in args.run:
        if "=" not in item:
            parser.error(f"Invalid --run value: {item}")
        condition, raw_path = item.split("=", maxsplit=1)
        rows.extend(parse_run(condition, Path(raw_path)))

    if not rows:
        raise SystemExit("No recorded rollout videos found in supplied logs")
    rows.sort(key=lambda row: (row["condition"], int(row["step"]), int(row["episode_index"])))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(row["success"] == "false" for row in rows)
    print(f"Indexed {len(rows)} recorded videos ({failures} failures) in {args.output_csv}")


if __name__ == "__main__":
    main()
