#!/usr/bin/env python3
"""Extract LeRobot evaluation milestones from a trainer console log."""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path

RESULT_MARKER = "Suite overall aggregated:"
STEP_PATTERN = re.compile(r"videos_step_(\d+)")
FIELDS = ["step", "n_episodes", "success_rate_pct", "avg_sum_reward", "avg_max_reward", "eval_seconds"]


def parse_log(log_path: Path) -> list[dict[str, str]]:
    """Return complete evaluation records; never invent interrupted metrics."""
    records: dict[int, dict[str, str]] = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if RESULT_MARKER not in line:
            continue
        try:
            metrics = ast.literal_eval(line.split(RESULT_MARKER, maxsplit=1)[1].strip())
        except (SyntaxError, ValueError):
            continue
        match = STEP_PATTERN.search(" ".join(metrics.get("video_paths", [])))
        if not match:
            continue
        step = int(match.group(1))
        records[step] = {
            "step": str(step),
            "n_episodes": str(metrics["n_episodes"]),
            "success_rate_pct": f"{float(metrics['pc_success']):.1f}",
            "avg_sum_reward": f"{float(metrics['avg_sum_reward']):.2f}",
            "avg_max_reward": f"{float(metrics['avg_max_reward']):.2f}",
            "eval_seconds": f"{float(metrics['eval_s']):.2f}",
        }
    return [records[step] for step in sorted(records)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()
    rows = parse_log(args.log)
    if not rows:
        raise SystemExit(f"No complete evaluation records found in {args.log}")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} evaluation milestones to {args.output_csv}")


if __name__ == "__main__":
    main()
