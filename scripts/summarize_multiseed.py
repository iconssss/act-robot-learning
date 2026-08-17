#!/usr/bin/env python3
"""Summarize final LeRobot evaluation metrics across independent training seeds."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from summarize_results import parse_log


RUN_FIELDS = ["condition", "seed", "step", "n_episodes", "success_rate_pct", "avg_sum_reward", "avg_max_reward"]
SUMMARY_FIELDS = ["condition", "n_seeds", "mean_success_rate_pct", "sample_std_success_rate_pct", "min_success_rate_pct", "max_success_rate_pct", "mean_avg_sum_reward"]


def split_label(label: str) -> tuple[str, str]:
    condition, separator, seed = label.rpartition("_seed")
    if not separator or not condition or not seed.isdigit():
        raise ValueError(f"Run label must use CONDITION_seedINTEGER, got: {label}")
    return condition, seed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, metavar="CONDITION_seedN=LOG_PATH")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    for item in args.run:
        if "=" not in item:
            parser.error(f"Invalid --run value: {item}")
        label, raw_path = item.split("=", maxsplit=1)
        condition, seed = split_label(label)
        milestones = parse_log(Path(raw_path))
        if not milestones:
            raise SystemExit(f"No complete evaluation in {raw_path}")
        final = milestones[-1]
        rows.append({"condition": condition, "seed": seed, **{key: final[key] for key in RUN_FIELDS[2:]}})

    rows.sort(key=lambda row: (row["condition"], int(row["seed"])))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    summary_rows: list[dict[str, str]] = []
    for condition, group in sorted(grouped.items()):
        rates = [float(row["success_rate_pct"]) for row in group]
        returns = [float(row["avg_sum_reward"]) for row in group]
        summary_rows.append(
            {
                "condition": condition,
                "n_seeds": str(len(group)),
                "mean_success_rate_pct": f"{statistics.mean(rates):.2f}",
                "sample_std_success_rate_pct": f"{statistics.stdev(rates):.2f}" if len(rates) > 1 else "",
                "min_success_rate_pct": f"{min(rates):.1f}",
                "max_success_rate_pct": f"{max(rates):.1f}",
                "mean_avg_sum_reward": f"{statistics.mean(returns):.2f}",
            }
        )
    with args.summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {len(rows)} runs and {len(summary_rows)} condition summaries")


if __name__ == "__main__":
    main()
