#!/usr/bin/env python3
"""Create labelled four-frame contact sheets from rollout MP4 files.

Uses ffprobe/ffmpeg for decoding and Pillow only for layout, so it works in a
headless cloud container without a GPU renderer.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def media_tool(name: str) -> str:
    """Find conda-bundled ffmpeg tools even when the environment is not activated."""
    configured_dir = os.environ.get("FFMPEG_BIN_DIR")
    candidates = [
        Path(configured_dir) / name if configured_dir else None,
        Path(sys.executable).parent / name,
        Path(shutil.which(name)) if shutil.which(name) else None,
    ]
    for candidate in candidates:
        if candidate and candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"Could not locate {name}; set FFMPEG_BIN_DIR explicitly")


def duration_seconds(video: Path) -> float:
    result = subprocess.run(
        [media_tool("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(video)],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def make_sheet(label: str, video: Path, output_dir: Path, width: int) -> Path:
    duration = duration_seconds(video)
    sample_times = [duration * fraction for fraction in (0.05, 0.35, 0.65, 0.95)]
    frames: list[Image.Image] = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        for index, timestamp in enumerate(sample_times):
            frame_path = temp_path / f"frame_{index}.png"
            subprocess.run(
                [media_tool("ffmpeg"), "-y", "-ss", f"{timestamp:.3f}", "-i", str(video), "-frames:v", "1", "-vf", f"scale={width}:-2", str(frame_path)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            frames.append(Image.open(frame_path).convert("RGB").copy())

    frame_w, frame_h = frames[0].size
    header_h = 32
    sheet = Image.new("RGB", (frame_w * 2, frame_h * 2 + header_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), f"{label} | {video.name} | duration={duration:.1f}s", fill="black")
    for index, frame in enumerate(frames):
        x, y = (index % 2) * frame_w, header_h + (index // 2) * frame_h
        sheet.paste(frame, (x, y))
        draw.text((x + 8, y + 8), f"t={sample_times[index]:.1f}s", fill="white", stroke_width=2, stroke_fill="black")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{label}.jpg"
    sheet.save(output_path, quality=92)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", required=True, metavar="LABEL=VIDEO_PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=320)
    args = parser.parse_args()
    for item in args.case:
        if "=" not in item:
            parser.error(f"Invalid --case: {item}")
        label, raw_video = item.split("=", maxsplit=1)
        output = make_sheet(label, Path(raw_video), args.output_dir, args.width)
        print(output)


if __name__ == "__main__":
    main()
