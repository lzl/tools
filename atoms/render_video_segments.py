# /// script
# requires-python = ">=3.11"
# ///

"""Render video segments described by a detection JSON file."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, check=False, capture_output=capture, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{stderr}")
    return result


def require_binary(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"{name} not found. Install ffmpeg first.")


def load_segments(path: Path) -> list[Segment]:
    if not path.exists():
        raise FileNotFoundError(f"Segments JSON not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Segments JSON is invalid: {path}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError("Segments JSON must contain a 'segments' list.")

    segments: list[Segment] = []
    for index, item in enumerate(payload["segments"]):
        if not isinstance(item, dict) or "start" not in item or "end" not in item:
            raise ValueError(f"Segment {index} must contain start and end.")
        start = float(item["start"])
        end = float(item["end"])
        if start < 0 or end < start:
            raise ValueError(f"Segment {index} has invalid bounds.")
        segments.append(Segment(start=start, end=end))
    return segments


def render_segments(
    input_path: Path,
    segments: list[Segment],
    work_dir: Path,
    output_path: Path,
    crf: int,
    preset: str,
) -> None:
    clips_dir = work_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clip_paths: list[Path] = []

    for index, segment in enumerate(segments):
        clip_path = clips_dir / f"clip_{index:04d}.mp4"
        clip_paths.append(clip_path)
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{segment.start:.3f}",
                "-to",
                f"{segment.end:.3f}",
                "-i",
                str(input_path),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-dn",
                "-c:v",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                "-y",
                str(clip_path),
            ]
        )

    concat_file = work_dir / "concat.txt"
    with concat_file.open("w", encoding="utf-8") as f:
        for clip_path in clip_paths:
            f.write(f"file '{clip_path.resolve().as_posix()}'\n")

    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-y",
            str(output_path),
        ]
    )


def write_empty_video(output_path: Path) -> None:
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=1280x720:d=0.2",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-y",
            str(output_path),
        ]
    )


def render_from_json(
    *,
    input_path: Path,
    segments_json: Path,
    output_path: Path,
    crf: int,
    preset: str,
    keep_work_dir: bool,
) -> dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"'{input_path}' is not a file")
    require_binary("ffmpeg")
    segments = load_segments(segments_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_context = tempfile.TemporaryDirectory(prefix="nude_segment_render_")
    work_dir = Path(temp_context.name)
    placeholder = not segments
    try:
        if segments:
            render_segments(input_path, segments, work_dir, output_path, crf, preset)
        else:
            write_empty_video(output_path)
        return {
            "input": str(input_path),
            "segments_json": str(segments_json),
            "output": str(output_path),
            "segment_count": len(segments),
            "segment_seconds": round(sum(segment.duration for segment in segments), 3),
            "placeholder": placeholder,
        }
    finally:
        if keep_work_dir:
            print(f"Kept work dir: {work_dir}", file=sys.stderr)
        else:
            temp_context.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render segments from a detection JSON file into one video."
    )
    parser.add_argument("input", type=Path, help="Path to the input video.")
    parser.add_argument("--segments-json", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--json", action="store_true", help="Write a single JSON object to stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = render_from_json(
            input_path=args.input.expanduser(),
            segments_json=args.segments_json.expanduser(),
            output_path=args.output.expanduser(),
            crf=args.crf,
            preset=args.preset,
            keep_work_dir=args.keep_work_dir,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
