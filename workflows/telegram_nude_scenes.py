# /// script
# requires-python = ">=3.11"
# ///

"""Download one Telegram video, detect explicit-nudity segments, and render them.

Only run this on material where every depicted person is an adult and where
you have the right to process the file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


DEFAULT_DOWNLOAD_ROOT = Path("input_dir")
DEFAULT_OUTPUT_DIR = Path("output_dir")


@dataclass(frozen=True)
class AtomFailure(RuntimeError):
    atom: str
    exit_code: int
    stderr: str

    def __str__(self) -> str:
        detail = self.stderr.strip() or "(no stderr)"
        return f"{self.atom} failed with exit code {self.exit_code}: {detail}"


def run_atom(atom: str, args: list[str]) -> dict[str, object]:
    command = ["uv", "run", atom, *args, "--json"]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise AtomFailure(atom=atom, exit_code=result.returncode, stderr=result.stderr)
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AtomFailure(
            atom=atom,
            exit_code=0,
            stderr=f"Atom produced invalid JSON on stdout: {result.stdout!r}",
        ) from exc
    if not isinstance(payload, dict):
        raise AtomFailure(atom=atom, exit_code=0, stderr="Atom JSON stdout was not an object.")
    return payload


def run_workflow(
    *,
    link: str,
    download_root: Path,
    output_dir: Path,
    output: Path | None,
    sample_fps: float,
    threshold: float,
    exposed_threshold: float,
    padding: float,
    merge_gap: float,
    min_segment: float,
    max_width: int,
    classes: list[str],
    keep_work_dir: bool,
    crf: int,
    preset: str,
) -> dict[str, object]:
    download_result = run_atom(
        "atoms/download_telegram_message_media.py",
        [
            link,
            "--output-root",
            str(download_root),
            "--media-type",
            "video",
        ],
    )
    file_path_value = download_result.get("file_path")
    if not isinstance(file_path_value, str) or not file_path_value:
        raise AtomFailure(
            atom="atoms/download_telegram_message_media.py",
            exit_code=0,
            stderr="Download atom JSON did not include a file_path.",
        )

    input_video = Path(file_path_value)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output or output_dir / f"{input_video.stem}_nude_scenes.mp4"
    manifest_csv = output_dir / f"{input_video.stem}_nude_scenes.csv"
    summary_json = output_dir / f"{input_video.stem}_nude_scenes.json"

    detect_args = [
        str(input_video),
        "--summary-json",
        str(summary_json),
        "--manifest-csv",
        str(manifest_csv),
        "--output-video",
        str(output_video),
        "--sample-fps",
        str(sample_fps),
        "--threshold",
        str(threshold),
        "--exposed-threshold",
        str(exposed_threshold),
        "--padding",
        str(padding),
        "--merge-gap",
        str(merge_gap),
        "--min-segment",
        str(min_segment),
        "--max-width",
        str(max_width),
        "--classes",
        *classes,
    ]
    if keep_work_dir:
        detect_args.append("--keep-work-dir")
    detect_result = run_atom("atoms/detect_nude_segments.py", detect_args)

    render_args = [
        str(input_video),
        "--segments-json",
        str(summary_json),
        "-o",
        str(output_video),
        "--crf",
        str(crf),
        "--preset",
        preset,
    ]
    if keep_work_dir:
        render_args.append("--keep-work-dir")
    render_result = run_atom("atoms/render_video_segments.py", render_args)

    return {
        "downloaded_video": str(input_video),
        "output_video": str(output_video),
        "manifest_csv": str(manifest_csv),
        "summary_json": str(summary_json),
        "atoms": {
            "download": download_result,
            "detect": detect_result,
            "render": render_result,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download one Telegram video, detect explicit-nudity scenes, and render them. "
            "Only process material you have rights to process and where all depicted people are adults."
        )
    )
    parser.add_argument("link", help="Telegram message link.")
    parser.add_argument(
        "--download-root",
        type=Path,
        default=DEFAULT_DOWNLOAD_ROOT,
        help="Root directory for downloaded Telegram media (default: input_dir).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for rendered video and detection artifacts (default: output_dir).",
    )
    parser.add_argument("-o", "--output", type=Path, default=None, help="Rendered output video path.")
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--exposed-threshold", type=float, default=0.3)
    parser.add_argument("--padding", type=float, default=3.0)
    parser.add_argument("--merge-gap", type=float, default=4.0)
    parser.add_argument("--min-segment", type=float, default=1.0)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument(
        "--classes",
        nargs="+",
        default=[
            "ANUS_EXPOSED",
            "FEMALE_BREAST_EXPOSED",
            "FEMALE_GENITALIA_EXPOSED",
            "MALE_GENITALIA_EXPOSED",
        ],
    )
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--json", action="store_true", help="Write workflow JSON to stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_workflow(
            link=args.link,
            download_root=args.download_root.expanduser(),
            output_dir=args.output_dir.expanduser(),
            output=args.output.expanduser() if args.output else None,
            sample_fps=args.sample_fps,
            threshold=args.threshold,
            exposed_threshold=args.exposed_threshold,
            padding=args.padding,
            merge_gap=args.merge_gap,
            min_segment=args.min_segment,
            max_width=args.max_width,
            classes=args.classes,
            keep_work_dir=args.keep_work_dir,
            crf=args.crf,
            preset=args.preset,
        )
    except AtomFailure as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["downloaded_video"])
        print(result["output_video"])
        print(result["manifest_csv"])
        print(result["summary_json"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
