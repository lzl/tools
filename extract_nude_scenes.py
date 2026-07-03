# /// script
# dependencies = [
#   "nudenet>=3.4.2",
#   "pillow>=10.0.0",
# ]
# ///

"""Extract detected explicit-nudity scenes from a video and merge them.

The tool samples frames with ffmpeg, runs NudeNet detection on those frames,
turns positive samples into padded/merged time ranges, then uses ffmpeg to
render those ranges into one output video.

Only run this on material where every depicted person is an adult and where
you have the right to process the file.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from nudenet import NudeDetector


DEFAULT_OUTPUT_DIR = Path("output_dir")
DEFAULT_INPUT_DIR = Path("input_dir")
MEDIA_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".avi",
}
DEFAULT_CLASSES = {
    "ANUS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
}
DEFAULT_EXPOSED_THRESHOLD = 0.3


def log(message: str) -> None:
    print(message, flush=True)


@dataclass(frozen=True)
class DetectionHit:
    time: float
    label: str
    score: float


@dataclass(frozen=True)
class Segment:
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        check=False,
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{stderr}")
    return result


def require_binary(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"{name} not found. Install ffmpeg first.")


def find_latest_media_file(directory: Path) -> Path:
    """Find the latest modified supported media file in a directory."""
    media_files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    ]

    if not media_files:
        raise FileNotFoundError(f"No media files found in {directory}")

    return max(media_files, key=lambda path: path.stat().st_mtime)


def resolve_input_file(input_path: Path | None) -> Path:
    """Resolve and validate the input video file."""
    if input_path is None:
        if not DEFAULT_INPUT_DIR.exists():
            raise FileNotFoundError(f"Input directory '{DEFAULT_INPUT_DIR}' does not exist")

        latest_file = find_latest_media_file(DEFAULT_INPUT_DIR)
        log(f"Using latest media file from {DEFAULT_INPUT_DIR}: {latest_file.name}")
        return latest_file

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"'{input_path}' is not a file")

    if input_path.suffix.lower() not in MEDIA_EXTENSIONS:
        supported = ", ".join(sorted(MEDIA_EXTENSIONS))
        raise ValueError(f"'{input_path}' is not a supported video format. Supported formats: {supported}")

    return input_path


def probe_duration(input_path: Path) -> float:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(input_path),
        ],
        capture=True,
    )
    return float(result.stdout.strip())


def extract_sample_frames(
    input_path: Path,
    frames_dir: Path,
    sample_fps: float,
    max_width: int,
) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={sample_fps},scale='min({max_width},iw)':-2"
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-vf",
            vf,
            "-q:v",
            "3",
            str(frames_dir / "frame_%08d.jpg"),
        ]
    )
    return sorted(frames_dir.glob("frame_*.jpg"))


def frame_time(frame_path: Path, sample_fps: float) -> float:
    # ffmpeg image2 starts numbering at 1.
    number = int(frame_path.stem.rsplit("_", 1)[1])
    return (number - 1) / sample_fps


def detect_hits(
    frames: Iterable[Path],
    detector: NudeDetector,
    sample_fps: float,
    labels: set[str],
    threshold: float,
    class_thresholds: dict[str, float],
) -> list[DetectionHit]:
    hits: list[DetectionHit] = []
    for index, frame in enumerate(frames, start=1):
        detections = detector.detect(str(frame))
        matching = [
            item
            for item in detections
            if item.get("class") in labels
            and float(item.get("score", 0.0)) >= class_thresholds.get(str(item.get("class")), threshold)
        ]
        if matching:
            best = max(matching, key=lambda item: float(item.get("score", 0.0)))
            hits.append(
                DetectionHit(
                    time=frame_time(frame, sample_fps),
                    label=str(best["class"]),
                    score=float(best["score"]),
                )
            )
        if index % 100 == 0:
            log(f"Scanned {index} sampled frames, positive frames: {len(hits)}")
    return hits


def build_segments(
    hits: list[DetectionHit],
    duration: float,
    sample_interval: float,
    padding: float,
    merge_gap: float,
    min_segment: float,
) -> list[Segment]:
    if not hits:
        return []

    raw_segments = [
        Segment(
            start=max(0.0, hit.time - padding),
            end=min(duration, hit.time + sample_interval + padding),
        )
        for hit in hits
    ]
    raw_segments.sort(key=lambda segment: segment.start)

    merged: list[Segment] = []
    for segment in raw_segments:
        if not merged or segment.start > merged[-1].end + merge_gap:
            merged.append(segment)
            continue
        previous = merged[-1]
        merged[-1] = Segment(previous.start, max(previous.end, segment.end))

    return [segment for segment in merged if segment.duration >= min_segment]


def write_manifest(path: Path, input_path: Path, hits: list[DetectionHit], segments: list[Segment]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "start", "end", "duration", "label", "score", "source"])
        for segment in segments:
            writer.writerow(
                [
                    "segment",
                    f"{segment.start:.3f}",
                    f"{segment.end:.3f}",
                    f"{segment.duration:.3f}",
                    "",
                    "",
                    input_path.name,
                ]
            )
        for hit in hits:
            writer.writerow(
                [
                    "hit",
                    f"{hit.time:.3f}",
                    "",
                    "",
                    hit.label,
                    f"{hit.score:.4f}",
                    input_path.name,
                ]
            )


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect explicit-nudity frames in a video, cut matching scenes, and merge them.",
    )
    parser.add_argument("input", nargs="?", type=Path, default=None, help="Path to video file (default: latest in input_dir/).")
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Frame sample rate for detection.")
    parser.add_argument("--threshold", type=float, default=0.45, help="Detection score threshold.")
    parser.add_argument(
        "--exposed-threshold",
        type=float,
        default=DEFAULT_EXPOSED_THRESHOLD,
        help="Default confidence threshold for exposed nudity classes.",
    )
    parser.add_argument("--padding", type=float, default=3.0, help="Seconds added before and after each hit.")
    parser.add_argument("--merge-gap", type=float, default=4.0, help="Merge segments separated by this many seconds.")
    parser.add_argument("--min-segment", type=float, default=1.0, help="Drop shorter merged segments.")
    parser.add_argument("--max-width", type=int, default=640, help="Scale sampled frames to this max width.")
    parser.add_argument("--classes", nargs="+", default=sorted(DEFAULT_CLASSES))
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_input_file(args.input.expanduser() if args.input else None)
    if args.sample_fps <= 0:
        raise SystemExit("--sample-fps must be greater than 0")
    if not 0 <= args.threshold <= 1:
        raise SystemExit("--threshold must be between 0 and 1")
    if not 0 <= args.exposed_threshold <= 1:
        raise SystemExit("--exposed-threshold must be between 0 and 1")

    require_binary("ffmpeg")
    require_binary("ffprobe")
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output or output_dir / f"{input_path.stem}_nude_scenes.mp4"
    manifest_path = output_path.with_suffix(".csv")
    summary_path = output_path.with_suffix(".json")

    duration = probe_duration(input_path)
    sample_interval = 1.0 / args.sample_fps
    log(f"Input: {input_path}")
    log(f"Duration: {duration:.1f}s")
    log(f"Sampling: {args.sample_fps:g} fps ({math.ceil(duration * args.sample_fps)} frames expected)")

    temp_context = tempfile.TemporaryDirectory(prefix="nude_scene_extract_")
    work_dir = Path(temp_context.name)
    try:
        frames = extract_sample_frames(input_path, work_dir / "frames", args.sample_fps, args.max_width)
        log(f"Extracted {len(frames)} sampled frames")
        detector = NudeDetector()
        class_thresholds = {
            label: args.exposed_threshold
            for label in DEFAULT_CLASSES
            if label in set(args.classes)
        }
        hits = detect_hits(frames, detector, args.sample_fps, set(args.classes), args.threshold, class_thresholds)
        segments = build_segments(
            hits,
            duration,
            sample_interval,
            args.padding,
            args.merge_gap,
            args.min_segment,
        )
        write_manifest(manifest_path, input_path, hits, segments)

        if segments:
            log(f"Rendering {len(segments)} merged segments to {output_path}")
            render_segments(input_path, segments, work_dir, output_path, args.crf, args.preset)
        else:
            log("No matching segments detected; writing a short empty placeholder video.")
            write_empty_video(output_path)

        summary = {
            "input": str(input_path),
            "output": str(output_path),
            "manifest": str(manifest_path),
            "duration": duration,
            "sample_fps": args.sample_fps,
            "threshold": args.threshold,
            "exposed_threshold": args.exposed_threshold,
            "class_thresholds": class_thresholds,
            "classes": args.classes,
            "hit_count": len(hits),
            "segment_count": len(segments),
            "segment_seconds": round(sum(segment.duration for segment in segments), 3),
            "segments": [
                {"start": round(segment.start, 3), "end": round(segment.end, 3)}
                for segment in segments
            ],
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"Output: {output_path}")
        log(f"Manifest: {manifest_path}")
        log(f"Summary: {summary_path}")
    finally:
        if args.keep_work_dir:
            log(f"Kept work dir: {work_dir}")
        else:
            temp_context.cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted")
    except Exception as exc:
        sys.exit(f"Error: {exc}")
