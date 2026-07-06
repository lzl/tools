# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "nudenet>=3.4.2",
#   "pillow>=10.0.0",
# ]
# ///

"""Detect explicit-nudity segments in a video without rendering clips.

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
from typing import Iterable, Protocol, Sequence


DEFAULT_CLASSES = {
    "ANUS_EXPOSED",
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
}
DEFAULT_EXPOSED_THRESHOLD = 0.3
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


class Detector(Protocol):
    def detect(self, image_path: str) -> list[dict[str, object]]: ...


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


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, check=False, capture_output=capture, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{stderr}")
    return result


def require_binary(name: str) -> None:
    if not shutil.which(name):
        raise RuntimeError(f"{name} not found. Install ffmpeg first.")


def validate_video(input_path: Path) -> Path:
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
    number = int(frame_path.stem.rsplit("_", 1)[1])
    return (number - 1) / sample_fps


def detect_hits(
    frames: Iterable[Path],
    detector: Detector,
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
            and float(item.get("score", 0.0))
            >= class_thresholds.get(str(item.get("class")), threshold)
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


def write_manifest(
    path: Path,
    input_path: Path,
    hits: list[DetectionHit],
    segments: list[Segment],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def build_summary(
    *,
    input_path: Path,
    output_video: Path | None,
    manifest_path: Path,
    duration: float,
    sample_fps: float,
    threshold: float,
    exposed_threshold: float,
    class_thresholds: dict[str, float],
    classes: list[str],
    hits: list[DetectionHit],
    segments: list[Segment],
) -> dict[str, object]:
    return {
        "input": str(input_path),
        "output": str(output_video) if output_video is not None else "",
        "manifest": str(manifest_path),
        "duration": duration,
        "sample_fps": sample_fps,
        "threshold": threshold,
        "exposed_threshold": exposed_threshold,
        "class_thresholds": class_thresholds,
        "classes": classes,
        "hit_count": len(hits),
        "segment_count": len(segments),
        "segment_seconds": round(sum(segment.duration for segment in segments), 3),
        "segments": [
            {"start": round(segment.start, 3), "end": round(segment.end, 3)}
            for segment in segments
        ],
        "hits": [
            {
                "time": round(hit.time, 3),
                "label": hit.label,
                "score": round(hit.score, 4),
            }
            for hit in hits
        ],
    }


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def detect_video_segments(
    *,
    input_path: Path,
    summary_json: Path,
    manifest_csv: Path,
    output_video: Path | None,
    sample_fps: float,
    threshold: float,
    exposed_threshold: float,
    padding: float,
    merge_gap: float,
    min_segment: float,
    max_width: int,
    classes: list[str],
    keep_work_dir: bool,
    work_dir_root: Path | None,
) -> dict[str, object]:
    if sample_fps <= 0:
        raise ValueError("--sample-fps must be greater than 0")
    if not 0 <= threshold <= 1:
        raise ValueError("--threshold must be between 0 and 1")
    if not 0 <= exposed_threshold <= 1:
        raise ValueError("--exposed-threshold must be between 0 and 1")

    input_path = validate_video(input_path)
    require_binary("ffmpeg")
    require_binary("ffprobe")

    duration = probe_duration(input_path)
    sample_interval = 1.0 / sample_fps
    log(f"Input: {input_path}")
    log(f"Duration: {duration:.1f}s")
    log(f"Sampling: {sample_fps:g} fps ({math.ceil(duration * sample_fps)} frames expected)")

    if work_dir_root is not None:
        work_dir_root.mkdir(parents=True, exist_ok=True)
    temp_context = tempfile.TemporaryDirectory(prefix="nude_segment_detect_", dir=work_dir_root)
    work_dir = Path(temp_context.name)
    try:
        frames = extract_sample_frames(input_path, work_dir / "frames", sample_fps, max_width)
        log(f"Extracted {len(frames)} sampled frames")
        from nudenet import NudeDetector

        class_set = set(classes)
        class_thresholds = {
            label: exposed_threshold
            for label in DEFAULT_CLASSES
            if label in class_set
        }
        hits = detect_hits(
            frames,
            NudeDetector(),
            sample_fps,
            class_set,
            threshold,
            class_thresholds,
        )
        segments = build_segments(
            hits,
            duration,
            sample_interval,
            padding,
            merge_gap,
            min_segment,
        )
        write_manifest(manifest_csv, input_path, hits, segments)
        summary = build_summary(
            input_path=input_path,
            output_video=output_video,
            manifest_path=manifest_csv,
            duration=duration,
            sample_fps=sample_fps,
            threshold=threshold,
            exposed_threshold=exposed_threshold,
            class_thresholds=class_thresholds,
            classes=classes,
            hits=hits,
            segments=segments,
        )
        write_summary(summary_json, summary)
        log(f"Manifest: {manifest_csv}")
        log(f"Summary: {summary_json}")
        return summary
    finally:
        if keep_work_dir:
            log(f"Kept work dir: {work_dir}")
        else:
            temp_context.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Detect explicit-nudity segments in a video. Only process material "
            "you have rights to process and where all depicted people are adults."
        )
    )
    parser.add_argument("video", type=Path, help="Path to the input video.")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    parser.add_argument(
        "--output-video",
        type=Path,
        default=None,
        help="Optional output video path to include in JSON metadata.",
    )
    parser.add_argument("--sample-fps", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--exposed-threshold", type=float, default=DEFAULT_EXPOSED_THRESHOLD)
    parser.add_argument("--padding", type=float, default=3.0)
    parser.add_argument("--merge-gap", type=float, default=4.0)
    parser.add_argument("--min-segment", type=float, default=1.0)
    parser.add_argument("--max-width", type=int, default=640)
    parser.add_argument("--classes", nargs="+", default=sorted(DEFAULT_CLASSES))
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument(
        "--work-dir-root",
        type=Path,
        default=None,
        help="Directory where temporary work directories are created.",
    )
    parser.add_argument("--json", action="store_true", help="Write summary JSON to stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = detect_video_segments(
            input_path=args.video.expanduser(),
            summary_json=args.summary_json.expanduser(),
            manifest_csv=args.manifest_csv.expanduser(),
            output_video=args.output_video.expanduser() if args.output_video else None,
            sample_fps=args.sample_fps,
            threshold=args.threshold,
            exposed_threshold=args.exposed_threshold,
            padding=args.padding,
            merge_gap=args.merge_gap,
            min_segment=args.min_segment,
            max_width=args.max_width,
            classes=args.classes,
            keep_work_dir=args.keep_work_dir,
            work_dir_root=args.work_dir_root.expanduser() if args.work_dir_root else None,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
