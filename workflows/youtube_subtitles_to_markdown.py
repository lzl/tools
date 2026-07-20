#!/usr/bin/env python3
"""Download public YouTube captions and turn them into readable Markdown."""

from __future__ import annotations

import argparse
import html
import re
import subprocess
from pathlib import Path


TIMESTAMP = re.compile(r"^(?P<start>\d\d?:\d\d(?::\d\d)?\.\d\d\d)\s+-->")
TAG = re.compile(r"<[^>]+>")


def clean_caption_text(vtt_path: Path) -> list[tuple[str, str]]:
    """Extract timestamped VTT cues, collapsing only adjacent repetitions."""
    cues: list[tuple[str, str]] = []
    start: str | None = None
    text_lines: list[str] = []

    def save_cue() -> None:
        if not start or not text_lines:
            return
        text = html.unescape(TAG.sub("", " ".join(text_lines))).strip()
        timestamp = start.split(".")[0].zfill(8)
        if text and (not cues or cues[-1][1] != text):
            cues.append((timestamp, text))

    for raw_line in [*vtt_path.read_text(encoding="utf-8").splitlines(), ""]:
        line = raw_line.strip()
        match = TIMESTAMP.match(line)
        if match:
            save_cue()
            start = match.group("start")
            text_lines = []
        elif not line:
            save_cue()
            start = None
            text_lines = []
        elif start and not line.startswith(("Kind:", "Language:", "NOTE", "STYLE", "REGION")):
            text_lines.append(line)
    return cues


def publication_date(video_url: str) -> str:
    """Return YouTube's upload date in ISO format."""
    result = subprocess.run(
        ["yt-dlp", "--skip-download", "--print", "%(upload_date)s", video_url],
        check=True,
        capture_output=True,
        text=True,
    )
    raw_date = result.stdout.strip().splitlines()[-1]
    if not re.fullmatch(r"\d{8}", raw_date):
        raise RuntimeError(f"Unexpected upload date: {raw_date!r}")
    return f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("--title", required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vtt_template = args.output.parent / ".%(id)s.%(ext)s"
    subprocess.run(
        [
            "yt-dlp", args.url, "--skip-download", "--write-subs",
            "--write-auto-subs", "--sub-langs", args.language,
            "--sub-format", "vtt", "--no-mtime", "-o", str(vtt_template),
        ],
        check=True,
    )
    vtt_files = list(args.output.parent.glob(".*.vtt"))
    if len(vtt_files) != 1:
        raise RuntimeError(f"Expected one VTT file, found {len(vtt_files)}")

    transcript = clean_caption_text(vtt_files[0])
    vtt_files[0].unlink()
    published = publication_date(args.url)
    args.output.write_text(
        "\n".join(
            [
                f"# {args.title}",
                "",
                f"- Channel: {args.channel}",
                f"- Video: {args.url}",
                f"- Captions: {args.language}",
                f"- Published: {published}",
                "- Note: Public captions are retained for research and personal study; consult the source video for context.",
                "",
                "## Transcript",
                "",
                *(f"[{timestamp}] {text}" for timestamp, text in transcript),
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
