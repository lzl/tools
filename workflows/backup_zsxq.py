# /// script
# requires-python = ">=3.11"
# ///

"""Orchestrate a complete backup of one subscribed ZSXQ group."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence
from uuid import uuid4


DEFAULT_ARTIFACTS_ROOT = Path("artifacts")
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
VIEWER_TEMPLATE = REPOSITORY_ROOT / "templates/zsxq_viewer.html"


@dataclass(frozen=True)
class AtomFailure(RuntimeError):
    atom: str
    exit_code: int
    stderr: str

    def __str__(self) -> str:
        return f"{self.atom} failed with exit code {self.exit_code}: {self.stderr.strip() or '(no stderr)'}"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def create_run_dir(artifacts_root: Path) -> Path:
    run_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid4().hex[:8]}"
    run_dir = artifacts_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def run_atom(atom: str, args: list[str]) -> dict[str, object]:
    log(f"[workflow] Starting {atom}")
    process = subprocess.Popen(["uv", "run", atom, *args, "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if process.stdout is None or process.stderr is None:
        raise AtomFailure(atom, 0, "Could not capture atom output.")
    stderr_chunks: list[str] = []

    def relay_stderr() -> None:
        for line in process.stderr:
            stderr_chunks.append(line)
            print(line, end="", file=sys.stderr, flush=True)

    relay_thread = threading.Thread(target=relay_stderr, daemon=True)
    relay_thread.start()
    stdout = process.stdout.read()
    return_code = process.wait()
    relay_thread.join()
    stderr = "".join(stderr_chunks)
    if return_code:
        raise AtomFailure(atom, return_code, stderr)
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AtomFailure(atom, 0, f"Atom produced invalid JSON on stdout: {stdout!r}") from exc
    if not isinstance(payload, dict):
        raise AtomFailure(atom, 0, "Atom JSON stdout was not an object.")
    log(f"[workflow] Finished {atom}")
    return payload


def load_json_lines(path: Path) -> list[object]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_viewer_artifacts(run_dir: Path, backup: dict[str, object]) -> tuple[Path, Path]:
    """Write a self-contained viewer beside the backup files it reads."""
    if not VIEWER_TEMPLATE.is_file():
        raise RuntimeError(f"ZSXQ viewer template is missing: {VIEWER_TEMPLATE}")
    group_path = run_dir / "group.json"
    topics_path = run_dir / "topics.jsonl"
    media_path = run_dir / "media.jsonl"
    payload = {
        "group": json.loads(group_path.read_text(encoding="utf-8")),
        "topics": load_json_lines(topics_path),
        "media": load_json_lines(media_path),
        "artifactRoot": ".",
    }
    payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    payload_json = payload_json.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    viewer_html = run_dir / "zsxq_viewer.html"
    viewer_data = run_dir / "zsxq_viewer_data.js"
    viewer_html.write_text(VIEWER_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    viewer_data.write_text(f"window.ZSXQ_BACKUP = {payload_json};\n", encoding="utf-8")
    backup["viewer_html"] = str(viewer_html)
    backup["viewer_data"] = str(viewer_data)
    (run_dir / "summary.json").write_text(json.dumps(backup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return viewer_html, viewer_data


def run_workflow(*, group_id: str, artifacts_root: Path, run_dir: Path | None, scope: str, count: int, delay: float, no_media: bool, max_pages: int | None) -> dict[str, object]:
    actual_run_dir = run_dir or create_run_dir(artifacts_root)
    actual_run_dir.mkdir(parents=True, exist_ok=True)
    log(f"[workflow] Run artifacts: {actual_run_dir}")
    atom_args = [group_id, "--output-dir", str(actual_run_dir), "--scope", scope, "--count", str(count), "--delay", str(delay)]
    if no_media:
        atom_args.append("--no-media")
    if max_pages is not None:
        atom_args.extend(["--max-pages", str(max_pages)])
    backup = run_atom("atoms/backup_zsxq.py", atom_args)
    viewer_html, viewer_data = write_viewer_artifacts(actual_run_dir, backup)
    log(f"[workflow] Viewer: {viewer_html}")
    return {"status": backup.get("status", "completed"), "artifacts_dir": str(actual_run_dir), "group_id": group_id, "viewer_html": str(viewer_html), "viewer_data": str(viewer_data), "backup": backup}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Back up all API-visible topics and full articles from one subscribed ZSXQ group.")
    parser.add_argument("group_id", help="Numeric group ID from the ZSXQ URL")
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None, help="Use an explicit empty run directory.")
    parser.add_argument("--scope", choices=("all", "digests", "by_owner", "questions", "tasks"), default="all")
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--delay", type=float, default=1.0)
    parser.add_argument("--no-media", action="store_true", help="Keep media metadata but do not download images/files.")
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after N pages; useful for a smoke test.")
    parser.add_argument("--json", action="store_true", help="Write one JSON result object to stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.count < 1 or args.count > 30:
        log("Error: --count must be between 1 and 30")
        return 1
    if args.delay < 0:
        log("Error: --delay must be >= 0")
        return 1
    if args.max_pages is not None and args.max_pages <= 0:
        log("Error: --max-pages must be greater than 0")
        return 1
    try:
        result = run_workflow(group_id=args.group_id, artifacts_root=args.artifacts_root.expanduser(), run_dir=args.run_dir.expanduser() if args.run_dir else None, scope=args.scope, count=args.count, delay=args.delay, no_media=args.no_media, max_pages=args.max_pages)
    except Exception as exc:
        log(f"Error: {exc}")
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["artifacts_dir"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
