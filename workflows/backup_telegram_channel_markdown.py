# /// script
# requires-python = ">=3.11"
# ///

"""Back up one Telegram channel and export messages added by this run."""

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
DEFAULT_BACKUP_DATABASE = Path("data/telegram_channel_backup.sqlite3")


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
    process = subprocess.Popen(
        ["uv", "run", atom, *args, "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
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


def run_workflow(
    *, channel: str, database: Path, artifacts_root: Path, run_dir: Path | None, output: Path | None, full: bool
) -> dict[str, object]:
    actual_run_dir = run_dir or create_run_dir(artifacts_root)
    outputs_dir = actual_run_dir / "outputs"
    new_message_ids = actual_run_dir / "new_message_ids.json"
    markdown_output = output or outputs_dir / "new_messages.md"
    log(f"[workflow] Run artifacts: {actual_run_dir}")
    backup_args = [
        channel,
        "--database",
        str(database),
        "--new-messages-json",
        str(new_message_ids),
    ]
    if full:
        backup_args.append("--full")
    backup = run_atom("atoms/backup_telegram_channel.py", backup_args)
    channel_id = backup.get("channel_id")
    if not isinstance(channel_id, str) or not channel_id:
        raise AtomFailure("atoms/backup_telegram_channel.py", 0, "Backup JSON did not include channel_id.")
    export = run_atom(
        "atoms/export_telegram_messages_markdown.py",
        [
            "--database",
            str(database),
            "--channel-id",
            channel_id,
            "--message-ids-json",
            str(new_message_ids),
            "--output",
            str(markdown_output),
        ],
    )
    return {
        "status": "completed",
        "artifacts_dir": str(actual_run_dir),
        "database": str(database),
        "new_messages_json": str(new_message_ids),
        "markdown_output": str(markdown_output),
        "atoms": {"backup": backup, "export": export},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Back up one Telegram channel and export messages newly added during this run."
    )
    parser.add_argument("channel", help="Channel username or numeric ID, including -100... IDs.")
    parser.add_argument("--database", type=Path, default=DEFAULT_BACKUP_DATABASE)
    parser.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("-o", "--output", type=Path, default=None)
    parser.add_argument("--full", action="store_true", help="Scan full channel; export only rows new to database.")
    parser.add_argument("--json", action="store_true", help="Write workflow JSON to stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_workflow(
            channel=args.channel,
            database=args.database.expanduser(),
            artifacts_root=args.artifacts_root.expanduser(),
            run_dir=args.run_dir.expanduser() if args.run_dir else None,
            output=args.output.expanduser() if args.output else None,
            full=args.full,
        )
    except AtomFailure as exc:
        log(f"Error: {exc}")
        return 1
    except Exception as exc:
        log(f"Error: {exc}")
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["markdown_output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
