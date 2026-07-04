# Atoms and Workflows Guide

This guide captures conventions for writing standalone atoms and workflow
orchestrators. Use it for future scripts that need predictable composition,
clear ownership, and low coupling.

## Core Model

An atom is a standalone executable script that performs one bounded operation.
It owns its inputs, validation, dependencies, side effects, and output contract.
It must be usable without importing any local project code.

A workflow is an orchestrator script that combines atoms into a higher-level
task. It should call atoms as subprocesses, pass explicit arguments, parse their
machine-readable output, and decide what happens next.

Keep this boundary strict:

- Atom = one operation, self-contained.
- Workflow = sequencing, defaults, path derivation, failure handling.
- Shared local packages = not allowed inside atoms.
- Shared local packages may exist for legacy tools, but atoms must not import
  them.

## Directory Layout

Use plural collection directories:

```text
atoms/
workflows/
docs/
tests/
```

Naming:

- Atom file names should be verb-first: `download_telegram_message_media.py`,
  `detect_nude_segments.py`, `render_video_segments.py`.
- Workflow file names should describe the composed task:
  `telegram_nude_scenes.py`.
- Avoid generic names like `run.py`, `main.py`, or `process.py`.

## Atom Rules

Atoms must be self-contained.

Allowed dependencies:

- Python standard library.
- Third-party packages declared in the script's PEP 723 metadata.
- System binaries explicitly checked at runtime, such as `ffmpeg`.

Not allowed:

- Importing project packages such as `telegram_media`.
- Importing other atoms.
- Importing workflows.
- Adding `sys.path` hacks to reach repo code.
- Depending on current working directory for imports.
- Mutating global project config such as `pyproject.toml` only to make the atom
  run.

Each atom should include PEP 723 metadata when it needs third-party packages:

```python
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "telethon>=1.41.2",
# ]
# ///
```

If an atom needs logic that already exists elsewhere, copy or reimplement the
small needed subset inside the atom. Duplication is acceptable when it preserves
atom independence. Avoid deep shared abstractions unless the project explicitly
introduces a versioned external package for atoms.

## Workflow Rules

Workflows may depend on atoms, but only by invoking them as commands:

```python
subprocess.run(
    ["uv", "run", "atoms/example_atom.py", "--json"],
    check=False,
    capture_output=True,
    text=True,
)
```

Do not import atom business logic into workflows. Subprocess boundaries make
contracts visible and prevent hidden coupling.

Workflows should handle:

- Call order.
- Default output paths.
- Passing CLI arguments through to atoms.
- Parsing atom JSON.
- Stopping on first failed atom.
- Producing one workflow-level summary.

Workflows should not duplicate atom internals. If a workflow needs an atom
result, it should read it from the atom's JSON output or artifact files.

## CLI Contracts

Every atom and workflow should have a stable CLI contract.

Prefer this shape:

```text
atoms/do_one_thing.py INPUT --output OUTPUT --json
workflows/do_many_things.py INPUT --output-dir output_dir --json
```

Rules:

- Required inputs should be positional only when there is exactly one obvious
  primary input.
- Artifact paths should be explicit flags: `--summary-json`,
  `--manifest-csv`, `--output`.
- Defaults should be safe and predictable.
- `--json` means stdout contains one JSON object and nothing else.
- Human progress, warnings, and errors go to stderr.
- Nonzero exit means failure.

Never mix logs with JSON stdout. Workflows parse stdout; one stray progress line
breaks composition.

## JSON Output

Use stable JSON objects, not ad hoc text.

Good atom result:

```json
{
  "input": "input.mp4",
  "output": "output.mp4",
  "status": "downloaded",
  "segment_count": 3
}
```

Guidelines:

- Use strings for paths.
- Use numbers for counts, durations, thresholds, and scores.
- Use booleans for flags such as `placeholder`.
- Include enough metadata for workflows and tests.
- Keep field names stable once published.
- Round floats when output must be deterministic.
- Prefer empty arrays over omitted fields.

## Error Handling

Atoms should fail closed:

- Validate inputs before side effects.
- Check required binaries before running long work.
- Reject unsupported media or schema mismatches early.
- Clean temporary files on failure.
- Return nonzero exit on error.
- Print concise error messages to stderr.

Workflows should preserve atom failure context:

```text
Error: atoms/example.py failed with exit code 1: missing input file
```

Include:

- Atom path/name.
- Exit code.
- Stderr content.

Stop workflow at first failed atom unless there is a clear recovery path.

## File Output

Atoms should write only files they explicitly own.

Use atomic or cleanup-aware patterns:

- Write downloads to `.part`, then move to final path.
- Create parent directories explicitly.
- Remove stale temp files before retry.
- Remove temp files after failure.
- Keep `--keep-work-dir` for debugging when useful.

If an atom writes a manifest or summary, it should write valid output even for
empty results. Empty success is still success.

## Dependencies

Atom dependency policy:

- Declare third-party Python packages in PEP 723 metadata.
- Probe system binaries with `shutil.which`.
- Do not rely on project install state.
- Do not rely on another local module being importable.

This command must work from repo root:

```bash
uv run atoms/example_atom.py --help
```

If it fails because a local package is missing from `sys.path`, the atom is not
self-contained enough.

## Tests

Test atoms at their real boundaries.

Recommended tests:

- CLI parse and help behavior.
- JSON stdout is valid and uncontaminated by logs.
- stderr receives progress and errors.
- Input validation failures return nonzero.
- Empty output still writes valid artifacts.
- Status mapping: `downloaded`, `skipped`, `manifest_backfilled`, or domain
  equivalents.
- Schema stability for JSON and CSV outputs.
- External runners such as `ffmpeg` mocked at command boundary.
- Network APIs mocked inside the atom, not through old project packages.

Workflow tests should mock the subprocess runner and verify:

- Atom call order.
- Arguments passed to each atom.
- Default path derivation.
- First failing atom stops the workflow.
- Error includes atom name, exit code, and stderr.

When a test imports an atom, avoid importing legacy project types to build fake
inputs. Use atom-owned dataclasses or small local fakes. Tests should not
normalize away forbidden dependencies.

## Review Checklist

Before accepting a new atom:

- `uv run atoms/name.py --help` works.
- No imports from local project packages.
- No imports from other atoms or workflows.
- No `sys.path` mutation to reach repo code.
- Third-party packages declared in PEP 723.
- Logs go to stderr.
- `--json` stdout is one JSON object.
- Errors exit nonzero.
- Temporary files cleaned.
- Tests cover success, validation failure, and empty result.

Before accepting a new workflow:

- `uv run workflows/name.py --help` works.
- Atoms invoked through subprocess, not imported.
- Atom failures preserve atom path, exit code, stderr.
- Default artifacts have deterministic paths.
- Workflow JSON includes key artifact paths and nested atom results when useful.
- Tests verify call order and failure stop behavior.

## Common Pitfalls

Local package import in atom:

```python
from telegram_media.cli import parse_message_link
```

Problem: atom now depends on repo install state and hidden package behavior.
Fix: implement the required parsing inside the atom or move dependency into a
real external package with explicit versioning.

Workflow importing atom logic:

```python
from atoms.detect_nude_segments import detect_video_segments
```

Problem: workflow bypasses CLI contract and couples to internals.
Fix: call `uv run atoms/detect_nude_segments.py --json`.

JSON stdout mixed with logs:

```text
Downloaded 20%
{"output": "file.mp4"}
```

Problem: workflow JSON parse fails.
Fix: progress to stderr, JSON only to stdout.

Implicit output paths:

```python
output = Path("output.mp4")
```

Problem: repeated runs collide and workflows cannot predict artifacts.
Fix: derive paths from input stem or require explicit output flags.

## Minimal Atom Skeleton

```python
# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Do one bounded operation.")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    return parser


def run_atom(input_path: Path, output_path: Path) -> dict[str, object]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Do work here.
    return {
        "input": str(input_path),
        "output": str(output_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = run_atom(args.input.expanduser(), args.output.expanduser())
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
```

## Minimal Workflow Skeleton

```python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class AtomFailure(RuntimeError):
    atom: str
    exit_code: int
    stderr: str

    def __str__(self) -> str:
        detail = self.stderr.strip() or "(no stderr)"
        return f"{self.atom} failed with exit code {self.exit_code}: {detail}"


def run_atom(atom: str, args: list[str]) -> dict[str, object]:
    result = subprocess.run(
        ["uv", "run", atom, *args, "--json"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AtomFailure(atom, result.returncode, result.stderr)
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise AtomFailure(atom, 0, "Atom JSON stdout was not an object.")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compose atoms into one task.")
    parser.add_argument("input")
    args = parser.parse_args(argv)

    try:
        first = run_atom("atoms/first.py", [args.input])
        second = run_atom("atoms/second.py", [str(first["output"])])
    except AtomFailure as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps({"first": first, "second": second}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

