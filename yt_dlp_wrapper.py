"""Helpers for resolving a usable yt-dlp binary across uv-managed script envs."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class YtDlpBinary:
    """Resolved yt-dlp binary metadata."""

    path: str
    version: Optional[str]
    version_tuple: tuple[int, ...]


def parse_version(version: str) -> tuple[int, ...]:
    """Parse yt-dlp version strings like 2026.03.17 into an integer tuple."""
    parts = tuple(int(part) for part in re.findall(r"\d+", version))
    return parts or (0,)


def _probe_binary(binary_path: str) -> Optional[YtDlpBinary]:
    """Return binary metadata if the binary is executable and reports a version."""
    try:
        result = subprocess.run(
            [binary_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError):
        return None

    version = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else None
    return YtDlpBinary(
        path=binary_path,
        version=version,
        version_tuple=parse_version(version or ""),
    )


def _iter_candidate_names() -> list[str]:
    """Return executable names that may resolve to yt-dlp on this system."""
    names = ["yt-dlp"]
    seen = set(names)

    for ext in os.environ.get("PATHEXT", "").split(";"):
        ext = ext.strip()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        candidate_name = f"yt-dlp{ext.lower()}"
        if candidate_name in seen:
            continue
        names.append(candidate_name)
        seen.add(candidate_name)

    return names


def _iter_binary_candidates() -> list[str]:
    """Collect yt-dlp binaries visible on PATH, preserving PATH order."""
    candidates: list[str] = []
    seen: set[str] = set()

    current_binary_path = shutil.which("yt-dlp")
    if current_binary_path:
        candidates.append(current_binary_path)
        seen.add(current_binary_path)
        seen.add(str(Path(current_binary_path).resolve()))

    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry:
            continue

        for candidate_name in _iter_candidate_names():
            candidate_path = Path(entry) / candidate_name
            if not candidate_path.is_file() or not os.access(candidate_path, os.X_OK):
                continue

            raw_candidate = str(candidate_path)
            resolved_candidate = str(candidate_path.resolve())
            if raw_candidate in seen or resolved_candidate in seen:
                continue

            candidates.append(raw_candidate)
            seen.add(raw_candidate)
            seen.add(resolved_candidate)

    return candidates


@lru_cache(maxsize=1)
def resolve_yt_dlp() -> YtDlpBinary:
    """Resolve the best yt-dlp binary available on PATH."""
    candidates: list[YtDlpBinary] = []
    for candidate_path in _iter_binary_candidates():
        candidate = _probe_binary(candidate_path)
        if candidate:
            candidates.append(candidate)

    if not candidates:
        raise FileNotFoundError("yt-dlp not found on PATH")

    candidates.sort(key=lambda candidate: candidate.version_tuple, reverse=True)
    return candidates[0]


def resolve_yt_dlp_binary() -> str:
    """Return the best yt-dlp binary path."""
    return resolve_yt_dlp().path


def yt_dlp_command(*args: str) -> list[str]:
    """Build a yt-dlp command using the best available binary."""
    return [resolve_yt_dlp_binary(), *args]
