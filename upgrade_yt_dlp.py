# /// script
# dependencies = []
# ///

"""Manually upgrade a standalone yt-dlp executable from GitHub releases."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from yt_dlp_wrapper import parse_version, resolve_yt_dlp


LATEST_RELEASE_URL = "https://github.com/yt-dlp/yt-dlp/releases/latest"
DOWNLOAD_BASE_URL = "https://github.com/yt-dlp/yt-dlp/releases/download"
DEFAULT_ASSET_NAME = "yt-dlp"
CHECKSUMS_ASSET_NAME = "SHA2-256SUMS"


@dataclass(frozen=True)
class LocalYtDlp:
    path: Path
    version: str | None


def fetch_latest_tag(latest_url: str = LATEST_RELEASE_URL) -> str:
    """Resolve the latest release redirect and return its tag."""
    request = urllib.request.Request(
        latest_url,
        headers={"User-Agent": "tools-upgrade-yt-dlp/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        final_url = response.geturl()

    tag = parse_tag_from_release_url(final_url)
    if not tag:
        raise RuntimeError(f"Could not determine latest yt-dlp release from {final_url}")
    return tag


def parse_tag_from_release_url(url: str) -> str | None:
    """Extract a GitHub release tag from a release URL."""
    parts = [part for part in urlparse(url).path.split("/") if part]
    if len(parts) >= 5 and parts[-2] == "tag":
        return parts[-1]
    return None


def should_upgrade(current_version: str | None, latest_tag: str, force: bool = False) -> bool:
    """Return True when the local binary should be replaced."""
    if force:
        return True
    if not current_version:
        return True
    return parse_version(current_version) < parse_version(latest_tag)


def release_asset_url(tag: str, asset_name: str) -> str:
    """Build a GitHub release asset URL."""
    return f"{DOWNLOAD_BASE_URL}/{tag}/{asset_name}"


def download_file(url: str, destination: Path) -> None:
    """Download a URL to a local file."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "tools-upgrade-yt-dlp/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def checksum_for_asset(checksums_text: str, asset_name: str) -> str:
    """Return the SHA256 checksum for an asset listed in SHA2-256SUMS."""
    for line in checksums_text.splitlines():
        fields = line.strip().split()
        if len(fields) != 2:
            continue
        checksum, filename = fields
        if filename == asset_name:
            return checksum.lower()
    raise RuntimeError(f"Could not find checksum for {asset_name!r}")


def sha256sum(path: Path) -> str:
    """Compute a file's SHA256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksum(path: Path, expected_sha256: str) -> None:
    """Raise if a downloaded file does not match its expected SHA256."""
    actual_sha256 = sha256sum(path)
    if actual_sha256.lower() != expected_sha256.lower():
        raise RuntimeError(
            f"Checksum mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )


def probe_local_yt_dlp(path: Path) -> LocalYtDlp:
    """Read the version reported by a yt-dlp executable."""
    try:
        result = subprocess.run(
            [str(path), "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, PermissionError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Could not run yt-dlp at {path}") from exc

    version = result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else None
    return LocalYtDlp(path=path, version=version)


def resolve_target(target: str | None) -> LocalYtDlp:
    """Resolve the yt-dlp executable to inspect and upgrade."""
    if target:
        return probe_local_yt_dlp(Path(target).expanduser())

    resolved = resolve_yt_dlp()
    return LocalYtDlp(path=Path(resolved.path), version=resolved.version)


def backup_path_for(target: Path, version: str | None) -> Path:
    """Return a non-conflicting backup path next to the target executable."""
    suffix = version or "unknown"
    base = target.with_name(f"{target.name}.{suffix}.bak")
    if not base.exists():
        return base

    index = 2
    while True:
        candidate = target.with_name(f"{target.name}.{suffix}.bak.{index}")
        if not candidate.exists():
            return candidate
        index += 1


def install_asset(downloaded_asset: Path, target: Path, current_version: str | None) -> Path:
    """Back up the current executable and atomically install the downloaded one."""
    backup_path = backup_path_for(target, current_version)
    shutil.copy2(target, backup_path)

    replacement = target.with_name(f".{target.name}.upgrade-tmp")
    shutil.copy2(downloaded_asset, replacement)
    os.chmod(replacement, 0o755)
    os.replace(replacement, target)
    return backup_path


def upgrade_yt_dlp(
    target: str | None = None,
    force: bool = False,
    dry_run: bool = False,
    latest_url: str = LATEST_RELEASE_URL,
    asset_name: str = DEFAULT_ASSET_NAME,
) -> int:
    """Check GitHub releases and upgrade yt-dlp only when needed."""
    local = resolve_target(target)
    print(f"Local yt-dlp: {local.path}")
    print(f"Local version: {local.version or 'unknown'}")

    latest_tag = fetch_latest_tag(latest_url)
    print(f"Latest version: {latest_tag}")

    if not should_upgrade(local.version, latest_tag, force=force):
        print("Already up to date; no download needed.")
        return 0

    if dry_run:
        print("Upgrade available; dry run enabled, no files changed.")
        return 0

    asset_url = release_asset_url(latest_tag, asset_name)
    checksums_url = release_asset_url(latest_tag, CHECKSUMS_ASSET_NAME)

    with tempfile.TemporaryDirectory(prefix="yt-dlp-upgrade-") as temp_dir:
        temp_path = Path(temp_dir)
        asset_path = temp_path / asset_name
        checksums_path = temp_path / CHECKSUMS_ASSET_NAME

        print(f"Downloading: {asset_url}")
        download_file(asset_url, asset_path)
        print(f"Downloading: {checksums_url}")
        download_file(checksums_url, checksums_path)

        expected_sha256 = checksum_for_asset(
            checksums_path.read_text(encoding="utf-8"),
            asset_name,
        )
        verify_checksum(asset_path, expected_sha256)
        print(f"SHA256 verified: {expected_sha256}")

        backup_path = install_asset(asset_path, local.path, local.version)

    upgraded = probe_local_yt_dlp(local.path)
    print(f"Backup created: {backup_path}")
    print(f"Installed version: {upgraded.version or 'unknown'}")

    if upgraded.version != latest_tag:
        raise RuntimeError(
            f"Installed yt-dlp reports {upgraded.version!r}, expected {latest_tag!r}"
        )

    return 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manually upgrade a standalone yt-dlp executable from GitHub releases.",
    )
    parser.add_argument(
        "--target",
        help="Path to the yt-dlp executable to upgrade. Defaults to the best yt-dlp on PATH.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download and reinstall the latest release even if the local version matches.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check versions without downloading or changing files.",
    )
    parser.add_argument(
        "--latest-url",
        default=LATEST_RELEASE_URL,
        help=f"Latest release URL to resolve. Defaults to {LATEST_RELEASE_URL}.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        return upgrade_yt_dlp(
            target=args.target,
            force=args.force,
            dry_run=args.dry_run,
            latest_url=args.latest_url,
        )
    except (RuntimeError, urllib.error.URLError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
