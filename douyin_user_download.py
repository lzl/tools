# /// script
# dependencies = [
#   "f2",
#   "httpx",
#   "tqdm",
# ]
# ///

"""Download public posted videos from a Douyin user profile."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections.abc import AsyncIterator, Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote, urlparse


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
DOUYIN_HOSTS = {"douyin.com", "www.douyin.com", "iesdouyin.com", "www.iesdouyin.com"}
DEFAULT_PAGE_SIZE = 20
PAGE_DELAY_SECONDS = 2
DEFAULT_OUTPUT_ROOT = Path("data/douyin_downloads")
DEFAULT_COOKIE_FILE = Path("cookies.txt")


class LoginRequiredError(RuntimeError):
    """Raised when Douyin exposes only the first page without an authenticated cookie."""


@dataclass(frozen=True)
class DouyinVideo:
    aweme_id: str
    desc: str
    create_time: Optional[int]
    source_url: str
    download_url: str


@dataclass(frozen=True)
class DownloadConfig:
    sec_user_id: str
    output_root: Path
    cookie: Optional[str] = None
    page_size: int = DEFAULT_PAGE_SIZE
    max_count: Optional[int] = None
    overwrite: bool = False
    dry_run: bool = False


@dataclass
class DownloadResult:
    discovered: int = 0
    downloaded: int = 0
    skipped: int = 0
    dry_run: int = 0
    failed: int = 0


FetchUserPosts = Callable[[DownloadConfig], AsyncIterator[DouyinVideo]]


def parse_sec_user_id(value: str) -> str:
    """Return the Douyin sec_user_id from a profile URL or raw ID."""
    candidate = value.strip()
    if not candidate:
        raise ValueError("sec_user_id or Douyin user URL is required")

    parsed = urlparse(candidate)
    if parsed.scheme:
        host = parsed.netloc.lower()
        if host not in DOUYIN_HOSTS:
            raise ValueError(f"unsupported Douyin user URL host: {parsed.netloc}")

        path_parts = [unquote(part) for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] == "user":
            sec_user_id = path_parts[1]
        else:
            raise ValueError("expected a Douyin user URL like https://www.douyin.com/user/<sec_user_id>")
    else:
        sec_user_id = candidate

    if not re.fullmatch(r"[A-Za-z0-9_.=-]+", sec_user_id):
        raise ValueError("sec_user_id contains unsupported characters")

    return sec_user_id


def read_cookie_file(path: Path) -> Optional[str]:
    if not path.exists():
        raise FileNotFoundError(f"cookie file not found: {path}")

    raw_lines = []
    netscape_pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            parts = stripped.split("\t")
            if len(parts) >= 7:
                domain = parts[0].lstrip(".").lower()
                name = parts[5].strip()
                value = parts[6].strip()
                if domain.endswith("douyin.com") and name:
                    netscape_pairs.append(f"{name}={value}")
            else:
                raw_lines.append(stripped)

    if netscape_pairs:
        return "; ".join(netscape_pairs)

    return "\n".join(raw_lines) if raw_lines else None


def _sanitize_file_stem(value: str) -> str:
    stem = Path(value).name
    stem = re.sub(r"[^A-Za-z0-9_.=-]+", "_", stem).strip("._")
    return stem or "unknown"


def _extension_from_url(url: str) -> str:
    path = unquote(urlparse(url).path)
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in {"mp4", "mov", "m4v", "webm", "flv", "mkv"}:
        return suffix
    return "mp4"


def build_video_path(output_root: Path, sec_user_id: str, video: DouyinVideo) -> Path:
    """Return the destination path for a video."""
    safe_id = _sanitize_file_stem(video.aweme_id)
    ext = _extension_from_url(video.download_url)
    return output_root / sec_user_id / "videos" / f"{safe_id}.{ext}"


def _user_dir(config: DownloadConfig) -> Path:
    return config.output_root / config.sec_user_id


def _checkpoint_path(config: DownloadConfig) -> Path:
    return _user_dir(config) / "checkpoint.json"


def _manifest_path(config: DownloadConfig) -> Path:
    return _user_dir(config) / "manifest.jsonl"


def load_completed_aweme_ids(config: DownloadConfig) -> list[str]:
    path = _checkpoint_path(config)
    if not path.exists():
        return []

    try:
        checkpoint = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    completed = checkpoint.get("completed_aweme_ids", [])
    if not isinstance(completed, list):
        return []

    return [str(aweme_id) for aweme_id in completed]


def write_checkpoint(config: DownloadConfig, completed_aweme_ids: Iterable[str]) -> None:
    path = _checkpoint_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "sec_user_id": config.sec_user_id,
        "completed_aweme_ids": list(dict.fromkeys(completed_aweme_ids)),
        "last_run_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def append_manifest_row(config: DownloadConfig, video: DouyinVideo, output_path: Path, status: str) -> None:
    manifest_path = _manifest_path(config)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "aweme_id": video.aweme_id,
        "desc": video.desc,
        "create_time": video.create_time,
        "source_url": video.source_url,
        "download_url": video.download_url,
        "output_path": str(output_path),
        "byte_size": output_path.stat().st_size if output_path.exists() else 0,
        "status": status,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with manifest_path.open("a", encoding="utf-8") as manifest_file:
        manifest_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _as_dict(value: Any) -> Any:
    if isinstance(value, Mapping):
        return dict(value)

    for method_name in ("_to_dict", "_to_raw"):
        method = getattr(value, method_name, None)
        if callable(method):
            converted = method()
            if isinstance(converted, Mapping):
                return dict(converted)
            return converted

    return value


def iter_aweme_records(value: Any) -> Iterable[Mapping[str, Any]]:
    """Yield aweme dictionaries from F2 filter objects or raw API-shaped data."""
    if not isinstance(value, (Mapping, list)):
        to_list = getattr(value, "_to_list", None)
        if callable(to_list):
            yield from iter_aweme_records(to_list())
            return

    converted = _as_dict(value)

    if isinstance(converted, list):
        for item in converted:
            yield from iter_aweme_records(item)
        return

    if not isinstance(converted, Mapping):
        return

    for key in ("aweme_list", "aweme_data", "data", "list", "items", "videos"):
        nested = converted.get(key)
        if isinstance(nested, list):
            for item in nested:
                yield from iter_aweme_records(item)
            return

    if converted.get("aweme_id") or converted.get("id"):
        yield converted


def _walk_values(value: Any, key_hint: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            next_hint = f"{key_hint}.{key}" if key_hint else str(key)
            yield from _walk_values(nested, next_hint)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_values(nested, key_hint)
    else:
        yield key_hint, value


def _url_score(key_hint: str, url: str) -> tuple[int, int, int]:
    lowered_key = key_hint.lower()
    lowered_url = url.lower()
    downloadable = int("download" in lowered_key or "download" in lowered_url)
    playable = int("play" in lowered_key or "play" in lowered_url)
    clean = int("watermark" not in lowered_key and "watermark" not in lowered_url)
    return clean, downloadable, playable


def _iter_candidate_urls(record: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    for key_hint, value in _walk_values(record):
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            lowered_key = key_hint.lower()
            lowered_url = value.lower()
            suffix = Path(unquote(urlparse(value).path)).suffix.lower()
            if any(part in lowered_key for part in ("music", "audio", "cover", "avatar", "image")):
                continue
            if suffix in {".mp3", ".m4a", ".aac", ".wav", ".jpg", ".jpeg", ".png", ".webp"}:
                continue
            if (
                suffix in {".mp4", ".mov", ".m4v", ".webm", ".flv", ".mkv"}
                or any(part in lowered_key for part in ("video", "play_addr", "download_addr", "bit_rate"))
                or any(part in lowered_url for part in ("douyin", "byte", "ixigua"))
            ):
                yield key_hint, value


def choose_download_url(record: Mapping[str, Any]) -> Optional[str]:
    candidates = list(_iter_candidate_urls(record))
    if not candidates:
        return None
    return max(candidates, key=lambda item: _url_score(item[0], item[1]))[1]


def coerce_video_record(record: Mapping[str, Any]) -> Optional[DouyinVideo]:
    aweme_id = str(record.get("aweme_id") or record.get("id") or "").strip()
    if not aweme_id:
        return None

    download_url = choose_download_url(record)
    if not download_url:
        return None

    desc = str(record.get("desc") or record.get("title") or record.get("caption") or "")
    create_time_raw = record.get("create_time")
    try:
        create_time = int(create_time_raw) if create_time_raw is not None else None
    except (TypeError, ValueError):
        create_time = None

    return DouyinVideo(
        aweme_id=aweme_id,
        desc=desc,
        create_time=create_time,
        source_url=f"https://www.douyin.com/video/{aweme_id}",
        download_url=download_url,
    )


def is_login_limited_response(response: Mapping[str, Any]) -> bool:
    """Return True when Douyin indicates there are more posts behind login."""
    return bool(response.get("has_more")) and bool(response.get("not_login_module"))


def build_f2_kwargs(config: DownloadConfig) -> dict[str, Any]:
    """Build kwargs expected by F2's DouyinHandler."""
    return {
        "headers": {
            "User-Agent": USER_AGENT,
            "Referer": "https://www.douyin.com/",
        },
        "proxies": {"http://": None, "https://": None},
        "timeout": 15,
        "cookie": config.cookie,
    }


def config_with_guest_cookie(
    config: DownloadConfig,
    token_factory: Callable[[], str],
) -> DownloadConfig:
    """Return config with a generated visitor ttwid cookie when no cookie was supplied."""
    if config.cookie:
        return config

    token = token_factory().strip()
    cookie = token if token.startswith("ttwid=") else f"ttwid={token}"
    return replace(config, cookie=cookie)


async def fetch_f2_user_posts(config: DownloadConfig) -> AsyncIterator[DouyinVideo]:
    """Fetch posted videos from Douyin through F2."""
    try:
        from f2.apps.douyin.crawler import DouyinCrawler  # type: ignore[reportMissingImports]
        from f2.apps.douyin.filter import UserPostFilter  # type: ignore[reportMissingImports]
        from f2.apps.douyin.model import UserPost  # type: ignore[reportMissingImports]
        from f2.apps.douyin.utils import TokenManager  # type: ignore[reportMissingImports]
    except ImportError as exc:
        raise RuntimeError(
            "F2 is required for Douyin discovery. Run this script with `uv run "
            "douyin_user_download.py ...` so inline dependencies are installed."
        ) from exc

    try:
        config = config_with_guest_cookie(config, TokenManager.gen_ttwid)
    except Exception as exc:
        raise RuntimeError(
            "Could not generate a Douyin guest cookie. Export browser cookies to cookies.txt "
            "and rerun with `--cookie-file cookies.txt`."
        ) from exc

    kwargs = build_f2_kwargs(config)
    cursor = 0
    discovered = 0

    while config.max_count is None or discovered < config.max_count:
        request_count = config.page_size
        if config.max_count is not None:
            request_count = min(request_count, config.max_count - discovered)

        async with DouyinCrawler(kwargs) as crawler:
            response = await crawler.fetch_user_post(
                UserPost(
                    max_cursor=cursor,
                    count=request_count,
                    sec_user_id=config.sec_user_id,
                )
            )

        records = list(iter_aweme_records(UserPostFilter(response)))
        for record in records:
            video = coerce_video_record(record)
            if video:
                discovered += 1
                yield video
                if config.max_count is not None and discovered >= config.max_count:
                    return

        if is_login_limited_response(response):
            raise LoginRequiredError(
                "Douyin returned a login-limited page: only the first page of public posts "
                "is visible with a generated visitor cookie. Export cookies for a logged-in "
                "Douyin browser session and rerun with `--cookie-file cookies.txt`; existing "
                "downloads will be skipped from the checkpoint."
            )

        if not response.get("has_more"):
            return

        next_cursor = response.get("max_cursor")
        if not next_cursor or next_cursor == cursor:
            raise RuntimeError("Douyin pagination cursor did not advance")

        cursor = int(next_cursor)
        await asyncio.sleep(PAGE_DELAY_SECONDS)


async def _download_file(http_client: Any, url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".part")
    if temp_path.exists():
        temp_path.unlink()

    headers = {
        "User-Agent": USER_AGENT,
        "Referer": "https://www.douyin.com/",
    }
    async with http_client.stream("GET", url, headers=headers, follow_redirects=True) as response:
        response.raise_for_status()
        with temp_path.open("wb") as output_file:
            async for chunk in response.aiter_bytes():
                if chunk:
                    output_file.write(chunk)

    if temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise RuntimeError("download produced an empty file")

    temp_path.replace(output_path)


async def download_user_videos(
    config: DownloadConfig,
    fetch_user_posts: FetchUserPosts = fetch_f2_user_posts,
    http_client: Any = None,
) -> DownloadResult:
    result = DownloadResult()
    completed = load_completed_aweme_ids(config)
    completed_set = set(completed)
    owns_client = http_client is None

    if owns_client:
        try:
            import httpx  # type: ignore[reportMissingImports]
        except ImportError as exc:
            raise RuntimeError(
                "httpx is required for downloads. Run this script with `uv run "
                "douyin_user_download.py ...` so inline dependencies are installed."
            ) from exc
        http_client = httpx.AsyncClient(timeout=None)

    try:
        async for video in fetch_user_posts(config):
            if config.max_count is not None and result.discovered >= config.max_count:
                break

            result.discovered += 1
            output_path = build_video_path(config.output_root, config.sec_user_id, video)

            if config.dry_run:
                result.dry_run += 1
                print(f"[dry-run] {video.aweme_id}: {video.desc} -> {video.download_url}")
                continue

            if (
                not config.overwrite
                and video.aweme_id in completed_set
                and output_path.exists()
                and output_path.stat().st_size > 0
            ):
                result.skipped += 1
                print(f"[skip] {video.aweme_id}: already downloaded")
                continue

            try:
                print(f"[download] {video.aweme_id}: {video.desc}")
                await _download_file(http_client, video.download_url, output_path)
            except Exception as exc:
                result.failed += 1
                print(f"[error] {video.aweme_id}: {exc}", file=sys.stderr)
                continue

            result.downloaded += 1
            completed.append(video.aweme_id)
            completed = list(dict.fromkeys(completed))
            completed_set = set(completed)
            append_manifest_row(config, video, output_path, "complete")
            write_checkpoint(config, completed)
    finally:
        if owns_client:
            await http_client.aclose()

    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download public posted videos from a Douyin user profile.",
    )
    parser.add_argument("user_url_or_sec_user_id", help="Douyin user URL or raw sec_user_id")
    parser.add_argument(
        "output_dir",
        nargs="?",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=f"Output root directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument("--cookie", help="Raw Douyin Cookie header value")
    parser.add_argument("--cookie-file", type=Path, help="Path to a cookie text file")
    parser.add_argument("--max-count", type=int, help="Maximum number of videos to process")
    parser.add_argument("--page-size", type=int, default=DEFAULT_PAGE_SIZE, help="F2 page size")
    parser.add_argument("--overwrite", action="store_true", help="Redownload completed files")
    parser.add_argument("--dry-run", action="store_true", help="List discovered videos without downloading")
    return parser


def config_from_args(argv: list[str]) -> DownloadConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.max_count is not None and args.max_count < 1:
        parser.error("--max-count must be at least 1")
    if args.page_size < 1:
        parser.error("--page-size must be at least 1")

    cookie = args.cookie
    if args.cookie_file:
        cookie = read_cookie_file(args.cookie_file)
    elif cookie is None and DEFAULT_COOKIE_FILE.exists():
        cookie = read_cookie_file(DEFAULT_COOKIE_FILE)

    return DownloadConfig(
        sec_user_id=parse_sec_user_id(args.user_url_or_sec_user_id),
        output_root=Path(args.output_dir),
        cookie=cookie,
        page_size=args.page_size,
        max_count=args.max_count,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


async def async_main(argv: list[str]) -> int:
    try:
        config = config_from_args(argv)
        result = await download_user_videos(config)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        print(
            "If Douyin requires login, export browser cookies to cookies.txt and rerun with "
            "`--cookie-file cookies.txt`. The script will not automate login or bypass verification.",
            file=sys.stderr,
        )
        return 1
    except KeyboardInterrupt:
        print("\nDownload cancelled by user", file=sys.stderr)
        return 130

    print(
        "Summary: "
        f"discovered={result.discovered} "
        f"downloaded={result.downloaded} "
        f"skipped={result.skipped} "
        f"dry_run={result.dry_run} "
        f"failed={result.failed}"
    )
    return 1 if result.failed else 0


def main() -> None:
    raise SystemExit(asyncio.run(async_main(sys.argv[1:])))


if __name__ == "__main__":
    main()
