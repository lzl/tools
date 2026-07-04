# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "telethon>=1.41.2",
# ]
# ///

"""Download media from one Telegram message link.

This atom is intentionally self-contained: it does not import any local tools,
packages, workflows, or other atoms from this repository.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, Sequence

from telethon import TelegramClient
from telethon.errors import (
    ChannelPrivateError,
    RPCError,
    UsernameInvalidError,
    UsernameNotOccupiedError,
)
from telethon.sessions import StringSession
from telethon.tl import types

MediaType = Literal["image", "video"]
MediaTypeArg = Literal["video", "image", "any"]
ProcessMessageStatus = Literal["downloaded", "skipped", "manifest_backfilled"]


@dataclass(frozen=True)
class MessageLink:
    channel_id: int | str
    message_id: int


@dataclass(frozen=True)
class ChannelMessage:
    channel_id: int | str
    message_id: int
    media_type: MediaType | None
    file_id: str | None
    extension: str | None
    source: Any


@dataclass(frozen=True)
class ManifestRecord:
    channel_id: int | str
    message_id: int
    media_type: MediaType
    file_id: str
    file_path: str
    status: str


@dataclass(frozen=True)
class ProcessMessageResult:
    status: ProcessMessageStatus
    final_path: Path | None
    record: ManifestRecord | None


class TelegramMediaApiProtocol(Protocol):
    async def get_channel_message(
        self,
        channel_id: int | str,
        *,
        message_id: int,
    ) -> ChannelMessage | None: ...

    async def download_media(
        self,
        message: ChannelMessage,
        destination: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str | bytes | None: ...


def load_dotenv_if_present(start_dir: Path | None = None) -> Path | None:
    env_path = find_dotenv(start_dir or Path.cwd())
    if env_path is None:
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value

    return env_path


def find_dotenv(start_dir: Path) -> Path | None:
    current = start_dir.resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_api_credentials() -> tuple[int, str]:
    api_id_text = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    if not api_id_text:
        raise RuntimeError("TELEGRAM_API_ID is required")
    if not api_hash:
        raise RuntimeError("TELEGRAM_API_HASH is required")
    try:
        return int(api_id_text), api_hash
    except ValueError as exc:
        raise RuntimeError("TELEGRAM_API_ID must be an integer") from exc


def load_download_session() -> str:
    session = os.environ.get("TELEGRAM_STRING_SESSION")
    if not session:
        raise RuntimeError("TELEGRAM_STRING_SESSION is required for download runs.")
    return session


def normalize_channel_id(raw_channel_id: int | str) -> int | str:
    text = str(raw_channel_id).strip()
    if not text:
        raise ValueError("channel ID must not be empty")
    numeric = text.lstrip("-")
    if numeric.isdigit():
        if text.startswith("-100"):
            return int(text)
        return int(f"-100{numeric}")
    return text


def parse_message_link(url: str) -> MessageLink:
    base_url = re.sub(r"\?.*$", "", url.strip())
    private_match = re.fullmatch(r"https?://t\.me/c/(\d+)/(\d+)", base_url)
    if private_match:
        return MessageLink(
            channel_id=normalize_channel_id(private_match.group(1)),
            message_id=int(private_match.group(2)),
        )

    public_match = re.fullmatch(
        r"https?://t\.me/([a-zA-Z][a-zA-Z0-9_]{3,30}[a-zA-Z0-9])/(\d+)",
        base_url,
    )
    if public_match:
        return MessageLink(
            channel_id=public_match.group(1),
            message_id=int(public_match.group(2)),
        )

    raise ValueError(f"unsupported Telegram message link: {url}")


def resolve_extension(message: types.Message, default: str) -> str:
    file_wrapper = getattr(message, "file", None)
    extension = getattr(file_wrapper, "ext", None)
    if extension:
        return extension if extension.startswith(".") else f".{extension}"
    return default


def to_channel_message(channel_id: int | str, message: types.Message) -> ChannelMessage:
    if isinstance(message.media, types.MessageMediaPhoto) and message.photo is not None:
        return ChannelMessage(
            channel_id=channel_id,
            message_id=message.id,
            media_type="image",
            file_id=str(message.photo.id),
            extension=resolve_extension(message, ".jpg"),
            source=message,
        )

    if isinstance(message.media, types.MessageMediaDocument) and message.document is not None:
        mime_type = message.document.mime_type or ""
        if mime_type.startswith("video/"):
            return ChannelMessage(
                channel_id=channel_id,
                message_id=message.id,
                media_type="video",
                file_id=str(message.document.id),
                extension=resolve_extension(message, ".mp4"),
                source=message,
            )
        if mime_type.startswith("image/"):
            return ChannelMessage(
                channel_id=channel_id,
                message_id=message.id,
                media_type="image",
                file_id=str(message.document.id),
                extension=resolve_extension(message, ".jpg"),
                source=message,
            )

    return ChannelMessage(
        channel_id=channel_id,
        message_id=message.id,
        media_type=None,
        file_id=None,
        extension=None,
        source=message,
    )


class TelethonMediaApi:
    def __init__(self, client: TelegramClient, *, output_root: Path = Path("input_dir")) -> None:
        self.client = client
        self.output_root = output_root

    async def get_channel_message(
        self,
        channel_id: int | str,
        *,
        message_id: int,
    ) -> ChannelMessage | None:
        try:
            entity = await self.client.get_entity(channel_id)
            raw_message = await self.client.get_messages(entity, ids=message_id)
            if raw_message is None:
                return None
            return to_channel_message(channel_id, raw_message)
        except (ChannelPrivateError, UsernameInvalidError, UsernameNotOccupiedError) as exc:
            raise RuntimeError(
                "The authenticated account cannot access this channel. "
                "Check TELEGRAM_STRING_SESSION and channel visibility."
            ) from exc
        except ValueError as exc:
            raise RuntimeError(f"Could not resolve Telegram channel {channel_id!r}.") from exc
        except RPCError as exc:
            raise RuntimeError(f"Telegram request failed: {exc}") from exc

    async def download_media(
        self,
        message: ChannelMessage,
        destination: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str | bytes | None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            return await self.client.download_media(
                message.source,
                file=destination,
                progress_callback=progress_callback,
            )
        except RPCError as exc:
            if destination.exists():
                destination.unlink()
            raise RuntimeError(
                f"Telegram request failed while downloading message {message.message_id}: {exc}"
            ) from exc


async def create_download_client() -> TelegramClient:
    api_id, api_hash = load_api_credentials()
    session = load_download_session()
    client = TelegramClient(StringSession(session), api_id, api_hash)
    await client.connect()
    if not await client.is_user_authorized():
        await client.disconnect()
        raise RuntimeError("TELEGRAM_STRING_SESSION is invalid or expired.")
    return client


class ChannelStore:
    def __init__(self, output_root: Path, channel_id: int | str) -> None:
        self.channel_id = channel_id
        self.channel_root = output_root / str(channel_id)
        self.images_dir = self.channel_root / "images"
        self.videos_dir = self.channel_root / "videos"
        self.manifest_path = self.channel_root / "manifest.jsonl"
        self._manifest_index = self._load_manifest_index()

    def build_file_paths(self, message: ChannelMessage) -> tuple[Path, Path]:
        if message.media_type is None or message.extension is None:
            raise ValueError("message does not contain downloadable media")
        target_dir = self.images_dir if message.media_type == "image" else self.videos_dir
        extension = message.extension if message.extension.startswith(".") else f".{message.extension}"
        final_path = target_dir / f"{message.message_id}{extension}"
        temp_path = target_dir / f"{message.message_id}{extension}.part"
        return final_path, temp_path

    def get_complete_record(self, message_id: int) -> ManifestRecord | None:
        return self._manifest_index.get(message_id)

    def append_manifest(self, record: ManifestRecord) -> None:
        self.channel_root.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), ensure_ascii=False))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._manifest_index[record.message_id] = record

    def build_record(self, message: ChannelMessage, final_path: Path) -> ManifestRecord:
        if message.media_type is None or message.file_id is None:
            raise ValueError("message does not contain downloadable media")
        return ManifestRecord(
            channel_id=message.channel_id,
            message_id=message.message_id,
            media_type=message.media_type,
            file_id=message.file_id,
            file_path=str(final_path),
            status="complete",
        )

    def _load_manifest_index(self) -> dict[int, ManifestRecord]:
        if not self.manifest_path.exists():
            return {}
        index: dict[int, ManifestRecord] = {}
        for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("status") != "complete":
                continue
            index[int(payload["message_id"])] = ManifestRecord(
                channel_id=payload["channel_id"],
                message_id=int(payload["message_id"]),
                media_type=payload["media_type"],
                file_id=str(payload["file_id"]),
                file_path=payload["file_path"],
                status=payload["status"],
            )
        return index


class ConsoleProgressReporter:
    def __init__(self, stream: Any = None) -> None:
        self.stream = stream if stream is not None else sys.stderr

    def on_download_started(self, *, message: ChannelMessage, final_path: Path) -> None:
        print(
            f"Downloading {message.media_type or 'media'} from message {message.message_id} -> {final_path}",
            file=self.stream,
            flush=True,
        )

    def on_download_progress(
        self,
        *,
        message: ChannelMessage,
        received_bytes: int,
        total_bytes: int | None,
    ) -> None:
        if total_bytes:
            print(
                f"Message {message.message_id}: {received_bytes} / {total_bytes} bytes",
                file=self.stream,
                flush=True,
            )

    def on_download_completed(self, *, message: ChannelMessage, final_path: Path) -> None:
        print(f"Saved message {message.message_id} to {final_path}", file=self.stream, flush=True)

    def on_existing_file_skipped(self, *, message: ChannelMessage, final_path: Path) -> None:
        print(
            f"Skipping message {message.message_id}; already downloaded at {final_path}",
            file=self.stream,
            flush=True,
        )

    def on_manifest_backfilled(self, *, message: ChannelMessage, final_path: Path) -> None:
        print(
            f"Backfilled manifest for message {message.message_id} from existing file {final_path}",
            file=self.stream,
            flush=True,
        )


class DownloadMessageRunner:
    def __init__(
        self,
        api: TelegramMediaApiProtocol,
        *,
        reporter: ConsoleProgressReporter | None = None,
    ) -> None:
        self.api = api
        self.reporter = reporter or ConsoleProgressReporter()

    async def process_message(
        self,
        store: ChannelStore,
        message: ChannelMessage,
    ) -> ProcessMessageResult:
        if message.media_type is None or message.file_id is None or message.extension is None:
            return ProcessMessageResult(status="skipped", final_path=None, record=None)

        final_path, temp_path = store.build_file_paths(message)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        complete_record = store.get_complete_record(message.message_id)

        if complete_record is not None and final_path.exists():
            self.reporter.on_existing_file_skipped(message=message, final_path=final_path)
            return ProcessMessageResult("skipped", final_path, complete_record)

        if final_path.exists() and complete_record is None:
            record = store.build_record(message, final_path)
            store.append_manifest(record)
            self.reporter.on_manifest_backfilled(message=message, final_path=final_path)
            return ProcessMessageResult("manifest_backfilled", final_path, record)

        if temp_path.exists():
            temp_path.unlink()

        self.reporter.on_download_started(message=message, final_path=final_path)
        try:
            download_result = await self.api.download_media(
                message,
                temp_path,
                progress_callback=lambda current, total: self.reporter.on_download_progress(
                    message=message,
                    received_bytes=current,
                    total_bytes=total,
                ),
            )
            produced_path = self.resolve_produced_path(
                download_result=download_result,
                temp_path=temp_path,
                final_path=final_path,
            )
            if produced_path is None:
                raise RuntimeError(
                    f"download for message {message.message_id} did not produce {temp_path}"
                )
            if produced_path != final_path:
                produced_path.replace(final_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        record = store.build_record(message, final_path)
        store.append_manifest(record)
        self.reporter.on_download_completed(message=message, final_path=final_path)
        return ProcessMessageResult("downloaded", final_path, record)

    def resolve_produced_path(
        self,
        *,
        download_result: str | bytes | None,
        temp_path: Path,
        final_path: Path,
    ) -> Path | None:
        if isinstance(download_result, str):
            candidate = Path(download_result)
            if candidate.exists():
                return candidate
        if temp_path.exists():
            return temp_path
        if final_path.exists():
            return final_path
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download media from one Telegram message link.",
    )
    parser.add_argument("link", help="Telegram message link, for example https://t.me/c/123/456.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("input_dir"),
        help="Root directory for downloaded media (default: input_dir).",
    )
    parser.add_argument(
        "--media-type",
        choices=["video", "image", "any"],
        default="video",
        help="Required media type (default: video).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Write a single machine-readable JSON object to stdout.",
    )
    return parser


async def download_message_media(
    *,
    link_text: str,
    output_root: Path,
    media_type: MediaTypeArg,
) -> dict[str, object]:
    link = parse_message_link(link_text)
    client = await create_download_client()
    try:
        api = TelethonMediaApi(client, output_root=output_root)
        message = await api.get_channel_message(
            link.channel_id,
            message_id=link.message_id,
        )
        if message is None:
            raise RuntimeError(f"Message not found: {link.channel_id}/{link.message_id}")
        if message.media_type is None or message.file_id is None or message.extension is None:
            raise RuntimeError(f"Message has no downloadable media: {link.channel_id}/{link.message_id}")
        if media_type != "any" and message.media_type != media_type:
            raise RuntimeError(
                f"Message media type is {message.media_type!r}, expected {media_type!r}."
            )

        store = ChannelStore(output_root, link.channel_id)
        runner = DownloadMessageRunner(api=api, reporter=ConsoleProgressReporter())
        result = await runner.process_message(store, message)
        if result.final_path is None:
            raise RuntimeError(f"Message did not produce a file path: {link.channel_id}/{link.message_id}")

        return {
            "link": link_text,
            "channel_id": link.channel_id,
            "message_id": link.message_id,
            "media_type": message.media_type,
            "file_path": str(result.final_path),
            "status": result.status,
        }
    finally:
        await client.disconnect()


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv_if_present()
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = asyncio.run(
            download_message_media(
                link_text=args.link,
                output_root=args.output_root.expanduser(),
                media_type=args.media_type,
            )
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["file_path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
