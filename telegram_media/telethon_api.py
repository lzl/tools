from __future__ import annotations

import asyncio
import hashlib
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from telethon import TelegramClient, functions, utils
from telethon.crypto.authkey import AuthKey
from telethon.errors import (
    ChannelPrivateError,
    FloodWaitError,
    RPCError,
    SessionPasswordNeededError,
    UsernameInvalidError,
    UsernameNotOccupiedError,
)
from telethon.network.mtprotosender import MTProtoSender
from telethon.sessions import StringSession
from telethon.tl import types

from telegram_media.downloader import ChannelMessage, DownloadInterrupted
from telegram_media.session import (
    load_api_credentials,
    load_download_session,
    prompt_code,
    prompt_password,
    prompt_phone,
)

SAFE_PARALLEL_VIDEO_SHARD_COUNT = 4
PARALLEL_VIDEO_SHARD_COUNT = SAFE_PARALLEL_VIDEO_SHARD_COUNT
PARALLEL_VIDEO_MIN_BYTES = 64 * 1024 * 1024
PARALLEL_VIDEO_REQUEST_SIZE = 512 * 1024
RUNTIME_STATE_DIRNAME = "_runtime"
RATE_LIMIT_FILENAME = "rate_limits.json"


def _resolve_extension(message: types.Message, default: str) -> str:
    file_wrapper = getattr(message, "file", None)
    extension = getattr(file_wrapper, "ext", None)
    if extension:
        return extension if extension.startswith(".") else f".{extension}"
    return default


def _format_wait_duration(seconds: int) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {remaining_minutes}m {remaining_seconds}s"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


class RateLimitStop(RuntimeError):
    """Raised when Telegram rate limiting should fail the current run closed."""


class RateLimitStore:
    def __init__(self, output_root: Path, account_fingerprint: str) -> None:
        self.path = output_root / RUNTIME_STATE_DIRNAME / RATE_LIMIT_FILENAME
        self.account_fingerprint = account_fingerprint

    def get_remaining_seconds(
        self,
        *,
        source_dc_id: int,
        target_dc_id: int,
        request_kind: str,
    ) -> int | None:
        entries = self._load_entries()
        key = self._build_key(
            source_dc_id=source_dc_id,
            target_dc_id=target_dc_id,
            request_kind=request_kind,
        )
        entry = entries.get(key)
        if entry is None:
            return None

        remaining_seconds = int(entry["wait_until"]) - int(time.time())
        if remaining_seconds > 0:
            return remaining_seconds

        del entries[key]
        self._save_entries(entries)
        return None

    def record_cooldown(
        self,
        *,
        source_dc_id: int,
        target_dc_id: int,
        request_kind: str,
        wait_seconds: int,
    ) -> None:
        entries = self._load_entries()
        key = self._build_key(
            source_dc_id=source_dc_id,
            target_dc_id=target_dc_id,
            request_kind=request_kind,
        )
        entries[key] = {
            "account_fingerprint": self.account_fingerprint,
            "source_dc_id": source_dc_id,
            "target_dc_id": target_dc_id,
            "request_kind": request_kind,
            "wait_until": int(time.time()) + max(1, int(wait_seconds)),
        }
        self._save_entries(entries)

    def _build_key(
        self,
        *,
        source_dc_id: int,
        target_dc_id: int,
        request_kind: str,
    ) -> str:
        return (
            f"{self.account_fingerprint}:"
            f"{source_dc_id}:{target_dc_id}:{request_kind}"
        )

    def _load_entries(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {
            str(key): value
            for key, value in payload.items()
            if isinstance(value, dict)
        }

    def _save_entries(self, entries: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(entries, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.path)


class TelethonMediaApi:
    def __init__(self, client: TelegramClient, *, output_root: Path = Path("data/telegram")) -> None:
        self.client = client
        self.output_root = output_root
        self._parallel_auth_key_bytes: dict[int, bytes] = {}
        self._parallel_auth_key_lock = asyncio.Lock()
        self._rate_limit_store = RateLimitStore(
            output_root=output_root,
            account_fingerprint=self._build_account_fingerprint(),
        )

    async def iter_channel_messages(
        self,
        channel_id: int | str,
        *,
        min_message_id: int,
    ):
        try:
            entity = await self.client.get_entity(channel_id)
            async for raw_message in self.client.iter_messages(
                entity,
                min_id=min_message_id,
                reverse=True,
            ):
                yield self._to_channel_message(channel_id, raw_message)
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
        try:
            if self.should_use_parallel_video_download(message):
                try:
                    return await self._download_video_in_parallel(
                        message,
                        destination,
                        progress_callback=progress_callback,
                    )
                except DownloadInterrupted:
                    raise
                except RateLimitStop:
                    raise
                except RPCError as exc:
                    self._raise_for_rpc_error(
                        exc,
                        source_dc_id=self._session_dc_id(),
                        target_dc_id=message.dc_id or self._session_dc_id(),
                        request_kind="download",
                        message=message,
                    )
                except Exception:
                    return await self._download_media_single(
                        message,
                        destination,
                        progress_callback=progress_callback,
                    )

            return await self._download_media_single(
                message,
                destination,
                progress_callback=progress_callback,
            )
        except DownloadInterrupted:
            raise
        except RateLimitStop:
            raise
        except RPCError as exc:
            self._raise_for_rpc_error(
                exc,
                source_dc_id=self._session_dc_id(),
                target_dc_id=message.dc_id or self._session_dc_id(),
                request_kind="download",
                message=message,
            )

    def _describe_download_rpc_error(
        self,
        exc: RPCError,
        message: ChannelMessage,
    ) -> str:
        if isinstance(exc, FloodWaitError):
            session_dc_id = getattr(getattr(self.client, "session", None), "dc_id", None)
            if message.dc_id is not None and session_dc_id not in (None, message.dc_id):
                return (
                    "Telegram is temporarily throttling cross-DC downloads "
                    f"from session DC {session_dc_id} to media DC {message.dc_id}. "
                    f"Retry in about {_format_wait_duration(exc.seconds)}."
                )
            return (
                "Telegram is temporarily throttling downloads. "
                f"Retry in about {_format_wait_duration(exc.seconds)}."
            )

        return f"Telegram request failed while downloading message {message.message_id}: {exc}"

    async def _download_media_single(
        self,
        message: ChannelMessage,
        destination: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str | bytes | None:
        self._raise_if_rate_limited(
            source_dc_id=self._session_dc_id(),
            target_dc_id=message.dc_id or self._session_dc_id(),
            request_kind="download",
            message=message,
        )
        if self.is_cross_dc_message(message):
            self._raise_if_rate_limited(
                source_dc_id=self._session_dc_id(),
                target_dc_id=message.dc_id or self._session_dc_id(),
                request_kind="export_auth",
                message=message,
            )

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
            self._raise_for_rpc_error(
                exc,
                source_dc_id=self._session_dc_id(),
                target_dc_id=message.dc_id or self._session_dc_id(),
                request_kind="download",
                message=message,
            )

    async def _download_video_in_parallel(
        self,
        message: ChannelMessage,
        destination: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str:
        if message.file_size is None or message.input_location is None or message.dc_id is None:
            raise RuntimeError("parallel video download requires file_size, dc_id, and input_location")
        self._raise_if_rate_limited(
            source_dc_id=self._session_dc_id(),
            target_dc_id=message.dc_id,
            request_kind="download",
            message=message,
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        shard_dir = self._build_shard_dir(destination)
        shutil.rmtree(shard_dir, ignore_errors=True)
        shard_dir.mkdir(parents=True, exist_ok=True)

        progress_by_worker: dict[int, int] = {}
        progress_lock = asyncio.Lock()
        senders: list[Any] = []
        tasks: list[asyncio.Task[None]] = []

        total_chunks = max(1, math.ceil(message.file_size / PARALLEL_VIDEO_REQUEST_SIZE))
        worker_count = min(SAFE_PARALLEL_VIDEO_SHARD_COUNT, total_chunks)
        chunk_ranges = self._build_chunk_ranges(
            file_size=message.file_size,
            worker_count=worker_count,
        )

        try:
            for _ in range(worker_count):
                senders.append(await self._create_parallel_sender(message.dc_id))

            async def run_worker(worker_index: int, sender: Any) -> None:
                start_chunk, end_chunk = chunk_ranges[worker_index]
                shard_path = shard_dir / f"part-{worker_index:02}.bin"
                with shard_path.open("wb") as shard_handle:
                    for chunk_index in range(start_chunk, end_chunk):
                        offset = chunk_index * PARALLEL_VIDEO_REQUEST_SIZE
                        chunk = await self._fetch_file_chunk(
                            sender=sender,
                            input_location=message.input_location,
                            offset=offset,
                            limit=PARALLEL_VIDEO_REQUEST_SIZE,
                        )
                        if not chunk:
                            break
                        shard_handle.write(chunk)
                        async with progress_lock:
                            progress_by_worker[worker_index] = (
                                progress_by_worker.get(worker_index, 0) + len(chunk)
                            )
                            if progress_callback is not None:
                                progress_callback(
                                    sum(progress_by_worker.values()),
                                    message.file_size,
                                )
                        if len(chunk) < PARALLEL_VIDEO_REQUEST_SIZE:
                            break

            tasks = [
                asyncio.create_task(run_worker(worker_index, senders[worker_index]))
                for worker_index in range(worker_count)
            ]
            await asyncio.gather(*tasks)

            with destination.open("wb") as destination_handle:
                for worker_index in range(worker_count):
                    shard_path = shard_dir / f"part-{worker_index:02}.bin"
                    if not shard_path.exists():
                        continue
                    with shard_path.open("rb") as shard_handle:
                        shutil.copyfileobj(shard_handle, destination_handle)
        except Exception:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if destination.exists():
                destination.unlink()
            shutil.rmtree(shard_dir, ignore_errors=True)
            raise
        finally:
            for sender in senders:
                await self._disconnect_parallel_sender(sender, message.dc_id)

        shutil.rmtree(shard_dir, ignore_errors=True)
        return str(destination)

    def _to_channel_message(self, channel_id: int | str, message: types.Message) -> ChannelMessage:
        if isinstance(message.media, types.MessageMediaPhoto) and message.photo is not None:
            dc_id, input_location = utils.get_input_location(message.photo)
            photo_size = None
            if message.photo.sizes:
                photo_size = getattr(message.photo.sizes[-1], "size", None)
            return ChannelMessage(
                channel_id=channel_id,
                message_id=message.id,
                media_type="image",
                file_id=str(message.photo.id),
                extension=_resolve_extension(message, ".jpg"),
                source=message,
                file_size=photo_size,
                dc_id=dc_id,
                input_location=input_location,
            )

        if isinstance(message.media, types.MessageMediaDocument) and message.document is not None:
            mime_type = message.document.mime_type or ""
            dc_id, input_location = utils.get_input_location(message.document)
            if mime_type.startswith("video/"):
                return ChannelMessage(
                    channel_id=channel_id,
                    message_id=message.id,
                    media_type="video",
                    file_id=str(message.document.id),
                    extension=_resolve_extension(message, ".mp4"),
                    source=message,
                    file_size=message.document.size,
                    dc_id=dc_id,
                    input_location=input_location,
                )
            if mime_type.startswith("image/"):
                return ChannelMessage(
                    channel_id=channel_id,
                    message_id=message.id,
                    media_type="image",
                    file_id=str(message.document.id),
                    extension=_resolve_extension(message, ".jpg"),
                    source=message,
                    file_size=message.document.size,
                    dc_id=dc_id,
                    input_location=input_location,
                )

        return ChannelMessage(
            channel_id=channel_id,
            message_id=message.id,
            media_type=None,
            file_id=None,
            extension=None,
            source=message,
        )

    def should_use_parallel_video_download(self, message: ChannelMessage) -> bool:
        return (
            message.media_type == "video"
            and message.file_size is not None
            and message.file_size >= PARALLEL_VIDEO_MIN_BYTES
            and message.dc_id is not None
            and message.input_location is not None
            and not self.is_cross_dc_message(message)
            and not self._has_active_rate_limit(
                source_dc_id=self._session_dc_id(),
                target_dc_id=message.dc_id,
                request_kind="download",
            )
        )

    def is_cross_dc_message(self, message: ChannelMessage) -> bool:
        return message.dc_id is not None and message.dc_id != self._session_dc_id()

    def classify_rpc_error(self, exc: RPCError) -> str:
        error_name = type(exc).__name__.lower()
        if isinstance(exc, FloodWaitError) or "flood" in error_name:
            return "throttle"
        return "rpc"

    def _build_shard_dir(self, destination: Path) -> Path:
        return destination.parent / f"{destination.name.removesuffix('.part')}.parts"

    def _build_chunk_ranges(
        self,
        *,
        file_size: int,
        worker_count: int,
    ) -> list[tuple[int, int]]:
        total_chunks = max(1, math.ceil(file_size / PARALLEL_VIDEO_REQUEST_SIZE))
        base_chunk_count, remainder = divmod(total_chunks, worker_count)
        ranges: list[tuple[int, int]] = []
        current_chunk = 0
        for worker_index in range(worker_count):
            chunk_count = base_chunk_count + (1 if worker_index < remainder else 0)
            ranges.append((current_chunk, current_chunk + chunk_count))
            current_chunk += chunk_count
        return ranges

    async def _create_parallel_sender(self, dc_id: int):
        auth_key = await self._get_parallel_auth_key(dc_id)
        return await self._connect_parallel_sender(dc_id, auth_key)

    async def _get_parallel_auth_key(self, dc_id: int) -> AuthKey:
        if self.client.session.dc_id == dc_id:
            return AuthKey(bytes(self.client.session.auth_key.key))

        self._raise_if_rate_limited(
            source_dc_id=self._session_dc_id(),
            target_dc_id=dc_id,
            request_kind="export_auth",
        )

        async with self._parallel_auth_key_lock:
            cached_key = self._parallel_auth_key_bytes.get(dc_id)
            if cached_key is None:
                try:
                    borrowed_sender = await self.client._borrow_exported_sender(dc_id)
                except RPCError as exc:
                    self._raise_for_rpc_error(
                        exc,
                        source_dc_id=self._session_dc_id(),
                        target_dc_id=dc_id,
                        request_kind="export_auth",
                    )
                try:
                    if not borrowed_sender.auth_key or not borrowed_sender.auth_key.key:
                        raise RuntimeError(f"borrowed sender for dc {dc_id} has no auth key")
                    cached_key = bytes(borrowed_sender.auth_key.key)
                    self._parallel_auth_key_bytes[dc_id] = cached_key
                finally:
                    await self.client._return_exported_sender(borrowed_sender)

        return AuthKey(cached_key)

    async def _connect_parallel_sender(self, dc_id: int, auth_key: AuthKey):
        dc = await self.client._get_dc(dc_id)
        sender = MTProtoSender(
            auth_key,
            loggers=self.client._log,
            retries=self.client._connection_retries,
            delay=self.client._retry_delay,
            auto_reconnect=self.client._auto_reconnect,
            connect_timeout=self.client._timeout,
            auth_key_callback=self.client._auth_key_callback,
            updates_queue=self.client._updates_queue,
            auto_reconnect_callback=self.client._handle_auto_reconnect,
        )
        await sender.connect(
            self.client._connection(
                dc.ip_address,
                dc.port,
                dc.id,
                loggers=self.client._log,
                proxy=self.client._proxy,
                local_addr=self.client._local_addr,
            )
        )
        return sender

    async def _disconnect_parallel_sender(self, sender: Any, dc_id: int) -> None:
        try:
            await sender.disconnect()
        except Exception:
            return None

    async def _fetch_file_chunk(
        self,
        *,
        sender: Any,
        input_location: Any,
        offset: int,
        limit: int,
    ) -> bytes:
        result = await self.client._call(
            sender,
            functions.upload.GetFileRequest(
                location=input_location,
                offset=offset,
                limit=limit,
                precise=True,
                cdn_supported=True,
            ),
        )
        if isinstance(result, types.upload.FileCdnRedirect):
            raise RuntimeError("parallel downloader does not support CDN redirects")
        if isinstance(result, types.upload.CdnFileReuploadNeeded):
            raise RuntimeError("parallel downloader does not support CDN reupload")
        return result.bytes

    def _build_account_fingerprint(self) -> str:
        auth_key = getattr(getattr(self.client, "session", None), "auth_key", None)
        auth_key_bytes = getattr(auth_key, "key", None)
        if not auth_key_bytes:
            return "unknown-account"
        return hashlib.sha256(bytes(auth_key_bytes)).hexdigest()[:16]

    def _session_dc_id(self) -> int:
        return int(getattr(getattr(self.client, "session", None), "dc_id", 0))

    def _has_active_rate_limit(
        self,
        *,
        source_dc_id: int,
        target_dc_id: int,
        request_kind: str,
    ) -> bool:
        return (
            self._rate_limit_store.get_remaining_seconds(
                source_dc_id=source_dc_id,
                target_dc_id=target_dc_id,
                request_kind=request_kind,
            )
            is not None
        )

    def _raise_if_rate_limited(
        self,
        *,
        source_dc_id: int,
        target_dc_id: int,
        request_kind: str,
        message: ChannelMessage | None = None,
    ) -> None:
        remaining_seconds = self._rate_limit_store.get_remaining_seconds(
            source_dc_id=source_dc_id,
            target_dc_id=target_dc_id,
            request_kind=request_kind,
        )
        if remaining_seconds is None:
            return
        raise RateLimitStop(
            self._describe_rate_limit(
                source_dc_id=source_dc_id,
                target_dc_id=target_dc_id,
                request_kind=request_kind,
                wait_seconds=remaining_seconds,
                message=message,
            )
        )

    def _raise_for_rpc_error(
        self,
        exc: RPCError,
        *,
        source_dc_id: int,
        target_dc_id: int,
        request_kind: str,
        message: ChannelMessage | None = None,
    ) -> None:
        if self.classify_rpc_error(exc) == "throttle":
            wait_seconds = getattr(exc, "seconds", None)
            if isinstance(wait_seconds, int) and wait_seconds > 0:
                self._rate_limit_store.record_cooldown(
                    source_dc_id=source_dc_id,
                    target_dc_id=target_dc_id,
                    request_kind=request_kind,
                    wait_seconds=wait_seconds,
                )
            raise RateLimitStop(
                self._describe_rate_limit(
                    source_dc_id=source_dc_id,
                    target_dc_id=target_dc_id,
                    request_kind=request_kind,
                    wait_seconds=wait_seconds,
                    message=message,
                )
            ) from exc

        if message is None:
            raise RuntimeError(f"Telegram request failed: {exc}") from exc
        raise RuntimeError(f"Telegram request failed while downloading message {message.message_id}: {exc}") from exc

    def _describe_rate_limit(
        self,
        *,
        source_dc_id: int,
        target_dc_id: int,
        request_kind: str,
        wait_seconds: int | None,
        message: ChannelMessage | None = None,
    ) -> str:
        if wait_seconds is None:
            wait_clause = "Try again later."
        else:
            wait_clause = f"Retry in about {_format_wait_duration(wait_seconds)}."

        if request_kind == "export_auth" or source_dc_id != target_dc_id:
            return (
                "Telegram is temporarily throttling cross-DC downloads "
                f"from session DC {source_dc_id} to media DC {target_dc_id}. "
                f"{wait_clause}"
            )

        if message is None:
            return f"Telegram is temporarily throttling downloads. {wait_clause}"
        return (
            f"Telegram is temporarily throttling downloads for message {message.message_id}. "
            f"{wait_clause}"
        )


async def create_download_client() -> TelegramClient:
    api_id, api_hash = load_api_credentials()
    session = load_download_session()
    client = TelegramClient(StringSession(session), api_id, api_hash)
    await client.connect()
    if not await client.is_user_authorized():
        await client.disconnect()
        raise RuntimeError(
            "TELEGRAM_STRING_SESSION is invalid or expired. "
            "Generate a fresh one with `uv run telegram-media generate-session`."
        )
    return client


async def generate_string_session(phone: str | None = None) -> str:
    api_id, api_hash = load_api_credentials()
    client = TelegramClient(StringSession(), api_id, api_hash)
    await client.connect()
    try:
        login_phone = phone or prompt_phone()
        await client.send_code_request(login_phone)
        try:
            await client.sign_in(phone=login_phone, code=prompt_code())
        except SessionPasswordNeededError:
            await client.sign_in(password=prompt_password())
        session = client.session.save()
    finally:
        await client.disconnect()

    if not session:
        raise RuntimeError("Telethon did not return a string session.")
    return session
