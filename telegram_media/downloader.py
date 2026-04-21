from __future__ import annotations

import json
import os
import signal
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

MediaType = Literal["image", "video"]


class DownloadInterrupted(Exception):
    """Raised when the current download should stop immediately."""


@dataclass(frozen=True)
class ChannelMessage:
    channel_id: int | str
    message_id: int
    media_type: MediaType | None
    file_id: str | None
    extension: str | None
    source: Any
    file_size: int | None = None
    dc_id: int | None = None
    input_location: Any = None


@dataclass(frozen=True)
class ManifestRecord:
    channel_id: int | str
    message_id: int
    media_type: MediaType
    file_id: str
    file_path: str
    status: str


class TelegramMediaApi(Protocol):
    async def iter_channel_messages(
        self,
        channel_id: int | str,
        *,
        min_message_id: int,
    ): ...

    async def download_media(
        self,
        message: ChannelMessage,
        destination: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str | bytes | None: ...


class ProgressReporter(Protocol):
    def on_run_started(
        self,
        *,
        channel_id: int | str,
        output_root: Path,
        min_message_id: int,
        full: bool,
    ) -> None: ...

    def on_download_started(self, *, message: ChannelMessage, final_path: Path) -> None: ...

    def on_download_progress(
        self,
        *,
        message: ChannelMessage,
        received_bytes: int,
        total_bytes: int | None,
    ) -> None: ...

    def on_download_completed(self, *, message: ChannelMessage, final_path: Path) -> None: ...

    def on_existing_file_skipped(self, *, message: ChannelMessage, final_path: Path) -> None: ...

    def on_manifest_backfilled(self, *, message: ChannelMessage, final_path: Path) -> None: ...

    def on_interrupt_requested(self) -> None: ...

    def on_run_finished(
        self,
        *,
        last_message_id: int | None,
        interrupted: bool,
    ) -> None: ...


class NullProgressReporter:
    def on_run_started(
        self,
        *,
        channel_id: int | str,
        output_root: Path,
        min_message_id: int,
        full: bool,
    ) -> None:
        return None

    def on_download_started(self, *, message: ChannelMessage, final_path: Path) -> None:
        return None

    def on_download_progress(
        self,
        *,
        message: ChannelMessage,
        received_bytes: int,
        total_bytes: int | None,
    ) -> None:
        return None

    def on_download_completed(self, *, message: ChannelMessage, final_path: Path) -> None:
        return None

    def on_existing_file_skipped(self, *, message: ChannelMessage, final_path: Path) -> None:
        return None

    def on_manifest_backfilled(self, *, message: ChannelMessage, final_path: Path) -> None:
        return None

    def on_interrupt_requested(self) -> None:
        return None

    def on_run_finished(
        self,
        *,
        last_message_id: int | None,
        interrupted: bool,
    ) -> None:
        return None


class ConsoleProgressReporter(NullProgressReporter):
    def __init__(self, stream: Any = None) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self._use_in_place_updates = bool(
            hasattr(self.stream, "isatty") and self.stream.isatty()
        )
        self._progress_line_active = False
        self._last_progress_key: tuple[int, int, int | None] | None = None

    def on_run_started(
        self,
        *,
        channel_id: int | str,
        output_root: Path,
        min_message_id: int,
        full: bool,
    ) -> None:
        mode = "full scan" if full else f"resume after message {min_message_id}"
        self._write(
            f"Starting download for channel {channel_id} into {output_root} ({mode})."
        )

    def on_download_started(self, *, message: ChannelMessage, final_path: Path) -> None:
        media_label = message.media_type or "media"
        self._last_progress_key = None
        self._flush_progress_line()
        self._write(
            f"Downloading {media_label} from message {message.message_id} -> {final_path}"
        )

    def on_download_progress(
        self,
        *,
        message: ChannelMessage,
        received_bytes: int,
        total_bytes: int | None,
    ) -> None:
        progress_key = (message.message_id, received_bytes, total_bytes)
        if progress_key == self._last_progress_key:
            return
        self._last_progress_key = progress_key
        if total_bytes:
            message_text = (
                f"Message {message.message_id}: "
                f"{_format_bytes(received_bytes)} / {_format_bytes(total_bytes)} "
                f"({(received_bytes / total_bytes) * 100:.1f}%)"
            )
        else:
            message_text = (
                f"Message {message.message_id}: downloaded {_format_bytes(received_bytes)}"
            )

        if self._use_in_place_updates:
            self.stream.write(f"\r{message_text}")
            flush = getattr(self.stream, "flush", None)
            if callable(flush):
                flush()
            self._progress_line_active = True
            return

        if total_bytes:
            self._write(message_text)
            return
        self._write(message_text)

    def on_download_completed(self, *, message: ChannelMessage, final_path: Path) -> None:
        self._flush_progress_line()
        self._write(f"Saved message {message.message_id} to {final_path}")

    def on_existing_file_skipped(self, *, message: ChannelMessage, final_path: Path) -> None:
        self._flush_progress_line()
        self._write(f"Skipping message {message.message_id}; already downloaded at {final_path}")

    def on_manifest_backfilled(self, *, message: ChannelMessage, final_path: Path) -> None:
        self._flush_progress_line()
        self._write(
            f"Backfilled manifest for message {message.message_id} from existing file {final_path}"
        )

    def on_interrupt_requested(self) -> None:
        self._flush_progress_line()
        self._write("Ctrl+C received. Stopping now.")

    def on_run_finished(
        self,
        *,
        last_message_id: int | None,
        interrupted: bool,
    ) -> None:
        self._flush_progress_line()
        if interrupted:
            if last_message_id is None:
                self._write("Stopped immediately. No new messages were committed.")
            else:
                self._write(f"Stopped after checkpointing message {last_message_id}.")
            return
        if last_message_id is None:
            self._write("No new downloadable media found.")
        else:
            self._write(f"Finished through message {last_message_id}.")

    def _write(self, text: str) -> None:
        self.stream.write(text + "\n")
        flush = getattr(self.stream, "flush", None)
        if callable(flush):
            flush()

    def _flush_progress_line(self) -> None:
        if not self._progress_line_active:
            return
        self.stream.write("\n")
        flush = getattr(self.stream, "flush", None)
        if callable(flush):
            flush()
        self._progress_line_active = False


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num_bytes} B"


def normalize_channel_id(raw_channel_id: int | str) -> int | str:
    if isinstance(raw_channel_id, int):
        text = str(raw_channel_id)
    else:
        text = raw_channel_id.strip()

    if not text:
        raise ValueError("channel ID must not be empty")

    numeric = text.lstrip("-")
    if numeric.isdigit():
        if text.startswith("-100"):
            return int(text)
        return int(f"-100{numeric}")

    return text


class StopSignal:
    def __init__(self, on_stop_requested: Callable[[], None] | None = None) -> None:
        self.should_stop = False
        self._on_stop_requested = on_stop_requested
        self._notified = False
        self._previous_sigint_handler = None

    def install(self) -> None:
        self._previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)

    def restore(self) -> None:
        if self._previous_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._previous_sigint_handler)
            self._previous_sigint_handler = None

    def _handle_signal(self, _signum: int, _frame: object) -> None:
        if not self._notified and self._on_stop_requested is not None:
            self._on_stop_requested()
            self._notified = True
        self.should_stop = True


class ChannelStore:
    def __init__(self, output_root: Path, channel_id: int | str) -> None:
        self.channel_id = channel_id
        self.channel_root = output_root / str(channel_id)
        self.images_dir = self.channel_root / "images"
        self.videos_dir = self.channel_root / "videos"
        self.manifest_path = self.channel_root / "manifest.jsonl"
        self.checkpoint_path = self.channel_root / "checkpoint.json"
        self._manifest_index = self._load_manifest_index()

    def load_checkpoint(self) -> int:
        if not self.checkpoint_path.exists():
            return 0

        payload = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        return int(payload.get("last_message_id", 0))

    def checkpoint(self, message_id: int) -> None:
        self.channel_root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.checkpoint_path.with_suffix(".json.tmp")
        payload = {
            "channel_id": self.channel_id,
            "last_message_id": message_id,
        }
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        tmp_path.replace(self.checkpoint_path)

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


class DownloadChannelMediaRunner:
    def __init__(
        self,
        api: TelegramMediaApi,
        output_root: Path,
        *,
        reporter: ProgressReporter | None = None,
        stop_signal: StopSignal | None = None,
    ) -> None:
        self.api = api
        self.output_root = output_root
        self.reporter = reporter or NullProgressReporter()
        self.stop_signal = stop_signal or StopSignal(
            on_stop_requested=self.reporter.on_interrupt_requested
        )

    async def run(self, channel_id: int | str, *, full: bool) -> None:
        normalized_channel_id = normalize_channel_id(channel_id)
        store = ChannelStore(self.output_root, normalized_channel_id)
        min_message_id = 0 if full else store.load_checkpoint()
        last_message_id: int | None = None

        self.reporter.on_run_started(
            channel_id=normalized_channel_id,
            output_root=self.output_root,
            min_message_id=min_message_id,
            full=full,
        )

        self.stop_signal.install()
        try:
            async for message in self.api.iter_channel_messages(
                normalized_channel_id,
                min_message_id=min_message_id,
            ):
                if self.stop_signal.should_stop:
                    break
                try:
                    await self._process_message(store, message)
                except DownloadInterrupted:
                    break
                store.checkpoint(message.message_id)
                last_message_id = message.message_id
                if self.stop_signal.should_stop:
                    break
        finally:
            self.stop_signal.restore()
            self.reporter.on_run_finished(
                last_message_id=last_message_id,
                interrupted=self.stop_signal.should_stop,
            )

    async def _process_message(self, store: ChannelStore, message: ChannelMessage) -> None:
        if message.media_type is None or message.file_id is None or message.extension is None:
            return

        final_path, temp_path = store.build_file_paths(message)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        complete_record = store.get_complete_record(message.message_id)

        if complete_record is not None and final_path.exists():
            self.reporter.on_existing_file_skipped(message=message, final_path=final_path)
            return

        if final_path.exists() and complete_record is None:
            store.append_manifest(store.build_record(message, final_path))
            self.reporter.on_manifest_backfilled(message=message, final_path=final_path)
            return

        if temp_path.exists():
            temp_path.unlink()

        self.reporter.on_download_started(message=message, final_path=final_path)
        try:
            download_result = await self.api.download_media(
                message,
                temp_path,
                progress_callback=lambda current, total: self._handle_download_progress(
                    message=message,
                    received_bytes=current,
                    total_bytes=total,
                ),
            )
            produced_path = self._resolve_produced_path(
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
        except DownloadInterrupted:
            if temp_path.exists():
                temp_path.unlink()
            if final_path.exists():
                final_path.unlink()
            raise
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        store.append_manifest(store.build_record(message, final_path))
        self.reporter.on_download_completed(message=message, final_path=final_path)

    def _resolve_produced_path(
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

    def _handle_download_progress(
        self,
        *,
        message: ChannelMessage,
        received_bytes: int,
        total_bytes: int | None,
    ) -> None:
        if self.stop_signal.should_stop:
            raise DownloadInterrupted()
        self.reporter.on_download_progress(
            message=message,
            received_bytes=received_bytes,
            total_bytes=total_bytes,
        )
        if self.stop_signal.should_stop:
            raise DownloadInterrupted()
