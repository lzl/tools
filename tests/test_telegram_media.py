import asyncio
import io
import json
import os
import tempfile
import unittest
from importlib import import_module
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from telegram_media.cli import build_parser
from telegram_media.downloader import (
    ChannelMessage,
    ConsoleProgressReporter,
    DownloadChannelMediaRunner,
    StopSignal,
    normalize_channel_id,
)
from telegram_media.session import load_dotenv_if_present


class FakeTelegramApi:
    def __init__(self, messages: list[ChannelMessage]) -> None:
        self._messages = messages
        self.download_calls: list[tuple[int, str]] = []

    async def iter_channel_messages(
        self,
        channel_id: int | str,
        *,
        min_message_id: int,
    ):
        for message in self._messages:
            if message.message_id > min_message_id:
                yield message

    async def download_media(
        self,
        message: ChannelMessage,
        destination: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.download_calls.append((message.message_id, destination.name))
        payload = message.source if isinstance(message.source, bytes) else b"payload"
        if progress_callback is not None:
            progress_callback(len(payload), len(payload))
        destination.write_bytes(payload)


class FinalPathTelegramApi(FakeTelegramApi):
    async def download_media(
        self,
        message: ChannelMessage,
        destination: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> str:
        self.download_calls.append((message.message_id, destination.name))
        final_path = Path(str(destination).removesuffix(".part"))
        payload = message.source if isinstance(message.source, bytes) else b"payload"
        if progress_callback is not None:
            progress_callback(len(payload), len(payload))
        final_path.write_bytes(payload)
        return str(final_path)


class ProgressCallbackTelegramApi(FakeTelegramApi):
    async def download_media(
        self,
        message: ChannelMessage,
        destination: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.download_calls.append((message.message_id, destination.name))
        payload = message.source if isinstance(message.source, bytes) else b"payload"
        if progress_callback is not None:
            progress_callback(2, len(payload))
            progress_callback(len(payload), len(payload))
        destination.write_bytes(payload)


class InterruptingTelegramApi(FakeTelegramApi):
    def __init__(
        self,
        messages: list[ChannelMessage],
        *,
        interrupt: Callable[[], None],
    ) -> None:
        super().__init__(messages)
        self._interrupt = interrupt

    async def download_media(
        self,
        message: ChannelMessage,
        destination: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.download_calls.append((message.message_id, destination.name))
        self._interrupt()
        if progress_callback is not None:
            progress_callback(1, 10)
        destination.write_bytes(b"should-not-complete")


class RecordingReporter:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def on_run_started(
        self,
        *,
        channel_id: int | str,
        output_root: Path,
        min_message_id: int,
        full: bool,
    ) -> None:
        self.events.append(("run_started", channel_id, min_message_id, full))

    def on_download_started(self, *, message: ChannelMessage, final_path: Path) -> None:
        self.events.append(("download_started", message.message_id, str(final_path)))

    def on_download_progress(
        self,
        *,
        message: ChannelMessage,
        received_bytes: int,
        total_bytes: int | None,
    ) -> None:
        self.events.append(
            ("download_progress", message.message_id, received_bytes, total_bytes)
        )

    def on_download_completed(self, *, message: ChannelMessage, final_path: Path) -> None:
        self.events.append(("download_completed", message.message_id, str(final_path)))

    def on_existing_file_skipped(self, *, message: ChannelMessage, final_path: Path) -> None:
        self.events.append(("existing_skipped", message.message_id, str(final_path)))

    def on_manifest_backfilled(self, *, message: ChannelMessage, final_path: Path) -> None:
        self.events.append(("manifest_backfilled", message.message_id, str(final_path)))

    def on_interrupt_requested(self) -> None:
        self.events.append(("interrupt_requested",))

    def on_run_finished(
        self,
        *,
        last_message_id: int | None,
        interrupted: bool,
    ) -> None:
        self.events.append(("run_finished", last_message_id, interrupted))


class FakeTTYStream(io.StringIO):
    def isatty(self) -> bool:
        return True


class NormalizeChannelIdTests(unittest.TestCase):
    def test_adds_full_channel_prefix_for_short_negative_ids(self) -> None:
        self.assertEqual(normalize_channel_id("-1234567890"), -1001234567890)

    def test_preserves_full_channel_ids(self) -> None:
        self.assertEqual(normalize_channel_id("-1001234567890"), -1001234567890)

    def test_preserves_usernames(self) -> None:
        self.assertEqual(normalize_channel_id("my_channel"), "my_channel")


class CliTests(unittest.TestCase):
    def test_telethon_api_module_imports_with_installed_telethon(self) -> None:
        module = import_module("telegram_media.telethon_api")

        self.assertTrue(hasattr(module, "TelethonMediaApi"))

    def test_download_subcommand_uses_expected_defaults(self) -> None:
        parser = build_parser()

        args = parser.parse_args(
            ["download-channel-media", "--channel-id", "-1234567890"]
        )

        self.assertEqual(args.command, "download-channel-media")
        self.assertEqual(args.channel_id, "-1234567890")
        self.assertEqual(args.output_root, Path("data/telegram"))
        self.assertFalse(args.full)


class DotenvLoadingTests(unittest.TestCase):
    def test_loads_telegram_values_from_cwd_dotenv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cwd = Path(temp_dir)
            (cwd / ".env").write_text(
                "TELEGRAM_API_ID=12345\n"
                "TELEGRAM_API_HASH=hash-from-dotenv\n"
                "TELEGRAM_STRING_SESSION=session-from-dotenv\n",
                encoding="utf-8",
            )

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("TELEGRAM_API_ID", None)
                os.environ.pop("TELEGRAM_API_HASH", None)
                os.environ.pop("TELEGRAM_STRING_SESSION", None)

                load_dotenv_if_present(cwd)

                self.assertEqual(os.environ["TELEGRAM_API_ID"], "12345")
                self.assertEqual(os.environ["TELEGRAM_API_HASH"], "hash-from-dotenv")
                self.assertEqual(
                    os.environ["TELEGRAM_STRING_SESSION"],
                    "session-from-dotenv",
                )

    def test_does_not_override_existing_environment_variables(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cwd = Path(temp_dir)
            (cwd / ".env").write_text(
                "TELEGRAM_STRING_SESSION=session-from-dotenv\n",
                encoding="utf-8",
            )

            with patch.dict(
                os.environ,
                {"TELEGRAM_STRING_SESSION": "session-from-env"},
                clear=False,
            ):
                load_dotenv_if_present(cwd)

                self.assertEqual(
                    os.environ["TELEGRAM_STRING_SESSION"],
                    "session-from-env",
                )


class StopSignalTests(unittest.TestCase):
    def test_interrupt_callback_fires_once(self) -> None:
        events: list[str] = []
        stop_signal = StopSignal(on_stop_requested=lambda: events.append("interrupt"))

        stop_signal._handle_signal(2, None)
        stop_signal._handle_signal(2, None)

        self.assertTrue(stop_signal.should_stop)
        self.assertEqual(events, ["interrupt"])


class ConsoleProgressReporterTests(unittest.TestCase):
    def test_uses_single_line_progress_updates_on_tty(self) -> None:
        stream = FakeTTYStream()
        reporter = ConsoleProgressReporter(stream=stream)
        message = ChannelMessage(
            channel_id=-1001234567890,
            message_id=425,
            media_type="video",
            file_id="video-425",
            extension=".mp4",
            source=None,
        )
        final_path = Path("data/telegram/-1001234567890/videos/425.mp4")

        reporter.on_download_started(message=message, final_path=final_path)
        reporter.on_download_progress(message=message, received_bytes=128 * 1024, total_bytes=1024 * 1024)
        reporter.on_download_progress(message=message, received_bytes=256 * 1024, total_bytes=1024 * 1024)
        reporter.on_download_completed(message=message, final_path=final_path)

        output = stream.getvalue()
        self.assertIn("Downloading video from message 425", output)
        self.assertIn("\rMessage 425: 128.0 KiB / 1.0 MiB (12.5%)", output)
        self.assertIn("\rMessage 425: 256.0 KiB / 1.0 MiB (25.0%)", output)
        self.assertIn("Saved message 425", output)
        self.assertEqual(output.count("\r"), 2)
        self.assertEqual(output.count("\n"), 3)


class DownloadChannelMediaRunnerTests(unittest.TestCase):
    def test_first_run_downloads_media_and_updates_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            channel_id = -1001234567890
            messages = [
                ChannelMessage(
                    channel_id=channel_id,
                    message_id=1,
                    media_type=None,
                    file_id=None,
                    extension=None,
                    source=None,
                ),
                ChannelMessage(
                    channel_id=channel_id,
                    message_id=2,
                    media_type="image",
                    file_id="image-2",
                    extension=".jpg",
                    source=b"image-2",
                ),
                ChannelMessage(
                    channel_id=channel_id,
                    message_id=3,
                    media_type="video",
                    file_id="video-3",
                    extension=".mp4",
                    source=b"video-3",
                ),
            ]
            api = FakeTelegramApi(messages)
            runner = DownloadChannelMediaRunner(api=api, output_root=output_root)

            asyncio.run(runner.run(channel_id=channel_id, full=False))

            channel_root = output_root / str(channel_id)
            self.assertEqual(
                api.download_calls,
                [(2, "2.jpg.part"), (3, "3.mp4.part")],
            )
            self.assertTrue((channel_root / "images" / "2.jpg").exists())
            self.assertTrue((channel_root / "videos" / "3.mp4").exists())
            checkpoint = json.loads((channel_root / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["last_message_id"], 3)
            manifest_rows = [
                json.loads(line)
                for line in (channel_root / "manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [row["message_id"] for row in manifest_rows],
                [2, 3],
            )

    def test_second_run_with_no_new_media_downloads_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            channel_id = -1001234567890
            messages = [
                ChannelMessage(
                    channel_id=channel_id,
                    message_id=2,
                    media_type="image",
                    file_id="image-2",
                    extension=".jpg",
                    source=b"image-2",
                )
            ]
            api = FakeTelegramApi(messages)
            runner = DownloadChannelMediaRunner(api=api, output_root=output_root)

            asyncio.run(runner.run(channel_id=channel_id, full=False))
            api.download_calls.clear()

            asyncio.run(runner.run(channel_id=channel_id, full=False))

            self.assertEqual(api.download_calls, [])

    def test_existing_file_without_manifest_is_backfilled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            channel_id = -1001234567890
            channel_root = output_root / str(channel_id)
            image_path = channel_root / "images" / "2.jpg"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"existing")

            api = FakeTelegramApi(
                [
                    ChannelMessage(
                        channel_id=channel_id,
                        message_id=2,
                        media_type="image",
                        file_id="image-2",
                        extension=".jpg",
                        source=b"image-2",
                    )
                ]
            )
            runner = DownloadChannelMediaRunner(api=api, output_root=output_root)

            asyncio.run(runner.run(channel_id=channel_id, full=True))

            self.assertEqual(api.download_calls, [])
            manifest_rows = [
                json.loads(line)
                for line in (channel_root / "manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(manifest_rows), 1)
            self.assertEqual(manifest_rows[0]["status"], "complete")

    def test_missing_file_is_redownloaded_even_if_manifest_claims_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            channel_id = -1001234567890
            channel_root = output_root / str(channel_id)
            channel_root.mkdir(parents=True, exist_ok=True)
            manifest_path = channel_root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "channel_id": channel_id,
                        "message_id": 2,
                        "media_type": "image",
                        "file_id": "image-2",
                        "file_path": str(channel_root / "images" / "2.jpg"),
                        "status": "complete",
                    }
                )
                + "\n"
            )

            api = FakeTelegramApi(
                [
                    ChannelMessage(
                        channel_id=channel_id,
                        message_id=2,
                        media_type="image",
                        file_id="image-2",
                        extension=".jpg",
                        source=b"fresh",
                    )
                ]
            )
            runner = DownloadChannelMediaRunner(api=api, output_root=output_root)

            asyncio.run(runner.run(channel_id=channel_id, full=True))

            self.assertEqual(api.download_calls, [(2, "2.jpg.part")])
            self.assertEqual((channel_root / "images" / "2.jpg").read_bytes(), b"fresh")

    def test_stale_temp_file_is_removed_before_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            channel_id = -1001234567890
            channel_root = output_root / str(channel_id)
            stale_temp = channel_root / "images" / "2.jpg.part"
            stale_temp.parent.mkdir(parents=True, exist_ok=True)
            stale_temp.write_bytes(b"stale")

            api = FakeTelegramApi(
                [
                    ChannelMessage(
                        channel_id=channel_id,
                        message_id=2,
                        media_type="image",
                        file_id="image-2",
                        extension=".jpg",
                        source=b"fresh",
                    )
                ]
            )
            runner = DownloadChannelMediaRunner(api=api, output_root=output_root)

            asyncio.run(runner.run(channel_id=channel_id, full=True))

            self.assertFalse(stale_temp.exists())
            self.assertEqual((channel_root / "images" / "2.jpg").read_bytes(), b"fresh")

    def test_accepts_actual_download_path_returned_by_api(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            channel_id = -1001234567890
            channel_root = output_root / str(channel_id)
            api = FinalPathTelegramApi(
                [
                    ChannelMessage(
                        channel_id=channel_id,
                        message_id=339,
                        media_type="video",
                        file_id="video-339",
                        extension=".mp4",
                        source=b"video-339",
                    )
                ]
            )
            runner = DownloadChannelMediaRunner(api=api, output_root=output_root)

            asyncio.run(runner.run(channel_id=channel_id, full=True))

            self.assertEqual(api.download_calls, [(339, "339.mp4.part")])
            self.assertTrue((channel_root / "videos" / "339.mp4").exists())
            self.assertFalse((channel_root / "videos" / "339.mp4.part").exists())
            manifest_rows = [
                json.loads(line)
                for line in (channel_root / "manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(manifest_rows[0]["message_id"], 339)

    def test_reports_download_progress_and_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            channel_id = -1001234567890
            reporter = RecordingReporter()
            api = ProgressCallbackTelegramApi(
                [
                    ChannelMessage(
                        channel_id=channel_id,
                        message_id=12,
                        media_type="video",
                        file_id="video-12",
                        extension=".mp4",
                        source=b"0123456789",
                    )
                ]
            )
            runner = DownloadChannelMediaRunner(
                api=api,
                output_root=output_root,
                reporter=reporter,
            )

            asyncio.run(runner.run(channel_id=channel_id, full=False))

            self.assertEqual(
                reporter.events,
                [
                    ("run_started", channel_id, 0, False),
                    (
                        "download_started",
                        12,
                        str(output_root / str(channel_id) / "videos" / "12.mp4"),
                    ),
                    ("download_progress", 12, 2, 10),
                    ("download_progress", 12, 10, 10),
                    (
                        "download_completed",
                        12,
                        str(output_root / str(channel_id) / "videos" / "12.mp4"),
                    ),
                    ("run_finished", 12, False),
                ],
            )

    def test_interrupt_aborts_current_download_without_checkpointing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            channel_id = -1001234567890
            reporter = RecordingReporter()
            stop_signal = StopSignal(on_stop_requested=reporter.on_interrupt_requested)
            api = InterruptingTelegramApi(
                [
                    ChannelMessage(
                        channel_id=channel_id,
                        message_id=425,
                        media_type="video",
                        file_id="video-425",
                        extension=".mp4",
                        source=None,
                    )
                ],
                interrupt=lambda: stop_signal._handle_signal(2, None),
            )
            runner = DownloadChannelMediaRunner(
                api=api,
                output_root=output_root,
                reporter=reporter,
                stop_signal=stop_signal,
            )

            asyncio.run(runner.run(channel_id=channel_id, full=False))

            channel_root = output_root / str(channel_id)
            self.assertFalse((channel_root / "videos" / "425.mp4").exists())
            self.assertFalse((channel_root / "videos" / "425.mp4.part").exists())
            self.assertFalse((channel_root / "manifest.jsonl").exists())
            self.assertFalse((channel_root / "checkpoint.json").exists())
            self.assertEqual(
                reporter.events,
                [
                    ("run_started", channel_id, 0, False),
                    (
                        "download_started",
                        425,
                        str(channel_root / "videos" / "425.mp4"),
                    ),
                    ("interrupt_requested",),
                    ("run_finished", None, True),
                ],
            )


if __name__ == "__main__":
    unittest.main()
