import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from atoms import backup_telegram_channel as backup_atom
from atoms import export_telegram_messages_markdown as export_atom
from workflows import backup_telegram_channel_markdown as workflow


class BackupTelegramChannelAtomTests(unittest.TestCase):
    def test_normalize_channel_id_supports_private_ids_and_usernames(self) -> None:
        self.assertEqual(backup_atom.normalize_channel_id("-1445373305"), -1001445373305)
        self.assertEqual(backup_atom.normalize_channel_id("-1001445373305"), -1001445373305)
        self.assertEqual(backup_atom.normalize_channel_id("@example_channel"), "example_channel")

    def test_database_last_id_is_scoped_to_channel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            connection = backup_atom.create_database(Path(temp_dir) / "backup.sqlite3")
            connection.execute(
                "INSERT INTO telegram_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("one", 50, "2026-01-01T00:00:00+00:00", None, None, None, 0, None),
            )
            connection.execute(
                "INSERT INTO telegram_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                ("two", 80, "2026-01-01T00:00:00+00:00", None, None, None, 0, None),
            )
            connection.commit()
            self.assertEqual(backup_atom.get_last_message_id(connection, "one"), 50)
            self.assertEqual(backup_atom.get_last_message_id(connection, "missing"), 0)
            connection.close()

    def test_message_record_keeps_text_media_and_utc_timestamp(self) -> None:
        message = type(
            "Message",
            (),
            {
                "id": 7,
                "date": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
                "message": "saved text",
                "sender_id": 9,
                "grouped_id": None,
                "media": object(),
                "photo": object(),
                "video": None,
                "voice": None,
                "audio": None,
                "document": None,
            },
        )()

        record = backup_atom.message_record("one", message)

        self.assertEqual(record[0:4], ("one", 7, "2026-01-02T03:04:05+00:00", "saved text"))
        self.assertEqual(record[6:], (1, "photo"))


class ExportTelegramMessagesAtomTests(unittest.TestCase):
    def make_database(self, path: Path) -> None:
        connection = backup_atom.create_database(path)
        connection.execute(
            "INSERT INTO telegram_channels VALUES (?, ?, ?, datetime('now'))",
            ("one", "First channel", "first"),
        )
        connection.execute(
            "INSERT INTO telegram_channels VALUES (?, ?, ?, datetime('now'))",
            ("two", "Second channel", None),
        )
        connection.execute(
            "INSERT INTO telegram_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("one", 7, "2026-01-02T00:00:00+00:00", "first text", None, None, 0, None),
        )
        connection.execute(
            "INSERT INTO telegram_messages VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("two", 7, "2026-01-03T00:00:00+00:00", "wrong channel", None, None, 0, None),
        )
        connection.commit()
        connection.close()

    def test_export_uses_channel_id_to_isolate_reused_message_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database = root / "backup.sqlite3"
            message_ids = root / "ids.json"
            output = root / "new.md"
            self.make_database(database)
            message_ids.write_text("[7]", encoding="utf-8")

            result = export_atom.export_messages(
                database_path=database,
                channel_id="one",
                message_ids_path=message_ids,
                output_path=output,
            )

            self.assertEqual(result["message_count"], 1)
            markdown = output.read_text(encoding="utf-8")
            self.assertIn("first text", markdown)
            self.assertNotIn("wrong channel", markdown)
            self.assertIn("https://t.me/first/7", markdown)

    def test_export_writes_valid_empty_incremental_document(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database = root / "backup.sqlite3"
            message_ids = root / "ids.json"
            output = root / "new.md"
            self.make_database(database)
            message_ids.write_text("[]", encoding="utf-8")

            result = export_atom.export_messages(
                database_path=database,
                channel_id="one",
                message_ids_path=message_ids,
                output_path=output,
            )

            self.assertEqual(result["message_count"], 0)
            self.assertIn("No new messages.", output.read_text(encoding="utf-8"))


class BackupWorkflowTests(unittest.TestCase):
    def test_workflow_runs_backup_before_export_and_returns_paths(self) -> None:
        calls: list[tuple[str, list[str]]] = []

        def fake_run_atom(atom: str, args: list[str]) -> dict[str, object]:
            calls.append((atom, args))
            if atom.endswith("backup_telegram_channel.py"):
                return {"channel_id": "123", "new_message_count": 2}
            return {"output": "ignored.md", "message_count": 2}

        with tempfile.TemporaryDirectory() as temp_dir, patch.object(
            workflow, "run_atom", new=fake_run_atom
        ):
            root = Path(temp_dir)
            result = workflow.run_workflow(
                channel="-100123",
                database=root / "state.sqlite3",
                artifacts_root=root / "artifacts",
                run_dir=root / "run",
                output=None,
                full=False,
            )

        self.assertEqual([call[0] for call in calls], [
            "atoms/backup_telegram_channel.py",
            "atoms/export_telegram_messages_markdown.py",
        ])
        self.assertIn("--channel-id", calls[1][1])
        self.assertIn("123", calls[1][1])
        self.assertTrue(result["markdown_output"].endswith("run/outputs/new_messages.md"))

    def test_workflow_relays_atom_stderr_without_contaminating_stdout(self) -> None:
        class FakeProcess:
            def __init__(self) -> None:
                self.stdout = io.StringIO('{"status": "ok"}')
                self.stderr = io.StringIO("[backup] visible progress\\n")

            def wait(self) -> int:
                return 0

        stderr = io.StringIO()
        with patch.object(workflow.subprocess, "Popen", new=lambda *args, **kwargs: FakeProcess()), contextlib.redirect_stderr(stderr):
            payload = workflow.run_atom("atoms/example.py", [])

        self.assertEqual(payload, {"status": "ok"})
        self.assertIn("[backup] visible progress", stderr.getvalue())
