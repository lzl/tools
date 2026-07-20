# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "telethon>=1.41.2",
# ]
# ///

"""Back up one Telegram channel into a local SQLite database."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sqlite3
import sys
from datetime import timezone
from pathlib import Path
from typing import Any, Sequence

from telethon import TelegramClient
from telethon.sessions import StringSession


SCHEMA = """
CREATE TABLE IF NOT EXISTS telegram_channels (
    channel_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    username TEXT,
    backed_up_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS telegram_messages (
    channel_id TEXT NOT NULL,
    message_id INTEGER NOT NULL,
    posted_at TEXT NOT NULL,
    text_content TEXT,
    sender_id TEXT,
    grouped_id TEXT,
    has_media INTEGER NOT NULL,
    media_type TEXT,
    PRIMARY KEY (channel_id, message_id),
    FOREIGN KEY (channel_id) REFERENCES telegram_channels(channel_id)
);
CREATE INDEX IF NOT EXISTS idx_telegram_messages_channel_date
    ON telegram_messages(channel_id, posted_at, message_id);
"""


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def find_dotenv(start_dir: Path) -> Path | None:
    current = start_dir.resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


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


def normalize_channel_id(value: str) -> int | str:
    stripped = value.strip()
    try:
        numeric = int(stripped)
    except ValueError:
        return stripped.lstrip("@")
    if numeric < 0 and not str(numeric).startswith("-100"):
        return int(f"-100{str(numeric)[1:]}")
    return numeric


def require_credentials() -> tuple[int, str, str]:
    api_id_text = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")
    session = os.environ.get("TELEGRAM_STRING_SESSION", "")
    if not api_id_text:
        raise RuntimeError("TELEGRAM_API_ID is required")
    if not api_hash:
        raise RuntimeError("TELEGRAM_API_HASH is required")
    if not session:
        raise RuntimeError("TELEGRAM_STRING_SESSION is required")
    try:
        return int(api_id_text), api_hash, session
    except ValueError as exc:
        raise RuntimeError("TELEGRAM_API_ID must be an integer") from exc


def create_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def get_last_message_id(connection: sqlite3.Connection, channel_id: str) -> int:
    row = connection.execute(
        "SELECT COALESCE(MAX(message_id), 0) FROM telegram_messages WHERE channel_id = ?",
        (channel_id,),
    ).fetchone()
    return int(row[0]) if row else 0


def media_type(message: Any) -> str | None:
    if not getattr(message, "media", None):
        return None
    if getattr(message, "photo", None):
        return "photo"
    if getattr(message, "video", None):
        return "video"
    if getattr(message, "voice", None):
        return "voice"
    if getattr(message, "audio", None):
        return "audio"
    if getattr(message, "document", None):
        return "document"
    return "media"


def message_record(channel_id: str, message: Any) -> tuple[object, ...]:
    date = getattr(message, "date", None)
    if date is None:
        raise ValueError(f"Message {message.id} has no date")
    if date.tzinfo is None:
        date = date.replace(tzinfo=timezone.utc)
    text = getattr(message, "message", None)
    return (
        channel_id,
        int(message.id),
        date.astimezone(timezone.utc).isoformat(),
        text if text else None,
        str(getattr(message, "sender_id", "") or "") or None,
        str(getattr(message, "grouped_id", "") or "") or None,
        int(bool(getattr(message, "media", None))),
        media_type(message),
    )


async def backup_channel(
    *, channel_input: str, database_path: Path, new_messages_path: Path, full: bool
) -> dict[str, object]:
    api_id, api_hash, session = require_credentials()
    requested_channel = normalize_channel_id(channel_input)
    log("[backup] Connecting to Telegram")
    client = TelegramClient(StringSession(session), api_id, api_hash)
    connection = create_database(database_path)
    try:
        await client.connect()
        if not await client.is_user_authorized():
            raise RuntimeError("Telegram session is not authorized")
        log(f"[backup] Resolving channel {channel_input}")
        entity = await client.get_entity(requested_channel)
        if not hasattr(entity, "id") or not hasattr(entity, "title"):
            raise RuntimeError("Input did not resolve to a Telegram channel")
        channel_id = str(entity.id)
        title = str(entity.title)
        username = getattr(entity, "username", None)
        last_message_id = 0 if full else get_last_message_id(connection, channel_id)
        mode = "full" if full else "incremental"
        log(f"[backup] {mode} sync for {title}; after message {last_message_id}")

        new_message_ids: list[int] = []
        scanned = 0
        inserted = 0
        insert_sql = """
            INSERT OR IGNORE INTO telegram_messages
            (channel_id, message_id, posted_at, text_content, sender_id, grouped_id, has_media, media_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """
        # Oldest first makes MAX(message_id) a valid resumable checkpoint after Ctrl+C.
        async for message in client.iter_messages(entity, min_id=last_message_id, reverse=True):
            scanned += 1
            cursor = connection.execute(insert_sql, message_record(channel_id, message))
            if cursor.rowcount:
                inserted += 1
                new_message_ids.append(int(message.id))
            if scanned % 100 == 0:
                connection.commit()
                log(f"[backup] Scanned {scanned}; added {inserted}")

        connection.execute(
            """
            INSERT INTO telegram_channels(channel_id, title, username, backed_up_at)
            VALUES (?, ?, ?, datetime('now'))
            ON CONFLICT(channel_id) DO UPDATE SET
                title = excluded.title,
                username = excluded.username,
                backed_up_at = excluded.backed_up_at
            """,
            (channel_id, title, username),
        )
        connection.commit()
        new_messages_path.parent.mkdir(parents=True, exist_ok=True)
        new_messages_path.write_text(json.dumps(new_message_ids), encoding="utf-8")
        log(f"[backup] Done. Scanned {scanned}; added {inserted}")
        return {
            "status": "backed_up",
            "database": str(database_path),
            "channel_id": channel_id,
            "channel_title": title,
            "channel_username": username,
            "sync_mode": mode,
            "last_message_id_before": last_message_id,
            "scanned_count": scanned,
            "new_message_count": inserted,
            "new_messages_json": str(new_messages_path),
        }
    finally:
        connection.close()
        await client.disconnect()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Back up one Telegram channel into SQLite.")
    parser.add_argument("channel", help="Channel username or numeric ID, including -100... IDs.")
    parser.add_argument("--database", type=Path, required=True, help="SQLite backup database path.")
    parser.add_argument(
        "--new-messages-json", type=Path, required=True, help="Write IDs first stored by this run here."
    )
    parser.add_argument("--full", action="store_true", help="Fetch all messages; existing rows stay unchanged.")
    parser.add_argument("--json", action="store_true", help="Write result JSON to stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv_if_present()
    try:
        result = asyncio.run(
            backup_channel(
                channel_input=args.channel,
                database_path=args.database.expanduser(),
                new_messages_path=args.new_messages_json.expanduser(),
                full=args.full,
            )
        )
    except KeyboardInterrupt:
        log("Error: interrupted")
        return 130
    except Exception as exc:
        log(f"Error: {exc}")
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["database"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
