# /// script
# requires-python = ">=3.11"
# ///

"""Export selected backed-up Telegram messages from SQLite to Markdown."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Sequence


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def read_message_ids(path: Path) -> list[int]:
    try:
        raw_ids = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid message-ID JSON: {path}") from exc
    if not isinstance(raw_ids, list) or any(not isinstance(item, int) for item in raw_ids):
        raise ValueError("Message-ID JSON must contain an array of integers")
    return raw_ids


def render_markdown(
    *, channel_title: str, channel_id: str, channel_username: str | None, rows: list[sqlite3.Row]
) -> str:
    lines = [
        "# Telegram Incremental Export",
        "",
        f"- Channel: {channel_title}",
        f"- Channel ID: {channel_id}",
        f"- New messages: {len(rows)}",
        "",
        "## Messages",
        "",
    ]
    if not rows:
        lines.extend(["No new messages.", ""])
        return "\n".join(lines)
    for row in rows:
        lines.append(f"### {row['posted_at']} (message {row['message_id']})")
        if channel_username:
            lines.append(f"Source: https://t.me/{channel_username}/{row['message_id']}")
        details: list[str] = []
        if row["media_type"]:
            details.append(f"media: {row['media_type']}")
        if row["grouped_id"]:
            details.append(f"album: {row['grouped_id']}")
        if details:
            lines.append("; ".join(details))
        lines.extend(["", row["text_content"] or "[No text content.]"])
        if row["transcription_text"]:
            lines.extend(["", "#### Transcript", "", row["transcription_text"]])
        lines.append("")
    return "\n".join(lines)


def export_messages(
    *, database_path: Path, channel_id: str, message_ids_path: Path, output_path: Path
) -> dict[str, object]:
    if not database_path.is_file():
        raise FileNotFoundError(f"Backup database not found: {database_path}")
    message_ids = read_message_ids(message_ids_path)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    try:
        if message_ids:
            placeholders = ", ".join("?" for _ in message_ids)
            rows = connection.execute(
                f"""
                SELECT m.*, c.title, c.username
                FROM telegram_messages AS m
                JOIN telegram_channels AS c ON c.channel_id = m.channel_id
                WHERE m.channel_id = ? AND m.message_id IN ({placeholders})
                ORDER BY m.posted_at ASC, m.message_id ASC
                """,
                [channel_id, *message_ids],
            ).fetchall()
        else:
            rows = []
        if rows:
            channel_id = str(rows[0]["channel_id"])
            title = str(rows[0]["title"])
            username = rows[0]["username"]
        else:
            channel = connection.execute(
                "SELECT channel_id, title, username FROM telegram_channels WHERE channel_id = ?",
                (channel_id,),
            ).fetchone()
            if channel is None:
                raise ValueError("Backup database has no Telegram channel metadata")
            channel_id, title, username = str(channel["channel_id"]), str(channel["title"]), channel["username"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            render_markdown(
                channel_title=title,
                channel_id=channel_id,
                channel_username=str(username) if username else None,
                rows=rows,
            ),
            encoding="utf-8",
        )
    finally:
        connection.close()
    log(f"[export] Wrote {len(rows)} new messages to {output_path}")
    return {
        "status": "exported",
        "database": str(database_path),
        "message_ids_json": str(message_ids_path),
        "output": str(output_path),
        "message_count": len(rows),
        "channel_id": channel_id,
        "channel_title": title,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export newly backed-up Telegram messages to Markdown.")
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--channel-id", required=True, help="Canonical channel ID from backup atom JSON.")
    parser.add_argument("--message-ids-json", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--json", action="store_true", help="Write result JSON to stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_messages(
            database_path=args.database.expanduser(),
            channel_id=args.channel_id,
            message_ids_path=args.message_ids_json.expanduser(),
            output_path=args.output.expanduser(),
        )
    except Exception as exc:
        log(f"Error: {exc}")
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["output"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
