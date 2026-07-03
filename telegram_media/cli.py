from __future__ import annotations

import argparse
import asyncio
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from telegram_media.downloader import (
    ChannelStore,
    ConsoleProgressReporter,
    DownloadChannelMediaRunner,
    normalize_channel_id,
)
from telegram_media.session import load_dotenv_if_present


@dataclass(frozen=True)
class MessageLink:
    channel_id: int | str
    message_id: int


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m telegram_media",
        description="Download Telegram channel images and videos with resume support.",
    )
    subparsers = parser.add_subparsers(dest="command")

    generate_session = subparsers.add_parser(
        "generate-session",
        help="Interactively generate TELEGRAM_STRING_SESSION.",
    )
    generate_session.add_argument(
        "--phone",
        help="Phone number to use for login. If omitted, prompt interactively.",
    )

    download = subparsers.add_parser(
        "download-channel-media",
        help="Download channel images and videos with manifest/checkpoint resume support.",
    )
    download.add_argument("--channel-id", required=True, help="Telegram channel ID or username.")
    download.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/telegram"),
        help="Root directory for channel downloads (default: data/telegram).",
    )
    download.add_argument(
        "--full",
        action="store_true",
        help="Ignore checkpoint.json and scan the full message history.",
    )

    download_messages = subparsers.add_parser(
        "download-message-media",
        help="Download media from specific Telegram message links.",
    )
    download_messages.add_argument("links", nargs="+", help="Telegram message links.")
    download_messages.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/telegram"),
        help="Root directory for channel downloads (default: data/telegram).",
    )

    return parser


async def _run_download_command(channel_id: str, output_root: Path, *, full: bool) -> None:
    from telegram_media.telethon_api import TelethonMediaApi, create_download_client

    client = await create_download_client()
    try:
        runner = DownloadChannelMediaRunner(
            api=TelethonMediaApi(client, output_root=output_root),
            output_root=output_root,
            reporter=ConsoleProgressReporter(),
        )
        await runner.run(channel_id=channel_id, full=full)
    finally:
        await client.disconnect()


async def _run_download_messages_command(links: Sequence[str], output_root: Path) -> None:
    from telegram_media.telethon_api import TelethonMediaApi, create_download_client

    parsed_links = [parse_message_link(link) for link in links]
    client = await create_download_client()
    reporter = ConsoleProgressReporter()
    try:
        api = TelethonMediaApi(client, output_root=output_root)
        runner = DownloadChannelMediaRunner(
            api=api,
            output_root=output_root,
            reporter=reporter,
        )
        stores: dict[int | str, ChannelStore] = {}
        last_message_id: int | None = None
        reporter.on_run_started(
            channel_id="message-links",
            output_root=output_root,
            min_message_id=0,
            full=True,
        )
        for link in parsed_links:
            store = stores.setdefault(
                link.channel_id,
                ChannelStore(output_root, link.channel_id),
            )
            message = await api.get_channel_message(
                link.channel_id,
                message_id=link.message_id,
            )
            if message is None:
                print(f"Message not found: {link.channel_id}/{link.message_id}", file=sys.stderr)
                continue
            await runner._process_message(store, message)
            last_message_id = link.message_id
        reporter.on_run_finished(
            last_message_id=last_message_id,
            interrupted=False,
            failed=False,
        )
    except Exception:
        reporter.on_run_finished(
            last_message_id=None,
            interrupted=False,
            failed=True,
        )
        raise
    finally:
        await client.disconnect()


def main(argv: Sequence[str] | None = None) -> int:
    load_dotenv_if_present()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    try:
        if args.command == "generate-session":
            from telegram_media.telethon_api import generate_string_session

            session = asyncio.run(generate_string_session(phone=args.phone))
            print(session)
            return 0

        if args.command == "download-channel-media":
            asyncio.run(
                _run_download_command(
                    channel_id=args.channel_id,
                    output_root=args.output_root,
                    full=args.full,
                )
            )
            return 0

        if args.command == "download-message-media":
            asyncio.run(
                _run_download_messages_command(
                    links=args.links,
                    output_root=args.output_root,
                )
            )
            return 0
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1
