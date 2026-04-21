from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Sequence

from telegram_media.downloader import ConsoleProgressReporter, DownloadChannelMediaRunner
from telegram_media.session import load_dotenv_if_present


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
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    parser.print_help()
    return 1
