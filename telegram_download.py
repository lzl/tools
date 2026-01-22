# /// script
# dependencies = [
#   "telethon",
#   "tqdm",
#   "typeguard",
# ]
# ///

"""A tool to download videos from Telegram message links (including private channels)"""

import asyncio
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from telethon import TelegramClient
from telethon.errors import SessionPasswordNeededError
from telethon.tl.types import Message, MessageMediaDocument, DocumentAttributeFilename
from tqdm import tqdm
from typeguard import typechecked


# Session file name (unique to avoid conflicts with other tools)
SESSION_NAME = "telegram_downloader"


@dataclass
class ParsedArgs:
    """Parsed command line arguments."""

    message_links: list[str]
    output_dir: Path


@dataclass
class ParsedLink:
    """Parsed Telegram message link."""

    channel_id: Union[int, str]  # int for private (-100xxx), str for public username
    message_id: int


@typechecked
def parse_args(args: list[str]) -> ParsedArgs:
    """Parse command line arguments."""
    message_links: list[str] = []
    output_dir: Path = Path("output_dir")

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--out":
            if i + 1 >= len(args):
                print("Error: --out requires a directory path argument")
                sys.exit(1)
            output_dir = Path(args[i + 1])
            i += 2
        elif arg == "--help" or arg == "-h":
            print_usage()
            sys.exit(0)
        elif arg.startswith("-"):
            print(f"Error: Unknown option '{arg}'")
            sys.exit(1)
        else:
            message_links.append(arg)
            i += 1

    return ParsedArgs(message_links=message_links, output_dir=output_dir)


def print_usage() -> None:
    """Print usage instructions."""
    print("Usage: uv run telegram_download.py <message_link> [message_link2 ...] [options]")
    print()
    print("Download videos from Telegram message links (including private channels).")
    print()
    print("Options:")
    print("  --out <dir>     Output directory (default: output_dir)")
    print("  --help, -h      Show this help message")
    print()
    print("Examples:")
    print("  # Download single video")
    print('  uv run telegram_download.py "https://t.me/channel_name/123"')
    print()
    print("  # Download from private channel")
    print('  uv run telegram_download.py "https://t.me/c/1234567890/456"')
    print()
    print("  # Download multiple videos")
    print('  uv run telegram_download.py "https://t.me/ch1/123" "https://t.me/ch2/456"')
    print()
    print("  # Custom output directory")
    print('  uv run telegram_download.py "https://t.me/channel/123" --out ./downloads')
    print()
    print("Environment Variables:")
    print("  TELEGRAM_API_ID      Your Telegram API ID (from https://my.telegram.org)")
    print("  TELEGRAM_API_HASH    Your Telegram API Hash")
    print()
    print("First Run:")
    print("  On first run, you'll be prompted to enter your phone number and")
    print("  verification code. A session file will be saved for future use.")
    print()
    print("Message Link Formats:")
    print("  Public channel:  https://t.me/channel_name/12345")
    print("  Private channel: https://t.me/c/1234567890/12345")


@typechecked
def parse_message_link(url: str) -> Optional[ParsedLink]:
    """Parse a Telegram message link and extract channel/message IDs.

    Supports:
    - Public channels: https://t.me/channel_name/12345
    - Private channels: https://t.me/c/1234567890/12345
    """
    # Pattern for private channels: https://t.me/c/CHANNEL_ID/MESSAGE_ID
    private_pattern = r"https?://t\.me/c/(\d+)/(\d+)"
    private_match = re.match(private_pattern, url)
    if private_match:
        channel_id = private_match.group(1)
        message_id = int(private_match.group(2))
        # Convert to the format Telethon expects for private channels
        # Private channel IDs need -100 prefix (e.g., 1234567890 -> -1001234567890)
        return ParsedLink(channel_id=int(f"-100{channel_id}"), message_id=message_id)

    # Pattern for public channels: https://t.me/channel_name/MESSAGE_ID
    public_pattern = r"https?://t\.me/([a-zA-Z][a-zA-Z0-9_]{3,30}[a-zA-Z0-9])/(\d+)"
    public_match = re.match(public_pattern, url)
    if public_match:
        channel_name = public_match.group(1)
        message_id = int(public_match.group(2))
        return ParsedLink(channel_id=channel_name, message_id=message_id)

    return None


def get_video_filename(message: Message, channel_title: str) -> str:
    """Extract or generate filename for the video."""
    # Try to get original filename from document attributes
    if message.media and isinstance(message.media, MessageMediaDocument):
        doc = message.media.document
        if doc and hasattr(doc, "attributes"):
            for attr in doc.attributes:
                if isinstance(attr, DocumentAttributeFilename):
                    return attr.file_name

    # Generate filename from channel title and message ID
    ext = "mp4"  # Default extension for videos
    if message.media and isinstance(message.media, MessageMediaDocument):
        doc = message.media.document
        if doc and doc.mime_type:
            mime_to_ext = {
                "video/mp4": "mp4",
                "video/quicktime": "mov",
                "video/x-matroska": "mkv",
                "video/webm": "webm",
                "video/avi": "avi",
            }
            ext = mime_to_ext.get(doc.mime_type, "mp4")

    # Sanitize channel title for filename
    safe_title = re.sub(r'[<>:"/\\|?*]', "_", channel_title)
    safe_title = safe_title.strip()[:50]  # Limit length

    return f"{safe_title}_{message.id}.{ext}"


async def download_video_async(
    client: TelegramClient,
    link: ParsedLink,
    output_dir: Path,
) -> bool:
    """Download video from a Telegram message."""
    try:
        # Get the message
        message = await client.get_messages(link.channel_id, ids=link.message_id)

        if message is None:
            print(f"  Error: Message not found (ID: {link.message_id})")
            return False

        # Check if message has video/document media
        if not message.media:
            print(f"  Error: Message has no media")
            return False

        # Get video size for progress bar
        file_size = 0
        if isinstance(message.media, MessageMediaDocument):
            doc = message.media.document
            if doc:
                file_size = doc.size
        elif hasattr(message.media, "video"):
            video = message.media.video
            if video:
                file_size = video.size

        if file_size == 0:
            print(f"  Warning: Could not determine file size")

        # Get channel info for filename
        try:
            entity = await client.get_entity(link.channel_id)
            channel_title = getattr(entity, "title", None) or getattr(
                entity, "username", None
            ) or str(link.channel_id)
        except Exception:
            channel_title = str(link.channel_id)

        # Generate filename
        filename = get_video_filename(message, channel_title)
        output_path = output_dir / filename

        # Check if file already exists
        if output_path.exists():
            print(f"  File already exists: {filename}")
            return True

        print(f"  Downloading: {filename}")
        if file_size > 0:
            print(f"  Size: {file_size / (1024 * 1024):.1f} MB")

        # Download with progress bar
        with tqdm(
            total=file_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc="  Progress",
            leave=True,
        ) as pbar:

            def progress_callback(current: int, total: int) -> None:
                pbar.update(current - pbar.n)

            await client.download_media(
                message,
                file=str(output_path),
                progress_callback=progress_callback,
            )

        print(f"  Saved to: {output_path}")
        return True

    except Exception as e:
        print(f"  Error downloading: {e}")
        return False


async def main_async() -> None:
    """Main async entry point."""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    parsed: ParsedArgs = parse_args(sys.argv[1:])

    if not parsed.message_links:
        print("Error: At least one message link is required")
        print()
        print_usage()
        sys.exit(1)

    # Check for API credentials
    api_id_str = os.environ.get("TELEGRAM_API_ID")
    api_hash = os.environ.get("TELEGRAM_API_HASH")

    if not api_id_str or not api_hash:
        print("Error: Telegram API credentials not found")
        print()
        print("Please set the following environment variables:")
        print("  export TELEGRAM_API_ID='your_api_id'")
        print("  export TELEGRAM_API_HASH='your_api_hash'")
        print()
        print("Get your credentials from: https://my.telegram.org")
        sys.exit(1)

    try:
        api_id = int(api_id_str)
    except ValueError:
        print(f"Error: TELEGRAM_API_ID must be a number, got: {api_id_str}")
        sys.exit(1)

    # Parse all message links first
    links: list[tuple[str, ParsedLink]] = []
    for url in parsed.message_links:
        link = parse_message_link(url)
        if link is None:
            print(f"Warning: Invalid message link format: {url}")
            print("  Expected: https://t.me/channel_name/12345")
            print("        or: https://t.me/c/1234567890/12345")
            continue
        links.append((url, link))

    if not links:
        print("Error: No valid message links provided")
        sys.exit(1)

    # Ensure output directory exists
    parsed.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {parsed.output_dir.absolute()}")
    print(f"Videos to download: {len(links)}")
    print()

    # Create Telegram client
    client = TelegramClient(SESSION_NAME, api_id, api_hash)

    try:
        # Connect and handle authentication
        await client.connect()

        if not await client.is_user_authorized():
            print("First-time login required.")
            print()

            phone = input("Please enter your phone number: ")
            await client.send_code_request(phone)

            code = input("Please enter the code you received: ")

            try:
                await client.sign_in(phone, code)
            except SessionPasswordNeededError:
                # 2FA is enabled, ask for password
                print("Two-factor authentication is enabled on your account.")
                password = input("Please enter your 2FA password: ")
                await client.sign_in(password=password)

            print("Login successful!")
            print()

        # Download each video
        success_count = 0
        fail_count = 0

        for i, (url, link) in enumerate(links, 1):
            print(f"[{i}/{len(links)}] {url}")
            if await download_video_async(client, link, parsed.output_dir):
                success_count += 1
            else:
                fail_count += 1
            print()

        # Summary
        print("=" * 50)
        print(f"Download complete: {success_count} succeeded, {fail_count} failed")

    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
    finally:
        await client.disconnect()


@typechecked
def main() -> None:
    """Main entry point."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
