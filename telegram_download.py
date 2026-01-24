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
from telethon.sessions import StringSession
from telethon.tl.types import Message, MessageMediaDocument, DocumentAttributeFilename
from telethon.tl.functions.messages import GetDiscussionMessageRequest
from tqdm import tqdm
from typeguard import typechecked


@dataclass
class ParsedArgs:
    """Parsed command line arguments."""

    message_links: list[str]
    output_dir: Path
    max_concurrent: int


@dataclass
class ParsedLink:
    """Parsed Telegram message link."""

    channel_id: Union[int, str]  # int for private (-100xxx), str for public username
    message_id: int
    comment_id: Optional[int] = None  # Comment ID for discussion replies


@typechecked
def parse_args(args: list[str]) -> ParsedArgs:
    """Parse command line arguments."""
    message_links: list[str] = []
    output_dir: Path = Path("output_dir")
    max_concurrent: int = 5

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--out":
            if i + 1 >= len(args):
                print("Error: --out requires a directory path argument")
                sys.exit(1)
            output_dir = Path(args[i + 1])
            i += 2
        elif arg == "--concurrency" or arg == "-j":
            if i + 1 >= len(args):
                print("Error: --concurrency requires a number argument")
                sys.exit(1)
            try:
                max_concurrent = int(args[i + 1])
                if max_concurrent < 1:
                    print("Error: --concurrency must be at least 1")
                    sys.exit(1)
            except ValueError:
                print(f"Error: --concurrency requires a number, got: {args[i + 1]}")
                sys.exit(1)
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

    return ParsedArgs(
        message_links=message_links, output_dir=output_dir, max_concurrent=max_concurrent
    )


def print_usage() -> None:
    """Print usage instructions."""
    print("Usage: uv run telegram_download.py <message_link> [message_link2 ...] [options]")
    print()
    print("Download videos from Telegram message links (including private channels).")
    print()
    print("Options:")
    print("  --out <dir>         Output directory (default: output_dir)")
    print("  --concurrency, -j   Max concurrent downloads (default: 5)")
    print("  --help, -h          Show this help message")
    print()
    print("Examples:")
    print("  # Download single video")
    print('  uv run telegram_download.py "https://t.me/channel_name/123"')
    print()
    print("  # Download from private channel")
    print('  uv run telegram_download.py "https://t.me/c/1234567890/456"')
    print()
    print("  # Download video from a comment/reply")
    print('  uv run telegram_download.py "https://t.me/channel_name/123?comment=456"')
    print()
    print("  # Download multiple videos")
    print('  uv run telegram_download.py "https://t.me/ch1/123" "https://t.me/ch2/456"')
    print()
    print("  # Download with 10 concurrent downloads")
    print('  uv run telegram_download.py "https://t.me/ch1/123" "https://t.me/ch2/456" -j 10')
    print()
    print("  # Custom output directory")
    print('  uv run telegram_download.py "https://t.me/channel/123" --out ./downloads')
    print()
    print("Environment Variables:")
    print("  TELEGRAM_API_ID      Your Telegram API ID (from https://my.telegram.org)")
    print("  TELEGRAM_API_HASH    Your Telegram API Hash")
    print("  TELEGRAM_SESSION     Session string (obtained after first login)")
    print()
    print("First Run:")
    print("  On first run, you'll be prompted to enter your phone number and")
    print("  verification code. A session string will be printed for you to save.")
    print("  Add it to your environment: export TELEGRAM_SESSION='<session_string>'")
    print()
    print("Message Link Formats:")
    print("  Public channel:  https://t.me/channel_name/12345")
    print("  Private channel: https://t.me/c/1234567890/12345")
    print("  Comment/Reply:   https://t.me/channel_name/12345?comment=67890")


@typechecked
def parse_message_link(url: str) -> Optional[ParsedLink]:
    """Parse a Telegram message link and extract channel/message IDs.

    Supports:
    - Public channels: https://t.me/channel_name/12345
    - Public channel comments: https://t.me/channel_name/12345?comment=67890
    - Private channels: https://t.me/c/1234567890/12345
    - Private channel comments: https://t.me/c/1234567890/12345?comment=67890
    """
    # Extract comment ID if present
    comment_id: Optional[int] = None
    comment_match = re.search(r"[?&]comment=(\d+)", url)
    if comment_match:
        comment_id = int(comment_match.group(1))

    # Remove query parameters for pattern matching
    base_url = re.sub(r"\?.*$", "", url)

    # Pattern for private channels: https://t.me/c/CHANNEL_ID/MESSAGE_ID
    private_pattern = r"https?://t\.me/c/(\d+)/(\d+)"
    private_match = re.match(private_pattern, base_url)
    if private_match:
        channel_id = private_match.group(1)
        message_id = int(private_match.group(2))
        # Convert to the format Telethon expects for private channels
        # Private channel IDs need -100 prefix (e.g., 1234567890 -> -1001234567890)
        return ParsedLink(
            channel_id=int(f"-100{channel_id}"),
            message_id=message_id,
            comment_id=comment_id,
        )

    # Pattern for public channels: https://t.me/channel_name/MESSAGE_ID
    public_pattern = r"https?://t\.me/([a-zA-Z][a-zA-Z0-9_]{3,30}[a-zA-Z0-9])/(\d+)"
    public_match = re.match(public_pattern, base_url)
    if public_match:
        channel_name = public_match.group(1)
        message_id = int(public_match.group(2))
        return ParsedLink(
            channel_id=channel_name,
            message_id=message_id,
            comment_id=comment_id,
        )

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


@dataclass
class DownloadResult:
    """Result of a download operation."""

    url: str
    success: bool
    filename: str
    error: Optional[str] = None


@typechecked
async def download_video_async(
    client: TelegramClient,
    url: str,
    link: ParsedLink,
    output_dir: Path,
    semaphore: asyncio.Semaphore,
    task_id: int,
    total_tasks: int,
) -> DownloadResult:
    """Download video from a Telegram message."""
    async with semaphore:
        try:
            # Get the message (or comment if comment_id is specified)
            if link.comment_id is not None:
                # For comments, we need to get the discussion group first
                try:
                    # Get the entity for the channel
                    channel_entity = await client.get_entity(link.channel_id)

                    # Get discussion message info using the raw API
                    result = await client(
                        GetDiscussionMessageRequest(
                            peer=channel_entity, msg_id=link.message_id
                        )
                    )

                    if not result.messages:
                        return DownloadResult(
                            url=url,
                            success=False,
                            filename="",
                            error="Discussion not found for this message",
                        )

                    # The discussion group is where comments live
                    # Get the comment by its ID from the discussion group
                    discussion_chat_id = result.messages[0].peer_id
                    message = await client.get_messages(
                        discussion_chat_id, ids=link.comment_id
                    )
                except Exception as e:
                    return DownloadResult(
                        url=url,
                        success=False,
                        filename="",
                        error=f"Failed to get comment: {e}",
                    )
            else:
                message = await client.get_messages(link.channel_id, ids=link.message_id)

            if message is None:
                return DownloadResult(
                    url=url,
                    success=False,
                    filename="",
                    error=f"Message not found (ID: {link.comment_id or link.message_id})",
                )

            # Check if message has video/document media
            if not message.media:
                return DownloadResult(
                    url=url, success=False, filename="", error="Message has no media"
                )

            # Get video size for progress bar
            file_size = 0
            if isinstance(message.media, MessageMediaDocument):
                doc = message.media.document
                if doc and doc.size:
                    file_size = doc.size
            elif hasattr(message.media, "video"):
                video = message.media.video
                if video and video.size:
                    file_size = video.size

            # Get channel info for filename
            try:
                entity = await client.get_entity(link.channel_id)
                channel_title = (
                    getattr(entity, "title", None)
                    or getattr(entity, "username", None)
                    or str(link.channel_id)
                )
            except Exception:
                channel_title = str(link.channel_id)

            # Generate filename
            filename = get_video_filename(message, channel_title)
            output_path = output_dir / filename

            # Check for existing partial download (resume support)
            existing_size = 0
            if output_path.exists():
                existing_size = output_path.stat().st_size
                if file_size > 0 and existing_size >= file_size:
                    # File already fully downloaded
                    return DownloadResult(url=url, success=True, filename=filename)

            # Prepare progress bar description
            if file_size > 0:
                size_str = f"{file_size / (1024 * 1024):.1f}MB"
                if existing_size > 0:
                    existing_str = f"{existing_size / (1024 * 1024):.1f}MB"
                    desc = f"[{task_id}/{total_tasks}] {filename[:25]}... (resume {existing_str}/{size_str})"
                else:
                    desc = f"[{task_id}/{total_tasks}] {filename[:30]}... ({size_str})"
            else:
                desc = f"[{task_id}/{total_tasks}] {filename[:30]}... (unknown size)"

            # Download with resume support using iter_download
            with tqdm(
                total=file_size if file_size > 0 else None,
                initial=existing_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=desc,
                leave=True,
                position=None,
            ) as pbar:
                # Use append mode for resume, write mode for new downloads
                mode = "ab" if existing_size > 0 else "wb"
                with open(output_path, mode) as f:
                    async for chunk in client.iter_download(
                        message.media,
                        offset=existing_size,
                    ):
                        f.write(chunk)
                        pbar.update(len(chunk))

            return DownloadResult(url=url, success=True, filename=filename)

        except Exception as e:
            return DownloadResult(url=url, success=False, filename="", error=str(e))


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

    # Get session string from environment (empty string for first-time login)
    session_str = os.environ.get("TELEGRAM_SESSION", "")

    # Create Telegram client with StringSession
    client = TelegramClient(StringSession(session_str), api_id, api_hash)

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
            print("Save this session string to TELEGRAM_SESSION environment variable:")
            print(client.session.save())
            print()

        # Create semaphore for concurrent downloads
        semaphore = asyncio.Semaphore(parsed.max_concurrent)
        print(f"Max concurrent downloads: {parsed.max_concurrent}")
        print()

        # Create download tasks
        tasks = [
            download_video_async(
                client,
                url,
                link,
                parsed.output_dir,
                semaphore,
                i,
                len(links),
            )
            for i, (url, link) in enumerate(links, 1)
        ]

        # Run all downloads concurrently (limited by semaphore)
        results: list[DownloadResult] = await asyncio.gather(*tasks)

        # Print summary
        print()
        print("=" * 50)
        success_count = sum(1 for r in results if r.success)
        fail_count = sum(1 for r in results if not r.success)
        print(f"Download complete: {success_count} succeeded, {fail_count} failed")

        # Print failed downloads if any
        failed = [r for r in results if not r.success]
        if failed:
            print()
            print("Failed downloads:")
            for r in failed:
                print(f"  - {r.url}")
                if r.error:
                    print(f"    Error: {r.error}")

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
