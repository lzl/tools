# /// script
# dependencies = [
#   "yt-dlp",
# ]
# ///

"""A tool to download audio from YouTube and other platforms using yt-dlp"""

import sys
import subprocess
from pathlib import Path
from typing import Optional

from yt_dlp_wrapper import resolve_yt_dlp, yt_dlp_command

STREAMING_FRAGMENT_CONCURRENCY = 20
MIN_SIZE_FORMAT_SORT = "+size,+br,+res,+fps"


def parse_fragment_concurrency(value: str) -> int:
    """Parse a positive fragment concurrency value."""
    try:
        concurrency = int(value)
    except ValueError:
        print(f"Invalid --concurrent-fragments value: {value}")
        sys.exit(1)

    if concurrency < 1:
        print("--concurrent-fragments must be at least 1")
        sys.exit(1)

    return concurrency


def parse_args(argv: list[str]) -> tuple[str, Path, Optional[str], int]:
    """
    Parse command line arguments.
    
    Returns:
        Tuple of (video_url, output_dir, browser_name, concurrent_fragments)
    """
    if len(argv) < 2:
        print("Usage: download_audio <video_url> [output_directory] [--browser BROWSER] [--concurrent-fragments N]")
        print("\nExample:")
        print("  download_audio https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("  download_audio https://www.youtube.com/watch?v=dQw4w9WgXcQ ./custom_dir")
        print("  download_audio https://www.youtube.com/watch?v=dQw4w9WgXcQ --browser chrome")
        print("  download_audio https://www.youtube.com/watch?v=dQw4w9WgXcQ ./custom_dir --browser firefox")
        print("  download_audio https://www.youtube.com/watch?v=dQw4w9WgXcQ --concurrent-fragments 40")
        print("\nSupported browsers: chrome, firefox, safari, edge, opera, brave, chromium, vivaldi")
        print(f"\nDefault fragment concurrency for HLS/DASH downloads: {STREAMING_FRAGMENT_CONCURRENCY}")
        sys.exit(1)
    
    video_url = argv[1]
    output_dir = Path("output_dir")
    browser: Optional[str] = None
    concurrent_fragments = STREAMING_FRAGMENT_CONCURRENCY
    
    # Parse remaining arguments
    i = 2
    while i < len(argv):
        arg = argv[i]
        if arg == "--browser" and i + 1 < len(argv):
            browser = argv[i + 1]
            i += 2
        elif arg in ("--concurrent-fragments", "-N") and i + 1 < len(argv):
            concurrent_fragments = parse_fragment_concurrency(argv[i + 1])
            i += 2
        elif not arg.startswith("--"):
            output_dir = Path(arg)
            i += 1
        else:
            print(f"Unknown argument: {arg}")
            sys.exit(1)
    
    return video_url, output_dir, browser, concurrent_fragments


def build_download_command(
    video_url: str,
    output_dir: Path,
    *,
    concurrent_fragments: int = STREAMING_FRAGMENT_CONCURRENCY,
) -> list[str]:
    """
    Build the yt-dlp download command.

    `-N` is only meaningful for fragmented downloads, so keeping it on the
    command line preserves the simple audio-first flow while speeding up
    HLS/DASH cases. Prefer the smallest audio-only format; if a site exposes
    only combined video/audio formats, fall back to the smallest combined one.
    """
    return yt_dlp_command(
        video_url,
        "-f", "bestaudio/best",
        "-S", MIN_SIZE_FORMAT_SORT,
        "--format-sort-force",
        "-o", str(output_dir / "%(title)s.%(ext)s"),
        "--no-mtime",
        "--remote-components", "ejs:github",
        "-N", str(concurrent_fragments),
    )


def main() -> None:
    """Download audio from provided URL using yt-dlp"""
    video_url, output_dir, browser, concurrent_fragments = parse_args(sys.argv)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading audio from: {video_url}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"HLS/DASH fragment concurrency: {concurrent_fragments}")
    
    # Check for cookies.txt file in current directory (only if --browser not specified)
    cookies_file = Path("cookies.txt") if browser is None else None
    if browser:
        print(f"Using cookies from browser: {browser}")
    elif cookies_file and cookies_file.exists():
        print(f"Using cookies file: {cookies_file.absolute()}")
    
    try:
        yt_dlp = resolve_yt_dlp()
        version_suffix = f" ({yt_dlp.version})" if yt_dlp.version else ""
        print(f"Using yt-dlp: {yt_dlp.path}{version_suffix}")

        # yt-dlp only uses `-N` for fragmented downloads, so this stays simple
        # for normal files and speeds up streaming formats when needed.
        cmd = build_download_command(
            video_url,
            output_dir,
            concurrent_fragments=concurrent_fragments,
        )

        # Add cookies from browser or file
        if browser:
            cmd.extend(["--cookies-from-browser", browser])
        elif cookies_file and cookies_file.exists():
            cmd.extend(["--cookies", str(cookies_file)])
        
        # Run yt-dlp
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        print("\nDownload completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Failed to download audio")
        print(f"yt-dlp exited with code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: yt-dlp not found. Please install it:")
        print("  pip install yt-dlp")
        print("  or")
        print("  brew install yt-dlp")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
