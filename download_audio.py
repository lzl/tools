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


def parse_args(argv: list[str]) -> tuple[str, Path, Optional[str]]:
    """
    Parse command line arguments.
    
    Returns:
        Tuple of (video_url, output_dir, browser_name)
    """
    if len(argv) < 2:
        print("Usage: download_audio <video_url> [output_directory] [--browser BROWSER]")
        print("\nExample:")
        print("  download_audio https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("  download_audio https://www.youtube.com/watch?v=dQw4w9WgXcQ ./custom_dir")
        print("  download_audio https://www.youtube.com/watch?v=dQw4w9WgXcQ --browser chrome")
        print("  download_audio https://www.youtube.com/watch?v=dQw4w9WgXcQ ./custom_dir --browser firefox")
        print("\nSupported browsers: chrome, firefox, safari, edge, opera, brave, chromium, vivaldi")
        sys.exit(1)
    
    video_url = argv[1]
    output_dir = Path("output_dir")
    browser: Optional[str] = None
    
    # Parse remaining arguments
    i = 2
    while i < len(argv):
        arg = argv[i]
        if arg == "--browser" and i + 1 < len(argv):
            browser = argv[i + 1]
            i += 2
        elif not arg.startswith("--"):
            output_dir = Path(arg)
            i += 1
        else:
            print(f"Unknown argument: {arg}")
            sys.exit(1)
    
    return video_url, output_dir, browser


def main() -> None:
    """Download audio from provided URL using yt-dlp"""
    video_url, output_dir, browser = parse_args(sys.argv)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading audio from: {video_url}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Check for cookies.txt file in current directory (only if --browser not specified)
    cookies_file = Path("cookies.txt") if browser is None else None
    if browser:
        print(f"Using cookies from browser: {browser}")
    elif cookies_file and cookies_file.exists():
        print(f"Using cookies file: {cookies_file.absolute()}")
    
    try:
        # Build yt-dlp command
        cmd = [
            "yt-dlp",
            video_url,
            "-f", "worstaudio",  # Select worst audio quality (smallest file size)
            "-o", str(output_dir / "%(title)s.%(ext)s"),
            "--no-mtime",  # Don't set file modification time
            "--remote-components", "ejs:github",  # Enable EJS script downloads from GitHub
        ]
        
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

