# /// script
# dependencies = [
#   "yt-dlp",
# ]
# ///

"""A tool to download subtitles from Bilibili videos (including AI-generated subtitles)"""

import sys
import subprocess
from pathlib import Path


def parse_args(args):
    """Parse command line arguments"""
    video_url = None
    lang = "ai-zh"
    output_dir = Path("output_dir")
    
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--lang":
            if i + 1 >= len(args):
                print("Error: --lang requires a language code argument")
                sys.exit(1)
            lang = args[i + 1]
            i += 2
        elif arg == "--out":
            if i + 1 >= len(args):
                print("Error: --out requires a directory path argument")
                sys.exit(1)
            output_dir = Path(args[i + 1])
            i += 2
        elif arg.startswith("-"):
            print(f"Error: Unknown option '{arg}'")
            sys.exit(1)
        else:
            if video_url is None:
                video_url = arg
            else:
                print(f"Error: Unexpected argument '{arg}'")
                sys.exit(1)
            i += 1
    
    return video_url, lang, output_dir


def print_usage():
    """Print usage instructions"""
    print("Usage: uv run bilibili_subtitle.py <video_url> [options]")
    print()
    print("Options:")
    print("  --lang <code>   Subtitle language code (default: ai-zh)")
    print("  --out <dir>     Output directory (default: output_dir)")
    print()
    print("Examples:")
    print("  uv run bilibili_subtitle.py https://www.bilibili.com/video/BV1xx411c7XW")
    print("  uv run bilibili_subtitle.py https://www.bilibili.com/video/BV1xx411c7XW --lang ai-en")
    print("  uv run bilibili_subtitle.py https://www.bilibili.com/video/BV1xx411c7XW --out ./subtitles")
    print()
    print("Common language codes:")
    print("  ai-zh    AI-generated Chinese subtitles (default)")
    print("  ai-en    AI-generated English subtitles")
    print("  zh-Hans  Simplified Chinese (manual)")
    print("  zh-Hant  Traditional Chinese (manual)")
    print()
    print("Note: cookies.txt is required for most videos with AI subtitles")


def download_subtitle(video_url, lang, output_dir, cookies_file=None):
    """Download subtitles using yt-dlp"""
    cmd = [
        "yt-dlp",
        video_url,
        "--write-subs",           # Download subtitles
        "--write-auto-subs",      # Download AI-generated subtitles if available
        "--sub-format", "srt",    # Output format SRT
        "--sub-langs", lang,      # Language selection
        "--skip-download",        # Skip downloading video/audio
        "-o", str(output_dir / "%(title)s.%(ext)s"),
        "--no-mtime",             # Don't set file modification time
    ]
    
    # Add cookies file if it exists
    if cookies_file and cookies_file.exists():
        cmd.extend(["--cookies", str(cookies_file)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError: Failed to download subtitles")
        print(f"yt-dlp exited with code {e.returncode}")
        return False


def main():
    """Download subtitles from Bilibili video"""
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)
    
    video_url, lang, output_dir = parse_args(sys.argv[1:])
    
    if video_url is None:
        print("Error: Video URL is required")
        print()
        print_usage()
        sys.exit(1)
    
    # Validate URL (basic check for Bilibili)
    if "bilibili.com" not in video_url and "b23.tv" not in video_url:
        print("Warning: URL does not appear to be a Bilibili video URL")
        print("         This tool is designed for Bilibili videos.")
        print()
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Video URL: {video_url}")
    print(f"Language: {lang}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Check for cookies.txt file in current directory
    cookies_file = Path("cookies.txt")
    if cookies_file.exists():
        print(f"Using cookies file: {cookies_file.absolute()}")
    
    try:
        print("\nDownloading subtitles...")
        success = download_subtitle(video_url, lang, output_dir, cookies_file)
        
        if success:
            print("\nDownload completed successfully!")
        else:
            print("\nDownload failed. Possible reasons:")
            print("  - The video has no subtitles available")
            print("  - The specified language is not available")
            print("  - Network or authentication issues")
            print()
            print("Tips:")
            print("  - AI subtitles require cookies.txt - place it in current directory")
            print("  - Try different language codes: ai-zh, ai-en, zh-Hans, zh-Hant")
            print("  - Use 'yt-dlp --list-subs <url>' to see available subtitles")
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
