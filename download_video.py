# /// script
# dependencies = [
#   "yt-dlp",
# ]
# ///

"""A tool to download videos from YouTube and other platforms using yt-dlp"""

import sys
import subprocess
from pathlib import Path


def main():
    """Download video from provided URL using yt-dlp"""
    if len(sys.argv) < 2:
        print("Usage: download_video <video_url> [output_directory]")
        print("\nExample:")
        print("  download_video https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("  download_video https://www.youtube.com/watch?v=dQw4w9WgXcQ ./custom_dir")
        sys.exit(1)
    
    video_url = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("downloads")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading video from: {video_url}")
    print(f"Output directory: {output_dir.absolute()}")
    
    try:
        # Build yt-dlp command
        cmd = [
            "yt-dlp",
            video_url,
            "-o", str(output_dir / "%(title)s.%(ext)s"),
            "--no-mtime",  # Don't set file modification time
            "--remote-components", "ejs:github",  # Enable EJS script downloads from GitHub
        ]
        
        # Run yt-dlp
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        print("\nDownload completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Failed to download video")
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

