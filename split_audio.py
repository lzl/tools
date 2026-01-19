# /// script
# dependencies = [
#   "typeguard",
# ]
# ///

"""A tool to split audio files into segments using ffmpeg"""

import sys
import subprocess
from pathlib import Path
from typing import Optional

from typeguard import typechecked


# Supported audio formats
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus', '.webm'}

# Default segment duration: 25 minutes in seconds
DEFAULT_SEGMENT_SECONDS = 25 * 60


def find_latest_audio_file(directory: Path) -> Path:
    """Find the latest modified audio file in the directory"""
    audio_files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]
    
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {directory}")
    
    # Return the file with the latest modification time
    latest_file = max(audio_files, key=lambda f: f.stat().st_mtime)
    return latest_file


def split_audio(input_path: Path, segment_seconds: int, output_dir: Path) -> list[Path]:
    """
    Split audio file into segments using ffmpeg.
    
    Args:
        input_path: Path to the input audio file
        segment_seconds: Duration of each segment in seconds
        output_dir: Directory to save the segments
    
    Returns:
        List of paths to the generated segment files, sorted by index
    """
    stem = input_path.stem
    suffix = input_path.suffix
    
    # Output pattern: stem_part_000.ext, stem_part_001.ext, etc.
    output_pattern = str(output_dir / f"{stem}_part_%03d{suffix}")
    
    print(f"Splitting audio into {segment_seconds}s segments...")
    
    # Build ffmpeg command using segment muxer
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-f", "segment",
        "-segment_time", str(segment_seconds),
        "-reset_timestamps", "1",
        "-c", "copy",  # Copy without re-encoding for speed
        "-y",  # Overwrite output files
        output_pattern
    ]
    
    # Run ffmpeg
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    # Find and sort generated segment files
    segment_files: list[Path] = []
    for f in output_dir.iterdir():
        if f.is_file() and f.stem.startswith(f"{stem}_part_") and f.suffix == suffix:
            segment_files.append(f)
    
    # Sort by the numeric index in the filename
    segment_files.sort(key=lambda p: int(p.stem.split("_part_")[-1]))
    
    if not segment_files:
        raise RuntimeError("No segment files were created")
    
    print(f"Created {len(segment_files)} segment(s)")
    for seg in segment_files:
        print(f"  - {seg.name}")
    
    return segment_files


@typechecked
def main() -> None:
    """Split an audio file into segments"""
    # Parse arguments
    input_file: Optional[Path] = None
    segment_seconds: int = DEFAULT_SEGMENT_SECONDS
    output_dir: Path = Path("output_dir")
    
    # Usage: split_audio <input_file> [segment_seconds] [output_dir]
    if len(sys.argv) < 2:
        print("Usage: split_audio <input_file> [segment_seconds] [output_dir]")
        print(f"\nDefault segment duration: {DEFAULT_SEGMENT_SECONDS} seconds (25 minutes)")
        print("\nExample:")
        print("  split_audio audio.mp3")
        print("  split_audio audio.mp3 1500")
        print("  split_audio audio.mp3 1500 ./segments")
        sys.exit(1)
    
    input_file = Path(sys.argv[1])
    
    if len(sys.argv) > 2:
        try:
            segment_seconds = int(sys.argv[2])
            if segment_seconds <= 0:
                raise ValueError("Segment duration must be positive")
        except ValueError as e:
            print(f"Error: Invalid segment duration '{sys.argv[2]}'. Must be a positive integer.")
            sys.exit(1)
    
    if len(sys.argv) > 3:
        output_dir = Path(sys.argv[3])
    
    # Validate input file
    if not input_file.exists():
        print(f"Error: File '{input_file}' does not exist")
        sys.exit(1)
    
    if not input_file.is_file():
        print(f"Error: '{input_file}' is not a file")
        sys.exit(1)
    
    if input_file.suffix.lower() not in AUDIO_EXTENSIONS:
        print(f"Error: '{input_file}' is not a supported audio format")
        print(f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input file: {input_file}")
    print(f"Segment duration: {segment_seconds} seconds")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    try:
        segments = split_audio(input_file, segment_seconds, output_dir)
        print(f"\nSuccess! Created {len(segments)} segment(s) in: {output_dir.absolute()}")
    except FileNotFoundError:
        print("\nError: ffmpeg not found. Please install it:")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt-get install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
