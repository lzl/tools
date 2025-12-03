# /// script
# dependencies = []
# ///

"""A tool to speed up audio files"""

import sys
import subprocess
from pathlib import Path


def find_latest_audio_file(directory: Path) -> Path:
    """Find the latest modified audio file in the directory"""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma'}
    audio_files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]
    
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {directory}")
    
    # Return the file with the latest modification time
    latest_file = max(audio_files, key=lambda f: f.stat().st_mtime)
    return latest_file


def speed_up_audio(input_path: Path, speed_factor: float, output_dir: Path) -> Path:
    """Speed up audio file by the given factor using ffmpeg atempo filter"""
    # Generate output filename with speed factor
    stem = input_path.stem
    suffix = input_path.suffix
    output_filename = f"{stem}_speed_{speed_factor}x{suffix}"
    output_path = output_dir / output_filename
    
    # ffmpeg atempo filter supports range 0.5 to 2.0
    # For values > 2.0, we need to chain multiple atempo filters
    atempo_filters = []
    remaining_speed = speed_factor
    
    while remaining_speed > 2.0:
        atempo_filters.append("atempo=2.0")
        remaining_speed /= 2.0
    
    if remaining_speed != 1.0:
        atempo_filters.append(f"atempo={remaining_speed:.3f}")
    
    filter_chain = ",".join(atempo_filters) if atempo_filters else "anull"
    
    print(f"Processing audio with speed factor: {speed_factor}x")
    
    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-filter:a", filter_chain,
        "-y",  # Overwrite output file if it exists
        str(output_path)
    ]
    
    # Run ffmpeg (suppress normal output, show errors)
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")
    
    return output_path


def main():
    """Speed up an audio file"""
    # Parse arguments
    input_file = None
    speed_factor = 2.0
    output_dir = Path("output")
    
    # Usage: speed_up_audio [file] [speed_factor] [output_dir]
    # If file is not provided, use latest file from downloads/
    # If speed_factor is not provided, default to 2.0
    
    if len(sys.argv) > 1:
        arg1 = sys.argv[1]
        # Check if it's a number (speed factor) or a file path
        try:
            speed_factor = float(arg1)
            if speed_factor <= 0:
                raise ValueError("Speed factor must be positive")
            # First arg is speed factor, file will be from downloads
        except ValueError:
            # Not a number, treat as file path
            input_file = Path(arg1)
    
    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
        if input_file is None:
            # First arg was speed factor, second is file path
            input_file = Path(arg2)
        else:
            # First arg was file path, second is speed factor
            try:
                speed_factor = float(arg2)
                if speed_factor <= 0:
                    raise ValueError("Speed factor must be positive")
            except ValueError:
                print(f"Error: Invalid speed factor '{arg2}'. Must be a positive number.")
                print("\nUsage: speed_up_audio [file] [speed_factor] [output_dir]")
                print("\nExamples:")
                print("  speed_up_audio                                    # Use latest from downloads/, 2x speed")
                print("  speed_up_audio 1.5                                 # Use latest from downloads/, 1.5x speed")
                print("  speed_up_audio audio.mp3                           # Use audio.mp3, 2x speed")
                print("  speed_up_audio audio.mp3 1.5                       # Use audio.mp3, 1.5x speed")
                print("  speed_up_audio audio.mp3 1.5 ./custom_output       # Custom output directory")
                sys.exit(1)
    
    if len(sys.argv) > 3:
        output_dir = Path(sys.argv[3])
    
    # Determine input file
    if input_file is None:
        downloads_dir = Path("downloads")
        if not downloads_dir.exists():
            print(f"Error: Downloads directory '{downloads_dir}' does not exist")
            sys.exit(1)
        try:
            input_file = find_latest_audio_file(downloads_dir)
            print(f"Using latest audio file: {input_file}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        if not input_file.exists():
            print(f"Error: File '{input_file}' does not exist")
            sys.exit(1)
        if not input_file.is_file():
            print(f"Error: '{input_file}' is not a file")
            sys.exit(1)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Input file: {input_file}")
    print(f"Speed factor: {speed_factor}x")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    try:
        output_path = speed_up_audio(input_file, speed_factor, output_dir)
        print(f"\nSuccess! Sped-up audio saved to: {output_path.absolute()}")
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

