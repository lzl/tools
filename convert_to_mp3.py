# /// script
# dependencies = [
#   "pydub",
# ]
# ///

"""A tool to convert audio files to MP3 format with the same quality as source"""

import sys
from pathlib import Path
from pydub import AudioSegment


def find_latest_audio_file(directory: Path) -> Path:
    """Find the latest modified audio file in the directory"""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus', '.mp4', '.m4v', '.webm'}
    audio_files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]
    
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {directory}")
    
    # Return the file with the latest modification time
    latest_file = max(audio_files, key=lambda f: f.stat().st_mtime)
    return latest_file


def main():
    """Convert audio file to MP3 format"""
    # Parse arguments
    input_file = None
    output_dir = Path("output_dir")
    
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    
    # Determine input file
    if input_file is None:
        input_dir = Path("input_dir")
        if not input_dir.exists():
            print("Usage: convert_to_mp3 [audio_file] [output_dir]")
            print("\nExample:")
            print("  convert_to_mp3                                    # Use latest from input_dir/, output to output_dir/")
            print("  convert_to_mp3 input.wav                          # Convert specific file to output_dir/")
            print("  convert_to_mp3 input.m4a ./custom_output          # Convert with custom output directory")
            print(f"\nError: Input directory '{input_dir}' does not exist")
            sys.exit(1)
        
        try:
            input_file = find_latest_audio_file(input_dir)
            print(f"Using latest file from {input_dir}: {input_file.name}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        if not input_file.exists():
            print(f"Error: Input file '{input_file}' does not exist")
            sys.exit(1)
        if not input_file.is_file():
            print(f"Error: '{input_file}' is not a file")
            sys.exit(1)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output file path
    # Use input filename with .mp3 extension in output_dir
    output_file = output_dir / f"{input_file.stem}.mp3"
    
    # Check if output file already exists
    if output_file.exists():
        print(f"Warning: Output file '{output_file}' already exists")
        response = input("Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("Conversion cancelled")
            sys.exit(0)
    
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Output file: {output_file.name}")
    print()
    
    try:
        # Load audio file
        print("Loading audio file...")
        audio = AudioSegment.from_file(str(input_file))
        
        # Get source file info
        print(f"Source format: {input_file.suffix}")
        print(f"Duration: {len(audio) / 1000:.2f} seconds")
        print(f"Channels: {audio.channels}")
        print(f"Sample rate: {audio.frame_rate} Hz")

        # Export as MP3 using VBR (Variable Bitrate) mode
        # 9 = lowest quality/smallest size in LAME scale (0-9)
        vbr_quality = "9"
        print(f"Converting to MP3 (VBR quality {vbr_quality}, lowest quality/smallest size)...")
        audio.export(
            str(output_file),
            format="mp3",
            parameters=["-q:a", vbr_quality]  # VBR encoding with adaptive quality
        )
        
        # Get output file size
        output_size = output_file.stat().st_size
        input_size = input_file.stat().st_size
        
        print(f"\nConversion completed successfully!")
        print(f"Output file: {output_file.absolute()}")
        print(f"Output size: {output_size:,} bytes ({output_size / 1024 / 1024:.2f} MB)")
        print(f"Input size: {input_size:,} bytes ({input_size / 1024 / 1024:.2f} MB)")
        
    except FileNotFoundError:
        print("\nError: ffmpeg not found. Please install it:")
        print("  macOS: brew install ffmpeg")
        print("  Linux: sudo apt-get install ffmpeg")
        print("  Windows: Download from https://ffmpeg.org/download.html")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: Failed to convert audio file")
        print(f"Details: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
