# /// script
# dependencies = [
#   "google-generativeai",
#   "python-dotenv",
# ]
# ///

"""A tool to transcribe audio files to text using Google Gemini AI"""

import sys
import os
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai


def find_latest_audio_file(directory: Path) -> Path:
    """Find the latest modified audio file in the directory"""
    audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus'}
    audio_files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in audio_extensions
    ]
    
    if not audio_files:
        raise FileNotFoundError(f"No audio files found in {directory}")
    
    # Return the file with the latest modification time
    latest_file = max(audio_files, key=lambda f: f.stat().st_mtime)
    return latest_file


def format_time(seconds: float) -> str:
    """Convert seconds to VTT time format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def transcribe_audio_with_gemini(audio_path: Path, api_key: str, output_dir: Path) -> Path:
    """Transcribe audio file using Google Gemini AI"""
    # Configure Gemini API
    genai.configure(api_key=api_key)
    
    # Load the model
    model = genai.GenerativeModel(model_name='gemini-2.5-flash-lite-preview-09-2025')
    
    print(f"Loading audio file: {audio_path}")
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Upload audio file to Gemini
    print("\nUploading audio to Gemini API...")
    audio_file = genai.upload_file(path=str(audio_path))
    
    print("Waiting for file to be processed...")
    # Wait for the file to be processed
    while audio_file.state.name == "PROCESSING":
        print(".", end="", flush=True)
        time.sleep(2)
        audio_file = genai.get_file(audio_file.name)
    print()
    
    if audio_file.state.name == "FAILED":
        genai.delete_file(audio_file.name)
        raise RuntimeError("Failed to process audio file")
    
    print("Transcribing audio (this may take a while)...")
    
    # Generate transcription with prompt for VTT format
    prompt = """Please transcribe this audio file and output the transcription in WebVTT format with timestamps.
The output should follow this format:

WEBVTT
Kind: captions
Language: auto

00:00:00.000 --> 00:00:05.000
[transcribed text]

00:00:05.000 --> 00:00:10.000
[next segment]

Please provide accurate timestamps and break the transcription into readable segments of 3-10 seconds each.
Make sure the timestamps are accurate and the text matches the audio content.
"""
    
    try:
        response = model.generate_content(
            contents=[audio_file, prompt]
        )
        
        transcript = response.text
        
        # Generate output filename
        output_filename = f"{audio_path.stem}.vtt"
        output_path = output_dir / output_filename
        
        # Save transcript to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(transcript)
        
        # Clean up uploaded file
        print("Cleaning up uploaded file...")
        genai.delete_file(audio_file.name)
        
        return output_path
        
    except Exception as e:
        # Clean up uploaded file on error
        try:
            genai.delete_file(audio_file.name)
        except:
            pass
        raise e


def main():
    """Transcribe audio file to text using Google Gemini AI"""
    # Load environment variables from .env file
    load_dotenv()
    
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
            print(f"Error: Input directory '{input_dir}' does not exist")
            sys.exit(1)
        try:
            input_file = find_latest_audio_file(input_dir)
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
    
    # Get API key from environment variable (loaded from .env file)
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set")
        print("\nPlease set your Google Gemini API key:")
        print("  Option 1: Create a .env file with: GEMINI_API_KEY='your-api-key-here'")
        print("  Option 2: Export environment variable: export GEMINI_API_KEY='your-api-key-here'")
        print("\nYou can get an API key from: https://aistudio.google.com/")
        sys.exit(1)
    
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    try:
        output_path = transcribe_audio_with_gemini(input_file, api_key, output_dir)
        print(f"\nSuccess! Transcription saved to: {output_path.absolute()}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

