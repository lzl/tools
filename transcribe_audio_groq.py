# /// script
# dependencies = [
#   "requests",
#   "python-dotenv",
# ]
# ///

"""A tool to transcribe audio files to text using Groq Whisper API"""

import sys
import os
import time
import shutil
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import requests


# Groq Whisper API limits
MAX_FILE_SIZE_MB = 25
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"

# Supported audio formats per Groq SKILL.md
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.webm'}


def format_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def convert_to_vtt(verbose_json: dict) -> str:
    """Convert Groq verbose_json response to WebVTT format"""
    lines = ["WEBVTT", ""]
    
    segments = verbose_json.get("segments", [])
    
    for i, segment in enumerate(segments):
        start = segment.get("start", 0)
        end = segment.get("end", 0)
        text = segment.get("text", "").strip()
        
        if text:
            # Add cue number (optional but common in VTT)
            lines.append(str(i + 1))
            # Add timestamp line
            lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
            # Add text
            lines.append(text)
            # Add blank line between cues
            lines.append("")
    
    return "\n".join(lines)


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


def is_ascii_safe(filename: str) -> bool:
    """Check if filename contains only ASCII characters"""
    try:
        filename.encode('ascii')
        return True
    except UnicodeEncodeError:
        return False


def create_ascii_safe_temp_file(source_path: Path) -> Path:
    """Create a temporary file with ASCII-safe filename from source file"""
    # Generate ASCII-safe filename: use stem hash + extension
    stem_hash = hashlib.md5(str(source_path.stem).encode('utf-8')).hexdigest()[:8]
    safe_filename = f"audio_{stem_hash}{source_path.suffix}"
    
    # Create temporary file in same directory as source (to preserve permissions)
    temp_path = source_path.parent / safe_filename
    
    # Copy source file to temporary file
    shutil.copy2(source_path, temp_path)
    
    return temp_path


def is_retryable_status(status_code: int) -> bool:
    """Check if HTTP status code is retryable"""
    return status_code in {429, 500, 502, 503, 504}


def transcribe_audio_with_groq(audio_path: Path, api_key: str, output_dir: Path) -> Path:
    """Transcribe audio file using Groq Whisper API"""
    
    print(f"Loading audio file: {audio_path}")
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")
    
    # Check file size limit
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(
            f"File size ({file_size_mb:.2f} MB) exceeds Groq's limit of {MAX_FILE_SIZE_MB} MB. "
            f"Please compress or split the audio file."
        )
    
    # Check if filename is ASCII-safe, create temp file if not
    temp_file = None
    upload_path = audio_path
    
    if not is_ascii_safe(str(audio_path.name)):
        print(f"Filename contains non-ASCII characters, creating temporary file with ASCII-safe name...")
        temp_file = create_ascii_safe_temp_file(audio_path)
        upload_path = temp_file
        print(f"Using temporary file: {temp_file.name}")
    
    # Retry logic with exponential backoff
    max_retries = 3
    retry_delays = [10, 30]  # seconds between retries
    
    response = None
    last_error = None
    
    print("\nTranscribing audio with Groq Whisper API...")
    
    try:
        for attempt in range(max_retries):
            try:
                with open(upload_path, 'rb') as audio_file:
                    files = {
                        'file': (upload_path.name, audio_file, 'audio/mpeg')
                    }
                    data = {
                        'model': 'whisper-large-v3-turbo',
                        'response_format': 'verbose_json'  # Use verbose_json to get timestamps
                    }
                    headers = {
                        'Authorization': f'Bearer {api_key}'
                    }
                    
                    response = requests.post(
                        GROQ_API_URL,
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=300  # 5 minute timeout for large files
                    )
                
                # Check for success
                if response.status_code == 200:
                    break
                
                # Check if retryable
                if attempt < max_retries - 1 and is_retryable_status(response.status_code):
                    delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
                    print(f"\n⚠ Attempt {attempt + 1} failed: HTTP {response.status_code}")
                    print(f"  Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    # Not retryable or out of retries
                    error_msg = response.text[:500] if response.text else "No error message"
                    raise RuntimeError(
                        f"Groq API error (HTTP {response.status_code}): {error_msg}"
                    )
                    
            except requests.exceptions.RequestException as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt] if attempt < len(retry_delays) else retry_delays[-1]
                    print(f"\n⚠ Attempt {attempt + 1} failed: {e}")
                    print(f"  Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_error}")
    
    finally:
        # Clean up temporary file if it was created
        if temp_file is not None and temp_file.exists():
            temp_file.unlink()
            print(f"Cleaned up temporary file")
    
    if response is None or response.status_code != 200:
        raise RuntimeError(f"Failed to transcribe after {max_retries} attempts")
    
    # Parse JSON response and convert to VTT
    verbose_json = response.json()
    vtt_content = convert_to_vtt(verbose_json)
    
    # Generate output filename
    output_filename = f"{audio_path.stem}.vtt"
    output_path = output_dir / output_filename
    
    # Save transcript to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(vtt_content)
    
    return output_path


def main():
    """Transcribe audio file to text using Groq Whisper API"""
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
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("Error: GROQ_API_KEY environment variable is not set")
        print("\nPlease set your Groq API key:")
        print("  Option 1: Create a .env file with: GROQ_API_KEY='your-api-key-here'")
        print("  Option 2: Export environment variable: export GROQ_API_KEY='your-api-key-here'")
        print("\nYou can get an API key from: https://console.groq.com/")
        sys.exit(1)
    
    print(f"Input file: {input_file}")
    print(f"Output directory: {output_dir.absolute()}")
    print()
    
    try:
        output_path = transcribe_audio_with_groq(input_file, api_key, output_dir)
        print(f"\nSuccess! Transcription saved to: {output_path.absolute()}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"\nError: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

