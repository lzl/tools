# /// script
# dependencies = [
#   "yt-dlp",
# ]
# ///

"""A tool to get subtitles for a video URL by downloading or transcribing"""

import sys
import subprocess
import tempfile
import shutil
import re
import hashlib
import time
from pathlib import Path


def retry_step(func, *args, max_retries=2, delay=5, step_name="step", **kwargs):
    """Retry a step with exponential backoff"""
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries:
                print(f"\n⚠ {step_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"  Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise


def get_safe_filename_from_url(video_url, extension=".mp3"):
    """Generate a safe ASCII filename from video URL"""
    # Try to extract YouTube video ID
    youtube_patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([a-zA-Z0-9_-]{11})',
        r'youtube\.com/watch\?.*v=([a-zA-Z0-9_-]{11})',
    ]
    
    for pattern in youtube_patterns:
        match = re.search(pattern, video_url)
        if match:
            video_id = match.group(1)
            return f"video_{video_id}{extension}"
    
    # If not YouTube or can't extract ID, use URL hash
    url_hash = hashlib.md5(video_url.encode('utf-8')).hexdigest()[:12]
    return f"video_{url_hash}{extension}"


def check_subtitle_available(video_url, cookies_file=None):
    """Check if subtitles are available for the video"""
    cmd = [
        "yt-dlp",
        video_url,
        "--list-subs",
        "--remote-components", "ejs:github",
    ]
    
    if cookies_file and cookies_file.exists():
        cmd.extend(["--cookies", str(cookies_file)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        output = result.stdout.lower()
        # Check if any subtitles are available (manual or auto)
        return ("available subtitles" in output or 
                "available automatic" in output or
                "available manual" in output)
    except subprocess.CalledProcessError:
        return False


def download_subtitle_direct(video_url, output_dir, cookies_file=None):
    """Try to download subtitle directly using yt-dlp"""
    print("\n=== Step 1: Attempting to download subtitle directly ===")
    
    # Record existing files before download
    existing_files = set(output_dir.glob("*"))
    
    cmd = [
        "yt-dlp",
        video_url,
        "--write-subs",  # Download subtitles
        "--write-auto-subs",  # Download auto-generated subtitles
        "--skip-download",  # Skip downloading video/audio
        "--sub-langs", "en,en-US,en-GB,en-CA,en-AU",  # Auto-select English variants
        "-o", str(output_dir / "%(title)s.%(ext)s"),
        "--no-mtime",
        "--remote-components", "ejs:github",
    ]
    
    if cookies_file and cookies_file.exists():
        cmd.extend(["--cookies", str(cookies_file)])
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        
        # Check for newly created subtitle files
        subtitle_extensions = {'.vtt', '.srt', '.ttml', '.dfxp'}
        all_files = set(output_dir.glob("*"))
        new_files = all_files - existing_files
        
        for new_file in new_files:
            if new_file.suffix.lower() in subtitle_extensions:
                print(f"\n✓ Successfully downloaded subtitle: {new_file.name}")
                return new_file
        
        print("\n✗ No subtitle files found after download attempt")
        return None
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to download subtitle (exit code {e.returncode})")
        return None


def get_audio_duration(audio_path):
    """Get audio duration in seconds using ffprobe"""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = float(result.stdout.strip())
        return duration
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Warning: Could not determine audio duration: {e}")
        return None


def download_audio(video_url, temp_dir, cookies_file=None):
    """Download audio file using download_audio.py tool"""
    print("\n=== Step 2: Downloading audio ===")
    
    # Get the script directory to find download_audio.py
    script_dir = Path(__file__).parent
    download_audio_script = script_dir / "download_audio.py"
    
    if not download_audio_script.exists():
        raise FileNotFoundError(f"download_audio.py not found at {download_audio_script}")
    
    cmd = ["uv", "run", str(download_audio_script), video_url, str(temp_dir)]
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        
        # Find the downloaded audio file
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus', '.webm'}
        audio_files = [
            f for f in temp_dir.iterdir()
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]
        
        if audio_files:
            # Get the latest one (most recently modified)
            original_file = max(audio_files, key=lambda f: f.stat().st_mtime)
            print(f"✓ Audio downloaded: {original_file.name}")
            
            # Rename to a safe ASCII filename based on video URL to avoid encoding issues
            safe_filename = get_safe_filename_from_url(video_url, original_file.suffix)
            safe_file = temp_dir / safe_filename
            
            # If a file with the safe name already exists, remove it first
            if safe_file.exists():
                safe_file.unlink()
            
            original_file.rename(safe_file)
            print(f"  Renamed to: {safe_filename}")
            return safe_file
        else:
            raise FileNotFoundError("No audio file found after download")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to download audio (exit code {e.returncode})")


def speed_up_audio_if_needed(audio_path, temp_dir):
    """Speed up audio if duration > 1 hour"""
    print("\n=== Step 3: Checking audio duration ===")
    
    duration = get_audio_duration(audio_path)
    if duration is None:
        print("Warning: Could not determine duration, skipping speed-up")
        return audio_path
    
    duration_hours = duration / 3600
    print(f"Audio duration: {duration_hours:.2f} hours ({duration:.0f} seconds)")
    
    if duration > 3600:  # More than 1 hour
        print(f"\n=== Step 4: Speeding up audio (1.5x) ===")
        
        # Get the script directory to find speed_up_audio.py
        script_dir = Path(__file__).parent
        speed_up_script = script_dir / "speed_up_audio.py"
        
        if not speed_up_script.exists():
            raise FileNotFoundError(f"speed_up_audio.py not found at {speed_up_script}")
        
        cmd = ["uv", "run", str(speed_up_script), str(audio_path), "1.5", str(temp_dir)]
        
        try:
            subprocess.run(cmd, check=True, capture_output=False)
            
            # Find the sped-up audio file
            stem = audio_path.stem
            suffix = audio_path.suffix
            sped_up_filename = f"{stem}_speed_1.5x{suffix}"
            sped_up_path = temp_dir / sped_up_filename
            
            if sped_up_path.exists():
                print(f"✓ Audio sped up: {sped_up_filename}")
                return sped_up_path
            else:
                print("Warning: Sped-up audio file not found, using original")
                return audio_path
                
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to speed up audio (exit code {e.returncode}), using original")
            return audio_path
    else:
        print(f"\n=== Step 4: Speeding up audio ===")
        print("Audio is less than 1 hour, skipping speed-up")
        return audio_path


def convert_to_mp3_if_needed(audio_path, temp_dir):
    """Convert audio to MP3 by delegating to convert_to_mp3.py"""
    print(f"\n=== Step 5: Converting audio format ===")
    
    if audio_path.suffix.lower() == '.mp3':
        print("Audio is already in MP3 format, skipping conversion")
        return audio_path
    
    script_dir = Path(__file__).parent
    convert_script = script_dir / "convert_to_mp3.py"
    
    if not convert_script.exists():
        raise FileNotFoundError(f"convert_to_mp3.py not found at {convert_script}")
    
    mp3_path = temp_dir / f"{audio_path.stem}.mp3"
    
    # Avoid interactive overwrite prompt in convert_to_mp3.py
    if mp3_path.exists():
        mp3_path.unlink()
    
    cmd = ["uv", "run", str(convert_script), str(audio_path), str(temp_dir)]
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        
        if mp3_path.exists():
            print(f"✓ Audio converted to MP3: {mp3_path.name}")
            return mp3_path
        
        print("Warning: Converted MP3 file not found, using original")
        return audio_path
        
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to convert to MP3 (exit code {e.returncode})")
        print("Using original file format")
        return audio_path
    except Exception as e:
        print(f"Warning: Error during conversion: {e}")
        print("Using original file format")
        return audio_path


def add_video_url_to_vtt(vtt_path, video_url):
    """Add video URL as a comment in the VTT file header"""
    if not vtt_path.exists():
        return
    
    # Read the existing VTT file
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if URL already exists in the file
    if video_url in content:
        return
    
    # Split into lines
    lines = content.split('\n')
    new_lines = []
    
    # Process VTT file header
    if lines and lines[0].strip() == 'WEBVTT':
        new_lines.append('WEBVTT')
        idx = 1
        
        # Add metadata lines (Kind, Language, etc.) if they exist
        while idx < len(lines) and lines[idx].strip() and ':' in lines[idx]:
            new_lines.append(lines[idx])
            idx += 1
        
        # Add video URL as a NOTE comment in the header section
        new_lines.append(f'NOTE Video URL: {video_url}')
        
        # Add empty line if not already present
        if idx < len(lines) and lines[idx].strip() == '':
            new_lines.append('')
            idx += 1
        else:
            new_lines.append('')
        
        # Add the rest of the lines (subtitle content)
        new_lines.extend(lines[idx:])
    else:
        # If format is unexpected, prepend the URL comment
        new_lines.append('WEBVTT')
        new_lines.append(f'NOTE Video URL: {video_url}')
        new_lines.append('')
        new_lines.extend(lines)
    
    # Write back to file
    with open(vtt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))


def transcribe_audio(audio_path, output_dir, video_url=None):
    """Transcribe audio to VTT using transcribe_audio.py tool"""
    print(f"\n=== Step 6: Transcribing audio ===")
    
    # Get the script directory to find transcribe_audio.py
    script_dir = Path(__file__).parent
    transcribe_script = script_dir / "transcribe_audio.py"
    
    if not transcribe_script.exists():
        raise FileNotFoundError(f"transcribe_audio.py not found at {transcribe_script}")
    
    cmd = ["uv", "run", str(transcribe_script), str(audio_path), str(output_dir)]
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        
        # Find the generated VTT file
        expected_vtt = output_dir / f"{audio_path.stem}.vtt"
        if expected_vtt.exists():
            # Add video URL to the VTT file if provided
            if video_url:
                add_video_url_to_vtt(expected_vtt, video_url)
                print(f"✓ Video URL added to subtitle file")
            print(f"✓ Transcription completed: {expected_vtt.name}")
            return expected_vtt
        else:
            raise FileNotFoundError("VTT file not found after transcription")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to transcribe audio (exit code {e.returncode})")


def main():
    """Get subtitle for a video URL by downloading or transcribing"""
    if len(sys.argv) < 2:
        print("Usage: get_subtitle <video_url> [output_directory]")
        print("\nExample:")
        print("  get_subtitle https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("  get_subtitle https://www.youtube.com/watch?v=dQw4w9WgXcQ ./custom_dir")
        sys.exit(1)
    
    video_url = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output_dir")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for cookies.txt file
    cookies_file = Path("cookies.txt")
    
    print(f"Video URL: {video_url}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Create temporary directory for intermediate files
    temp_dir = None
    
    try:
        temp_dir = Path(tempfile.mkdtemp(prefix="get_subtitle_"))
        print(f"Temporary directory: {temp_dir}")
        
        # Step 1: Try to download subtitle directly
        subtitle_file = download_subtitle_direct(video_url, output_dir, cookies_file)
        
        if subtitle_file:
            # Add video URL to the downloaded subtitle file
            add_video_url_to_vtt(subtitle_file, video_url)
            print(f"\n✓ Success! Subtitle saved to: {subtitle_file.absolute()}")
            return
        
        # Step 2: No subtitle available, download audio
        audio_file = retry_step(
            download_audio,
            video_url,
            temp_dir,
            cookies_file,
            max_retries=2,
            delay=5,
            step_name="Downloading audio"
        )
        
        # Step 3 & 4: Check duration and speed up if needed
        final_audio = retry_step(
            speed_up_audio_if_needed,
            audio_file,
            temp_dir,
            max_retries=2,
            delay=5,
            step_name="Speeding up audio"
        )
        
        # Step 5: Convert to MP3 if needed (e.g., webm format)
        mp3_audio = retry_step(
            convert_to_mp3_if_needed,
            final_audio,
            temp_dir,
            max_retries=2,
            delay=5,
            step_name="Converting audio format"
        )
        
        # Step 6: Transcribe audio to VTT
        vtt_file = retry_step(
            transcribe_audio,
            mp3_audio,
            output_dir,
            video_url,
            max_retries=2,
            delay=5,
            step_name="Transcribing audio"
        )
        
        print(f"\n✓ Success! Subtitle saved to: {vtt_file.absolute()}")
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up temporary directory
        if temp_dir and temp_dir.exists():
            print(f"\nCleaning up temporary directory...")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

