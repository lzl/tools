# /// script
# dependencies = [
#   "yt-dlp",
#   "typeguard",
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

from typeguard import typechecked

T = TypeVar('T')


# Segment duration threshold: 25 minutes in seconds
SEGMENT_SECONDS = 25 * 60

# Maximum parallel transcription workers
MAX_TRANSCRIBE_WORKERS = 4


@dataclass
class VTTCue:
    """Represents a single VTT cue with start/end times and text"""
    start: float
    end: float
    text: str


def parse_vtt_timestamp(ts: str) -> float:
    """Parse VTT timestamp (HH:MM:SS.mmm) to seconds"""
    parts = ts.strip().split(':')
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    else:
        return float(ts)


def format_vtt_timestamp(seconds: float) -> str:
    """Convert seconds to VTT timestamp format (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def parse_vtt_file(vtt_path: Path) -> list[VTTCue]:
    """Parse a VTT file and extract cues"""
    cues: list[VTTCue] = []
    
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    lines = content.split('\n')
    i = 0
    
    # Skip header (WEBVTT and metadata)
    while i < len(lines):
        line = lines[i].strip()
        if '-->' in line:
            break
        i += 1
    
    # Parse cues
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for timestamp line
        if '-->' in line:
            # Parse timestamps
            parts = line.split('-->')
            if len(parts) == 2:
                start = parse_vtt_timestamp(parts[0].strip())
                # Handle potential positioning info after end time
                end_part = parts[1].strip().split()[0]
                end = parse_vtt_timestamp(end_part)
                
                # Collect text lines until empty line or next timestamp
                text_lines: list[str] = []
                i += 1
                while i < len(lines):
                    text_line = lines[i]
                    if text_line.strip() == '' or '-->' in text_line:
                        break
                    # Skip cue numbers (lines that are just digits)
                    if not text_line.strip().isdigit():
                        text_lines.append(text_line.strip())
                    i += 1
                
                if text_lines:
                    cues.append(VTTCue(start=start, end=end, text=' '.join(text_lines)))
                continue
        
        i += 1
    
    return cues


def merge_vtt_files(vtt_files: list[Path], segment_durations: list[float], output_path: Path) -> Path:
    """
    Merge multiple VTT files with timestamp correction.
    
    Args:
        vtt_files: List of VTT file paths in order
        segment_durations: Duration of each corresponding audio segment in seconds
        output_path: Path to write the merged VTT file
    
    Returns:
        Path to the merged VTT file
    """
    all_cues: list[VTTCue] = []
    cumulative_offset = 0.0
    
    for idx, (vtt_file, duration) in enumerate(zip(vtt_files, segment_durations)):
        cues = parse_vtt_file(vtt_file)
        
        # Shift timestamps by cumulative offset
        for cue in cues:
            shifted_cue = VTTCue(
                start=cue.start + cumulative_offset,
                end=cue.end + cumulative_offset,
                text=cue.text
            )
            all_cues.append(shifted_cue)
        
        # Add this segment's duration to offset for next segment
        cumulative_offset += duration
    
    # Write merged VTT
    lines = ["WEBVTT", ""]
    
    for i, cue in enumerate(all_cues):
        lines.append(str(i + 1))
        lines.append(f"{format_vtt_timestamp(cue.start)} --> {format_vtt_timestamp(cue.end)}")
        lines.append(cue.text)
        lines.append("")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    
    return output_path


def retry_step(
    func: Callable[..., T],
    *args: Any,
    max_retries: int = 2,
    delay: int = 5,
    step_name: str = "step",
    **kwargs: Any
) -> T:
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
    # This should never be reached, but satisfies the type checker
    raise RuntimeError(f"{step_name} failed after {max_retries + 1} attempts")


def get_safe_filename_from_url(video_url: str, extension: str = ".mp3") -> str:
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


def check_subtitle_available(video_url: str, cookies_file: Optional[Path] = None) -> bool:
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


def download_subtitle_direct(
    video_url: str,
    output_dir: Path,
    cookies_file: Optional[Path] = None
) -> Optional[Path]:
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
        "--sub-langs", "zh,en,en-US",  # Auto-select English variants
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


def get_audio_duration(audio_path: Path) -> Optional[float]:
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


def download_audio(
    video_url: str,
    temp_dir: Path,
    cookies_file: Optional[Path] = None
) -> Path:
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
        
        # Find the downloaded audio/video file
        # Include .mp4 since Groq API supports it directly for transcription
        audio_extensions = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus', '.webm', '.mp4', '.mpeg', '.mpga'}
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


def split_audio(audio_path: Path, segments_dir: Path) -> list[Path]:
    """
    Split audio file into segments using split_media.py tool.
    
    Args:
        audio_path: Path to the audio file to split
        segments_dir: Directory to save the segments
    
    Returns:
        List of paths to the generated segment files, sorted by index
    """
    print("\n=== Step 3: Splitting audio into segments ===")
    
    # Get the script directory to find split_media.py
    script_dir = Path(__file__).parent
    split_media_script = script_dir / "split_media.py"
    
    if not split_media_script.exists():
        raise FileNotFoundError(f"split_media.py not found at {split_media_script}")
    
    cmd = [
        "uv", "run", str(split_media_script),
        str(audio_path),
        str(SEGMENT_SECONDS),
        str(segments_dir)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        
        # Find and sort generated segment files
        stem = audio_path.stem
        suffix = audio_path.suffix
        segment_files: list[Path] = []
        
        for f in segments_dir.iterdir():
            if f.is_file() and f.stem.startswith(f"{stem}_part_") and f.suffix == suffix:
                segment_files.append(f)
        
        # Sort by the numeric index in the filename
        segment_files.sort(key=lambda p: int(p.stem.split("_part_")[-1]))
        
        if not segment_files:
            raise RuntimeError("No segment files were created")
        
        print(f"✓ Created {len(segment_files)} segment(s)")
        return segment_files
        
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to split audio (exit code {e.returncode})")


def transcribe_single_segment(
    segment_path: Path,
    segment_index: int,
    total_segments: int,
    output_dir: Path
) -> Path:
    """
    Transcribe a single audio segment.
    
    Args:
        segment_path: Path to the audio segment
        segment_index: Index of this segment (0-based)
        total_segments: Total number of segments
        output_dir: Directory to save the VTT file
    
    Returns:
        Path to the generated VTT file
    """
    print(f"  Transcribing segment {segment_index + 1}/{total_segments}: {segment_path.name}")
    
    # Get the script directory to find transcribe_media_groq.py
    script_dir = Path(__file__).parent
    transcribe_script = script_dir / "transcribe_media_groq.py"
    
    if not transcribe_script.exists():
        raise FileNotFoundError(f"transcribe_media_groq.py not found at {transcribe_script}")
    
    cmd = ["uv", "run", str(transcribe_script), str(segment_path), str(output_dir)]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Find the generated VTT file
        expected_vtt = output_dir / f"{segment_path.stem}.vtt"
        if expected_vtt.exists():
            print(f"  ✓ Segment {segment_index + 1}/{total_segments} completed")
            return expected_vtt
        else:
            raise FileNotFoundError(f"VTT file not found for segment {segment_index + 1}")
            
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to transcribe segment {segment_index + 1} (exit code {e.returncode})")


def transcribe_segments_parallel(
    segment_files: list[Path],
    vtt_output_dir: Path
) -> list[Path]:
    """
    Transcribe multiple audio segments in parallel.
    
    Args:
        segment_files: List of audio segment file paths
        vtt_output_dir: Directory to save the VTT files
    
    Returns:
        List of VTT file paths in the same order as segment_files
    """
    print(f"\n=== Step 4: Transcribing {len(segment_files)} segment(s) in parallel ===")
    
    total_segments = len(segment_files)
    num_workers = min(MAX_TRANSCRIBE_WORKERS, total_segments)
    
    print(f"Using {num_workers} parallel worker(s)")
    
    # Map to store results: segment_index -> vtt_path
    results: dict[int, Path] = {}
    errors: list[tuple[int, Exception]] = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(
                retry_step,
                transcribe_single_segment,
                segment_path,
                idx,
                total_segments,
                vtt_output_dir,
                max_retries=2,
                delay=5,
                step_name=f"Transcribing segment {idx + 1}"
            ): idx
            for idx, segment_path in enumerate(segment_files)
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                vtt_path = future.result()
                results[idx] = vtt_path
            except Exception as e:
                errors.append((idx, e))
    
    # Check for errors
    if errors:
        error_msgs = [f"Segment {idx + 1}: {e}" for idx, e in errors]
        raise RuntimeError(f"Failed to transcribe segments:\n" + "\n".join(error_msgs))
    
    # Return VTT files in order
    vtt_files = [results[i] for i in range(total_segments)]
    print(f"✓ All {total_segments} segment(s) transcribed successfully")
    
    return vtt_files


def add_video_url_to_vtt(vtt_path: Path, video_url: str) -> None:
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


def transcribe_audio(
    audio_path: Path,
    output_dir: Path,
    video_url: Optional[str] = None
) -> Path:
    """Transcribe audio to VTT using transcribe_media_groq.py tool"""
    print(f"\n=== Step 4: Transcribing audio ===")
    
    # Get the script directory to find transcribe_media_groq.py
    script_dir = Path(__file__).parent
    transcribe_script = script_dir / "transcribe_media_groq.py"
    
    if not transcribe_script.exists():
        raise FileNotFoundError(f"transcribe_media_groq.py not found at {transcribe_script}")
    
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


@typechecked
def main() -> None:
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
    temp_dir: Optional[Path] = None
    
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
        
        # Step 3: Check duration and decide whether to split
        duration = get_audio_duration(audio_file)
        
        if duration is not None:
            duration_minutes = duration / 60
            print(f"\n=== Step 3: Checking audio duration ===")
            print(f"Audio duration: {duration_minutes:.1f} minutes ({duration:.0f} seconds)")
        
        # Determine output filename (use original audio stem)
        output_vtt_name = f"{audio_file.stem}.vtt"
        final_vtt_path = output_dir / output_vtt_name
        
        if duration is not None and duration > SEGMENT_SECONDS:
            # Audio is longer than 25 minutes, split and transcribe in parallel
            print(f"Audio exceeds {SEGMENT_SECONDS // 60} minutes, will split and transcribe in parallel")
            
            # Create segments directory
            segments_dir = temp_dir / "segments"
            segments_dir.mkdir(exist_ok=True)
            
            # Create VTT output directory for segments
            vtt_segments_dir = temp_dir / "vtt_segments"
            vtt_segments_dir.mkdir(exist_ok=True)
            
            # Split audio into segments
            segment_files = retry_step(
                split_audio,
                audio_file,
                segments_dir,
                max_retries=2,
                delay=5,
                step_name="Splitting audio"
            )
            
            # Transcribe segments in parallel
            vtt_files = transcribe_segments_parallel(segment_files, vtt_segments_dir)
            
            # Get actual durations of each segment for accurate timestamp merging
            segment_durations: list[float] = []
            for seg_file in segment_files:
                seg_duration = get_audio_duration(seg_file)
                if seg_duration is not None:
                    segment_durations.append(seg_duration)
                else:
                    # Fallback to expected segment duration
                    segment_durations.append(float(SEGMENT_SECONDS))
            
            # Merge VTT files with timestamp correction
            print(f"\n=== Step 5: Merging {len(vtt_files)} VTT file(s) ===")
            merge_vtt_files(vtt_files, segment_durations, final_vtt_path)
            print(f"✓ Merged VTT saved to: {final_vtt_path.name}")
            
            # Add video URL to the merged VTT file
            add_video_url_to_vtt(final_vtt_path, video_url)
            print(f"✓ Video URL added to subtitle file")
            
        else:
            # Audio is 25 minutes or less, transcribe directly
            if duration is not None:
                print(f"Audio is under {SEGMENT_SECONDS // 60} minutes, transcribing directly")
            else:
                print("Could not determine duration, transcribing directly")
            
            # Step 4: Transcribe audio to VTT
            vtt_file = retry_step(
                transcribe_audio,
                audio_file,
                output_dir,
                video_url,
                max_retries=2,
                delay=5,
                step_name="Transcribing audio"
            )
            final_vtt_path = vtt_file
        
        print(f"\n✓ Success! Subtitle saved to: {final_vtt_path.absolute()}")
        
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

