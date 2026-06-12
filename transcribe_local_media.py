# /// script
# dependencies = [
#   "typeguard",
# ]
# ///

"""A tool to transcribe local audio/video files to WebVTT subtitles."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from typeguard import typechecked

from get_subtitle import (
    SEGMENT_SECONDS,
    add_video_url_to_vtt,
    get_audio_duration,
    merge_vtt_files,
    retry_step,
    split_audio,
    transcribe_audio,
    transcribe_segments_parallel,
)


MEDIA_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
    ".opus",
    ".webm",
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".mpeg",
    ".mpga",
}

MP3_BITRATE = "32k"
MP3_SAMPLE_RATE = "16000"


def find_latest_media_file(directory: Path) -> Path:
    """Find the latest modified supported media file in a directory."""
    media_files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS
    ]

    if not media_files:
        raise FileNotFoundError(f"No media files found in {directory}")

    return max(media_files, key=lambda path: path.stat().st_mtime)


def parse_args(argv: list[str]) -> tuple[Optional[Path], Path]:
    """
    Parse command line arguments.

    Returns:
        Tuple of (input_file, output_dir). If input_file is None, use input_dir/.
    """
    if len(argv) > 3:
        print("Usage: transcribe_local_media [media_file] [output_directory]")
        print("\nExample:")
        print("  transcribe_local_media")
        print("  transcribe_local_media input_dir/demo.mp4")
        print("  transcribe_local_media input_dir/demo.mp4 ./output_dir")
        sys.exit(1)

    input_file: Optional[Path] = None
    output_dir = Path("output_dir")

    if len(argv) > 1:
        input_file = Path(argv[1])

    if len(argv) > 2:
        output_dir = Path(argv[2])

    return input_file, output_dir


def resolve_input_file(input_file: Optional[Path]) -> Path:
    """Resolve and validate the input media file."""
    if input_file is None:
        input_dir = Path("input_dir")
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory '{input_dir}' does not exist")

        latest_file = find_latest_media_file(input_dir)
        print(f"Using latest media file from {input_dir}: {latest_file.name}")
        return latest_file

    if not input_file.exists():
        raise FileNotFoundError(f"File '{input_file}' does not exist")

    if not input_file.is_file():
        raise ValueError(f"'{input_file}' is not a file")

    if input_file.suffix.lower() not in MEDIA_EXTENSIONS:
        supported = ", ".join(sorted(MEDIA_EXTENSIONS))
        raise ValueError(f"'{input_file}' is not a supported media format. Supported formats: {supported}")

    return input_file


def compress_to_small_mp3(input_file: Path, output_dir: Path) -> Path:
    """Compress local audio/video media into a small MP3 file for transcription."""
    output_mp3 = output_dir / f"{input_file.stem}.mp3"

    print("\n=== Step 1: Compressing local media to small MP3 ===")
    print(f"Input file: {input_file}")
    print(f"Compressed MP3: {output_mp3.name}")
    print(f"Encoding: mono, {MP3_SAMPLE_RATE} Hz, {MP3_BITRATE}")

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_file),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        MP3_SAMPLE_RATE,
        "-b:a",
        MP3_BITRATE,
        str(output_mp3),
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg first.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "No ffmpeg error output"
        raise RuntimeError(f"Failed to compress media with ffmpeg: {stderr}") from exc

    input_size_mb = input_file.stat().st_size / 1024 / 1024
    output_size_mb = output_mp3.stat().st_size / 1024 / 1024
    print(f"✓ Compression completed: {input_size_mb:.2f} MB -> {output_size_mb:.2f} MB")

    return output_mp3


def transcribe_compressed_mp3(mp3_file: Path, output_dir: Path, source_file: Path, temp_dir: Path) -> Path:
    """Transcribe a compressed MP3 directly or by splitting it into segments."""
    duration = get_audio_duration(mp3_file)

    if duration is not None:
        duration_minutes = duration / 60
        print("\n=== Step 2: Checking compressed audio duration ===")
        print(f"Audio duration: {duration_minutes:.1f} minutes ({duration:.0f} seconds)")

    final_vtt_path = output_dir / f"{source_file.stem}.vtt"

    if duration is not None and duration > SEGMENT_SECONDS:
        print(f"Audio exceeds {SEGMENT_SECONDS // 60} minutes, will split and transcribe in parallel")

        segments_dir = temp_dir / "segments"
        segments_dir.mkdir(exist_ok=True)

        vtt_segments_dir = temp_dir / "vtt_segments"
        vtt_segments_dir.mkdir(exist_ok=True)

        segment_files = retry_step(
            split_audio,
            mp3_file,
            segments_dir,
            max_retries=2,
            delay=5,
            step_name="Splitting audio",
        )

        vtt_files = transcribe_segments_parallel(segment_files, vtt_segments_dir)

        segment_durations: list[float] = []
        for segment_file in segment_files:
            segment_duration = get_audio_duration(segment_file)
            segment_durations.append(segment_duration if segment_duration is not None else float(SEGMENT_SECONDS))

        print(f"\n=== Step 5: Merging {len(vtt_files)} VTT file(s) ===")
        merge_vtt_files(vtt_files, segment_durations, final_vtt_path)
        print(f"✓ Merged VTT saved to: {final_vtt_path.name}")
    else:
        if duration is not None:
            print(f"Audio is under {SEGMENT_SECONDS // 60} minutes, transcribing directly")
        else:
            print("Could not determine duration, transcribing directly")

        vtt_file = retry_step(
            transcribe_audio,
            mp3_file,
            output_dir,
            None,
            max_retries=2,
            delay=5,
            step_name="Transcribing audio",
        )
        if vtt_file != final_vtt_path:
            if final_vtt_path.exists():
                final_vtt_path.unlink()
            vtt_file.rename(final_vtt_path)

    add_video_url_to_vtt(final_vtt_path, str(source_file))
    print("✓ Source file added to subtitle file")

    return final_vtt_path


@typechecked
def main() -> None:
    """Transcribe a local audio/video file by first compressing it to MP3."""
    input_arg, output_dir = parse_args(sys.argv)
    temp_dir: Optional[Path] = None

    try:
        input_file = resolve_input_file(input_arg)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Local media file: {input_file}")
        print(f"Output directory: {output_dir.absolute()}")

        temp_dir = Path(tempfile.mkdtemp(prefix="transcribe_local_media_"))
        print(f"Temporary directory: {temp_dir}")

        compressed_mp3 = retry_step(
            compress_to_small_mp3,
            input_file,
            temp_dir,
            max_retries=2,
            delay=5,
            step_name="Compressing media",
        )

        final_vtt_path = transcribe_compressed_mp3(compressed_mp3, output_dir, input_file, temp_dir)

        print(f"\n✓ Success! Subtitle saved to: {final_vtt_path.absolute()}")
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user")
        sys.exit(1)
    except Exception as exc:
        print(f"\n✗ Error: {exc}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        if temp_dir and temp_dir.exists():
            print("\nCleaning up temporary directory...")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
