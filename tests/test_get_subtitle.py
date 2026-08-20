import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from get_subtitle import (
    GROQ_MAX_FILE_SIZE_BYTES,
    SEGMENT_SECONDS,
    calculate_segment_seconds,
    download_audio,
    get_safe_filename_from_url,
    media_requires_splitting,
    parse_args,
    split_audio,
)


class ParseArgsTests(unittest.TestCase):
    def test_accepts_custom_fragment_concurrency(self) -> None:
        video_url, output_dir, browser, concurrent_fragments = parse_args(
            [
                "get_subtitle.py",
                "https://example.com/video",
                "/tmp/subtitles",
                "--browser",
                "chrome",
                "--concurrent-fragments",
                "40",
            ]
        )

        self.assertEqual(video_url, "https://example.com/video")
        self.assertEqual(output_dir, Path("/tmp/subtitles"))
        self.assertEqual(browser, "chrome")
        self.assertEqual(concurrent_fragments, 40)


class DownloadAudioTests(unittest.TestCase):
    def test_forwards_fragment_concurrency_to_download_audio_script(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            downloaded = temp_dir / "example.mp4"

            def fake_run(cmd: list[str], **_: object) -> None:
                self.assertIn("--concurrent-fragments", cmd)
                option_index = cmd.index("--concurrent-fragments")
                self.assertEqual(cmd[option_index + 1], "40")
                downloaded.write_bytes(b"audio")

            with patch("get_subtitle.subprocess.run", side_effect=fake_run):
                result = download_audio(
                    "https://example.com/video",
                    temp_dir,
                    browser="chrome",
                    concurrent_fragments=40,
                )

        self.assertEqual(
            result.name,
            get_safe_filename_from_url("https://example.com/video", ".mp4"),
        )


class MediaSplittingTests(unittest.TestCase):
    def test_short_oversized_media_requires_splitting(self) -> None:
        duration = 983.0
        file_size_bytes = 64 * 1024 * 1024

        self.assertTrue(media_requires_splitting(duration, file_size_bytes))
        self.assertEqual(calculate_segment_seconds(duration, file_size_bytes), 307)

    def test_small_media_uses_duration_limit(self) -> None:
        duration = 10 * 60.0
        file_size_bytes = 10 * 1024 * 1024

        self.assertFalse(media_requires_splitting(duration, file_size_bytes))
        self.assertEqual(
            calculate_segment_seconds(duration, file_size_bytes),
            SEGMENT_SECONDS,
        )

    def test_exact_file_size_limit_does_not_require_splitting(self) -> None:
        self.assertFalse(
            media_requires_splitting(60.0, GROQ_MAX_FILE_SIZE_BYTES)
        )

    def test_split_audio_retries_with_shorter_segments_when_output_is_too_large(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            audio_path = temp_dir / "audio.mp4"
            segments_dir = temp_dir / "segments"
            audio_path.touch()
            requested_durations: list[int] = []

            def fake_run(cmd: list[str], **_: object) -> None:
                requested_durations.append(int(cmd[-2]))
                attempt_dir = Path(cmd[-1])
                segment_path = attempt_dir / "audio_part_000.mp4"
                output_size = (26 if len(requested_durations) == 1 else 20) * 1024 * 1024
                with segment_path.open("wb") as segment_file:
                    segment_file.truncate(output_size)

            with patch("get_subtitle.subprocess.run", side_effect=fake_run):
                segments = split_audio(audio_path, segments_dir, segment_seconds=600)

        self.assertEqual(requested_durations, [600, 461])
        self.assertEqual(len(segments), 1)


if __name__ == "__main__":
    unittest.main()
