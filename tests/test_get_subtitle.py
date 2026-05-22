import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from get_subtitle import download_audio, get_safe_filename_from_url, parse_args


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


if __name__ == "__main__":
    unittest.main()
