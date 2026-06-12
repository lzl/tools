import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from transcribe_local_media import (
    compress_to_small_mp3,
    find_latest_media_file,
    parse_args,
    resolve_input_file,
)


class ParseArgsTests(unittest.TestCase):
    def test_accepts_input_and_output_directory(self) -> None:
        input_file, output_dir = parse_args(
            ["transcribe_local_media.py", "input_dir/demo.mp4", "output_dir"]
        )

        self.assertEqual(input_file, Path("input_dir/demo.mp4"))
        self.assertEqual(output_dir, Path("output_dir"))

    def test_defaults_to_input_dir_and_output_dir(self) -> None:
        input_file, output_dir = parse_args(["transcribe_local_media.py"])

        self.assertIsNone(input_file)
        self.assertEqual(output_dir, Path("output_dir"))


class InputFileTests(unittest.TestCase):
    def test_find_latest_media_file_ignores_unsupported_files(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            ignored = temp_dir / "book.epub"
            older = temp_dir / "older.mp3"
            newer = temp_dir / "newer.mp4"

            ignored.write_text("book")
            older.write_bytes(b"old")
            newer.write_bytes(b"new")

            self.assertEqual(find_latest_media_file(temp_dir), newer)

    def test_resolve_input_file_rejects_unsupported_extension(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            path = Path(temp_dir_name) / "book.epub"
            path.write_text("book")

            with self.assertRaises(ValueError):
                resolve_input_file(path)


class CompressionTests(unittest.TestCase):
    def test_compress_to_small_mp3_uses_ffmpeg_audio_only_settings(self) -> None:
        with TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_file = temp_dir / "demo.mp4"
            input_file.write_bytes(b"input")

            def fake_run(cmd: list[str], **_: object) -> None:
                self.assertEqual(cmd[0], "ffmpeg")
                self.assertIn("-vn", cmd)
                self.assertIn("-map", cmd)
                self.assertIn("0:a:0", cmd)
                self.assertIn("-ac", cmd)
                self.assertIn("1", cmd)
                self.assertIn("-ar", cmd)
                self.assertIn("16000", cmd)
                self.assertIn("-b:a", cmd)
                self.assertIn("32k", cmd)
                Path(cmd[-1]).write_bytes(b"mp3")

            with patch("transcribe_local_media.subprocess.run", side_effect=fake_run):
                result = compress_to_small_mp3(input_file, temp_dir)

            self.assertEqual(result, temp_dir / "demo.mp3")


if __name__ == "__main__":
    unittest.main()
