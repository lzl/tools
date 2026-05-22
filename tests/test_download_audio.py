import unittest
from pathlib import Path

from download_audio import (
    MIN_SIZE_FORMAT_SORT,
    STREAMING_FRAGMENT_CONCURRENCY,
    build_download_command,
    parse_args,
)


class BuildDownloadCommandTests(unittest.TestCase):
    def test_prefers_audio_with_combined_format_fallback(self) -> None:
        command = build_download_command(
            "https://example.com/video",
            Path("/tmp/output"),
        )

        format_index = command.index("-f")
        self.assertEqual(command[format_index + 1], "bestaudio/best")

    def test_sorts_formats_by_smallest_size_first(self) -> None:
        command = build_download_command(
            "https://example.com/video",
            Path("/tmp/output"),
        )

        format_sort_index = command.index("-S")
        self.assertEqual(command[format_sort_index + 1], MIN_SIZE_FORMAT_SORT)
        self.assertIn("--format-sort-force", command)

    def test_sets_configurable_fragment_concurrency(self) -> None:
        command = build_download_command(
            "https://example.com/video",
            Path("/tmp/output"),
            concurrent_fragments=12,
        )

        concurrency_index = command.index("-N")
        self.assertEqual(
            command[concurrency_index + 1],
            "12",
        )


class ParseArgsTests(unittest.TestCase):
    def test_accepts_custom_fragment_concurrency(self) -> None:
        video_url, output_dir, browser, concurrent_fragments = parse_args(
            [
                "download_audio.py",
                "https://example.com/video",
                "/tmp/output",
                "--browser",
                "chrome",
                "--concurrent-fragments",
                "32",
            ]
        )

        self.assertEqual(video_url, "https://example.com/video")
        self.assertEqual(output_dir, Path("/tmp/output"))
        self.assertEqual(browser, "chrome")
        self.assertEqual(concurrent_fragments, 32)

    def test_uses_default_fragment_concurrency(self) -> None:
        _, _, _, concurrent_fragments = parse_args(
            ["download_audio.py", "https://example.com/video"]
        )

        self.assertEqual(concurrent_fragments, STREAMING_FRAGMENT_CONCURRENCY)


if __name__ == "__main__":
    unittest.main()
