import unittest
from pathlib import Path

from download_audio import (
    MIN_SIZE_FORMAT_SORT,
    STREAMING_FRAGMENT_CONCURRENCY,
    build_download_command,
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
        )

        concurrency_index = command.index("-N")
        self.assertEqual(
            command[concurrency_index + 1],
            str(STREAMING_FRAGMENT_CONCURRENCY),
        )


if __name__ == "__main__":
    unittest.main()
