import unittest
from pathlib import Path

from download_audio import (
    STREAMING_FRAGMENT_CONCURRENCY,
    build_download_command,
)


class BuildDownloadCommandTests(unittest.TestCase):
    def test_uses_worstaudio_selector(self) -> None:
        command = build_download_command(
            "https://example.com/video",
            Path("/tmp/output"),
        )

        format_index = command.index("-f")
        self.assertEqual(command[format_index + 1], "worstaudio")

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
