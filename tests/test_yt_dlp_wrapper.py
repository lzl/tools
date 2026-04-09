import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from yt_dlp_wrapper import (
    _iter_binary_candidates,
    parse_version,
    resolve_yt_dlp,
    resolve_yt_dlp_binary,
)


class ParseVersionTests(unittest.TestCase):
    def test_parse_date_style_version(self) -> None:
        self.assertEqual(parse_version("2026.03.17"), (2026, 3, 17))

    def test_parse_unknown_version(self) -> None:
        self.assertEqual(parse_version("unknown"), (0,))


class IterBinaryCandidatesTests(unittest.TestCase):
    def test_discovers_exe_candidates_from_later_path_entries(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first_binary = Path(first_dir) / "yt-dlp.exe"
            second_binary = Path(second_dir) / "yt-dlp.exe"

            first_binary.write_text("")
            second_binary.write_text("")
            os.chmod(first_binary, 0o755)
            os.chmod(second_binary, 0o755)

            with patch.dict(
                os.environ,
                {
                    "PATH": os.pathsep.join([first_dir, second_dir]),
                    "PATHEXT": ".EXE;.BAT",
                },
                clear=False,
            ):
                with patch("shutil.which", return_value=str(first_binary)):
                    self.assertEqual(
                        _iter_binary_candidates(),
                        [str(first_binary), str(second_binary)],
                    )


class ResolveYtDlpBinaryTests(unittest.TestCase):
    def setUp(self) -> None:
        resolve_yt_dlp.cache_clear()

    def tearDown(self) -> None:
        resolve_yt_dlp.cache_clear()

    @patch("yt_dlp_wrapper._probe_binary")
    @patch(
        "yt_dlp_wrapper._iter_binary_candidates",
        return_value=["/warm"],
    )
    def test_a_warms_resolver_cache(
        self,
        _iter_candidates_mock: Mock,
        probe_binary_mock: Mock,
    ) -> None:
        probe_binary_mock.return_value = Mock(
            path="/warm",
            version="2025.01.01",
            version_tuple=(2025, 1, 1),
        )

        self.assertEqual(resolve_yt_dlp_binary(), "/warm")

    @patch("yt_dlp_wrapper._probe_binary")
    @patch(
        "yt_dlp_wrapper._iter_binary_candidates",
        return_value=[
            "/Users/test/uv-env/bin/yt-dlp",
            "/opt/homebrew/bin/yt-dlp",
        ],
    )
    def test_b_prefers_newer_external_binary_when_uv_env_is_older(
        self,
        _iter_candidates_mock: Mock,
        probe_binary_mock: Mock,
    ) -> None:
        probe_binary_mock.side_effect = [
            Mock(
                path="/Users/test/uv-env/bin/yt-dlp",
                version="2025.12.09",
                version_tuple=(2025, 12, 9),
            ),
            Mock(
                path="/opt/homebrew/bin/yt-dlp",
                version="2026.03.17",
                version_tuple=(2026, 3, 17),
            ),
        ]

        self.assertEqual(resolve_yt_dlp_binary(), "/opt/homebrew/bin/yt-dlp")


if __name__ == "__main__":
    unittest.main()
