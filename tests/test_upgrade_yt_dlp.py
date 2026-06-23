import unittest
from pathlib import Path

from upgrade_yt_dlp import (
    backup_path_for,
    checksum_for_asset,
    parse_tag_from_release_url,
    release_asset_url,
    should_upgrade,
)


class LatestReleaseUrlTests(unittest.TestCase):
    def test_parse_tag_from_release_url(self) -> None:
        self.assertEqual(
            parse_tag_from_release_url("https://github.com/yt-dlp/yt-dlp/releases/tag/2026.06.09"),
            "2026.06.09",
        )

    def test_parse_tag_from_release_url_rejects_non_tag_url(self) -> None:
        self.assertIsNone(
            parse_tag_from_release_url("https://github.com/yt-dlp/yt-dlp/releases/latest")
        )


class UpgradeDecisionTests(unittest.TestCase):
    def test_skips_when_current_matches_latest(self) -> None:
        self.assertFalse(should_upgrade("2026.06.09", "2026.06.09"))

    def test_skips_when_current_is_newer_than_latest(self) -> None:
        self.assertFalse(should_upgrade("2026.06.10", "2026.06.09"))

    def test_upgrades_when_current_is_older(self) -> None:
        self.assertTrue(should_upgrade("2026.03.17", "2026.06.09"))

    def test_force_upgrades_matching_version(self) -> None:
        self.assertTrue(should_upgrade("2026.06.09", "2026.06.09", force=True))


class ReleaseAssetUrlTests(unittest.TestCase):
    def test_builds_release_asset_url(self) -> None:
        self.assertEqual(
            release_asset_url("2026.06.09", "yt-dlp"),
            "https://github.com/yt-dlp/yt-dlp/releases/download/2026.06.09/yt-dlp",
        )


class ChecksumTests(unittest.TestCase):
    def test_finds_checksum_for_asset(self) -> None:
        checksums = "\n".join(
            [
                "abc123  yt-dlp_linux",
                "def456  yt-dlp",
                "789abc  yt-dlp.exe",
            ]
        )

        self.assertEqual(checksum_for_asset(checksums, "yt-dlp"), "def456")


class BackupPathTests(unittest.TestCase):
    def test_backup_path_includes_version(self) -> None:
        self.assertEqual(
            str(backup_path_for(Path("/tmp/yt-dlp"), "2026.03.17")),
            "/tmp/yt-dlp.2026.03.17.bak",
        )


if __name__ == "__main__":
    unittest.main()
