import json
import os
import tempfile
import unittest
from pathlib import Path

import douyin_user_download as douyin


class SecUserIdParsingTests(unittest.TestCase):
    def test_parse_sec_user_id_from_user_url(self) -> None:
        self.assertEqual(
            douyin.parse_sec_user_id(
                "https://www.douyin.com/user/"
                "MS4wLjABAAAAlCw8i2Klk3i7azZMk2lhdf4R3LXUSI4PP-e7DV0BfwcJNQeehP-VUwz4h3Bh-Y6v"
            ),
            "MS4wLjABAAAAlCw8i2Klk3i7azZMk2lhdf4R3LXUSI4PP-e7DV0BfwcJNQeehP-VUwz4h3Bh-Y6v",
        )

    def test_parse_raw_sec_user_id(self) -> None:
        self.assertEqual(
            douyin.parse_sec_user_id("MS4wLjABAAAATest-User_123"),
            "MS4wLjABAAAATest-User_123",
        )

    def test_reject_unsupported_urls(self) -> None:
        with self.assertRaises(ValueError):
            douyin.parse_sec_user_id("https://example.com/user/MS4wLjABAAAATest")


class CookieFileTests(unittest.TestCase):
    def test_read_cookie_file_parses_netscape_cookie_export(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cookie_file = Path(temp_dir) / "cookies.txt"
            cookie_file.write_text(
                "# Netscape HTTP Cookie File\n"
                ".douyin.com\tTRUE\t/\tTRUE\t1893456000\tttwid\tabc\n"
                "www.douyin.com\tFALSE\t/\tTRUE\t1893456000\tmsToken\tdef\n",
                encoding="utf-8",
            )

            self.assertEqual(
                douyin.read_cookie_file(cookie_file),
                "ttwid=abc; msToken=def",
            )

    def test_read_cookie_file_preserves_raw_cookie_header(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            cookie_file = Path(temp_dir) / "cookies.txt"
            cookie_file.write_text("ttwid=abc; msToken=def\n", encoding="utf-8")

            self.assertEqual(douyin.read_cookie_file(cookie_file), "ttwid=abc; msToken=def")


class ConfigFromArgsTests(unittest.TestCase):
    def test_default_output_dir_is_data_douyin_downloads(self) -> None:
        config = douyin.config_from_args(["MS4wLjABAAAATest"])

        self.assertEqual(config.output_root, Path("data/douyin_downloads"))

    def test_loads_cwd_cookies_txt_by_default(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                Path("cookies.txt").write_text("ttwid=abc; msToken=def\n", encoding="utf-8")

                config = douyin.config_from_args(["MS4wLjABAAAATest"])
            finally:
                os.chdir(original_cwd)

        self.assertEqual(config.cookie, "ttwid=abc; msToken=def")

    def test_explicit_cookie_overrides_cwd_cookies_txt(self) -> None:
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            try:
                Path("cookies.txt").write_text("ttwid=abc\n", encoding="utf-8")

                config = douyin.config_from_args(
                    ["MS4wLjABAAAATest", "--cookie", "sessionid=explicit"]
                )
            finally:
                os.chdir(original_cwd)

        self.assertEqual(config.cookie, "sessionid=explicit")


class OutputPathTests(unittest.TestCase):
    def test_build_video_path_uses_aweme_id_and_extension(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = douyin.build_video_path(
                Path(temp_dir),
                "MS4wLjABAAAATest",
                douyin.DouyinVideo(
                    aweme_id="../123",
                    desc="ignored",
                    create_time=1700000000,
                    source_url="https://www.douyin.com/video/../123",
                    download_url="https://example.test/media/video.mp4?token=1",
                ),
            )

            self.assertEqual(path.name, "123.mp4")
            self.assertEqual(path.parent.name, "videos")


class F2KwargsTests(unittest.TestCase):
    def test_build_f2_kwargs_includes_cookie_key_when_anonymous(self) -> None:
        kwargs = douyin.build_f2_kwargs(
            douyin.DownloadConfig(sec_user_id="MS4wLjABAAAATest", output_root=Path("out"))
        )

        self.assertIn("cookie", kwargs)
        self.assertIsNone(kwargs["cookie"])

    def test_config_with_guest_cookie_generates_ttwid_when_cookie_missing(self) -> None:
        config = douyin.DownloadConfig(sec_user_id="MS4wLjABAAAATest", output_root=Path("out"))

        resolved = douyin.config_with_guest_cookie(config, token_factory=lambda: "guest-token")

        self.assertEqual(resolved.cookie, "ttwid=guest-token")

    def test_config_with_guest_cookie_preserves_explicit_cookie(self) -> None:
        config = douyin.DownloadConfig(
            sec_user_id="MS4wLjABAAAATest",
            output_root=Path("out"),
            cookie="sessionid=explicit",
        )

        resolved = douyin.config_with_guest_cookie(config, token_factory=lambda: "guest-token")

        self.assertEqual(resolved.cookie, "sessionid=explicit")


class F2RecordExtractionTests(unittest.TestCase):
    def test_iter_aweme_records_prefers_to_list_over_column_dict(self) -> None:
        records = list(douyin.iter_aweme_records(FakeF2UserPostFilter()))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["aweme_id"], "1001")

    def test_coerce_video_record_accepts_f2_video_play_addr(self) -> None:
        video = douyin.coerce_video_record(
            {
                "aweme_id": "1001",
                "desc": "first",
                "create_time": "1700000000",
                "video_play_addr": [["https://example.test/play.mp4"]],
            }
        )

        self.assertIsNotNone(video)
        assert video is not None
        self.assertEqual(video.download_url, "https://example.test/play.mp4")
        self.assertEqual(video.create_time, 1700000000)

    def test_coerce_video_record_prefers_video_over_music_urls(self) -> None:
        video = douyin.coerce_video_record(
            {
                "aweme_id": "1001",
                "music": {"play_url": {"url_list": ["https://example.test/sound.mp3"]}},
                "video": {
                    "bit_rate": [
                        {
                            "play_addr": {
                                "url_list": ["https://example.test/video.mp4?token=1"]
                            }
                        }
                    ]
                },
            }
        )

        self.assertIsNotNone(video)
        assert video is not None
        self.assertEqual(video.download_url, "https://example.test/video.mp4?token=1")


class LoginGateTests(unittest.TestCase):
    def test_login_limited_response_detects_anonymous_more_prompt(self) -> None:
        self.assertTrue(
            douyin.is_login_limited_response(
                {
                    "has_more": 1,
                    "not_login_module": {
                        "guide_login_tip_text_biserial": "看更多最新作品",
                    },
                }
            )
        )

    def test_login_limited_response_ignores_complete_public_page(self) -> None:
        self.assertFalse(
            douyin.is_login_limited_response(
                {
                    "has_more": 0,
                    "not_login_module": {
                        "guide_login_tip_text_biserial": "看更多最新作品",
                    },
                }
            )
        )


class FakeF2UserPostFilter:
    def _to_dict(self):
        return {
            "aweme_id": ["1001"],
            "desc": ["first"],
            "video_play_addr": [["https://example.test/play.mp4"]],
        }

    def _to_list(self):
        return [
            {
                "aweme_id": "1001",
                "desc": "first",
                "video_play_addr": [["https://example.test/play.mp4"]],
            }
        ]


class FakeResponse:
    def __init__(self, content: bytes = b"video-bytes", content_type: str = "video/mp4") -> None:
        self.content = content
        self.headers = {"content-type": content_type, "content-length": str(len(content))}
        self.status_code = 200

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    async def aiter_bytes(self):
        yield self.content


class FakeHttpClient:
    def __init__(self) -> None:
        self.urls: list[str] = []

    def stream(self, method: str, url: str, headers=None, follow_redirects: bool = True):
        self.urls.append(url)
        return FakeStreamContext(FakeResponse())


class FakeStreamContext:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    async def __aenter__(self) -> FakeResponse:
        return self.response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class DownloadRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_dry_run_lists_videos_without_writing_media(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = FakeSource(
                [
                    douyin.DouyinVideo(
                        aweme_id="1001",
                        desc="first",
                        create_time=1700000000,
                        source_url="https://www.douyin.com/video/1001",
                        download_url="https://example.test/1001.mp4",
                    )
                ]
            )

            result = await douyin.download_user_videos(
                douyin.DownloadConfig(
                    sec_user_id="MS4wLjABAAAATest",
                    output_root=Path(temp_dir),
                    dry_run=True,
                ),
                fetch_user_posts=source.fetch,
                http_client=FakeHttpClient(),
            )

            self.assertEqual(result.dry_run, 1)
            self.assertEqual(result.downloaded, 0)
            self.assertFalse((Path(temp_dir) / "MS4wLjABAAAATest" / "videos").exists())

    async def test_downloads_all_fake_pages_once_and_writes_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = FakeSource(
                [
                    douyin.DouyinVideo("1001", "first", 1700000000, "https://www.douyin.com/video/1001", "https://example.test/1001.mp4"),
                    douyin.DouyinVideo("1002", "second", 1700000001, "https://www.douyin.com/video/1002", "https://example.test/1002.mp4"),
                ]
            )
            client = FakeHttpClient()

            result = await douyin.download_user_videos(
                douyin.DownloadConfig(sec_user_id="MS4wLjABAAAATest", output_root=Path(temp_dir)),
                fetch_user_posts=source.fetch,
                http_client=client,
            )

            user_dir = Path(temp_dir) / "MS4wLjABAAAATest"
            self.assertEqual(result.downloaded, 2)
            self.assertEqual(client.urls, ["https://example.test/1001.mp4", "https://example.test/1002.mp4"])
            self.assertEqual((user_dir / "videos" / "1001.mp4").read_bytes(), b"video-bytes")
            self.assertEqual((user_dir / "videos" / "1002.mp4").read_bytes(), b"video-bytes")

            manifest_rows = [
                json.loads(line)
                for line in (user_dir / "manifest.jsonl").read_text().splitlines()
            ]
            checkpoint = json.loads((user_dir / "checkpoint.json").read_text())
            self.assertEqual([row["aweme_id"] for row in manifest_rows], ["1001", "1002"])
            self.assertEqual(checkpoint["completed_aweme_ids"], ["1001", "1002"])

    async def test_resume_skips_existing_completed_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = FakeSource(
                [
                    douyin.DouyinVideo("1001", "first", 1700000000, "https://www.douyin.com/video/1001", "https://example.test/1001.mp4"),
                    douyin.DouyinVideo("1002", "second", 1700000001, "https://www.douyin.com/video/1002", "https://example.test/1002.mp4"),
                ]
            )
            config = douyin.DownloadConfig(sec_user_id="MS4wLjABAAAATest", output_root=Path(temp_dir))

            first_client = FakeHttpClient()
            second_client = FakeHttpClient()

            await douyin.download_user_videos(config, fetch_user_posts=source.fetch, http_client=first_client)
            result = await douyin.download_user_videos(config, fetch_user_posts=source.fetch, http_client=second_client)

            self.assertEqual(result.skipped, 2)
            self.assertEqual(result.downloaded, 0)
            self.assertEqual(second_client.urls, [])

    async def test_max_count_limits_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = FakeSource(
                [
                    douyin.DouyinVideo("1001", "first", 1700000000, "https://www.douyin.com/video/1001", "https://example.test/1001.mp4"),
                    douyin.DouyinVideo("1002", "second", 1700000001, "https://www.douyin.com/video/1002", "https://example.test/1002.mp4"),
                ]
            )
            client = FakeHttpClient()

            result = await douyin.download_user_videos(
                douyin.DownloadConfig(
                    sec_user_id="MS4wLjABAAAATest",
                    output_root=Path(temp_dir),
                    max_count=1,
                ),
                fetch_user_posts=source.fetch,
                http_client=client,
            )

            self.assertEqual(result.downloaded, 1)
            self.assertEqual(client.urls, ["https://example.test/1001.mp4"])


class FakeSource:
    def __init__(self, videos: list[douyin.DouyinVideo]) -> None:
        self.videos = videos

    async def fetch(self, config: douyin.DownloadConfig):
        for video in self.videos:
            yield video


if __name__ == "__main__":
    unittest.main()
