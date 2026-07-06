import asyncio
import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Callable
from unittest.mock import patch

from atoms import detect_nude_segments as detect_atom
from atoms import download_telegram_message_media as download_atom
from atoms import render_video_segments as render_atom
from workflows import telegram_nude_scenes as workflow_atom


class FakeClient:
    def __init__(self) -> None:
        self.disconnected = False

    async def disconnect(self) -> None:
        self.disconnected = True


class FakeTelegramApi:
    def __init__(
        self,
        client: FakeClient,
        *,
        output_root: Path,
        message: download_atom.ChannelMessage | None,
    ) -> None:
        self.client = client
        self.output_root = output_root
        self.message = message

    async def get_channel_message(
        self,
        channel_id: int | str,
        *,
        message_id: int,
    ) -> download_atom.ChannelMessage | None:
        return self.message

    async def download_media(
        self,
        message: download_atom.ChannelMessage,
        destination: Path,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if progress_callback is not None:
            progress_callback(7, 7)
        destination.write_bytes(b"payload")


class FakeDetector:
    def __init__(self, detections: dict[str, list[dict[str, object]]]) -> None:
        self.detections = detections

    def detect(self, image_path: str) -> list[dict[str, object]]:
        return self.detections.get(Path(image_path).name, [])


class DownloadTelegramMessageAtomTests(unittest.TestCase):
    def run_download_main(
        self,
        message: download_atom.ChannelMessage | None,
        args: list[str],
    ) -> tuple[int, str, str]:
        fake_client = FakeClient()

        def api_factory(client: FakeClient, *, output_root: Path) -> FakeTelegramApi:
            return FakeTelegramApi(client, output_root=output_root, message=message)

        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch.object(
            download_atom,
            "create_download_client",
            new=lambda: asyncio.sleep(0, result=fake_client),
        ), patch.object(
            download_atom,
            "TelethonMediaApi",
            new=api_factory,
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = download_atom.main(args)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_json_stdout_is_parseable_and_private_link_uses_expected_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            message = download_atom.ChannelMessage(
                channel_id=-1002334435280,
                message_id=8711,
                media_type="video",
                file_id="video-8711",
                extension=".mp4",
                source=None,
            )

            code, stdout, stderr = self.run_download_main(
                message,
                [
                    "https://t.me/c/2334435280/8711",
                    "--output-root",
                    temp_dir,
                    "--media-type",
                    "video",
                    "--json",
                ],
            )

            payload = json.loads(stdout)
            self.assertEqual(code, 0)
            self.assertEqual(payload["status"], "downloaded")
            self.assertEqual(
                payload["file_path"],
                str(Path(temp_dir) / "-1002334435280" / "videos" / "8711.mp4"),
            )
            self.assertNotIn("{", stderr)

    def test_public_link_uses_username_output_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            message = download_atom.ChannelMessage(
                channel_id="example_channel",
                message_id=42,
                media_type="video",
                file_id="video-42",
                extension=".mp4",
                source=None,
            )

            code, stdout, _stderr = self.run_download_main(
                message,
                [
                    "https://t.me/example_channel/42",
                    "--output-root",
                    temp_dir,
                    "--json",
                ],
            )

            self.assertEqual(code, 0)
            self.assertEqual(
                json.loads(stdout)["file_path"],
                str(Path(temp_dir) / "example_channel" / "videos" / "42.mp4"),
            )

    def test_video_mode_rejects_image_and_empty_messages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            image_message = download_atom.ChannelMessage(
                channel_id="example_channel",
                message_id=42,
                media_type="image",
                file_id="image-42",
                extension=".jpg",
                source=None,
            )
            image_code, _stdout, image_stderr = self.run_download_main(
                image_message,
                ["https://t.me/example_channel/42", "--output-root", temp_dir],
            )
            empty_code, _stdout, empty_stderr = self.run_download_main(
                download_atom.ChannelMessage(
                    channel_id="example_channel",
                    message_id=43,
                    media_type=None,
                    file_id=None,
                    extension=None,
                    source=None,
                ),
                ["https://t.me/example_channel/43", "--output-root", temp_dir],
            )

            self.assertEqual(image_code, 1)
            self.assertIn("expected 'video'", image_stderr)
            self.assertEqual(empty_code, 1)
            self.assertIn("no downloadable media", empty_stderr)


class DetectNudeSegmentsAtomTests(unittest.TestCase):
    def test_build_segments_handles_padding_merge_minimum_and_clipping(self) -> None:
        hits = [
            detect_atom.DetectionHit(time=1.0, label="A", score=0.9),
            detect_atom.DetectionHit(time=2.2, label="A", score=0.8),
            detect_atom.DetectionHit(time=9.8, label="A", score=0.7),
        ]

        segments = detect_atom.build_segments(
            hits,
            duration=10.0,
            sample_interval=0.5,
            padding=1.0,
            merge_gap=0.5,
            min_segment=1.0,
        )

        self.assertEqual(segments, [detect_atom.Segment(0.0, 3.7), detect_atom.Segment(8.8, 10.0)])

    def test_build_segments_returns_empty_for_empty_hits(self) -> None:
        self.assertEqual(
            detect_atom.build_segments([], 10.0, 0.5, 1.0, 1.0, 1.0),
            [],
        )

    def test_detect_hits_applies_thresholds_class_filtering_and_best_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "frame_00000001.jpg"
            second = root / "frame_00000002.jpg"
            first.write_bytes(b"x")
            second.write_bytes(b"x")
            detector = FakeDetector(
                {
                    first.name: [
                        {"class": "SAFE", "score": 0.99},
                        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.31},
                        {"class": "MALE_GENITALIA_EXPOSED", "score": 0.7},
                    ],
                    second.name: [
                        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.29},
                        {"class": "MALE_GENITALIA_EXPOSED", "score": 0.44},
                    ],
                }
            )

            hits = detect_atom.detect_hits(
                [first, second],
                detector,
                sample_fps=2.0,
                labels={"FEMALE_BREAST_EXPOSED", "MALE_GENITALIA_EXPOSED"},
                threshold=0.45,
                class_thresholds={"FEMALE_BREAST_EXPOSED": 0.3},
            )

            self.assertEqual(
                hits,
                [detect_atom.DetectionHit(time=0.0, label="MALE_GENITALIA_EXPOSED", score=0.7)],
            )

    def test_csv_and_json_fields_are_stable_for_empty_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            video = root / "input.mp4"
            video.write_bytes(b"x")
            csv_path = root / "out.csv"
            json_path = root / "out.json"
            summary = detect_atom.build_summary(
                input_path=video,
                output_video=root / "out.mp4",
                manifest_path=csv_path,
                duration=12.0,
                sample_fps=2.0,
                threshold=0.45,
                exposed_threshold=0.3,
                class_thresholds={"FEMALE_BREAST_EXPOSED": 0.3},
                classes=["FEMALE_BREAST_EXPOSED"],
                hits=[],
                segments=[],
            )
            detect_atom.write_manifest(csv_path, video, [], [])
            detect_atom.write_summary(json_path, summary)

            with csv_path.open(newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            self.assertEqual(rows, [["kind", "start", "end", "duration", "label", "score", "source"]])
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(
                list(payload.keys()),
                [
                    "input",
                    "output",
                    "manifest",
                    "duration",
                    "sample_fps",
                    "threshold",
                    "exposed_threshold",
                    "class_thresholds",
                    "classes",
                    "hit_count",
                    "segment_count",
                    "segment_seconds",
                    "segments",
                    "hits",
                ],
            )
            self.assertEqual(payload["segments"], [])
            self.assertEqual(payload["hits"], [])


class RenderVideoSegmentsAtomTests(unittest.TestCase):
    def test_segment_render_builds_clip_and_concat_commands(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.mp4"
            input_path.write_bytes(b"x")
            segments_json = root / "segments.json"
            segments_json.write_text(
                json.dumps({"segments": [{"start": 1.0, "end": 2.5}, {"start": 4.0, "end": 5.0}]}),
                encoding="utf-8",
            )
            output_path = root / "out.mp4"
            commands: list[list[str]] = []

            with patch.object(render_atom, "require_binary", new=lambda _name: None), patch.object(
                render_atom,
                "run",
                new=lambda cmd, capture=False: commands.append(cmd),
            ):
                result = render_atom.render_from_json(
                    input_path=input_path,
                    segments_json=segments_json,
                    output_path=output_path,
                    crf=21,
                    preset="fast",
                    keep_work_dir=False,
                    work_dir_root=None,
                )

            self.assertEqual(result["segment_count"], 2)
            self.assertFalse(result["placeholder"])
            self.assertEqual(len(commands), 3)
            self.assertEqual(commands[0][0], "ffmpeg")
            self.assertIn("-ss", commands[0])
            self.assertIn("1.000", commands[0])
            self.assertEqual(commands[-1][commands[-1].index("-f") + 1], "concat")

    def test_empty_segments_render_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "input.mp4"
            input_path.write_bytes(b"x")
            segments_json = root / "segments.json"
            segments_json.write_text(json.dumps({"segments": []}), encoding="utf-8")
            commands: list[list[str]] = []

            with patch.object(render_atom, "require_binary", new=lambda _name: None), patch.object(
                render_atom,
                "run",
                new=lambda cmd, capture=False: commands.append(cmd),
            ):
                result = render_atom.render_from_json(
                    input_path=input_path,
                    segments_json=segments_json,
                    output_path=root / "out.mp4",
                    crf=20,
                    preset="veryfast",
                    keep_work_dir=False,
                    work_dir_root=None,
                )

            self.assertTrue(result["placeholder"])
            self.assertEqual(len(commands), 1)
            self.assertIn("color=c=black:s=1280x720:d=0.2", commands[0])

    def test_missing_input_and_invalid_segments_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bad_json = root / "bad.json"
            bad_json.write_text(json.dumps({"segments": [{"start": 2, "end": 1}]}), encoding="utf-8")

            with self.assertRaises(FileNotFoundError):
                render_atom.render_from_json(
                    input_path=root / "missing.mp4",
                    segments_json=bad_json,
                    output_path=root / "out.mp4",
                    crf=20,
                    preset="veryfast",
                    keep_work_dir=False,
                    work_dir_root=None,
                )

            input_path = root / "input.mp4"
            input_path.write_bytes(b"x")
            with patch.object(render_atom, "require_binary", new=lambda _name: None):
                with self.assertRaises(ValueError):
                    render_atom.render_from_json(
                        input_path=input_path,
                        segments_json=bad_json,
                        output_path=root / "out.mp4",
                        crf=20,
                        preset="veryfast",
                        keep_work_dir=False,
                        work_dir_root=None,
                    )


class TelegramNudeScenesWorkflowTests(unittest.TestCase):
    def test_workflow_calls_atoms_in_order_and_derives_default_paths(self) -> None:
        calls: list[tuple[str, list[str]]] = []

        def fake_run_atom(atom: str, args: list[str]) -> dict[str, object]:
            calls.append((atom, args))
            if atom.endswith("download_telegram_message_media.py"):
                return {
                    "file_path": str(
                        Path("artifacts/run-1/downloads/-1002334435280/videos/8711.mp4")
                    )
                }
            if atom.endswith("detect_nude_segments.py"):
                return {"segment_count": 1}
            return {"output": "artifacts/run-1/outputs/8711_nude_scenes.mp4"}

        with patch.object(workflow_atom, "run_atom", new=fake_run_atom):
            result = workflow_atom.run_workflow(
                link="https://t.me/c/2334435280/8711",
                artifacts_root=Path("artifacts"),
                run_dir=Path("artifacts/run-1"),
                output=None,
                sample_fps=2.0,
                threshold=0.45,
                exposed_threshold=0.3,
                padding=3.0,
                merge_gap=4.0,
                min_segment=1.0,
                max_width=640,
                classes=["FEMALE_BREAST_EXPOSED"],
                keep_work_dir=False,
                crf=20,
                preset="veryfast",
            )

        self.assertEqual(
            [call[0] for call in calls],
            [
                "atoms/download_telegram_message_media.py",
                "atoms/detect_nude_segments.py",
                "atoms/render_video_segments.py",
            ],
        )
        self.assertEqual(result["artifacts_dir"], "artifacts/run-1")
        self.assertEqual(result["output_video"], "artifacts/run-1/outputs/8711_nude_scenes.mp4")
        self.assertEqual(result["manifest_csv"], "artifacts/run-1/outputs/8711_nude_scenes.csv")
        self.assertEqual(result["summary_json"], "artifacts/run-1/outputs/8711_nude_scenes.json")
        self.assertIn("artifacts/run-1/downloads", calls[0][1])
        self.assertIn("artifacts/run-1/outputs/8711_nude_scenes.json", calls[1][1])
        self.assertIn("artifacts/run-1/work", calls[1][1])
        self.assertIn("artifacts/run-1/work", calls[2][1])

    def test_workflow_stops_on_first_failing_atom(self) -> None:
        calls: list[str] = []

        def fake_run_atom(atom: str, args: list[str]) -> dict[str, object]:
            calls.append(atom)
            raise workflow_atom.AtomFailure(atom=atom, exit_code=7, stderr="boom")

        with patch.object(workflow_atom, "run_atom", new=fake_run_atom):
            with self.assertRaises(workflow_atom.AtomFailure) as caught:
                workflow_atom.run_workflow(
                    link="https://t.me/c/2334435280/8711",
                    artifacts_root=Path("artifacts"),
                    run_dir=Path("artifacts/run-1"),
                    output=None,
                    sample_fps=2.0,
                    threshold=0.45,
                    exposed_threshold=0.3,
                    padding=3.0,
                    merge_gap=4.0,
                    min_segment=1.0,
                    max_width=640,
                    classes=["FEMALE_BREAST_EXPOSED"],
                    keep_work_dir=False,
                    crf=20,
                    preset="veryfast",
                )

        self.assertEqual(calls, ["atoms/download_telegram_message_media.py"])
        self.assertIn("boom", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
