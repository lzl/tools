import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atoms import backup_zsxq as atom
from workflows import backup_zsxq as workflow


class FakeResponse:
    def __init__(self, payload: dict[str, object] | bytes) -> None:
        self.body = json.dumps(payload, ensure_ascii=False).encode("utf-8") if isinstance(payload, dict) else payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size == -1:
            body, self.body = self.body, b""
            return body
        body, self.body = self.body[:size], self.body[size:]
        return body


class ZsxqAtomTests(unittest.TestCase):
    def test_render_entities_and_topic_content(self) -> None:
        self.assertEqual(
            atom.render_entities('<e type="web" href="https%3A%2F%2Fexample.com" title="例子" />'),
            "[例子](https://example.com)",
        )
        kind, content = atom.topic_content({"talk": {"text": "hello"}})
        self.assertEqual((kind, content["text"]), ("talk", "hello"))

    def test_article_parser_extracts_full_content_and_removes_active_markup(self) -> None:
        parser = atom.ArticleContentParser("https://articles.zsxq.com/inline_form/example.html")
        parser.feed(
            '<html><body><script>outside()</script><div class="content ql-editor">'
            '<h2>完整标题</h2><p>第一段<strong>重点</strong></p>'
            '<img src="/images/example.jpg" onerror="bad()"><script>bad()</script>'
            '</div><p>outside</p></body></html>'
        )
        self.assertIn("完整标题", parser.content_text)
        self.assertIn("第一段重点", parser.content_text)
        self.assertNotIn("outside", parser.content_text)
        self.assertNotIn("<script", parser.content_html)
        self.assertNotIn("onerror", parser.content_html)
        self.assertIn('src="https://articles.zsxq.com/images/example.jpg"', parser.content_html)

    def test_hydrate_article_saves_full_html_and_searchable_text(self) -> None:
        topic = {
            "topic_id": "2",
            "type": "talk",
            "talk": {
                "text": "truncated...",
                "article": {"inline_article_url": "https://articles.zsxq.com/inline_form/example.html"},
            },
        }

        def opener(request: object, timeout: int = 60) -> FakeResponse:
            return FakeResponse('<div class="content ql-editor"><p>完整正文内容</p></div>'.encode())

        self.assertTrue(atom.hydrate_article(topic, token="token", user_agent="agent", opener=opener))
        article = topic["talk"]["article"]  # type: ignore[index]
        self.assertEqual(article["content_text"], "完整正文内容")  # type: ignore[index]
        self.assertIn("<p>完整正文内容</p>", article["content_html"])  # type: ignore[index]
        self.assertEqual(atom.content_text(topic["talk"]), "完整正文内容")  # type: ignore[arg-type]

    def test_backup_paginates_deduplicates_and_writes_outputs(self) -> None:
        group = {"group_id": "123", "name": "测试星球", "description": "desc"}
        first = {
            "topic_id": "2",
            "create_time": "2026-08-02T00:00:00.000+0800",
            "type": "talk",
            "talk": {"text": "new", "owner": {"name": "A"}},
        }
        older = {
            "topic_id": "1",
            "create_time": "2026-08-01T00:00:00.000+0800",
            "type": "talk",
            "talk": {"text": "old", "owner": {"name": "B"}},
        }
        requests: list[str] = []

        def opener(request: object, timeout: int = 60) -> FakeResponse:
            url = request.full_url  # type: ignore[attr-defined]
            requests.append(url)
            if url.endswith("/groups/123"):
                return FakeResponse({"succeeded": True, "resp_data": {"group": group}})
            if "end_time=" not in url:
                return FakeResponse({"succeeded": True, "resp_data": {"topics": [first, older]}})
            return FakeResponse({"succeeded": True, "resp_data": {"topics": [older]}})

        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"ZSXQ_ACCESS_TOKEN": "token"}, clear=False):
            result = atom.backup_planet(group_id="123", output_dir=Path(directory), delay=0, download_media=False, opener=opener)
            self.assertEqual(result["topic_count"], 2)
            self.assertEqual(result["page_count"], 2)
            self.assertEqual(len(requests), 3)
            self.assertIn("测试星球", (Path(directory) / "topics.md").read_text(encoding="utf-8"))
            raw_lines = (Path(directory) / "topics.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual([json.loads(line)["topic_id"] for line in raw_lines], ["2", "1"])
            self.assertEqual(json.loads((Path(directory) / "summary.json").read_text(encoding="utf-8"))["status"], "completed")


class ZsxqWorkflowTests(unittest.TestCase):
    def test_default_artifacts_root_matches_other_workflows(self) -> None:
        self.assertEqual(workflow.DEFAULT_ARTIFACTS_ROOT, Path("artifacts"))

    def test_workflow_passes_atom_contract_and_returns_artifacts(self) -> None:
        calls: list[tuple[str, list[str]]] = []

        def fake_run_atom(atom_path: str, args: list[str]) -> dict[str, object]:
            calls.append((atom_path, args))
            output_dir = Path(args[args.index("--output-dir") + 1])
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "group.json").write_text(json.dumps({"group_id": "123", "name": "测试星球"}), encoding="utf-8")
            (output_dir / "topics.jsonl").write_text(json.dumps({"topic_id": "1", "type": "talk", "talk": {"text": "全文"}}, ensure_ascii=False) + "\n", encoding="utf-8")
            (output_dir / "media.jsonl").write_text("", encoding="utf-8")
            result: dict[str, object] = {"status": "completed", "topic_count": 1, "output_dir": str(output_dir)}
            (output_dir / "summary.json").write_text(json.dumps(result), encoding="utf-8")
            return result

        with tempfile.TemporaryDirectory() as directory, patch.object(workflow, "run_atom", new=fake_run_atom):
            result = workflow.run_workflow(group_id="123", artifacts_root=Path(directory), run_dir=None, scope="all", count=30, delay=0, no_media=True, max_pages=1)
            self.assertEqual(result["group_id"], "123")
            self.assertEqual(result["backup"]["topic_count"], 1)  # type: ignore[index]
            self.assertEqual(calls[0][0], "atoms/backup_zsxq.py")
            self.assertIn("--no-media", calls[0][1])
            self.assertIn("--max-pages", calls[0][1])
            run_dir = Path(result["artifacts_dir"])  # type: ignore[arg-type]
            self.assertTrue((run_dir / "zsxq_viewer.html").is_file())
            viewer_data = (run_dir / "zsxq_viewer_data.js").read_text(encoding="utf-8")
            self.assertIn('\"artifactRoot\":\".\"', viewer_data)
            self.assertIn("全文", viewer_data)
            summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["viewer_html"], str(run_dir / "zsxq_viewer.html"))


if __name__ == "__main__":
    unittest.main()
