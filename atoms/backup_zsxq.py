# /// script
# requires-python = ">=3.11"
# ///

"""Back up topics and complete articles from one ZSXQ (知识星球) group.

The API is intentionally used only with an access token belonging to the
caller.  It does not log in, bypass membership checks, or expose the token in
the generated artifacts.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urljoin, urlparse
from urllib.request import Request, urlopen


API_ROOT = "https://api.zsxq.com/v2"
WEB_ROOT = "https://wx.zsxq.com"
DEFAULT_COUNT = 30
MAX_COUNT = 30
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; zsxq-backup/1.0)"
TOKEN_ENV = "ZSXQ_ACCESS_TOKEN"
USER_AGENT_ENV = "ZSXQ_USER_AGENT"
TOPIC_TYPES = ("talk", "question", "task", "solution")
_ENTITY_RE = re.compile(r'<e\s+type="(?P<type>[^"]+)"(?P<attrs>.*?)\s*/>', re.DOTALL)
_ATTR_RE = re.compile(r'(?P<name>[a-zA-Z_]+)="(?P<value>.*?)"')
_ARTICLE_BLOCK_TAGS = {"blockquote", "div", "figcaption", "h1", "h2", "h3", "h4", "hr", "li", "ol", "p", "pre", "ul"}
_ARTICLE_VOID_TAGS = {"br", "hr", "img"}
_ARTICLE_ALLOWED_TAGS = _ARTICLE_BLOCK_TAGS | {"a", "br", "code", "em", "img", "s", "strong"}


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def find_dotenv(start_dir: Path) -> Path | None:
    current = start_dir.resolve()
    for directory in (current, *current.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_dotenv_if_present(start_dir: Path | None = None) -> Path | None:
    """Load simple KEY=value lines without overriding exported variables."""
    env_path = find_dotenv(start_dir or Path.cwd())
    if env_path is None:
        return None
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value
    return env_path


def require_credentials() -> tuple[str, str]:
    token = os.environ.get(TOKEN_ENV, "").strip()
    user_agent = os.environ.get(USER_AGENT_ENV, "").strip() or DEFAULT_USER_AGENT
    if token.startswith("zsxq_access_token="):
        token = token.split("=", 1)[1].split(";", 1)[0].strip()
    if not token:
        raise RuntimeError(
            f"{TOKEN_ENV} is required; copy zsxq_access_token from your logged-in wx.zsxq.com cookie"
        )
    return token, user_agent


def validate_group_id(value: str) -> str:
    group_id = value.strip()
    if not group_id.isdigit():
        raise ValueError("group_id must be the numeric ID from the ZSXQ URL")
    return group_id


def choose_image_url(image: dict[str, Any]) -> str | None:
    for field in ("original", "large", "thumbnail"):
        candidate = image.get(field)
        if isinstance(candidate, dict) and isinstance(candidate.get("url"), str):
            return candidate["url"]
    return None


def topic_content(topic: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for key in TOPIC_TYPES:
        value = topic.get(key)
        if isinstance(value, dict):
            return key, value
    return "unknown", {}


def render_entities(text: str) -> str:
    """Turn ZSXQ's inline XML-ish entities into readable Markdown."""

    def replace(match: re.Match[str]) -> str:
        attrs = {item.group("name"): item.group("value") for item in _ATTR_RE.finditer(match.group("attrs"))}
        entity_type = match.group("type")
        title = unquote(attrs.get("title", ""))
        if entity_type == "web":
            href = unquote(attrs.get("href", ""))
            return f"[{title or href}]({href})" if href else title
        return title or attrs.get("name", "") or f"[{entity_type}]"

    return _ENTITY_RE.sub(replace, text)


def safe_name(value: str, fallback: str = "file") -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", value).strip(" .")
    return cleaned[:180] or fallback


def response_json(request: Request, *, retries: int = 3, opener: Callable[..., Any] = urlopen) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with opener(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not isinstance(payload, dict):
                raise RuntimeError("ZSXQ API returned a non-object JSON response")
            code = payload.get("code")
            if payload.get("succeeded") is True:
                return payload
            if code != 1059 or attempt >= retries:
                raise RuntimeError(f"ZSXQ API rejected the request (code={code!r})")
            last_error = RuntimeError(f"temporary API response code {code}")
        except HTTPError as exc:
            last_error = exc
            if exc.code not in {429, 500, 502, 503, 504} or attempt >= retries:
                raise RuntimeError(f"ZSXQ API HTTP {exc.code}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt >= retries:
                raise RuntimeError(f"ZSXQ API request failed: {exc}") from exc
        if attempt < retries:
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"ZSXQ API request failed: {last_error}")


def api_request(
    path: str,
    *,
    token: str,
    user_agent: str,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    request = Request(
        API_ROOT + path,
        headers={
            "Accept": "application/json",
            "Cookie": f"zsxq_access_token={token}",
            "Origin": WEB_ROOT,
            "Referer": f"{WEB_ROOT}/dweb2/",
            "User-Agent": user_agent,
        },
    )
    return response_json(request, opener=opener)


def content_text(content: dict[str, Any]) -> str:
    article = content.get("article")
    if isinstance(article, dict) and isinstance(article.get("content_text"), str):
        return article["content_text"]
    text = content.get("text", "")
    return render_entities(text) if isinstance(text, str) else ""


class ArticleContentParser(HTMLParser):
    """Extract safe, portable markup from a ZSXQ inline article page."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.capture_depth = 0
        self.ignored_depth = 0
        self.parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if self.capture_depth == 0:
            if tag == "div" and {"content", "ql-editor"}.issubset(classes):
                self.capture_depth = 1
            return
        if self.ignored_depth:
            if tag not in _ARTICLE_VOID_TAGS:
                self.capture_depth += 1
                self.ignored_depth += 1
            return
        if tag in {"script", "style"}:
            self.capture_depth += 1
            self.ignored_depth = 1
            return
        if tag not in _ARTICLE_VOID_TAGS:
            self.capture_depth += 1
        if tag not in _ARTICLE_ALLOWED_TAGS:
            return
        safe_attrs: list[tuple[str, str]] = []
        if tag == "a" and attributes.get("href"):
            href = urljoin(self.base_url, attributes["href"])
            if urlparse(href).scheme in {"http", "https"}:
                safe_attrs.extend((("href", href), ("target", "_blank"), ("rel", "noreferrer")))
        elif tag == "img" and attributes.get("src"):
            src = urljoin(self.base_url, attributes["src"])
            if urlparse(src).scheme in {"http", "https"}:
                safe_attrs.append(("src", src))
                for name in ("alt", "title"):
                    if attributes.get(name):
                        safe_attrs.append((name, attributes[name]))
                safe_attrs.append(("loading", "lazy"))
        rendered_attrs = "".join(f' {name}="{html.escape(value, quote=True)}"' for name, value in safe_attrs)
        self.parts.append(f"<{tag}{rendered_attrs}>")
        if tag == "br":
            self.text_parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)

    def handle_endtag(self, tag: str) -> None:
        if self.capture_depth == 0:
            return
        if self.ignored_depth:
            if tag not in _ARTICLE_VOID_TAGS:
                self.capture_depth -= 1
                self.ignored_depth -= 1
            return
        if self.capture_depth == 1 and tag == "div":
            self.capture_depth = 0
            return
        if tag in _ARTICLE_ALLOWED_TAGS and tag not in _ARTICLE_VOID_TAGS:
            self.parts.append(f"</{tag}>")
        if tag in _ARTICLE_BLOCK_TAGS:
            self.text_parts.append("\n")
        if tag not in _ARTICLE_VOID_TAGS:
            self.capture_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.capture_depth and not self.ignored_depth:
            self.parts.append(html.escape(data))
            self.text_parts.append(data)

    @property
    def content_html(self) -> str:
        return "".join(self.parts).strip()

    @property
    def content_text(self) -> str:
        lines = (re.sub(r"[ \t]+", " ", line).strip() for line in "".join(self.text_parts).splitlines())
        return "\n".join(line for line in lines if line)


def fetch_article_content(
    url: str,
    *,
    token: str,
    user_agent: str,
    opener: Callable[..., Any] = urlopen,
    retries: int = 3,
) -> tuple[str, str]:
    request = Request(url, headers={"Cookie": f"zsxq_access_token={token}", "Referer": WEB_ROOT, "User-Agent": user_agent})
    body = ""
    for attempt in range(retries + 1):
        try:
            with opener(request, timeout=60) as response:
                body = response.read().decode("utf-8")
            break
        except (HTTPError, URLError, TimeoutError, UnicodeDecodeError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"ZSXQ article request failed: {exc}") from exc
            time.sleep(min(2 ** attempt, 8))
    parser = ArticleContentParser(url)
    parser.feed(body)
    if not parser.content_html or not parser.content_text:
        raise RuntimeError("ZSXQ article page did not include full article content")
    return parser.content_html, parser.content_text


def hydrate_article(
    topic: dict[str, Any],
    *,
    token: str,
    user_agent: str,
    opener: Callable[..., Any] = urlopen,
) -> bool:
    _, content = topic_content(topic)
    article = content.get("article")
    if not isinstance(article, dict):
        return False
    url = article.get("inline_article_url") or article.get("article_url")
    if not isinstance(url, str) or not url:
        raise RuntimeError(f"topic {topic.get('topic_id')} article did not include a URL")
    content_html, full_text = fetch_article_content(url, token=token, user_agent=user_agent, opener=opener)
    article["content_html"] = content_html
    article["content_text"] = full_text
    return True


def iter_topic_media(topic: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    _, content = topic_content(topic)
    for image in content.get("images", []) if isinstance(content.get("images"), list) else []:
        if isinstance(image, dict):
            url = choose_image_url(image)
            if url:
                yield "image", {"url": url, "image_id": str(image.get("image_id", "image"))}
    for file_info in content.get("files", []) if isinstance(content.get("files"), list) else []:
        if isinstance(file_info, dict) and file_info.get("file_id") is not None:
            yield "file", file_info


def download_url(
    url: str,
    path: Path,
    *,
    token: str,
    user_agent: str,
    opener: Callable[..., Any] = urlopen,
) -> int:
    request = Request(url, headers={"Cookie": f"zsxq_access_token={token}", "User-Agent": user_agent})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    try:
        with opener(request, timeout=120) as response, temporary.open("wb") as output:
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                total += len(chunk)
        temporary.replace(path)
        return total
    finally:
        temporary.unlink(missing_ok=True)


def fetch_file_download_url(
    file_id: str,
    *,
    token: str,
    user_agent: str,
    opener: Callable[..., Any] = urlopen,
) -> str:
    payload = api_request(f"/files/{quote(file_id, safe='')}/download_url", token=token, user_agent=user_agent, opener=opener)
    data = payload.get("resp_data")
    if not isinstance(data, dict) or not isinstance(data.get("download_url"), str):
        raise RuntimeError(f"file {file_id} response did not include download_url")
    return data["download_url"]


def render_topic(topic: dict[str, Any], media_paths: list[str]) -> str:
    topic_id = str(topic.get("topic_id", "unknown"))
    topic_type, content = topic_content(topic)
    owner = content.get("owner") if isinstance(content.get("owner"), dict) else {}
    author = "匿名用户" if content.get("anonymous") else owner.get("name", "未知作者")
    created = str(topic.get("create_time", ""))
    title = str(topic.get("title") or content_text(content).splitlines()[0] if content_text(content) else topic_type)
    lines = [f"## {title}", "", f"- topic_id: `{topic_id}`", f"- type: `{topic_type}`", f"- author: {author}", f"- created: `{created}`", f"- source: https://wx.zsxq.com/topic/{topic_id}", "", content_text(content) or "_(no text)_", ""]
    answer = topic.get("answer")
    if isinstance(answer, dict):
        answer_owner = answer.get("owner") if isinstance(answer.get("owner"), dict) else {}
        lines.extend([f"### Answer ({answer_owner.get('name', '未知作者')})", "", content_text(answer) or "_(no text)_", ""])
    comments = topic.get("show_comments")
    if isinstance(comments, list) and comments:
        lines.extend(["### Comments", ""])
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            comment_owner = comment.get("owner") if isinstance(comment.get("owner"), dict) else {}
            lines.append(f"- **{comment_owner.get('name', '未知作者')}**: {content_text(comment)}")
        lines.append("")
    if media_paths:
        lines.extend(["### Attachments", ""])
        lines.extend(f"- [{Path(path).name}]({path})" for path in media_paths)
        lines.append("")
    return "\n".join(lines)


def backup_planet(
    *,
    group_id: str,
    output_dir: Path,
    scope: str = "all",
    count: int = DEFAULT_COUNT,
    delay: float = 1.0,
    download_media: bool = True,
    max_pages: int | None = None,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    group_id = validate_group_id(group_id)
    if not 1 <= count <= MAX_COUNT:
        raise ValueError(f"count must be between 1 and {MAX_COUNT}")
    if delay < 0:
        raise ValueError("delay must be >= 0")
    token, user_agent = require_credentials()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "topics.jsonl"
    group_path = output_dir / "group.json"
    markdown_path = output_dir / "topics.md"
    manifest_path = output_dir / "media.jsonl"

    group_payload = api_request(f"/groups/{group_id}", token=token, user_agent=user_agent, opener=opener)
    group = group_payload.get("resp_data", {}).get("group") if isinstance(group_payload.get("resp_data"), dict) else None
    if not isinstance(group, dict):
        raise RuntimeError("ZSXQ group response did not include group metadata")
    group_path.write_text(json.dumps(group, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    topics: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    end_time: str | None = None
    pages = 0
    while True:
        query: dict[str, str | int] = {"scope": scope, "count": count}
        if end_time:
            query["end_time"] = end_time
        path = f"/groups/{group_id}/topics?{urlencode(query)}"
        payload = api_request(path, token=token, user_agent=user_agent, opener=opener)
        data = payload.get("resp_data")
        page_topics = data.get("topics") if isinstance(data, dict) else None
        if not isinstance(page_topics, list) or not page_topics:
            break
        pages += 1
        added = 0
        for topic in page_topics:
            if not isinstance(topic, dict):
                continue
            topic_id = str(topic.get("topic_id", ""))
            if topic_id and topic_id not in seen_ids:
                seen_ids.add(topic_id)
                topics.append(topic)
                added += 1
        log(f"[zsxq] page {pages}: received {len(page_topics)}, new {added}, total {len(topics)}")
        if max_pages is not None and pages >= max_pages:
            break
        next_end_time = page_topics[-1].get("create_time") if isinstance(page_topics[-1], dict) else None
        if not isinstance(next_end_time, str) or not next_end_time or next_end_time == end_time or added == 0:
            break
        end_time = next_end_time
        if delay:
            time.sleep(delay)

    topics.sort(key=lambda item: str(item.get("create_time", "")), reverse=True)
    article_count = 0
    for topic in topics:
        if hydrate_article(topic, token=token, user_agent=user_agent, opener=opener):
            article_count += 1
            log(f"[zsxq] article {article_count}: topic {topic.get('topic_id')} full text saved")
            if delay:
                time.sleep(min(delay, 0.5))
    with raw_path.open("w", encoding="utf-8") as raw_file, markdown_path.open("w", encoding="utf-8") as markdown_file, manifest_path.open("w", encoding="utf-8") as manifest_file:
        markdown_file.write(f"# {group.get('name', group_id)}\n\n")
        markdown_file.write(f"- group_id: `{group_id}`\n- topic_count: `{len(topics)}`\n\n")
        media_count = 0
        media_bytes = 0
        media_errors: list[str] = []
        for index, topic in enumerate(topics, start=1):
            raw_file.write(json.dumps(topic, ensure_ascii=False, separators=(",", ":")) + "\n")
            topic_media_paths: list[str] = []
            if download_media:
                for media_type, media in iter_topic_media(topic):
                    try:
                        if media_type == "image":
                            media_url = str(media["url"])
                            suffix = Path(urlparse(media_url).path).suffix or ".jpg"
                            filename = safe_name(f"{media.get('image_id', 'image')}{suffix}")
                        else:
                            file_id = str(media["file_id"])
                            media_url = fetch_file_download_url(file_id, token=token, user_agent=user_agent, opener=opener)
                            filename = safe_name(str(media.get("name") or file_id))
                        relative = Path("media") / safe_name(str(topic.get("topic_id", index))) / filename
                        target = output_dir / relative
                        size = download_url(media_url, target, token=token, user_agent=user_agent, opener=opener)
                        media_count += 1
                        media_bytes += size
                        relative_string = relative.as_posix()
                        topic_media_paths.append(relative_string)
                        manifest_file.write(json.dumps({"topic_id": topic.get("topic_id"), "type": media_type, "path": relative_string, "bytes": size}, ensure_ascii=False) + "\n")
                    except Exception as exc:
                        detail = f"topic {topic.get('topic_id')}: {media_type} failed: {exc}"
                        media_errors.append(detail)
                        log(f"[zsxq] warning: {detail}")
            markdown_file.write(render_topic(topic, topic_media_paths))
            if index != len(topics):
                markdown_file.write("\n---\n\n")
    result = {"status": "completed", "group_id": group_id, "group_name": group.get("name"), "topic_count": len(topics), "article_count": article_count, "page_count": pages, "output_dir": str(output_dir), "group_json": str(group_path), "topics_jsonl": str(raw_path), "markdown": str(markdown_path), "media_manifest": str(manifest_path), "media_count": media_count, "media_bytes": media_bytes, "media_errors": media_errors}
    if media_errors and download_media:
        result["status"] = "completed_with_media_errors"
    (output_dir / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Back up all API-visible topics and full articles from one subscribed ZSXQ group.")
    parser.add_argument("group_id", help="Numeric group ID from wx.zsxq.com/dweb2/index/group/<id>")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=("all", "digests", "by_owner", "questions", "tasks"), default="all")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help=f"Topics per request, max {MAX_COUNT}.")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between topic pages (default: 1).")
    parser.add_argument("--no-media", action="store_true", help="Keep media metadata but do not download images/files.")
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after N pages; useful for a smoke test.")
    parser.add_argument("--json", action="store_true", help="Write one JSON result object to stdout.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_dotenv_if_present()
    try:
        if args.max_pages is not None and args.max_pages <= 0:
            raise ValueError("max-pages must be greater than 0")
        result = backup_planet(group_id=args.group_id, output_dir=args.output_dir.expanduser(), scope=args.scope, count=args.count, delay=args.delay, download_media=not args.no_media, max_pages=args.max_pages)
    except Exception as exc:
        log(f"Error: {exc}")
        return 1
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["markdown"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
