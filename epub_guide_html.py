# /// script
# dependencies = [
#   "beautifulsoup4",
#   "EbookLib",
#   "google-genai",
#   "python-dotenv",
# ]
# ///

"""
Generate a reading guide (导读手册) for ESL learners from an English EPUB.

Usage:
  uv run epub_guide_html.py --list book.epub
  uv run epub_guide_html.py book.epub --chapters 1,3-5 --out output_dir/book_guide.html
  uv run epub_guide_html.py book.epub --chapters all --out output_dir/book_guide.html
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from bs4 import BeautifulSoup
from dotenv import load_dotenv
import ebooklib
from ebooklib import epub
from google import genai
from google.genai import errors


# ============================================================================
# Constants
# ============================================================================

MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"
TARGET_LANG = "简体中文"
PROMPT_VERSION = "v1"  # Increment when prompts change significantly

DEFAULT_MAX_CHUNK_CHARS = 6000  # Characters per chunk for AI processing


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Chapter:
    index: int
    title: str
    href: str
    text: str = ""
    char_count: int = 0


@dataclass
class ChunkNotes:
    """Notes extracted from a single chunk of text."""
    events: List[str] = field(default_factory=list)
    characters: List[Dict[str, str]] = field(default_factory=list)
    vocabulary: List[Dict[str, str]] = field(default_factory=list)
    key_sentences: List[Dict[str, str]] = field(default_factory=list)
    cultural_notes: List[str] = field(default_factory=list)


@dataclass
class ChapterGuide:
    """Complete guide for a single chapter."""
    title: str
    pre_reading: Dict[str, Any] = field(default_factory=dict)
    characters: List[Dict[str, Any]] = field(default_factory=list)
    plot_structure: List[Dict[str, Any]] = field(default_factory=list)
    vocabulary: List[Dict[str, Any]] = field(default_factory=list)
    key_sentences: List[Dict[str, Any]] = field(default_factory=list)
    grammar_style: List[Dict[str, Any]] = field(default_factory=list)
    cultural_background: List[Dict[str, Any]] = field(default_factory=list)
    quiz: List[Dict[str, Any]] = field(default_factory=list)
    post_reading: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BookGuide:
    """Book-level guide content."""
    global_characters: List[Dict[str, Any]] = field(default_factory=list)
    themes: List[str] = field(default_factory=list)
    vocabulary_summary: List[Dict[str, Any]] = field(default_factory=list)
    reading_plan: str = ""


# ============================================================================
# Utility Functions
# ============================================================================

def eprint(*args):
    print(*args, file=sys.stderr)


def find_latest_epub(directory: Path) -> Path:
    files = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() == ".epub"]
    if not files:
        raise FileNotFoundError(f"No .epub found in {directory}")
    return max(files, key=lambda p: p.stat().st_mtime)


def parse_chapter_spec(spec: str, max_chapters: int) -> List[int]:
    """Parse chapter spec like '1,3-5' or 'all' into list of 1-based indices."""
    if spec.strip().lower() == "all":
        return list(range(1, max_chapters + 1))
    
    result: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a.strip()), int(b.strip())
            if a > b:
                a, b = b, a
            for x in range(a, b + 1):
                if 1 <= x <= max_chapters:
                    result.add(x)
        else:
            x = int(part)
            if 1 <= x <= max_chapters:
                result.add(x)
    return sorted(result)


def clean_text(text: str) -> str:
    """Clean extracted text: fix hyphenation, normalize whitespace."""
    text = text.replace("\r", "\n")
    # Join hyphenated words across line breaks
    text = re.sub(r"([A-Za-z])-\n\s*([A-Za-z])", r"\1\2", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ============================================================================
# EPUB Chapter Extraction (TOC priority, spine fallback)
# ============================================================================

def flatten_toc(toc_items, depth: int = 1) -> List[Tuple[str, str, int]]:
    """Recursively flatten TOC into (title, href, depth) tuples."""
    result = []
    for item in toc_items:
        if isinstance(item, tuple):
            section, children = item
            if hasattr(section, 'title') and hasattr(section, 'href'):
                result.append((section.title, section.href, depth))
            result.extend(flatten_toc(children, depth + 1))
        elif hasattr(item, 'title') and hasattr(item, 'href'):
            result.append((item.title, item.href, depth))
    return result


def extract_text_from_html(html_content: bytes, fragment: Optional[str] = None) -> str:
    """Extract text from HTML content, optionally starting from a fragment anchor."""
    try:
        html_str = html_content.decode("utf-8")
    except UnicodeDecodeError:
        html_str = html_content.decode("utf-8", errors="ignore")
    
    soup = BeautifulSoup(html_str, "html.parser")
    
    for tag in soup(["script", "style"]):
        tag.decompose()
    
    if fragment:
        anchor = soup.find(id=fragment) or soup.find(attrs={"name": fragment})
        if anchor:
            text_parts = []
            for sibling in anchor.find_all_next(string=True):
                text_parts.append(sibling.strip())
            return clean_text(" ".join(text_parts))
    
    return clean_text(soup.get_text("\n", strip=True))


def infer_title_from_html(html_content: bytes, fallback: str) -> str:
    """Infer a title from HTML content."""
    try:
        html_str = html_content.decode("utf-8")
    except UnicodeDecodeError:
        html_str = html_content.decode("utf-8", errors="ignore")
    
    soup = BeautifulSoup(html_str, "html.parser")
    
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    
    for tag in ["h1", "h2", "h3"]:
        h = soup.find(tag)
        if h:
            return h.get_text(" ", strip=True)[:100]
    
    return fallback


def get_chapters_from_epub(epub_path: Path, max_toc_depth: int = 10) -> List[Chapter]:
    """Extract chapters from EPUB. Priority: TOC (depth <= max_toc_depth) -> spine fallback."""
    book = epub.read_epub(str(epub_path))
    
    href_to_item: Dict[str, epub.EpubItem] = {}
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            href = item.get_name()
            href_to_item[href] = item
    
    spine_order: List[str] = []
    for idref, _linear in book.spine:
        if idref:
            spine_item = book.get_item_with_id(idref)
            if spine_item and spine_item.get_type() == ebooklib.ITEM_DOCUMENT:
                spine_order.append(spine_item.get_name())
    
    chapters: List[Chapter] = []
    
    if book.toc:
        toc_entries = flatten_toc(book.toc)
        toc_entries = [(t, h, d) for t, h, d in toc_entries if d <= max_toc_depth]
        
        if toc_entries:
            seen_hrefs: Set[str] = set()
            for idx, (title, href, _depth) in enumerate(toc_entries, start=1):
                if "#" in href:
                    file_href, fragment = href.split("#", 1)
                else:
                    file_href, fragment = href, None
                
                item = href_to_item.get(file_href)
                if item is None:
                    for key in href_to_item:
                        if key.endswith(file_href) or file_href.endswith(key):
                            item = href_to_item[key]
                            file_href = key
                            break
                
                if item is None:
                    continue
                
                full_href = f"{file_href}#{fragment}" if fragment else file_href
                if full_href in seen_hrefs:
                    continue
                seen_hrefs.add(full_href)
                
                text = extract_text_from_html(item.get_content(), fragment)
                
                if not text and file_href in spine_order:
                    spine_idx = spine_order.index(file_href)
                    if spine_idx + 1 < len(spine_order):
                        next_href = spine_order[spine_idx + 1]
                        next_item = href_to_item.get(next_href)
                        if next_item:
                            text = extract_text_from_html(next_item.get_content())
                            if text:
                                full_href = next_href
                
                if text:
                    chapters.append(Chapter(
                        index=len(chapters) + 1,
                        title=title.strip() or f"Chapter {len(chapters) + 1}",
                        href=full_href,
                        text=text,
                        char_count=len(text),
                    ))
            
            if chapters:
                return chapters
    
    # Fallback to spine
    spine_ids = [idref for (idref, _linear) in book.spine if idref]
    
    for idref in spine_ids:
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        
        content = item.get_content()
        text = extract_text_from_html(content)
        
        if not text or len(text) < 100:
            continue
        
        title = infer_title_from_html(content, f"Section {len(chapters) + 1}")
        
        chapters.append(Chapter(
            index=len(chapters) + 1,
            title=title,
            href=item.get_name(),
            text=text,
            char_count=len(text),
        ))
    
    return chapters


# ============================================================================
# Text Chunking
# ============================================================================

def chunk_text(text: str, max_chars: int = DEFAULT_MAX_CHUNK_CHARS) -> List[str]:
    """Split text into chunks at paragraph boundaries."""
    paragraphs = re.split(r'\n\s*\n', text)
    
    chunks: List[str] = []
    current_chunk: List[str] = []
    current_size = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        para_len = len(para)
        
        # If single paragraph exceeds max, split it by sentences
        if para_len > max_chars:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split by sentence
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sent_chunk: List[str] = []
            sent_size = 0
            for sent in sentences:
                if sent_size + len(sent) > max_chars and sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                    sent_chunk = []
                    sent_size = 0
                sent_chunk.append(sent)
                sent_size += len(sent) + 1
            if sent_chunk:
                chunks.append(" ".join(sent_chunk))
        elif current_size + para_len > max_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [para]
            current_size = para_len
        else:
            current_chunk.append(para)
            current_size += para_len + 2
    
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    
    return chunks if chunks else [text]


# ============================================================================
# SQLite Cache
# ============================================================================

def open_cache(db_path: Path) -> sqlite3.Connection:
    """Open or create SQLite cache database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS guide_cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def make_cache_key(prefix: str, content: str) -> str:
    """Generate cache key from prefix and content."""
    h = hashlib.sha256()
    h.update(f"{MODEL_NAME}\n{TARGET_LANG}\n{PROMPT_VERSION}\n{prefix}\n{content}".encode("utf-8"))
    return h.hexdigest()


def cache_get(conn: sqlite3.Connection, key: str) -> Optional[dict]:
    """Get value from cache."""
    cur = conn.execute("SELECT value FROM guide_cache WHERE key = ?", (key,))
    row = cur.fetchone()
    if row:
        try:
            return json.loads(row[0])
        except Exception:
            pass
    return None


def cache_set(conn: sqlite3.Connection, key: str, value: dict) -> None:
    """Set value in cache."""
    conn.execute(
        "INSERT OR REPLACE INTO guide_cache (key, value) VALUES (?, ?)",
        (key, json.dumps(value, ensure_ascii=False))
    )
    conn.commit()


# ============================================================================
# Gemini API Helpers
# ============================================================================

def is_retryable_error(e: Exception) -> bool:
    """Check if error is retryable (transient HTTP errors)."""
    if isinstance(e, errors.ServerError):
        if hasattr(e, 'status_code') and e.status_code in {429, 500, 502, 503, 504}:
            return True
        error_str = str(e).upper()
        if any(kw in error_str for kw in ['503', 'UNAVAILABLE', 'OVERLOADED', '429', 'RATE']):
            return True
    
    error_str = str(e).upper()
    if any(kw in error_str for kw in ['503', 'UNAVAILABLE', 'OVERLOADED', 'RATE LIMIT', 'TIMEOUT', '429']):
        return True
    
    return False


def extract_json_object(text: str) -> dict:
    """Extract JSON object from model response."""
    t = (text or "").strip()
    # Strip code fences
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    
    start = t.find("{")
    end = t.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON object")
    
    payload = t[start:end + 1]
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data


def call_gemini(
    client: genai.Client,
    prompt: str,
    max_retries: int = 3,
) -> str:
    """Call Gemini API with retry logic."""
    last_error = None
    retry_delays = [5, 15, 30]
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            return response.text or ""
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1 and is_retryable_error(e):
                delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                eprint(f"  Attempt {attempt + 1} failed: {e}")
                eprint(f"  Retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise
    
    if last_error:
        raise last_error
    return ""


# ============================================================================
# AI Generation Pipeline
# ============================================================================

CHUNK_ANALYSIS_PROMPT = """你是一位专业的英语阅读教学专家，正在为英语作为第二语言的学习者准备阅读材料分析。

请分析以下英文文本片段，提取对 ESL 学习者有用的信息。

文本片段：
---
{chunk_text}
---

请用 JSON 格式返回分析结果，包含以下字段：
{{
  "events": ["事件1简述", "事件2简述", ...],  // 这段文本中发生的主要事件/信息点（简体中文）
  "characters": [
    {{"name": "人物英文名", "description": "简短描述（身份、关系）", "aliases": ["别称1", "别称2"]}}
  ],  // 出现的人物/专有名词
  "vocabulary": [
    {{"word": "英文词/短语", "context": "原文中出现的句子", "meaning": "语境义（简体中文）", "category": "核心词汇/可能阻碍理解/习语短语动词"}}
  ],  // 对 ESL 学习者可能有难度的词汇/表达（选最重要的 5-10 个）
  "key_sentences": [
    {{"sentence": "原文句子", "difficulty": "语法难点/修辞特点简述"}}
  ],  // 值得精讲的复杂句子（选 2-5 个最有代表性的）
  "cultural_notes": ["需要了解的文化背景1", ...]  // 文化背景/典故/隐喻提示（如有）
}}

只返回 JSON，不要其他解释。"""

CHAPTER_SYNTHESIS_PROMPT = """你是一位专业的英语阅读教学专家，正在为英语作为第二语言的学习者编写深度阅读导读手册。

章节标题：{chapter_title}

以下是对本章各部分的分析笔记：
---
{chunk_notes_json}
---

请基于以上笔记，为这一章生成完整的导读内容。全部使用简体中文，但保留必要的英文原词/原句以便对照。

返回 JSON 格式：
{{
  "pre_reading": {{
    "positioning": "章节定位与情绪基调",
    "focus_points": ["需要关注的线索1", "线索2", ...],
    "difficulty_hints": ["阅读难点提示1", ...],
    "reading_strategy": "建议阅读策略"
  }},
  "characters": [
    {{
      "name": "英文名",
      "chinese_name": "中文译名（如适用）",
      "first_appearance": "首次出现位置/情境",
      "description": "身份与关系描述",
      "aliases": ["别称/指代"]
    }}
  ],
  "plot_structure": [
    {{
      "segment": "场景/段落标识",
      "summary": "情节要点",
      "significance": "转折点/冲突点/伏笔（如适用）"
    }}
  ],
  "vocabulary": [
    {{
      "term": "英文词/短语",
      "meaning": "语境义",
      "example": "原文例句",
      "usage_note": "使用提醒/搭配说明",
      "category": "核心词汇/可能阻碍理解/习语短语动词"
    }}
  ],
  "key_sentences": [
    {{
      "english": "原文句子",
      "chinese": "中文翻译",
      "analysis": "句法拆解/难点解释"
    }}
  ],
  "grammar_style": [
    {{
      "feature": "语法/风格特征名称",
      "explanation": "解释说明",
      "examples": ["原文例句1", "例句2"]
    }}
  ],
  "cultural_background": [
    {{
      "topic": "文化背景/典故/隐喻主题",
      "explanation": "必要背景解释"
    }}
  ],
  "quiz": [
    {{
      "type": "理解题/细节题/推断题",
      "question": "题目",
      "options": ["A. 选项1", "B. 选项2", "C. 选项3", "D. 选项4"],
      "answer": "正确选项字母",
      "explanation": "答案解析"
    }}
  ],
  "post_reading": {{
    "plot_summary": "更完整的剧情梳理（可剧透）",
    "character_motivations": "人物动机/心理分析",
    "themes": ["主题解读1", "主题2"],
    "connections": "与后续章节的可能关联/伏笔"
  }}
}}

要求：
1. vocabulary 选取最重要的 15-25 个词汇/表达
2. key_sentences 选取 5-10 个最值得精讲的句子
3. quiz 包含 3-5 道测试题
4. 所有解释使用简体中文，但原文引用保持英文

只返回 JSON，不要其他解释。"""

BOOK_SYNTHESIS_PROMPT = """你是一位专业的英语阅读教学专家，正在为英语作为第二语言的学习者编写全书导读概览。

书名：{book_name}
已分析章节：{chapter_list}

以下是各章节的导读摘要：
---
{chapters_summary_json}
---

请生成全书级别的导读内容，帮助读者纵览全书。使用简体中文。

返回 JSON 格式：
{{
  "global_characters": [
    {{
      "name": "英文名",
      "chinese_name": "中文译名",
      "role": "角色定位（主角/配角/反派等）",
      "description": "人物简介",
      "relationships": ["与XX的关系", ...]
    }}
  ],
  "themes": ["核心主题1", "主题2", ...],
  "vocabulary_summary": [
    {{
      "term": "跨章节高频/核心词汇",
      "meaning": "含义",
      "chapters": ["出现的章节"]
    }}
  ],
  "reading_plan": "阅读计划建议（分段建议、预计时长、难度提示等）"
}}

要求：
1. global_characters 列出所有重要人物，去重合并
2. vocabulary_summary 从各章词汇中选取最核心的 20-30 个，去重
3. reading_plan 给出切实可行的阅读建议

只返回 JSON，不要其他解释。"""


def analyze_chunk(
    client: genai.Client,
    chunk_text: str,
    cache_conn: sqlite3.Connection,
) -> dict:
    """Analyze a single chunk of text."""
    cache_key = make_cache_key("chunk", chunk_text)
    cached = cache_get(cache_conn, cache_key)
    if cached:
        return cached
    
    prompt = CHUNK_ANALYSIS_PROMPT.format(chunk_text=chunk_text)
    response = call_gemini(client, prompt)
    
    try:
        result = extract_json_object(response)
    except Exception as e:
        eprint(f"  Warning: Failed to parse chunk analysis: {e}")
        result = {
            "events": [],
            "characters": [],
            "vocabulary": [],
            "key_sentences": [],
            "cultural_notes": []
        }
    
    cache_set(cache_conn, cache_key, result)
    return result


def synthesize_chapter(
    client: genai.Client,
    chapter_title: str,
    chunk_notes: List[dict],
    cache_conn: sqlite3.Connection,
) -> dict:
    """Synthesize chapter guide from chunk notes."""
    chunk_notes_json = json.dumps(chunk_notes, ensure_ascii=False, indent=2)
    cache_key = make_cache_key("chapter", f"{chapter_title}\n{chunk_notes_json}")
    
    cached = cache_get(cache_conn, cache_key)
    if cached:
        return cached
    
    prompt = CHAPTER_SYNTHESIS_PROMPT.format(
        chapter_title=chapter_title,
        chunk_notes_json=chunk_notes_json
    )
    response = call_gemini(client, prompt)
    
    try:
        result = extract_json_object(response)
    except Exception as e:
        eprint(f"  Warning: Failed to parse chapter synthesis: {e}")
        result = {
            "pre_reading": {},
            "characters": [],
            "plot_structure": [],
            "vocabulary": [],
            "key_sentences": [],
            "grammar_style": [],
            "cultural_background": [],
            "quiz": [],
            "post_reading": {}
        }
    
    cache_set(cache_conn, cache_key, result)
    return result


def synthesize_book(
    client: genai.Client,
    book_name: str,
    chapter_guides: List[Tuple[str, dict]],
    cache_conn: sqlite3.Connection,
) -> dict:
    """Synthesize book-level guide from chapter guides."""
    chapter_list = ", ".join([title for title, _ in chapter_guides])
    
    # Create summary for each chapter
    summaries = []
    for title, guide in chapter_guides:
        summary = {
            "title": title,
            "characters": guide.get("characters", [])[:5],
            "vocabulary_sample": [v.get("term", "") for v in guide.get("vocabulary", [])[:10]],
            "themes": guide.get("post_reading", {}).get("themes", [])
        }
        summaries.append(summary)
    
    chapters_summary_json = json.dumps(summaries, ensure_ascii=False, indent=2)
    cache_key = make_cache_key("book", f"{book_name}\n{chapters_summary_json}")
    
    cached = cache_get(cache_conn, cache_key)
    if cached:
        return cached
    
    prompt = BOOK_SYNTHESIS_PROMPT.format(
        book_name=book_name,
        chapter_list=chapter_list,
        chapters_summary_json=chapters_summary_json
    )
    response = call_gemini(client, prompt)
    
    try:
        result = extract_json_object(response)
    except Exception as e:
        eprint(f"  Warning: Failed to parse book synthesis: {e}")
        result = {
            "global_characters": [],
            "themes": [],
            "vocabulary_summary": [],
            "reading_plan": ""
        }
    
    cache_set(cache_conn, cache_key, result)
    return result


def generate_chapter_guide(
    client: genai.Client,
    chapter: Chapter,
    cache_conn: sqlite3.Connection,
    max_chunk_chars: int,
) -> dict:
    """Generate complete guide for a single chapter."""
    eprint(f"  Chunking chapter text ({chapter.char_count} chars)...")
    chunks = chunk_text(chapter.text, max_chunk_chars)
    eprint(f"  Split into {len(chunks)} chunks")
    
    # Analyze each chunk
    chunk_notes: List[dict] = []
    for i, chunk in enumerate(chunks):
        eprint(f"  Analyzing chunk {i + 1}/{len(chunks)}...")
        notes = analyze_chunk(client, chunk, cache_conn)
        chunk_notes.append(notes)
        if i < len(chunks) - 1:
            time.sleep(0.5)  # Small delay between API calls
    
    # Synthesize chapter guide
    eprint(f"  Synthesizing chapter guide...")
    chapter_guide = synthesize_chapter(client, chapter.title, chunk_notes, cache_conn)
    
    return chapter_guide


# ============================================================================
# HTML Rendering
# ============================================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>导读手册 - {book_name}</title>
  <style>
    :root {{
      --primary-color: #2563eb;
      --secondary-color: #64748b;
      --accent-color: #f59e0b;
      --bg-color: #f8fafc;
      --card-bg: #ffffff;
      --text-color: #1e293b;
      --border-color: #e2e8f0;
      --code-bg: #f1f5f9;
    }}
    
    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}
    
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Microsoft YaHei", sans-serif;
      line-height: 1.7;
      background: var(--bg-color);
      color: var(--text-color);
      padding: 20px;
    }}
    
    .container {{
      max-width: 900px;
      margin: 0 auto;
    }}
    
    header {{
      background: linear-gradient(135deg, var(--primary-color), #1d4ed8);
      color: white;
      padding: 40px 30px;
      border-radius: 12px;
      margin-bottom: 30px;
      box-shadow: 0 4px 20px rgba(37, 99, 235, 0.3);
    }}
    
    header h1 {{
      font-size: 2em;
      margin-bottom: 10px;
    }}
    
    header .meta {{
      opacity: 0.9;
      font-size: 0.95em;
    }}
    
    nav.toc {{
      background: var(--card-bg);
      padding: 25px;
      border-radius: 10px;
      margin-bottom: 30px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }}
    
    nav.toc h2 {{
      color: var(--primary-color);
      margin-bottom: 15px;
      font-size: 1.3em;
    }}
    
    nav.toc ul {{
      list-style: none;
    }}
    
    nav.toc li {{
      margin: 8px 0;
    }}
    
    nav.toc a {{
      color: var(--text-color);
      text-decoration: none;
      padding: 5px 0;
      display: inline-block;
      transition: color 0.2s;
    }}
    
    nav.toc a:hover {{
      color: var(--primary-color);
    }}
    
    .chapter {{
      background: var(--card-bg);
      padding: 30px;
      border-radius: 10px;
      margin-bottom: 30px;
      box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }}
    
    .chapter h2 {{
      color: var(--primary-color);
      border-bottom: 3px solid var(--primary-color);
      padding-bottom: 10px;
      margin-bottom: 25px;
    }}
    
    .section {{
      margin-bottom: 30px;
    }}
    
    .section h3 {{
      color: var(--secondary-color);
      font-size: 1.15em;
      margin-bottom: 15px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    
    .section h3::before {{
      content: "";
      display: inline-block;
      width: 4px;
      height: 1.2em;
      background: var(--accent-color);
      border-radius: 2px;
    }}
    
    .card {{
      background: var(--code-bg);
      padding: 15px 20px;
      border-radius: 8px;
      margin-bottom: 12px;
      border-left: 4px solid var(--primary-color);
    }}
    
    .card-title {{
      font-weight: 600;
      color: var(--primary-color);
      margin-bottom: 5px;
    }}
    
    .english {{
      font-family: Georgia, "Times New Roman", serif;
      font-style: italic;
      color: #475569;
      background: #fef3c7;
      padding: 2px 6px;
      border-radius: 3px;
    }}
    
    .vocab-item {{
      display: grid;
      grid-template-columns: auto 1fr;
      gap: 10px 20px;
      padding: 12px 15px;
      background: var(--code-bg);
      border-radius: 8px;
      margin-bottom: 10px;
    }}
    
    .vocab-term {{
      font-weight: 600;
      color: var(--primary-color);
    }}
    
    .vocab-category {{
      font-size: 0.8em;
      background: var(--accent-color);
      color: white;
      padding: 2px 8px;
      border-radius: 10px;
      margin-left: 10px;
    }}
    
    .sentence-item {{
      background: var(--code-bg);
      padding: 15px 20px;
      border-radius: 8px;
      margin-bottom: 15px;
    }}
    
    .sentence-en {{
      font-family: Georgia, "Times New Roman", serif;
      font-size: 1.05em;
      color: #334155;
      margin-bottom: 8px;
      line-height: 1.8;
    }}
    
    .sentence-zh {{
      color: var(--text-color);
      margin-bottom: 10px;
    }}
    
    .sentence-analysis {{
      font-size: 0.9em;
      color: var(--secondary-color);
      padding-top: 10px;
      border-top: 1px dashed var(--border-color);
    }}
    
    details {{
      background: #fef2f2;
      border-radius: 8px;
      margin-bottom: 15px;
    }}
    
    details summary {{
      padding: 12px 15px;
      cursor: pointer;
      font-weight: 600;
      color: #dc2626;
    }}
    
    details summary:hover {{
      background: #fee2e2;
      border-radius: 8px;
    }}
    
    details .content {{
      padding: 15px 20px;
      border-top: 1px solid #fecaca;
    }}
    
    .quiz-item {{
      background: #eff6ff;
      padding: 15px 20px;
      border-radius: 8px;
      margin-bottom: 15px;
    }}
    
    .quiz-question {{
      font-weight: 600;
      margin-bottom: 10px;
    }}
    
    .quiz-options {{
      list-style: none;
      margin-bottom: 10px;
    }}
    
    .quiz-options li {{
      padding: 5px 0;
    }}
    
    .quiz-answer {{
      background: #dcfce7;
      padding: 10px 15px;
      border-radius: 6px;
      margin-top: 10px;
    }}
    
    .book-section {{
      background: linear-gradient(135deg, #faf5ff, #f3e8ff);
      padding: 30px;
      border-radius: 10px;
      margin-bottom: 30px;
      border: 2px solid #e9d5ff;
    }}
    
    .book-section h2 {{
      color: #7c3aed;
      border-bottom: 3px solid #7c3aed;
      padding-bottom: 10px;
      margin-bottom: 25px;
    }}
    
    .character-card {{
      background: white;
      padding: 15px 20px;
      border-radius: 8px;
      margin-bottom: 12px;
      box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }}
    
    .character-name {{
      font-size: 1.1em;
      font-weight: 600;
      color: var(--primary-color);
    }}
    
    .character-role {{
      font-size: 0.85em;
      background: var(--secondary-color);
      color: white;
      padding: 2px 10px;
      border-radius: 10px;
      margin-left: 10px;
    }}
    
    ul, ol {{
      padding-left: 25px;
      margin: 10px 0;
    }}
    
    li {{
      margin: 6px 0;
    }}
    
    @media print {{
      body {{
        background: white;
        padding: 0;
      }}
      
      .container {{
        max-width: 100%;
      }}
      
      header {{
        background: var(--primary-color) !important;
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
      }}
      
      .chapter, .book-section, nav.toc {{
        break-inside: avoid;
        box-shadow: none;
        border: 1px solid var(--border-color);
      }}
      
      details {{
        display: block;
      }}
      
      details[open] summary {{
        border-bottom: 1px solid #fecaca;
      }}
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>📖 {book_name}</h1>
      <p class="meta">ESL 阅读导读手册 | 章节：{chapter_info}</p>
    </header>
    
    <nav class="toc">
      <h2>目录</h2>
      <ul>
        <li><a href="#book-overview">📚 全书概览</a></li>
{toc_items}
      </ul>
    </nav>
    
    <section id="book-overview" class="book-section">
      <h2>📚 全书概览</h2>
{book_overview_content}
    </section>
    
{chapters_content}
  </div>
</body>
</html>"""


def render_list(items: List[str]) -> str:
    """Render a list of strings as HTML ul."""
    if not items:
        return "<p>（无）</p>"
    html = "<ul>"
    for item in items:
        html += f"<li>{escape(str(item))}</li>"
    html += "</ul>"
    return html


def render_pre_reading(pre: dict) -> str:
    """Render pre-reading section."""
    html = '<div class="section"><h3>📌 读前导读</h3>'
    
    if pre.get("positioning"):
        html += f'<div class="card"><div class="card-title">章节定位</div>{escape(pre["positioning"])}</div>'
    
    if pre.get("focus_points"):
        html += '<div class="card"><div class="card-title">关注要点</div>'
        html += render_list(pre["focus_points"])
        html += '</div>'
    
    if pre.get("difficulty_hints"):
        html += '<div class="card"><div class="card-title">难点提示</div>'
        html += render_list(pre["difficulty_hints"])
        html += '</div>'
    
    if pre.get("reading_strategy"):
        html += f'<div class="card"><div class="card-title">阅读策略</div>{escape(pre["reading_strategy"])}</div>'
    
    html += '</div>'
    return html


def render_characters(chars: List[dict]) -> str:
    """Render characters section."""
    if not chars:
        return ""
    
    html = '<div class="section"><h3>👤 人物/专有名词</h3>'
    for char in chars:
        html += '<div class="card">'
        html += f'<div class="card-title">{escape(char.get("name", ""))}'
        if char.get("chinese_name"):
            html += f' ({escape(char["chinese_name"])})'
        html += '</div>'
        if char.get("first_appearance"):
            html += f'<p><strong>首次出现：</strong>{escape(char["first_appearance"])}</p>'
        if char.get("description"):
            html += f'<p>{escape(char["description"])}</p>'
        if char.get("aliases"):
            html += f'<p><strong>别称/指代：</strong>{", ".join(escape(str(a)) for a in char["aliases"])}</p>'
        html += '</div>'
    html += '</div>'
    return html


def render_plot_structure(plots: List[dict]) -> str:
    """Render plot structure section."""
    if not plots:
        return ""
    
    html = '<div class="section"><h3>📖 情节结构</h3>'
    for i, plot in enumerate(plots, 1):
        html += '<div class="card">'
        segment = plot.get("segment", f"段落 {i}")
        html += f'<div class="card-title">{escape(segment)}</div>'
        if plot.get("summary"):
            html += f'<p>{escape(plot["summary"])}</p>'
        if plot.get("significance"):
            html += f'<p><strong>📍 {escape(plot["significance"])}</strong></p>'
        html += '</div>'
    html += '</div>'
    return html


def render_vocabulary(vocab: List[dict]) -> str:
    """Render vocabulary section."""
    if not vocab:
        return ""
    
    html = '<div class="section"><h3>📝 重点词汇与表达</h3>'
    for v in vocab:
        html += '<div class="vocab-item">'
        html += f'<div><span class="vocab-term">{escape(v.get("term", ""))}</span>'
        if v.get("category"):
            html += f'<span class="vocab-category">{escape(v["category"])}</span>'
        html += '</div>'
        html += f'<div>{escape(v.get("meaning", ""))}</div>'
        if v.get("example"):
            html += f'<div style="grid-column: 1 / -1;"><span class="english">{escape(v["example"])}</span></div>'
        if v.get("usage_note"):
            html += f'<div style="grid-column: 1 / -1; font-size: 0.9em; color: var(--secondary-color);">💡 {escape(v["usage_note"])}</div>'
        html += '</div>'
    html += '</div>'
    return html


def render_key_sentences(sentences: List[dict]) -> str:
    """Render key sentences section."""
    if not sentences:
        return ""
    
    html = '<div class="section"><h3>💬 重点句精讲</h3>'
    for s in sentences:
        html += '<div class="sentence-item">'
        if s.get("english"):
            html += f'<div class="sentence-en">{escape(s["english"])}</div>'
        if s.get("chinese"):
            html += f'<div class="sentence-zh">→ {escape(s["chinese"])}</div>'
        if s.get("analysis"):
            html += f'<div class="sentence-analysis">📖 {escape(s["analysis"])}</div>'
        html += '</div>'
    html += '</div>'
    return html


def render_grammar_style(items: List[dict]) -> str:
    """Render grammar and style section."""
    if not items:
        return ""
    
    html = '<div class="section"><h3>📐 语法与写作风格</h3>'
    for item in items:
        html += '<div class="card">'
        html += f'<div class="card-title">{escape(item.get("feature", ""))}</div>'
        if item.get("explanation"):
            html += f'<p>{escape(item["explanation"])}</p>'
        if item.get("examples"):
            html += '<p><strong>例句：</strong></p><ul>'
            for ex in item["examples"]:
                html += f'<li><span class="english">{escape(ex)}</span></li>'
            html += '</ul>'
        html += '</div>'
    html += '</div>'
    return html


def render_cultural_background(items: List[dict]) -> str:
    """Render cultural background section."""
    if not items:
        return ""
    
    html = '<div class="section"><h3>🌍 文化背景/典故</h3>'
    for item in items:
        html += '<div class="card">'
        html += f'<div class="card-title">{escape(item.get("topic", ""))}</div>'
        if item.get("explanation"):
            html += f'<p>{escape(item["explanation"])}</p>'
        html += '</div>'
    html += '</div>'
    return html


def render_quiz(quizzes: List[dict]) -> str:
    """Render quiz section with collapsible answers."""
    if not quizzes:
        return ""
    
    html = '<div class="section"><h3>✏️ 章节自测</h3>'
    for i, q in enumerate(quizzes, 1):
        html += '<div class="quiz-item">'
        qtype = q.get("type", "理解题")
        html += f'<div class="quiz-question">{i}. [{qtype}] {escape(q.get("question", ""))}</div>'
        if q.get("options"):
            html += '<ul class="quiz-options">'
            for opt in q["options"]:
                html += f'<li>{escape(opt)}</li>'
            html += '</ul>'
        html += '<details><summary>查看答案</summary><div class="content">'
        html += f'<div class="quiz-answer"><strong>答案：</strong>{escape(q.get("answer", ""))}</div>'
        if q.get("explanation"):
            html += f'<p><strong>解析：</strong>{escape(q["explanation"])}</p>'
        html += '</div></details>'
        html += '</div>'
    html += '</div>'
    return html


def render_post_reading(post: dict) -> str:
    """Render post-reading section (spoiler)."""
    if not post:
        return ""
    
    html = '<details><summary>⚠️ 读后复盘（含剧透，点击展开）</summary><div class="content">'
    
    if post.get("plot_summary"):
        html += f'<div class="card"><div class="card-title">完整剧情</div><p>{escape(post["plot_summary"])}</p></div>'
    
    if post.get("character_motivations"):
        html += f'<div class="card"><div class="card-title">人物动机分析</div><p>{escape(post["character_motivations"])}</p></div>'
    
    if post.get("themes"):
        html += '<div class="card"><div class="card-title">主题解读</div>'
        html += render_list(post["themes"])
        html += '</div>'
    
    if post.get("connections"):
        html += f'<div class="card"><div class="card-title">后续关联</div><p>{escape(post["connections"])}</p></div>'
    
    html += '</div></details>'
    return html


def render_chapter(chapter_title: str, chapter_id: str, guide: dict) -> str:
    """Render a complete chapter section."""
    html = f'<section id="{chapter_id}" class="chapter">'
    html += f'<h2>{escape(chapter_title)}</h2>'
    
    html += render_pre_reading(guide.get("pre_reading", {}))
    html += render_characters(guide.get("characters", []))
    html += render_plot_structure(guide.get("plot_structure", []))
    html += render_vocabulary(guide.get("vocabulary", []))
    html += render_key_sentences(guide.get("key_sentences", []))
    html += render_grammar_style(guide.get("grammar_style", []))
    html += render_cultural_background(guide.get("cultural_background", []))
    html += render_quiz(guide.get("quiz", []))
    html += render_post_reading(guide.get("post_reading", {}))
    
    html += '</section>'
    return html


def render_book_overview(book_guide: dict) -> str:
    """Render book-level overview section."""
    html = ""
    
    # Global characters
    chars = book_guide.get("global_characters", [])
    if chars:
        html += '<div class="section"><h3>👥 主要人物</h3>'
        for char in chars:
            html += '<div class="character-card">'
            html += f'<span class="character-name">{escape(char.get("name", ""))}'
            if char.get("chinese_name"):
                html += f' ({escape(char["chinese_name"])})'
            html += '</span>'
            if char.get("role"):
                html += f'<span class="character-role">{escape(char["role"])}</span>'
            if char.get("description"):
                html += f'<p>{escape(char["description"])}</p>'
            if char.get("relationships"):
                html += f'<p><small>关系：{", ".join(escape(str(r)) for r in char["relationships"])}</small></p>'
            html += '</div>'
        html += '</div>'
    
    # Themes
    themes = book_guide.get("themes", [])
    if themes:
        html += '<div class="section"><h3>🎯 核心主题</h3>'
        html += render_list(themes)
        html += '</div>'
    
    # Vocabulary summary
    vocab = book_guide.get("vocabulary_summary", [])
    if vocab:
        html += '<div class="section"><h3>📚 核心词汇表</h3>'
        for v in vocab[:30]:
            html += '<div class="vocab-item">'
            html += f'<div class="vocab-term">{escape(v.get("term", ""))}</div>'
            html += f'<div>{escape(v.get("meaning", ""))}</div>'
            if v.get("chapters"):
                html += f'<div style="grid-column: 1 / -1; font-size: 0.85em; color: var(--secondary-color);">出现章节：{", ".join(escape(str(c)) for c in v["chapters"])}</div>'
            html += '</div>'
        html += '</div>'
    
    # Reading plan
    plan = book_guide.get("reading_plan", "")
    if plan:
        html += '<div class="section"><h3>📅 阅读计划建议</h3>'
        html += f'<div class="card"><p>{escape(plan)}</p></div>'
        html += '</div>'
    
    return html


def render_html(
    book_name: str,
    chapter_info: str,
    chapter_guides: List[Tuple[str, dict]],
    book_guide: dict,
) -> str:
    """Render the complete HTML document."""
    # Generate TOC items
    toc_items = ""
    for i, (title, _) in enumerate(chapter_guides):
        chapter_id = f"chapter-{i + 1}"
        toc_items += f'        <li><a href="#{chapter_id}">{escape(title)}</a></li>\n'
    
    # Generate book overview
    book_overview_content = render_book_overview(book_guide)
    
    # Generate chapter content
    chapters_content = ""
    for i, (title, guide) in enumerate(chapter_guides):
        chapter_id = f"chapter-{i + 1}"
        chapters_content += render_chapter(title, chapter_id, guide)
        chapters_content += "\n"
    
    return HTML_TEMPLATE.format(
        book_name=escape(book_name),
        chapter_info=escape(chapter_info),
        toc_items=toc_items,
        book_overview_content=book_overview_content,
        chapters_content=chapters_content,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Generate an ESL reading guide (导读手册) from an English EPUB."
    )
    parser.add_argument("epub", nargs="?", help="Path to .epub file (default: latest in input_dir/)")
    parser.add_argument("--list", action="store_true", help="List chapters and exit")
    parser.add_argument("--chapters", default="1", help="Chapter selection: '1', '1,3-5', or 'all'")
    parser.add_argument("--out", help="Output HTML path (default: output_dir/<book>_guide.html)")
    parser.add_argument("--cache", default="output_dir/epub_guide_cache.sqlite", help="Cache database path")
    parser.add_argument("--max-toc-depth", type=int, default=10, help="Max TOC depth for chapters (default: 10)")
    parser.add_argument("--max-chunk-chars", type=int, default=DEFAULT_MAX_CHUNK_CHARS, help=f"Max chars per chunk (default: {DEFAULT_MAX_CHUNK_CHARS})")
    parser.add_argument("--no-book-guide", action="store_true", help="Skip book-level guide generation")
    
    args = parser.parse_args()
    
    # Determine EPUB path
    if args.epub:
        epub_path = Path(args.epub)
    else:
        try:
            epub_path = find_latest_epub(Path("input_dir"))
            eprint(f"Using latest EPUB: {epub_path}")
        except FileNotFoundError as e:
            eprint(f"Error: {e}")
            sys.exit(1)
    
    if not epub_path.exists():
        eprint(f"Error: File not found: {epub_path}")
        sys.exit(1)
    
    if epub_path.suffix.lower() != ".epub":
        eprint("Error: Only .epub files are supported")
        sys.exit(1)
    
    # Extract chapters
    eprint(f"Reading EPUB: {epub_path.name}")
    chapters = get_chapters_from_epub(epub_path, args.max_toc_depth)
    
    if not chapters:
        eprint("Error: No chapters found in EPUB")
        sys.exit(1)
    
    # List mode
    if args.list:
        print(f"Chapters in '{epub_path.name}' ({len(chapters)} total):\n")
        for ch in chapters:
            print(f"  {ch.index:>3}. {ch.title} ({ch.char_count:,} chars)")
        print(f"\nUsage: uv run epub_guide_html.py '{epub_path}' --chapters 1,3-5")
        return
    
    # Select chapters
    selected_indices = parse_chapter_spec(args.chapters, len(chapters))
    if not selected_indices:
        eprint("Error: No valid chapters selected")
        sys.exit(1)
    
    selected_chapters = [ch for ch in chapters if ch.index in selected_indices]
    chapter_titles = [ch.title for ch in selected_chapters]
    chapter_info = ", ".join(chapter_titles[:5])
    if len(chapter_titles) > 5:
        chapter_info += f" ... ({len(chapter_titles)} total)"
    
    eprint(f"Selected {len(selected_chapters)} chapter(s): {chapter_info}")
    
    # Check API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        eprint("Error: GEMINI_API_KEY not set. Please set it in environment or .env file.")
        sys.exit(1)
    
    # Initialize
    client = genai.Client(api_key=api_key)
    cache_conn = open_cache(Path(args.cache))
    
    try:
        # Generate chapter guides
        chapter_guides: List[Tuple[str, dict]] = []
        
        for ch in selected_chapters:
            eprint(f"\n{'='*60}")
            eprint(f"Processing chapter {ch.index}: {ch.title}")
            eprint(f"{'='*60}")
            
            guide = generate_chapter_guide(client, ch, cache_conn, args.max_chunk_chars)
            chapter_guides.append((ch.title, guide))
            
            eprint(f"  Chapter {ch.index} complete!")
        
        # Generate book-level guide
        book_guide = {}
        if not args.no_book_guide and len(chapter_guides) > 0:
            eprint(f"\n{'='*60}")
            eprint("Generating book-level overview...")
            eprint(f"{'='*60}")
            book_guide = synthesize_book(client, epub_path.stem, chapter_guides, cache_conn)
            eprint("Book overview complete!")
        
        # Render HTML
        out_path = Path(args.out) if args.out else Path("output_dir") / f"{epub_path.stem}_guide.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        html = render_html(epub_path.stem, chapter_info, chapter_guides, book_guide)
        out_path.write_text(html, encoding="utf-8")
        
        print(str(out_path.absolute()))
        eprint(f"\nDone! Output saved to: {out_path}")
        
    finally:
        cache_conn.close()


if __name__ == "__main__":
    main()

