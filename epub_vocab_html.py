# /// script
# dependencies = [
#   "beautifulsoup4",
#   "EbookLib",
#   "google-genai",
#   "python-dotenv",
#   "wordfreq",
# ]
# ///

"""
Extract unknown words and phrases from an English EPUB and generate an HTML study page.

Usage:
  uv run epub_vocab_html.py --list book.epub
  uv run epub_vocab_html.py book.epub --chapters 1,3-5 --out output_dir/book_vocab.html
  uv run epub_vocab_html.py book.epub --chapters all --out output_dir/book_vocab.html
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

from bs4 import BeautifulSoup
from dotenv import load_dotenv
import ebooklib
from ebooklib import epub
from google import genai
from google.genai import errors
from wordfreq import zipf_frequency


# ============================================================================
# Constants
# ============================================================================

STOPWORDS_EN: Set[str] = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while", "as", "of", "to", "in", "on", "at", "by",
    "for", "from", "with", "without", "into", "over", "under", "between", "among", "about", "after", "before", "during",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did", "done", "doing",
    "have", "has", "had", "having", "can", "could", "may", "might", "must", "should", "would", "will", "shall",
    "i", "me", "my", "mine", "we", "us", "our", "ours", "you", "your", "yours", "he", "him", "his", "she", "her", "hers",
    "it", "its", "they", "them", "their", "theirs",
    "this", "that", "these", "those", "here", "there", "who", "whom", "whose", "which", "what", "why", "how",
    "not", "no", "nor", "so", "too", "very", "just", "only", "also", "more", "most", "less", "least",
    "all", "any", "some", "such", "own", "same", "other", "another", "each", "every", "both", "few", "many", "much",
    "than", "now", "even", "still", "already", "yet", "again", "once", "never", "always", "often", "sometimes",
    "where", "whenever", "wherever", "however", "whether", "though", "although", "because", "since", "until", "unless",
    "mr", "mrs", "ms", "dr", "etc", "ie", "eg", "vs",
}

# Phrasal verb particles
PARTICLES: Set[str] = {
    "up", "off", "out", "in", "on", "away", "over", "back", "down", "around", "through", "into", "about",
    "along", "across", "aside", "apart", "forth", "forward", "together",
}

# Common fixed expressions to detect (lowercase)
FIXED_EXPRESSIONS: List[str] = [
    "in spite of", "as well as", "in order to", "due to", "because of", "instead of", "in terms of",
    "as a result", "on the other hand", "in addition to", "with regard to", "in accordance with",
    "as opposed to", "in favor of", "in case of", "by means of", "on behalf of", "in front of",
    "at least", "at most", "at first", "at last", "at once", "so far", "so that", "such as",
    "no longer", "no matter", "as if", "as though", "even if", "even though", "rather than",
    "whether or not", "more or less", "sooner or later", "little by little", "one by one",
    "from time to time", "now and then", "again and again", "over and over", "back and forth",
    "up and down", "in and out", "on and off", "here and there", "to and fro",
    "all of a sudden", "all at once", "first of all", "after all", "above all", "most of all",
    "in fact", "in general", "in particular", "in reality", "in theory", "in practice",
    "for example", "for instance", "for the most part", "for the time being",
    "take place", "make sense", "make sure", "make up", "take care", "take advantage",
    "pay attention", "come across", "come up with", "get rid of", "give up", "look forward to",
    "put up with", "run out of", "take part in", "carry out", "bring about", "figure out",
    "point out", "turn out", "find out", "work out", "set up", "pick up", "keep up",
]

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT_END_RE = re.compile(r'[.!?。！？](?:\s|$)')

MODEL_NAME = "gemini-2.5-flash-lite-preview-09-2025"
TARGET_LANG = "简体中文"


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
class VocabItem:
    term: str
    sentence: str
    count: int
    zipf: float
    is_phrase: bool = False
    term_translation: str = ""
    sentence_translation: str = ""


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


def iter_sentences(text: str) -> Iterable[str]:
    """Split text into sentences (simple rule-based)."""
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return
    
    parts = SENT_END_RE.split(text)
    for i, part in enumerate(parts):
        part = part.strip()
        if part:
            # Re-add the sentence-ending punctuation if not the last part
            yield part


def tokenize_sentence(sentence: str) -> List[str]:
    """Tokenize sentence into lowercase words."""
    return [m.group(0).lower() for m in WORD_RE.finditer(sentence)]


# ============================================================================
# EPUB Chapter Extraction (TOC priority, spine fallback)
# ============================================================================

def flatten_toc(toc_items, depth: int = 1) -> List[Tuple[str, str, int]]:
    """Recursively flatten TOC into (title, href, depth) tuples."""
    result = []
    for item in toc_items:
        if isinstance(item, tuple):
            # Nested section: (Section, [children])
            section, children = item
            if hasattr(section, 'title') and hasattr(section, 'href'):
                result.append((section.title, section.href, depth))
            result.extend(flatten_toc(children, depth + 1))
        elif hasattr(item, 'title') and hasattr(item, 'href'):
            # epub.Link object
            result.append((item.title, item.href, depth))
    return result


def extract_text_from_html(html_content: bytes, fragment: Optional[str] = None) -> str:
    """Extract text from HTML content, optionally starting from a fragment anchor."""
    try:
        html_str = html_content.decode("utf-8")
    except UnicodeDecodeError:
        html_str = html_content.decode("utf-8", errors="ignore")
    
    soup = BeautifulSoup(html_str, "html.parser")
    
    # Remove script and style elements
    for tag in soup(["script", "style"]):
        tag.decompose()
    
    if fragment:
        # Try to find the anchor element
        anchor = soup.find(id=fragment) or soup.find(attrs={"name": fragment})
        if anchor:
            # Get text from anchor onwards (simplified approach)
            text_parts = []
            for sibling in anchor.find_all_next(string=True):
                text_parts.append(sibling.strip())
            return clean_text(" ".join(text_parts))
    
    # Fallback: get all text
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


def get_chapters_from_epub(epub_path: Path, max_toc_depth: int = 1) -> List[Chapter]:
    """
    Extract chapters from EPUB.
    Priority: TOC (depth <= max_toc_depth) -> spine fallback.
    """
    book = epub.read_epub(str(epub_path))
    
    # Build a map from href (without fragment) to item
    href_to_item: Dict[str, epub.EpubItem] = {}
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            # Normalize href
            href = item.get_name()
            href_to_item[href] = item
    
    # Build spine order list for fallback lookup
    spine_order: List[str] = []
    for idref, _linear in book.spine:
        if idref:
            spine_item = book.get_item_with_id(idref)
            if spine_item and spine_item.get_type() == ebooklib.ITEM_DOCUMENT:
                spine_order.append(spine_item.get_name())
    
    chapters: List[Chapter] = []
    
    # Try TOC first
    if book.toc:
        toc_entries = flatten_toc(book.toc)
        # Filter by depth
        toc_entries = [(t, h, d) for t, h, d in toc_entries if d <= max_toc_depth]
        
        if toc_entries:
            seen_hrefs: Set[str] = set()
            for idx, (title, href, _depth) in enumerate(toc_entries, start=1):
                # Parse href (may have fragment like "chapter1.xhtml#section2")
                if "#" in href:
                    file_href, fragment = href.split("#", 1)
                else:
                    file_href, fragment = href, None
                
                # Find the item
                item = href_to_item.get(file_href)
                if item is None:
                    # Try with different path variations
                    for key in href_to_item:
                        if key.endswith(file_href) or file_href.endswith(key):
                            item = href_to_item[key]
                            file_href = key
                            break
                
                if item is None:
                    continue
                
                # Skip if we've seen this exact href (avoid duplicates)
                full_href = f"{file_href}#{fragment}" if fragment else file_href
                if full_href in seen_hrefs:
                    continue
                seen_hrefs.add(full_href)
                
                text = extract_text_from_html(item.get_content(), fragment)
                
                # If text is empty (e.g., page only has images), try next spine item
                if not text and file_href in spine_order:
                    spine_idx = spine_order.index(file_href)
                    if spine_idx + 1 < len(spine_order):
                        next_href = spine_order[spine_idx + 1]
                        next_item = href_to_item.get(next_href)
                        if next_item:
                            text = extract_text_from_html(next_item.get_content())
                            if text:
                                full_href = next_href  # Update href to actual content file
                
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
        
        if not text or len(text) < 100:  # Skip very short sections
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
# Vocabulary Extraction
# ============================================================================

def load_known_words(path: Optional[str]) -> Set[str]:
    """Load known words from file (one per line)."""
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Known words file not found: {p}")
    
    result: Set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        w = line.strip().lower()
        if w and not w.startswith("#"):
            result.add(w)
    return result


def extract_words(
    text: str,
    known: Set[str],
    zipf_max: float,
    min_count: int,
    min_len: int,
    max_words: int,
) -> List[VocabItem]:
    """Extract unknown words from text."""
    word_count: Counter = Counter()
    word_example: Dict[str, str] = {}
    word_zipf: Dict[str, float] = {}
    
    for sent in iter_sentences(text):
        tokens = tokenize_sentence(sent)
        for i, token in enumerate(tokens):
            if len(token) < min_len:
                continue
            if token in STOPWORDS_EN:
                continue
            if token in known:
                continue
            
            # Get zipf frequency
            if token not in word_zipf:
                try:
                    word_zipf[token] = zipf_frequency(token, "en")
                except Exception:
                    word_zipf[token] = 0.0
            
            if word_zipf[token] > zipf_max:
                continue
            
            word_count[token] += 1
            if token not in word_example:
                word_example[token] = sent
    
    # Filter by min_count and build items
    items: List[VocabItem] = []
    for word, count in word_count.items():
        if count >= min_count:
            items.append(VocabItem(
                term=word,
                sentence=word_example[word],
                count=count,
                zipf=word_zipf.get(word, 0.0),
                is_phrase=False,
            ))
    
    # Sort by count (desc), then zipf (asc)
    items.sort(key=lambda x: (-x.count, x.zipf, x.term))
    return items[:max_words]


def extract_phrasal_verbs(text: str, known: Set[str], min_count: int) -> Dict[str, Tuple[int, str]]:
    """Extract phrasal verbs (verb + particle) from text."""
    phrase_count: Counter = Counter()
    phrase_example: Dict[str, str] = {}
    
    for sent in iter_sentences(text):
        tokens = tokenize_sentence(sent)
        for i in range(len(tokens) - 1):
            verb = tokens[i]
            particle = tokens[i + 1]
            
            # Check if this looks like a phrasal verb
            if particle in PARTICLES and verb not in STOPWORDS_EN and len(verb) >= 3:
                # Skip if verb is known
                if verb in known:
                    continue
                
                phrase = f"{verb} {particle}"
                phrase_count[phrase] += 1
                if phrase not in phrase_example:
                    phrase_example[phrase] = sent
    
    return {p: (c, phrase_example[p]) for p, c in phrase_count.items() if c >= min_count}


def extract_fixed_expressions(text: str, min_count: int) -> Dict[str, Tuple[int, str]]:
    """Extract fixed expressions from text."""
    text_lower = text.lower()
    result: Dict[str, Tuple[int, str]] = {}
    
    for expr in FIXED_EXPRESSIONS:
        count = text_lower.count(expr)
        if count >= min_count:
            # Find an example sentence
            for sent in iter_sentences(text):
                if expr in sent.lower():
                    result[expr] = (count, sent)
                    break
    
    return result


def extract_ngrams(
    text: str,
    known: Set[str],
    min_count: int,
    max_phrases: int,
    ngram_range: Tuple[int, int] = (2, 4),
) -> List[VocabItem]:
    """Extract high-frequency n-grams (conservative filtering)."""
    ngram_count: Counter = Counter()
    ngram_example: Dict[str, str] = {}
    
    for sent in iter_sentences(text):
        tokens = tokenize_sentence(sent)
        
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i:i + n])
                
                # Filter out noise
                # Skip if all stopwords
                if all(t in STOPWORDS_EN for t in gram):
                    continue
                # Skip if contains very short tokens
                if any(len(t) < 2 for t in gram):
                    continue
                # Skip if starts or ends with stopword (usually noise)
                if gram[0] in STOPWORDS_EN or gram[-1] in STOPWORDS_EN:
                    continue
                # Skip if all words are known
                if all(t in known for t in gram):
                    continue
                
                phrase = " ".join(gram)
                ngram_count[phrase] += 1
                if phrase not in ngram_example:
                    ngram_example[phrase] = sent
    
    # Score by: count * (1 / avg_zipf) to prefer frequent + rare-word phrases
    scored: List[Tuple[str, float, int, str]] = []
    for phrase, count in ngram_count.items():
        if count < min_count:
            continue
        
        tokens = phrase.split()
        zipfs = []
        for t in tokens:
            if t not in STOPWORDS_EN:
                try:
                    zipfs.append(zipf_frequency(t, "en"))
                except Exception:
                    zipfs.append(3.0)
        
        avg_zipf = sum(zipfs) / len(zipfs) if zipfs else 5.0
        # Lower avg_zipf = rarer words = higher score
        score = count * (1.0 / max(avg_zipf, 0.5))
        scored.append((phrase, avg_zipf, count, ngram_example[phrase]))
    
    # Sort by score descending
    scored.sort(key=lambda x: -x[2])  # Sort by count for now (simpler)
    
    items: List[VocabItem] = []
    for phrase, avg_zipf, count, example in scored[:max_phrases]:
        items.append(VocabItem(
            term=phrase,
            sentence=example,
            count=count,
            zipf=avg_zipf,
            is_phrase=True,
        ))
    
    return items


def extract_phrases(
    text: str,
    known: Set[str],
    min_phrase_count: int,
    max_phrases: int,
) -> List[VocabItem]:
    """Extract phrases (phrasal verbs + fixed expressions + n-grams)."""
    items: List[VocabItem] = []
    seen_phrases: Set[str] = set()
    
    # 1. Phrasal verbs
    phrasal = extract_phrasal_verbs(text, known, min_phrase_count)
    for phrase, (count, example) in phrasal.items():
        if phrase not in seen_phrases:
            seen_phrases.add(phrase)
            items.append(VocabItem(
                term=phrase,
                sentence=example,
                count=count,
                zipf=0.0,
                is_phrase=True,
            ))
    
    # 2. Fixed expressions
    fixed = extract_fixed_expressions(text, min_phrase_count)
    for phrase, (count, example) in fixed.items():
        if phrase not in seen_phrases:
            seen_phrases.add(phrase)
            items.append(VocabItem(
                term=phrase,
                sentence=example,
                count=count,
                zipf=0.0,
                is_phrase=True,
            ))
    
    # 3. N-grams (fill remaining slots)
    remaining = max_phrases - len(items)
    if remaining > 0:
        ngrams = extract_ngrams(text, known, min_phrase_count, remaining)
        for item in ngrams:
            if item.term not in seen_phrases:
                seen_phrases.add(item.term)
                items.append(item)
    
    # Sort by count
    items.sort(key=lambda x: -x.count)
    return items[:max_phrases]


# ============================================================================
# Gemini Translation with Cache and Retry
# ============================================================================

def open_cache(db_path: Path) -> sqlite3.Connection:
    """Open or create SQLite cache database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def cache_key(term: str, sentence: str) -> str:
    """Generate cache key from term and sentence."""
    h = hashlib.sha256()
    h.update(f"{MODEL_NAME}\n{TARGET_LANG}\n{term}\n{sentence}".encode("utf-8"))
    return h.hexdigest()


def cache_get(conn: sqlite3.Connection, key: str) -> Optional[dict]:
    """Get value from cache."""
    cur = conn.execute("SELECT value FROM cache WHERE key = ?", (key,))
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
        "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
        (key, json.dumps(value, ensure_ascii=False))
    )
    conn.commit()


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


def extract_json_array(text: str) -> List[dict]:
    """Extract JSON array from model response."""
    t = (text or "").strip()
    # Strip code fences
    t = re.sub(r"^```(?:json)?\s*", "", t)
    t = re.sub(r"\s*```$", "", t)
    
    start = t.find("[")
    end = t.rfind("]")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return a JSON array")
    
    payload = t[start:end + 1]
    data = json.loads(payload)
    if not isinstance(data, list):
        raise ValueError("JSON is not an array")
    return data


def translate_batch(
    client: genai.Client,
    items: List[VocabItem],
    max_retries: int = 3,
) -> None:
    """Translate a batch of items using Gemini (modifies items in place)."""
    if not items:
        return
    
    # Build request
    req = [{"term": it.term, "sentence": it.sentence, "is_phrase": it.is_phrase} for it in items]
    
    prompt = f"""You are a bilingual dictionary and translator.
Target language: {TARGET_LANG}

For each item in the input JSON array:
1. Translate the "term" (word or phrase) - provide 1-3 concise meanings in context
2. Translate the "sentence" completely to {TARGET_LANG}

Return ONLY a JSON array with the same length and order as input.
Each element must have exactly these keys:
- "term_translation": string (1-3 meanings, separated by "; " if multiple)
- "sentence_translation": string (complete sentence translation)

Do NOT add markdown, comments, or extra keys.

INPUT:
{json.dumps(req, ensure_ascii=False, indent=2)}"""

    last_error = None
    retry_delays = [5, 15, 30]
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
            )
            
            out = extract_json_array(response.text or "")
            if len(out) != len(items):
                raise ValueError(f"Response length mismatch: expected {len(items)}, got {len(out)}")
            
            for item, obj in zip(items, out):
                item.term_translation = str(obj.get("term_translation", "")).strip()
                item.sentence_translation = str(obj.get("sentence_translation", "")).strip()
            
            return  # Success
            
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


def translate_items(
    items: List[VocabItem],
    cache_conn: sqlite3.Connection,
    api_key: str,
    batch_size: int = 20,
) -> None:
    """Translate all items, using cache and batching."""
    # Check cache first
    to_translate: List[VocabItem] = []
    
    for item in items:
        key = cache_key(item.term, item.sentence)
        cached = cache_get(cache_conn, key)
        if cached:
            item.term_translation = cached.get("term_translation", "")
            item.sentence_translation = cached.get("sentence_translation", "")
        else:
            to_translate.append(item)
    
    if not to_translate:
        eprint(f"  All {len(items)} items found in cache")
        return
    
    eprint(f"  {len(items) - len(to_translate)} items from cache, {len(to_translate)} to translate")
    
    client = genai.Client(api_key=api_key)
    
    # Process in batches
    for i in range(0, len(to_translate), batch_size):
        batch = to_translate[i:i + batch_size]
        eprint(f"  Translating batch {i // batch_size + 1} ({len(batch)} items)...")
        
        translate_batch(client, batch)
        
        # Save to cache
        for item in batch:
            key = cache_key(item.term, item.sentence)
            cache_set(cache_conn, key, {
                "term_translation": item.term_translation,
                "sentence_translation": item.sentence_translation,
            })
        
        # Small delay between batches to avoid rate limits
        if i + batch_size < len(to_translate):
            time.sleep(1)


# ============================================================================
# HTML Output
# ============================================================================

def render_html(
    book_name: str,
    chapter_info: str,
    words: List[VocabItem],
    phrases: List[VocabItem],
) -> str:
    """Render vocabulary items to HTML."""
    
    def render_table(items: List[VocabItem], title: str) -> str:
        if not items:
            return ""
        
        rows = []
        for item in items:
            rows.append(f"""      <tr>
        <td class="term">{escape(item.term)}</td>
        <td class="translation">{escape(item.term_translation)}</td>
        <td class="sentence">{escape(item.sentence)}</td>
        <td class="translation">{escape(item.sentence_translation)}</td>
        <td class="count">{item.count}</td>
      </tr>""")
        
        return f"""
    <h2>{escape(title)} ({len(items)})</h2>
    <table>
      <thead>
        <tr>
          <th>Term</th>
          <th>Translation</th>
          <th>Example Sentence</th>
          <th>Sentence Translation</th>
          <th>Count</th>
        </tr>
      </thead>
      <tbody>
{chr(10).join(rows)}
      </tbody>
    </table>
"""
    
    words_html = render_table(words, "Words")
    phrases_html = render_table(phrases, "Phrases")
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Vocabulary - {escape(book_name)}</title>
  <style>
    * {{
      box-sizing: border-box;
    }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      line-height: 1.6;
      margin: 0;
      padding: 20px;
      background: #f5f5f5;
      color: #333;
    }}
    .container {{
      max-width: 1400px;
      margin: 0 auto;
      background: white;
      padding: 30px;
      border-radius: 8px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }}
    h1 {{
      margin-top: 0;
      color: #2c3e50;
      border-bottom: 2px solid #3498db;
      padding-bottom: 10px;
    }}
    h2 {{
      color: #2980b9;
      margin-top: 40px;
    }}
    .meta {{
      background: #ecf0f1;
      padding: 15px;
      border-radius: 5px;
      margin-bottom: 30px;
      font-size: 14px;
      color: #7f8c8d;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-bottom: 30px;
    }}
    th, td {{
      border: 1px solid #ddd;
      padding: 12px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #3498db;
      color: white;
      font-weight: 600;
    }}
    tr:nth-child(even) {{
      background: #f9f9f9;
    }}
    tr:hover {{
      background: #e8f4f8;
    }}
    .term {{
      font-weight: 600;
      color: #2c3e50;
      white-space: nowrap;
    }}
    .translation {{
      color: #27ae60;
    }}
    .sentence {{
      font-style: italic;
      color: #555;
    }}
    .count {{
      text-align: center;
      color: #7f8c8d;
      font-size: 13px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Vocabulary Study - {escape(book_name)}</h1>
    <div class="meta">
      <strong>Chapters:</strong> {escape(chapter_info)}<br>
      <strong>Words:</strong> {len(words)} | <strong>Phrases:</strong> {len(phrases)}
    </div>
{words_html}
{phrases_html}
  </div>
</body>
</html>
"""


# ============================================================================
# Main
# ============================================================================

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(
        description="Extract unknown words and phrases from an English EPUB and generate an HTML study page."
    )
    parser.add_argument("epub", nargs="?", help="Path to .epub file (default: latest in input_dir/)")
    parser.add_argument("--list", action="store_true", help="List chapters and exit")
    parser.add_argument("--chapters", default="1", help="Chapter selection: '1', '1,3-5', or 'all'")
    parser.add_argument("--out", help="Output HTML path (default: output_dir/<book>_vocab.html)")
    parser.add_argument("--cache", default="output_dir/epub_vocab_cache.sqlite", help="Cache database path")
    parser.add_argument("--known-words", help="File with known words (one per line)")
    parser.add_argument("--zipf-max", type=float, default=4.0, help="Max Zipf frequency for words (default: 4.0)")
    parser.add_argument("--min-count", type=int, default=1, help="Min word occurrence count (default: 1)")
    parser.add_argument("--min-len", type=int, default=3, help="Min word length (default: 3)")
    parser.add_argument("--max-words", type=int, default=200, help="Max words to output (default: 200)")
    parser.add_argument("--min-phrase-count", type=int, default=2, help="Min phrase occurrence count (default: 2)")
    parser.add_argument("--max-phrases", type=int, default=100, help="Max phrases to output (default: 100)")
    parser.add_argument("--max-toc-depth", type=int, default=10, help="Max TOC depth for chapters (default: 10)")
    parser.add_argument("--no-translate", action="store_true", help="Skip translation (output empty translations)")
    
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
        print(f"\nUsage: uv run epub_vocab_html.py '{epub_path}' --chapters 1,3-5")
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
    
    # Load known words
    known = load_known_words(args.known_words)
    if known:
        eprint(f"Loaded {len(known)} known words")
    
    # Process chapters one by one to avoid memory issues
    all_words: List[VocabItem] = []
    all_phrases: List[VocabItem] = []
    word_seen: Set[str] = set()
    phrase_seen: Set[str] = set()
    
    for ch in selected_chapters:
        eprint(f"\nProcessing chapter {ch.index}: {ch.title}")
        
        # Extract words
        words = extract_words(
            ch.text,
            known,
            args.zipf_max,
            args.min_count,
            args.min_len,
            args.max_words,
        )
        for w in words:
            if w.term not in word_seen:
                word_seen.add(w.term)
                all_words.append(w)
        
        # Extract phrases
        phrases = extract_phrases(
            ch.text,
            known,
            args.min_phrase_count,
            args.max_phrases,
        )
        for p in phrases:
            if p.term not in phrase_seen:
                phrase_seen.add(p.term)
                all_phrases.append(p)
    
    # Sort and limit
    all_words.sort(key=lambda x: (-x.count, x.zipf, x.term))
    all_words = all_words[:args.max_words]
    
    all_phrases.sort(key=lambda x: -x.count)
    all_phrases = all_phrases[:args.max_phrases]
    
    eprint(f"\nExtracted {len(all_words)} words and {len(all_phrases)} phrases")
    
    if not all_words and not all_phrases:
        eprint("No vocabulary items found. Try adjusting --zipf-max or --min-count.")
        sys.exit(0)
    
    # Translate
    if not args.no_translate:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            eprint("Error: GEMINI_API_KEY not set. Use --no-translate to skip translation.")
            sys.exit(1)
        
        cache_conn = open_cache(Path(args.cache))
        try:
            eprint("\nTranslating words...")
            translate_items(all_words, cache_conn, api_key)
            
            eprint("\nTranslating phrases...")
            translate_items(all_phrases, cache_conn, api_key)
        finally:
            cache_conn.close()
    
    # Generate HTML
    out_path = Path(args.out) if args.out else Path("output_dir") / f"{epub_path.stem}_vocab.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    html = render_html(epub_path.stem, chapter_info, all_words, all_phrases)
    out_path.write_text(html, encoding="utf-8")
    
    print(str(out_path.absolute()))
    eprint(f"\nDone! Output saved to: {out_path}")


if __name__ == "__main__":
    main()

