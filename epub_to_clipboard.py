# /// script
# dependencies = [
#   "beautifulsoup4",
#   "EbookLib",
# ]
# ///

"""
Copy EPUB chapter content to the system clipboard.

Usage:
  uv run epub_to_clipboard.py --list book.epub
  uv run epub_to_clipboard.py book.epub --chapters 1,3-5
  uv run epub_to_clipboard.py book.epub --chapters all
"""

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from bs4 import BeautifulSoup
import ebooklib
from ebooklib import epub


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
# Clipboard
# ============================================================================

def copy_to_clipboard(text: str) -> None:
    """Copy text to the clipboard using platform-specific tools."""
    commands: list[list[str]] = []

    if sys.platform == "darwin":
        commands = [["pbcopy"]]
    elif sys.platform.startswith("win"):
        commands = [["clip"], ["powershell", "-command", "Set-Clipboard"]]
    else:
        commands = [
            ["wl-copy"],
            ["xclip", "-selection", "clipboard"],
            ["xsel", "--clipboard", "--input"],
        ]

    for cmd in commands:
        try:
            subprocess.run(cmd, input=text, text=True, check=True)
            return
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError:
            continue

    eprint(
        "Error: No available clipboard command found. "
        "Install a clipboard utility (pbcopy/clip/xclip/xsel/wl-copy) and try again."
    )
    sys.exit(1)


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
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Copy EPUB chapter content to the system clipboard."
    )
    parser.add_argument("epub", nargs="?", help="Path to .epub file (default: latest in input_dir/)")
    parser.add_argument("--list", action="store_true", help="List chapters and exit")
    parser.add_argument("--chapters", default="1", help="Chapter selection: '1', '1,3-5', or 'all' (default: 1)")
    parser.add_argument("--max-toc-depth", type=int, default=10, help="Max TOC depth for chapters (default: 10)")
    
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
        print(f"\nUsage: uv run epub_to_clipboard.py '{epub_path}' --chapters 1,3-5")
        return
    
    # Select chapters
    selected_indices = parse_chapter_spec(args.chapters, len(chapters))
    if not selected_indices:
        eprint("Error: No valid chapters selected")
        sys.exit(1)
    
    selected_chapters = [ch for ch in chapters if ch.index in selected_indices]
    
    # Build output text with chapter separators
    parts: List[str] = []
    for ch in selected_chapters:
        separator = f"=== {ch.index}. {ch.title} ==="
        parts.append(separator)
        parts.append("")
        parts.append(ch.text)
        parts.append("")
    
    output_text = "\n".join(parts).strip()
    
    # Copy to clipboard
    copy_to_clipboard(output_text)
    
    total_chars = sum(ch.char_count for ch in selected_chapters)
    chapter_info = ", ".join(ch.title for ch in selected_chapters[:3])
    if len(selected_chapters) > 3:
        chapter_info += f" ... ({len(selected_chapters)} total)"
    
    eprint(f"Copied {len(selected_chapters)} chapter(s) to clipboard ({total_chars:,} chars)")
    eprint(f"Chapters: {chapter_info}")


if __name__ == "__main__":
    main()

