# Tools

A collection of utility tools. Each Python file is an independent tool that can be called separately via `uv`.

## Usage

Run tools directly using `uv run` with the Python file:

```bash
uv run file_size.py <file_or_directory_path>
```

Each tool uses PEP 723 inline script metadata to declare dependencies, making them self-contained and independent.

## Available Tools

### file-size

Calculate the size of a file or directory.

**Examples:**
```bash
# Calculate file size
uv run file_size.py README.md

# Calculate directory size
uv run file_size.py .
```

### epub-vocab-html

Extract unknown words and phrases from an English EPUB and generate an HTML study page with translations.

**Features:**
- Supports chapter selection (TOC-based, with spine fallback)
- Vocabulary filtering: Zipf frequency, known words list, minimum count
- Phrase extraction: phrasal verbs, fixed expressions, n-grams
- Translation via Gemini API (with SQLite caching)
- Bilingual output: English term/sentence + Chinese translation

**Examples:**
```bash
# List chapters in an EPUB
uv run epub_vocab_html.py --list book.epub

# Process specific chapters
uv run epub_vocab_html.py book.epub --chapters 1,3-5 --out output_dir/vocab.html

# Process all chapters
uv run epub_vocab_html.py book.epub --chapters all

# Use a known words list to filter out words you already know
uv run epub_vocab_html.py book.epub --chapters 1 --known-words known_words.txt

# Adjust word filtering
uv run epub_vocab_html.py book.epub --chapters 1 --zipf-max 5.0 --min-count 2
```

**Environment Variables:**
- `GEMINI_API_KEY`: Required for translation (get from https://aistudio.google.com/)

## Adding New Tools

1. Create a new Python file (e.g., `my_tool.py`)
2. Add PEP 723 inline script metadata at the top of the file:
   ```python
   # /// script
   # dependencies = []
   # ///
   ```
3. Implement a `main()` function in the file
4. If your tool has dependencies, use `uv add --script my_tool.py <dependency>` to add them

The script metadata makes each tool self-contained and independent.

## Design Principles

- Each tool focuses on one thing only, with decoupled functionality
- Each tool is an independent Python file
- Tools can be called individually via `uv`
