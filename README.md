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

### epub-guide-html

Generate a comprehensive reading guide (导读手册) from an English EPUB for ESL learners. The guide helps readers with limited English proficiency understand and enjoy the original text.

**Features:**
- Supports chapter selection (TOC-based, with spine fallback)
- Deep analysis via Gemini API with intelligent chunking for long chapters
- SQLite caching for efficient regeneration
- Beautiful single-file HTML output (can be printed to PDF via browser)

**Guide Contents (per chapter):**
- 📌 **读前导读**: Chapter positioning, focus points, difficulty hints, reading strategies
- 👤 **人物/专有名词**: Characters, locations, and key terms with explanations
- 📖 **情节结构**: Scene-by-scene plot breakdown with turning points marked
- 📝 **重点词汇**: 15-25 key vocabulary items with context and usage notes
- 💬 **重点句精讲**: 5-10 complex sentences with translations and analysis
- 📐 **语法与风格**: Grammar patterns and writing style features
- 🌍 **文化背景**: Cultural context, allusions, and metaphors explained
- ✏️ **章节自测**: Comprehension quiz with collapsible answers
- ⚠️ **读后复盘**: Full plot summary and character analysis (spoiler, collapsed)

**Book-level overview:**
- Global character list with relationships
- Core themes and motifs
- Cross-chapter vocabulary summary
- Reading plan suggestions

**Examples:**
```bash
# List chapters in an EPUB
uv run epub_guide_html.py --list book.epub

# Generate guide for specific chapters
uv run epub_guide_html.py book.epub --chapters 1,3-5 --out output_dir/book_guide.html

# Generate guide for all chapters
uv run epub_guide_html.py book.epub --chapters all

# Adjust chunk size for very long chapters
uv run epub_guide_html.py book.epub --chapters 1 --max-chunk-chars 8000

# Skip book-level overview (faster for single chapter)
uv run epub_guide_html.py book.epub --chapters 1 --no-book-guide
```

**Environment Variables:**
- `GEMINI_API_KEY`: Required for AI generation (get from https://aistudio.google.com/)

### epub-to-clipboard

Copy EPUB chapter content to the system clipboard.

**Features:**
- Supports chapter selection (TOC-based, with spine fallback)
- Cross-platform clipboard support (macOS/Windows/Linux)
- Chapter title separators for easy navigation in pasted content

**Examples:**
```bash
# List chapters in an EPUB
uv run epub_to_clipboard.py --list book.epub

# Copy specific chapters to clipboard
uv run epub_to_clipboard.py book.epub --chapters 1,3-5

# Copy all chapters
uv run epub_to_clipboard.py book.epub --chapters all

# Copy first chapter (default if --chapters not specified)
uv run epub_to_clipboard.py book.epub
```

### transcribe-audio-groq

Transcribe audio files to WebVTT subtitles using Groq's Whisper API (`whisper-large-v3-turbo` model).

**Features:**
- Fast transcription via Groq's optimized Whisper model
- Outputs WebVTT format with timestamps
- Auto-selects latest audio file from `input_dir/` if no file specified
- Supports: mp3, wav, m4a, flac, ogg, webm (max 25MB)

**Examples:**
```bash
# Transcribe a specific audio file
uv run transcribe_audio_groq.py audio.mp3

# Transcribe with custom output directory
uv run transcribe_audio_groq.py audio.mp3 ./subtitles

# Auto-select latest audio from input_dir/
uv run transcribe_audio_groq.py
```

**Environment Variables:**
- `GROQ_API_KEY`: Required (get from https://console.groq.com/)

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
