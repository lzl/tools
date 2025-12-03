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
