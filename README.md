# Tools

A collection of utility tools. Each Python file is an independent tool that can be called separately via `uv`.

## Usage

### Method 1: Via Script Entry Point (Recommended)

After configuring script entry points in `pyproject.toml`, you can call tools directly:

```bash
uv run file-size <file_or_directory_path>
```

### Method 2: Run Python File Directly

```bash
uv run python file_size.py <file_or_directory_path>
```

## Available Tools

### file-size

Calculate the size of a file or directory.

**Examples:**
```bash
# Calculate file size
uv run file-size README.md

# Calculate directory size
uv run file-size .
```

## Adding New Tools

1. Create a new Python file (e.g., `my_tool.py`)
2. Implement a `main()` function in the file
3. Add an entry point in `pyproject.toml` under `[project.scripts]`:
   ```toml
   [project.scripts]
   my-tool = "my_tool:main"
   ```

## Design Principles

- Each tool focuses on one thing only, with decoupled functionality
- Each tool is an independent Python file
- Tools can be called individually via `uv`
