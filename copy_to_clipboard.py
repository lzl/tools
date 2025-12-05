# /// script
# dependencies = []
# ///

"""Copy a text-based file's contents to the system clipboard."""

import subprocess
import sys
from pathlib import Path
from typing import Iterable

ALLOWED_EXTENSIONS = {".txt", ".md", ".srt", ".vtt"}
DEFAULT_OUTPUT_DIR = Path("output_dir")


def is_allowed_file(path: Path) -> bool:
    """Check if the path is a file with an allowed extension."""
    return path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS


def find_latest_file(directory: Path, candidates: Iterable[str]) -> Path | None:
    """Return the most recently modified allowed file in the directory."""
    if not directory.is_dir():
        return None

    allowed_files = [
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in candidates
    ]
    if not allowed_files:
        return None

    return max(allowed_files, key=lambda p: p.stat().st_mtime)


def copy_to_clipboard(text: str) -> None:
    """Copy text to the clipboard using platform-specific tools."""
    commands: list[list[str]] = []

    if sys.platform == "darwin":
        commands = [["pbcopy"]]
    elif sys.platform.startswith("win"):
        commands = [["clip"], ["powershell", "-command", "Set-Clipboard"]]
    else:
        commands = [["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]

    for cmd in commands:
        try:
            subprocess.run(cmd, input=text, text=True, check=True)
            return
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError:
            continue

    print(
        "Error: No available clipboard command found. "
        "Install a clipboard utility (pbcopy/clip/xclip/xsel) and try again."
    )
    sys.exit(1)


def read_file_text(path: Path) -> str:
    """Read text content from a file with UTF-8 decoding."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to replacing undecodable bytes to avoid failing the copy.
        return path.read_text(encoding="utf-8", errors="replace")


def usage() -> None:
    allowed = ", ".join(sorted(ext.lstrip(".") for ext in ALLOWED_EXTENSIONS))
    print("Usage: copy_to_clipboard [file_path]")
    print(f"Supported extensions: {allowed}")
    print(f"Default: latest {allowed} file in '{DEFAULT_OUTPUT_DIR}/'")


def main():
    if len(sys.argv) > 2:
        usage()
        sys.exit(1)

    if len(sys.argv) == 2:
        target = Path(sys.argv[1])
        if target.is_dir():
            print(f"Error: '{target}' is a directory; please provide a file.")
            sys.exit(1)
        if not target.exists():
            print(f"Error: File '{target}' does not exist.")
            sys.exit(1)
        if not is_allowed_file(target):
            allowed = ", ".join(sorted(ext.lstrip('.') for ext in ALLOWED_EXTENSIONS))
            print(f"Error: File type not supported. Allowed extensions: {allowed}")
            sys.exit(1)
    else:
        target = find_latest_file(DEFAULT_OUTPUT_DIR, ALLOWED_EXTENSIONS)
        if target is None:
            allowed = ", ".join(sorted(ext.lstrip('.') for ext in ALLOWED_EXTENSIONS))
            print(
                f"Error: No supported files found in '{DEFAULT_OUTPUT_DIR}'. "
                f"Allowed extensions: {allowed}"
            )
            sys.exit(1)

    content = read_file_text(target)
    copy_to_clipboard(content)
    print(f"Copied contents of '{target}' to clipboard.")


if __name__ == "__main__":
    main()
