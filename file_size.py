# /// script
# dependencies = []
# ///

"""A simple file size statistics tool"""

import sys
from pathlib import Path


def main():
    """Calculate the size of a file or directory"""
    if len(sys.argv) < 2:
        print("Usage: file_size <file_or_directory_path>")
        sys.exit(1)
    
    target = Path(sys.argv[1])
    
    if not target.exists():
        print(f"Error: Path '{target}' does not exist")
        sys.exit(1)
    
    if target.is_file():
        size = target.stat().st_size
        print(f"File size: {size:,} bytes ({size / 1024:.2f} KB)")
    elif target.is_dir():
        total_size = sum(f.stat().st_size for f in target.rglob('*') if f.is_file())
        print(f"Directory total size: {total_size:,} bytes ({total_size / 1024:.2f} KB)")
    else:
        print(f"Error: '{target}' is not a file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()

