# /// script
# dependencies = [
#   "yt-dlp",
# ]
# ///

"""A tool to download subtitles from YouTube and other platforms using yt-dlp"""

import sys
import subprocess
from pathlib import Path


def list_available_subtitles(video_url, cookies_file=None):
    """List available subtitles for the video"""
    cmd = [
        "yt-dlp",
        video_url,
        "--list-subs",
        "--remote-components", "ejs:github",
    ]
    
    if cookies_file and cookies_file.exists():
        cmd.extend(["--cookies", str(cookies_file)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error: Failed to list subtitles")
        print(f"yt-dlp exited with code {e.returncode}")
        if e.stderr:
            print(f"Error message: {e.stderr}")
        sys.exit(1)


def parse_subtitle_languages(output):
    """Parse yt-dlp --list-subs output to extract language codes"""
    languages = []
    lines = output.split('\n')
    
    # Look for subtitle language lines
    # yt-dlp output format can vary, examples:
    # "Available subtitles for <id>:"
    # "Language Name    Formats"
    # "zh-Hans          vtt, ttml, srv3, srv2, srv1"
    # or just language codes directly
    
    in_subtitle_section = False
    found_header = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if we're in the subtitle section
        if "Available subtitles" in line.lower() or "subtitles available" in line.lower():
            in_subtitle_section = True
            continue
        
        # Skip header lines
        if ("Language" in line and "Formats" in line) or line.startswith("Language"):
            found_header = True
            continue
        
        # Stop if we hit another section (like "Available automatic captions")
        if in_subtitle_section and ("Available automatic" in line.lower() or 
                                     "Available manual" in line.lower() or
                                     line.startswith("WARNING") or
                                     line.startswith("ERROR")):
            # Continue to also parse auto/manual captions
            continue
        
        if in_subtitle_section or found_header:
            # Parse language code (first column)
            # Format: "zh-Hans          vtt, ttml, ..."
            # or: "Language Name (lang_code)    Formats"
            parts = line.split()
            if parts:
                lang_code = parts[0]
                # Handle format like "Chinese (zh-Hans)" - extract from parentheses
                if '(' in line and ')' in line:
                    # Find content in parentheses
                    start = line.find('(')
                    end = line.find(')', start)
                    if start != -1 and end != -1:
                        lang_code = line[start+1:end]
                
                # Skip if it's not a language code
                if lang_code and len(lang_code) <= 20:  # Language codes are typically short
                    # Skip common non-language words
                    skip_words = ["Available", "Language", "Formats", "WARNING", "ERROR", "INFO"]
                    if lang_code not in skip_words and not lang_code.startswith("http"):
                        # Check if it looks like a language code (alphanumeric with possible hyphens)
                        if lang_code.replace('-', '').replace('_', '').isalnum():
                            if lang_code not in languages:
                                languages.append(lang_code)
    
    return languages


def select_language(languages):
    """Interactive language selection"""
    if not languages:
        print("No subtitles available for this video.")
        return None
    
    print("\nAvailable subtitle languages:")
    print("-" * 50)
    for i, lang in enumerate(languages, 1):
        print(f"  {i}. {lang}")
    print(f"  {len(languages) + 1}. All languages")
    print("-" * 50)
    
    while True:
        try:
            choice = input(f"\nSelect language (1-{len(languages) + 1}): ").strip()
            choice_num = int(choice)
            
            if choice_num == len(languages) + 1:
                return None  # All languages
            elif 1 <= choice_num <= len(languages):
                return languages[choice_num - 1]
            else:
                print(f"Please enter a number between 1 and {len(languages) + 1}")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\nSelection cancelled")
            sys.exit(1)


def main():
    """Download subtitles from provided URL using yt-dlp"""
    if len(sys.argv) < 2:
        print("Usage: download_subtitle <video_url> [output_directory]")
        print("\nExample:")
        print("  download_subtitle https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        print("  download_subtitle https://www.youtube.com/watch?v=dQw4w9WgXcQ ./custom_dir")
        sys.exit(1)
    
    video_url = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output_dir")
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Video URL: {video_url}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Check for cookies.txt file in current directory
    cookies_file = Path("cookies.txt")
    if cookies_file.exists():
        print(f"Using cookies file: {cookies_file.absolute()}")
    
    try:
        # List available subtitles
        print("\nFetching available subtitle languages...")
        subtitle_list_output = list_available_subtitles(video_url, cookies_file)
        
        # Parse available languages
        languages = parse_subtitle_languages(subtitle_list_output)
        
        if not languages:
            print("No subtitles available for this video.")
            sys.exit(1)
        
        # Let user select language
        selected_language = select_language(languages)
        
        if selected_language:
            print(f"\nSelected language: {selected_language}")
        else:
            print("\nSelected: All available languages")
        
        # Build yt-dlp command
        cmd = [
            "yt-dlp",
            video_url,
            "--write-subs",  # Download subtitles
            "--write-auto-subs",  # Download auto-generated subtitles if available
            "--skip-download",  # Skip downloading video/audio
            "-o", str(output_dir / "%(title)s.%(ext)s"),
            "--no-mtime",  # Don't set file modification time
            "--remote-components", "ejs:github",  # Enable EJS script downloads from GitHub
        ]
        
        # Add language filter if specified
        if selected_language:
            cmd.extend(["--sub-langs", selected_language])
        
        # Add cookies file if it exists
        if cookies_file.exists():
            cmd.extend(["--cookies", str(cookies_file)])
        
        # Run yt-dlp
        print("\nDownloading subtitles...")
        result = subprocess.run(cmd, check=True, capture_output=False)
        
        print("\nDownload completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\nError: Failed to download subtitles")
        print(f"yt-dlp exited with code {e.returncode}")
        sys.exit(1)
    except FileNotFoundError:
        print("\nError: yt-dlp not found. Please install it:")
        print("  pip install yt-dlp")
        print("  or")
        print("  brew install yt-dlp")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

