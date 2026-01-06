#!/usr/bin/env -S cargo +nightly -Zscript
---
[dependencies]
reqwest = { version = "0.12", features = ["blocking", "multipart", "rustls-tls"], default-features = false }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
dotenvy = "0.15"
md-5 = "0.10"
---

//! A tool to transcribe audio files to text using Groq Whisper API

use std::env;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::thread;
use std::time::{Duration, SystemTime};

use md5::{Md5, Digest};
use reqwest::blocking::Client;
use reqwest::blocking::multipart::{Form, Part};
use serde::Deserialize;

// Groq Whisper API limits
const MAX_FILE_SIZE_MB: u64 = 25;
const GROQ_API_URL: &str = "https://api.groq.com/openai/v1/audio/transcriptions";

// Supported audio formats per Groq API
const AUDIO_EXTENSIONS: &[&str] = &["mp3", "wav", "m4a", "flac", "ogg", "webm"];

/// Segment from Groq verbose_json response
#[derive(Debug, Deserialize)]
struct Segment {
    start: f64,
    end: f64,
    text: String,
}

/// Groq verbose_json response structure
#[derive(Debug, Deserialize)]
struct VerboseJsonResponse {
    segments: Vec<Segment>,
}

/// RAII guard for temporary file cleanup
struct TempFileGuard {
    path: Option<PathBuf>,
}

impl TempFileGuard {
    fn new(path: PathBuf) -> Self {
        TempFileGuard { path: Some(path) }
    }

    fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }
}

impl Drop for TempFileGuard {
    fn drop(&mut self) {
        if let Some(ref path) = self.path {
            if path.exists() {
                if let Err(e) = fs::remove_file(path) {
                    eprintln!("Warning: Failed to clean up temporary file: {}", e);
                } else {
                    println!("Cleaned up temporary file");
                }
            }
        }
    }
}

/// Convert seconds to VTT timestamp format (HH:MM:SS.mmm)
fn format_timestamp(seconds: f64) -> String {
    let total_secs = seconds.floor() as u64;
    let hours = total_secs / 3600;
    let minutes = (total_secs % 3600) / 60;
    let secs = total_secs % 60;
    let millis = ((seconds.fract()) * 1000.0) as u64;
    format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, secs, millis)
}

/// Convert Groq verbose_json response to WebVTT format
fn convert_to_vtt(segments: &[Segment]) -> String {
    let mut lines = vec!["WEBVTT".to_string(), String::new()];

    for (i, segment) in segments.iter().enumerate() {
        let text = segment.text.trim();
        if !text.is_empty() {
            // Add cue number (optional but common in VTT)
            lines.push((i + 1).to_string());
            // Add timestamp line
            lines.push(format!(
                "{} --> {}",
                format_timestamp(segment.start),
                format_timestamp(segment.end)
            ));
            // Add text
            lines.push(text.to_string());
            // Add blank line between cues
            lines.push(String::new());
        }
    }

    lines.join("\n")
}

/// Find the latest modified audio file in the directory
fn find_latest_audio_file(directory: &Path) -> Result<PathBuf, String> {
    let entries = fs::read_dir(directory)
        .map_err(|e| format!("Cannot read directory '{}': {}", directory.display(), e))?;

    let mut audio_files: Vec<(PathBuf, SystemTime)> = Vec::new();

    for entry in entries {
        let entry = entry.map_err(|e| format!("Cannot read entry: {}", e))?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                let ext_lower = ext.to_string_lossy().to_lowercase();
                if AUDIO_EXTENSIONS.contains(&ext_lower.as_str()) {
                    if let Ok(metadata) = fs::metadata(&path) {
                        if let Ok(modified) = metadata.modified() {
                            audio_files.push((path, modified));
                        }
                    }
                }
            }
        }
    }

    if audio_files.is_empty() {
        return Err(format!("No audio files found in {}", directory.display()));
    }

    // Return the file with the latest modification time
    audio_files.sort_by(|a, b| b.1.cmp(&a.1));
    Ok(audio_files.into_iter().next().unwrap().0)
}

/// Check if filename contains only ASCII characters
fn is_ascii_safe(filename: &str) -> bool {
    filename.is_ascii()
}

/// Create a temporary file with ASCII-safe filename from source file
fn create_ascii_safe_temp_file(source_path: &Path) -> Result<PathBuf, String> {
    // Generate ASCII-safe filename: use stem hash + extension
    let stem = source_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio");

    let mut hasher = Md5::new();
    hasher.update(stem.as_bytes());
    let result = hasher.finalize();
    let stem_hash = format!("{:x}", result);
    let stem_hash = &stem_hash[..8.min(stem_hash.len())];

    let extension = source_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("mp3");

    let safe_filename = format!("audio_{}.{}", stem_hash, extension);

    // Create temporary file in same directory as source
    let temp_path = source_path.parent().unwrap_or(Path::new(".")).join(&safe_filename);

    // Copy source file to temporary file
    fs::copy(source_path, &temp_path)
        .map_err(|e| format!("Failed to create temporary file: {}", e))?;

    Ok(temp_path)
}

/// Check if HTTP status code is retryable
fn is_retryable_status(status_code: u16) -> bool {
    matches!(status_code, 429 | 500 | 502 | 503 | 504)
}

/// Transcribe audio file using Groq Whisper API
fn transcribe_audio_with_groq(
    audio_path: &Path,
    api_key: &str,
    output_dir: &Path,
) -> Result<PathBuf, String> {
    println!("Loading audio file: {}", audio_path.display());

    let metadata = fs::metadata(audio_path)
        .map_err(|e| format!("Cannot read file metadata: {}", e))?;

    let file_size_bytes = metadata.len();
    let file_size_mb = file_size_bytes as f64 / (1024.0 * 1024.0);
    println!("File size: {:.2} MB", file_size_mb);

    // Check file size limit
    if file_size_bytes > MAX_FILE_SIZE_MB * 1024 * 1024 {
        return Err(format!(
            "File size ({:.2} MB) exceeds Groq's limit of {} MB. Please compress or split the audio file.",
            file_size_mb, MAX_FILE_SIZE_MB
        ));
    }

    // Check if filename is ASCII-safe, create temp file if not
    let filename = audio_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");

    let temp_guard: Option<TempFileGuard>;
    let upload_path: &Path;

    if !is_ascii_safe(filename) {
        println!("Filename contains non-ASCII characters, creating temporary file with ASCII-safe name...");
        let temp_path = create_ascii_safe_temp_file(audio_path)?;
        println!("Using temporary file: {}", temp_path.file_name().unwrap_or_default().to_string_lossy());
        temp_guard = Some(TempFileGuard::new(temp_path));
        upload_path = temp_guard.as_ref().unwrap().path().unwrap();
    } else {
        temp_guard = None;
        upload_path = audio_path;
    }

    // Retry logic with exponential backoff
    let max_retries = 3;
    let retry_delays = [10, 30]; // seconds between retries

    println!("\nTranscribing audio with Groq Whisper API...");

    let client = Client::builder()
        .timeout(Duration::from_secs(300)) // 5 minute timeout for large files
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let mut last_error: Option<String> = None;
    let mut response_text: Option<String> = None;

    for attempt in 0..max_retries {
        // Read file contents
        let mut file_contents = Vec::new();
        let mut file = File::open(upload_path)
            .map_err(|e| format!("Cannot open file: {}", e))?;
        file.read_to_end(&mut file_contents)
            .map_err(|e| format!("Cannot read file: {}", e))?;

        let upload_filename = upload_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("audio.mp3")
            .to_string();

        // Build multipart form
        let file_part = Part::bytes(file_contents)
            .file_name(upload_filename)
            .mime_str("audio/mpeg")
            .map_err(|e| format!("Failed to create file part: {}", e))?;

        let form = Form::new()
            .part("file", file_part)
            .text("model", "whisper-large-v3-turbo")
            .text("response_format", "verbose_json");

        let result = client
            .post(GROQ_API_URL)
            .header("Authorization", format!("Bearer {}", api_key))
            .multipart(form)
            .send();

        match result {
            Ok(response) => {
                let status = response.status();

                if status.is_success() {
                    response_text = Some(
                        response
                            .text()
                            .map_err(|e| format!("Failed to read response: {}", e))?,
                    );
                    break;
                }

                let status_code = status.as_u16();
                let error_body = response.text().unwrap_or_else(|_| "No error message".to_string());

                if attempt < max_retries - 1 && is_retryable_status(status_code) {
                    let delay = retry_delays.get(attempt).copied().unwrap_or(30);
                    println!("\n⚠ Attempt {} failed: HTTP {}", attempt + 1, status_code);
                    println!("  Retrying in {} seconds...", delay);
                    thread::sleep(Duration::from_secs(delay as u64));
                } else {
                    let truncated_error = if error_body.len() > 500 {
                        &error_body[..500]
                    } else {
                        &error_body
                    };
                    return Err(format!(
                        "Groq API error (HTTP {}): {}",
                        status_code, truncated_error
                    ));
                }
            }
            Err(e) => {
                last_error = Some(format!("{}", e));
                if attempt < max_retries - 1 {
                    let delay = retry_delays.get(attempt).copied().unwrap_or(30);
                    println!("\n⚠ Attempt {} failed: {}", attempt + 1, e);
                    println!("  Retrying in {} seconds...", delay);
                    thread::sleep(Duration::from_secs(delay as u64));
                } else {
                    return Err(format!(
                        "Request failed after {} attempts: {}",
                        max_retries,
                        last_error.unwrap_or_else(|| "Unknown error".to_string())
                    ));
                }
            }
        }
    }

    // temp_guard will be dropped here, cleaning up the temp file

    let response_text = response_text.ok_or_else(|| {
        format!(
            "Failed to transcribe after {} attempts",
            max_retries
        )
    })?;

    // Parse JSON response
    let verbose_json: VerboseJsonResponse = serde_json::from_str(&response_text)
        .map_err(|e| format!("Failed to parse API response: {}", e))?;

    // Convert to VTT
    let vtt_content = convert_to_vtt(&verbose_json.segments);

    // Generate output filename
    let stem = audio_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("audio");
    let output_filename = format!("{}.vtt", stem);
    let output_path = output_dir.join(&output_filename);

    // Save transcript to file
    let mut output_file = File::create(&output_path)
        .map_err(|e| format!("Cannot create output file: {}", e))?;
    output_file
        .write_all(vtt_content.as_bytes())
        .map_err(|e| format!("Cannot write output file: {}", e))?;

    Ok(output_path)
}

fn main() {
    // Load environment variables from .env file
    let _ = dotenvy::dotenv();

    // Parse arguments
    let args: Vec<String> = env::args().collect();

    let input_file: Option<PathBuf> = args.get(1).map(PathBuf::from);
    let output_dir: PathBuf = args
        .get(2)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("output_dir"));

    // Determine input file
    let input_file = match input_file {
        Some(path) => {
            if !path.exists() {
                eprintln!("Error: File '{}' does not exist", path.display());
                process::exit(1);
            }
            if !path.is_file() {
                eprintln!("Error: '{}' is not a file", path.display());
                process::exit(1);
            }
            path
        }
        None => {
            let input_dir = PathBuf::from("input_dir");
            if !input_dir.exists() {
                eprintln!("Error: Input directory '{}' does not exist", input_dir.display());
                process::exit(1);
            }
            match find_latest_audio_file(&input_dir) {
                Ok(path) => {
                    println!("Using latest audio file: {}", path.display());
                    path
                }
                Err(e) => {
                    eprintln!("Error: {}", e);
                    process::exit(1);
                }
            }
        }
    };

    // Ensure output directory exists
    if let Err(e) = fs::create_dir_all(&output_dir) {
        eprintln!("Error: Cannot create output directory: {}", e);
        process::exit(1);
    }

    // Get API key from environment variable (loaded from .env file)
    let api_key = match env::var("GROQ_API_KEY") {
        Ok(key) if !key.is_empty() => key,
        _ => {
            eprintln!("Error: GROQ_API_KEY environment variable is not set");
            eprintln!();
            eprintln!("Please set your Groq API key:");
            eprintln!("  Option 1: Create a .env file with: GROQ_API_KEY='your-api-key-here'");
            eprintln!("  Option 2: Export environment variable: export GROQ_API_KEY='your-api-key-here'");
            eprintln!();
            eprintln!("You can get an API key from: https://console.groq.com/");
            process::exit(1);
        }
    };

    println!("Input file: {}", input_file.display());
    println!(
        "Output directory: {}",
        fs::canonicalize(&output_dir)
            .unwrap_or(output_dir.clone())
            .display()
    );
    println!();

    match transcribe_audio_with_groq(&input_file, &api_key, &output_dir) {
        Ok(output_path) => {
            println!(
                "\nSuccess! Transcription saved to: {}",
                fs::canonicalize(&output_path)
                    .unwrap_or(output_path)
                    .display()
            );
        }
        Err(e) => {
            eprintln!("\nError: {}", e);
            process::exit(1);
        }
    }
}

