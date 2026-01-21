#!/usr/bin/env -S cargo +nightly -Zscript
---
[dependencies]
tokio = { version = "1", features = ["full"] }
chromiumoxide = { version = "0.7", features = ["tokio-runtime"], default-features = false }
futures = "0.3"
regex = "1"
url = "2"
reqwest = { version = "0.12", features = ["stream"] }
indicatif = "0.17"
aes = "0.8"
cbc = "0.1"
ctrlc = "3.4"
---

//! Jable.tv Video Downloader (with Cloudflare bypass)
//!
//! Usage:
//!   jable_download.rs <jable_video_url> [output_name] [output_directory]
//!
//! Example:
//!   jable_download.rs "https://jable.tv/videos/miaa-033/"
//!   jable_download.rs "https://jable.tv/videos/miaa-033/" MIAA-033
//!   jable_download.rs "https://jable.tv/videos/miaa-033/" MIAA-033 ./downloads
//!
//! Requirements:
//!   - Google Chrome or Chromium installed
//!   - ffmpeg installed (for merging video segments)

use aes::cipher::{BlockDecryptMut, KeyIvInit};
use chromiumoxide::browser::{Browser, BrowserConfig};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;
use url::Url;

/// Global temp directory path for cleanup on Ctrl+C
static TEMP_DIR_PATH: OnceLock<PathBuf> = OnceLock::new();

/// Clean up temp directory (called on Ctrl+C, error, or normal exit)
/// If `verbose` is true, prints a message when cleanup succeeds.
fn cleanup_temp_dir_impl(verbose: bool) {
    if let Some(path) = TEMP_DIR_PATH.get() {
        if path.exists() {
            if fs::remove_dir_all(path).is_ok() && verbose {
                eprintln!("Cleaned up temp directory: {}", path.display());
            }
        }
    }
}

/// Clean up temp directory (silent version for normal exit)
fn cleanup_temp_dir() {
    cleanup_temp_dir_impl(false);
}

/// Clean up temp directory with message (for Ctrl+C)
fn cleanup_temp_dir_verbose() {
    cleanup_temp_dir_impl(true);
}

type Aes128CbcDec = cbc::Decryptor<aes::Aes128>;

/// Extract m3u8 URL from page HTML
fn extract_m3u8_url(html: &str) -> Option<String> {
    let re = Regex::new(r#"var\s+hlsUrl\s*=\s*'([^']+)'"#).ok()?;
    re.captures(html).map(|c| c[1].to_string())
}

/// Extract video title from page HTML
fn extract_title(html: &str) -> Option<String> {
    let re = Regex::new(r#"<h4>([^<]+)</h4>"#).ok()?;
    re.captures(html).map(|c| c[1].trim().to_string())
}

/// Extract video ID from URL (e.g., "miaa-033" from "https://jable.tv/videos/miaa-033/")
fn extract_video_id(url: &str) -> Option<String> {
    let re = Regex::new(r"/videos/([^/]+)/?").ok()?;
    re.captures(url).map(|c| c[1].to_uppercase())
}

/// HLS encryption info
struct HlsEncryption {
    key_url: String,
    iv: Option<[u8; 16]>,
}

/// Parsed m3u8 playlist info
struct M3u8Info {
    segments: Vec<String>,
    encryption: Option<HlsEncryption>,
}

/// Parse m3u8 content and extract segments and encryption info
fn parse_m3u8(content: &str, base_url: &str) -> M3u8Info {
    let base = base_url.rsplit_once('/').map(|(b, _)| b).unwrap_or("");
    
    // Parse encryption info: #EXT-X-KEY:METHOD=AES-128,URI="...",IV=0x...
    let encryption = if let Some(key_line) = content.lines().find(|l| l.contains("#EXT-X-KEY")) {
        // Extract URI
        let uri_re = Regex::new(r#"URI="([^"]+)""#).ok();
        let key_url = uri_re.and_then(|re| {
            re.captures(key_line).map(|c| {
                let uri = c[1].to_string();
                if uri.starts_with("http") {
                    uri
                } else {
                    format!("{}/{}", base, uri)
                }
            })
        });

        // Extract IV (optional)
        let iv_re = Regex::new(r"IV=0x([0-9a-fA-F]+)").ok();
        let iv = iv_re.and_then(|re| {
            re.captures(key_line).and_then(|c| {
                let hex_str = &c[1];
                let mut iv = [0u8; 16];
                // Pad to 32 hex chars (16 bytes)
                let padded = format!("{:0>32}", hex_str);
                for (i, chunk) in padded.as_bytes().chunks(2).enumerate() {
                    if i < 16 {
                        if let Ok(byte) = u8::from_str_radix(std::str::from_utf8(chunk).unwrap_or("00"), 16) {
                            iv[i] = byte;
                        }
                    }
                }
                Some(iv)
            })
        });

        key_url.map(|url| HlsEncryption { key_url: url, iv })
    } else {
        None
    };

    // Parse segment list
    let segments: Vec<String> = content
        .lines()
        .filter(|line| line.ends_with(".ts"))
        .map(|s| s.to_string())
        .collect();

    M3u8Info { segments, encryption }
}

/// Build full URL for a .ts segment
fn build_ts_url(m3u8_url: &str, ts_name: &str) -> String {
    let base = m3u8_url.rsplit_once('/').map(|(b, _)| b).unwrap_or("");
    format!("{}/{}", base, ts_name)
}

/// HTTP client with proper headers for jable.tv CDN
fn create_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        .timeout(Duration::from_secs(60))
        .build()
        .expect("Failed to create HTTP client")
}

/// Fetch m3u8 playlist
async fn fetch_m3u8(client: &reqwest::Client, url: &str) -> Result<String, String> {
    client
        .get(url)
        .header("Referer", "https://jable.tv/")
        .header("Origin", "https://jable.tv")
        .send()
        .await
        .map_err(|e| format!("Failed to fetch m3u8: {}", e))?
        .text()
        .await
        .map_err(|e| format!("Failed to read m3u8: {}", e))
}

/// Fetch encryption key
async fn fetch_key(client: &reqwest::Client, url: &str) -> Result<[u8; 16], String> {
    let bytes = client
        .get(url)
        .header("Referer", "https://jable.tv/")
        .header("Origin", "https://jable.tv")
        .send()
        .await
        .map_err(|e| format!("Failed to fetch key: {}", e))?
        .bytes()
        .await
        .map_err(|e| format!("Failed to read key: {}", e))?;

    if bytes.len() != 16 {
        return Err(format!("Invalid key length: {} (expected 16)", bytes.len()));
    }

    let mut key = [0u8; 16];
    key.copy_from_slice(&bytes);
    Ok(key)
}

/// Decrypt AES-128-CBC encrypted data
fn decrypt_aes128(data: &[u8], key: &[u8; 16], iv: &[u8; 16]) -> Result<Vec<u8>, String> {
    // Data must be multiple of 16 bytes for AES block cipher
    if data.len() % 16 != 0 {
        return Err("Data length is not a multiple of 16".to_string());
    }

    let mut buf = data.to_vec();
    let decryptor = Aes128CbcDec::new(key.into(), iv.into());
    
    // Decrypt with NoPadding - HLS/MPEG-TS can handle any extra bytes
    // Using NoPadding is more reliable for HLS streams
    let decrypted = decryptor
        .decrypt_padded_mut::<aes::cipher::block_padding::NoPadding>(&mut buf)
        .map_err(|e| format!("Decryption failed: {}", e))?;

    Ok(decrypted.to_vec())
}

/// Generate IV from segment sequence number (used when IV not specified in m3u8)
fn sequence_to_iv(seq: u32) -> [u8; 16] {
    let mut iv = [0u8; 16];
    iv[12..16].copy_from_slice(&seq.to_be_bytes());
    iv
}

/// Download and decrypt a single segment
async fn download_segment(
    client: &reqwest::Client,
    url: &str,
    output_path: &std::path::Path,
    key: Option<&[u8; 16]>,
    iv: &[u8; 16],
    max_retries: usize,
) -> Result<(), String> {
    let mut last_error = String::new();

    for attempt in 0..max_retries {
        if attempt > 0 {
            tokio::time::sleep(Duration::from_millis(500 * attempt as u64)).await;
        }

        match client
            .get(url)
            .header("Referer", "https://jable.tv/")
            .header("Origin", "https://jable.tv")
            .send()
            .await
        {
            Ok(response) => match response.bytes().await {
                Ok(bytes) => {
                    // Decrypt if key is provided
                    let data = if let Some(k) = key {
                        match decrypt_aes128(&bytes, k, iv) {
                            Ok(decrypted) => decrypted,
                            Err(e) => {
                                last_error = format!("Decryption failed: {}", e);
                                continue;
                            }
                        }
                    } else {
                        bytes.to_vec()
                    };

                    // Write to file
                    if let Ok(mut file) = File::create(output_path) {
                        if file.write_all(&data).is_ok() {
                            return Ok(());
                        }
                    }
                    last_error = "Failed to write file".to_string();
                }
                Err(e) => last_error = format!("Failed to read bytes: {}", e),
            },
            Err(e) => last_error = format!("Request failed: {}", e),
        }
    }

    Err(last_error)
}

/// Merge all .ts segments into final video using ffmpeg
fn merge_segments(temp_dir: &Path, segments: &[String], output_path: &Path) -> Result<(), String> {
    // Create file list for ffmpeg
    let list_path = temp_dir.join("filelist.txt");
    let mut list_file =
        File::create(&list_path).map_err(|e| format!("Failed to create file list: {}", e))?;

    for seg in segments {
        writeln!(list_file, "file '{}'", seg)
            .map_err(|e| format!("Failed to write file list: {}", e))?;
    }
    drop(list_file);

    // Convert output path to absolute path
    let output_abs = if output_path.is_absolute() {
        output_path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| format!("Failed to get current dir: {}", e))?
            .join(output_path)
    };

    // Run ffmpeg to merge
    let status = Command::new("ffmpeg")
        .current_dir(temp_dir)
        .args([
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            "filelist.txt",
            "-c",
            "copy",
            "-y",
            output_abs
                .to_str()
                .ok_or_else(|| "Output path contains invalid UTF-8 characters".to_string())?,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map_err(|e| format!("Failed to run ffmpeg: {}. Is ffmpeg installed?", e))?;

    if !status.success() {
        return Err("ffmpeg failed to merge segments".to_string());
    }

    Ok(())
}

/// Fetch page HTML using Chrome (bypasses Cloudflare)
async fn fetch_page_with_browser(url: &str) -> Result<String, String> {
    println!("      Starting Chrome (a window will appear, please don't close it)...");

    // Use headed mode with anti-detection measures
    let config = BrowserConfig::builder()
        .no_sandbox()
        .window_size(1280, 720)
        .with_head() // Use headed mode - less likely to be detected
        .arg("--disable-blink-features=AutomationControlled")
        .arg("--disable-infobars")
        .arg("--disable-dev-shm-usage")
        .arg("--disable-gpu")
        .arg("--lang=en-US,en")
        .build()
        .map_err(|e| format!("Failed to build browser config: {}", e))?;

    let (browser, mut handler) = Browser::launch(config)
        .await
        .map_err(|e| format!("Failed to launch browser: {}", e))?;

    // Spawn handler task
    let handle = tokio::spawn(async move {
        while handler.next().await.is_some() {}
    });

    let page = browser
        .new_page("about:blank")
        .await
        .map_err(|e| format!("Failed to create page: {}", e))?;

    println!("      Navigating to page (waiting for Cloudflare verification)...");

    // Navigate to the URL
    page.goto(url)
        .await
        .map_err(|e| format!("Failed to navigate: {}", e))?;

    // Wait for Cloudflare challenge to complete
    // Check periodically if hlsUrl appears in the page
    let mut html = String::new();
    let mut found_hls_url = false;
    for i in 0..60 {
        tokio::time::sleep(Duration::from_secs(1)).await;

        html = page
            .content()
            .await
            .map_err(|e| format!("Failed to get page content: {}", e))?;

        if html.contains("hlsUrl") {
            println!("      Verification passed!");
            found_hls_url = true;
            break;
        }

        // Still on Cloudflare challenge page
        if html.contains("Just a moment") || html.contains("challenge-platform") {
            if i % 10 == 0 && i > 0 {
                println!("      Waiting for Cloudflare verification... ({}s)", i);
            }
        }
    }

    if !found_hls_url {
        return Err("Timeout: Could not find video URL on page (Cloudflare verification may have failed or page structure changed)".to_string());
    }

    // Close browser
    drop(page);
    drop(browser);
    handle.abort();

    Ok(html)
}

/// Clean filename by removing invalid characters
fn clean_filename(name: &str) -> String {
    name.chars()
        .map(|c| match c {
            '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|' => '_',
            _ => c,
        })
        .collect()
}

fn print_usage(program: &str) {
    eprintln!("Jable.tv Video Downloader (with Cloudflare bypass)");
    eprintln!();
    eprintln!("Usage: {} <jable_video_url> [output_name] [output_directory]", program);
    eprintln!();
    eprintln!("Arguments:");
    eprintln!("  jable_video_url   The jable.tv video page URL");
    eprintln!("  output_name       Output filename without extension (optional)");
    eprintln!("  output_directory  Output directory (default: ./output_dir)");
    eprintln!();
    eprintln!("Example:");
    eprintln!(
        "  {} \"https://jable.tv/videos/miaa-033/\"",
        program
    );
    eprintln!(
        "  {} \"https://jable.tv/videos/miaa-033/\" MIAA-033",
        program
    );
    eprintln!(
        "  {} \"https://jable.tv/videos/miaa-033/\" MIAA-033 ./downloads",
        program
    );
    eprintln!();
    eprintln!("Requirements:");
    eprintln!("  - Google Chrome or Chromium installed");
    eprintln!("  - ffmpeg installed");
}

#[tokio::main]
async fn main() -> Result<(), String> {
    // Set up Ctrl+C handler for cleanup
    ctrlc::set_handler(move || {
        eprintln!("\nInterrupted! Cleaning up...");
        cleanup_temp_dir_verbose();
        std::process::exit(130); // 128 + SIGINT (2)
    })
    .expect("Error setting Ctrl+C handler");

    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let video_url = &args[1];

    // Validate URL using proper URL parsing
    let parsed_url =
        Url::parse(video_url).map_err(|e| format!("Invalid URL format: {}", e))?;

    if parsed_url.host_str() != Some("jable.tv") {
        return Err("Invalid URL. Host must be jable.tv".to_string());
    }

    if !parsed_url.path().starts_with("/videos/") {
        return Err("Invalid URL. Path must start with /videos/".to_string());
    }

    // Step 1: Fetch video page with headless browser
    println!("[1/4] Fetching video page...");
    let html = fetch_page_with_browser(video_url).await?;

    // Extract m3u8 URL
    let m3u8_url = extract_m3u8_url(&html).ok_or("Failed to extract m3u8 URL from page")?;

    // Extract title for filename
    let title = extract_title(&html);
    let video_id = extract_video_id(video_url).unwrap_or_else(|| "video".to_string());

    // Determine output directory (default to "output_dir" folder)
    let output_dir = if args.len() >= 4 {
        Path::new(&args[3]).to_path_buf()
    } else {
        Path::new("output_dir").to_path_buf()
    };
    fs::create_dir_all(&output_dir).map_err(|e| format!("Failed to create output dir: {}", e))?;

    let output_name = if args.len() >= 3 {
        clean_filename(&args[2])
    } else if let Some(t) = &title {
        clean_filename(t)
    } else {
        video_id.clone()
    };
    let output_filename = format!("{}.mp4", output_name);

    println!("      Title: {}", title.as_deref().unwrap_or(&video_id));

    let client = create_client();

    // Step 2: Fetch and parse m3u8
    println!("[2/4] Parsing m3u8 playlist...");
    let m3u8_content = fetch_m3u8(&client, &m3u8_url).await?;
    let m3u8_info = parse_m3u8(&m3u8_content, &m3u8_url);

    if m3u8_info.segments.is_empty() {
        return Err("No video segments found in m3u8".to_string());
    }

    println!("      Found {} video segments", m3u8_info.segments.len());

    // Fetch encryption key if stream is encrypted
    let encryption_key = if let Some(ref enc) = m3u8_info.encryption {
        println!("      Stream is encrypted, fetching key...");
        Some(fetch_key(&client, &enc.key_url).await?)
    } else {
        None
    };

    // Create temp directory
    let temp_dir = env::temp_dir().join(format!("jable_download_{}", std::process::id()));
    // Store path for cleanup on Ctrl+C
    TEMP_DIR_PATH.set(temp_dir.clone()).ok();
    println!("      Temp directory: {}", temp_dir.display());
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir).map_err(|e| format!("Failed to clean temp dir: {}", e))?;
    }
    fs::create_dir_all(&temp_dir).map_err(|e| format!("Failed to create temp dir: {}", e))?;

    // Step 3: Download all segments with progress bar
    println!("[3/4] Downloading video segments...");

    let pb = ProgressBar::new(m3u8_info.segments.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("      [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
            .unwrap()
            .progress_chars("█▓░"),
    );

    let failed_count = Arc::new(AtomicUsize::new(0));
    let client = Arc::new(client);
    let encryption_key = Arc::new(encryption_key);
    let default_iv = m3u8_info.encryption.as_ref().and_then(|e| e.iv);

    // Download segments with limited concurrency (8 workers)
    let semaphore = Arc::new(tokio::sync::Semaphore::new(8));

    let download_tasks: Vec<_> = m3u8_info.segments
        .iter()
        .enumerate()
        .map(|(idx, seg)| {
            let url = build_ts_url(&m3u8_url, seg);
            let seg_output_path = temp_dir.join(seg);
            let client = Arc::clone(&client);
            let failed_count = Arc::clone(&failed_count);
            let pb = pb.clone();
            let semaphore = Arc::clone(&semaphore);
            let encryption_key = Arc::clone(&encryption_key);

            // Use provided IV or generate from sequence number
            let iv = default_iv.unwrap_or_else(|| sequence_to_iv(idx as u32));

            tokio::spawn(async move {
                let _permit = semaphore
                    .acquire()
                    .await
                    .expect("Semaphore closed unexpectedly");
                
                let key_ref = encryption_key.as_ref().as_ref();
                if download_segment(&client, &url, &seg_output_path, key_ref, &iv, 3)
                    .await
                    .is_err()
                {
                    failed_count.fetch_add(1, Ordering::SeqCst);
                }
                pb.inc(1);
            })
        })
        .collect();

    // Wait for all downloads
    for task in download_tasks {
        if let Err(e) = task.await {
            if e.is_panic() {
                eprintln!("      Warning: download task panicked");
            }
        }
    }

    pb.finish();

    let failed = failed_count.load(Ordering::SeqCst);
    if failed > 0 {
        eprintln!("      Warning: {} segments failed to download", failed);
        if failed > m3u8_info.segments.len() / 10 {
            cleanup_temp_dir();
            return Err("Too many segments failed to download".to_string());
        }
    }

    // Step 4: Merge segments
    println!("[4/4] Merging video...");
    let output_path = output_dir.join(&output_filename);
    let merge_result = merge_segments(&temp_dir, &m3u8_info.segments, &output_path);

    // Cleanup temp directory (always, regardless of merge success)
    cleanup_temp_dir();

    // Check merge result after cleanup
    merge_result?;

    println!("      Output: {}", output_path.display());
    println!("Done!");

    Ok(())
}
