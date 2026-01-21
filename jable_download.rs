#!/usr/bin/env -S cargo +nightly -Zscript
---
[dependencies]
tokio = { version = "1", features = ["full"] }
chromiumoxide = { version = "0.7", features = ["tokio-runtime"], default-features = false }
futures = "0.3"
reqwest = { version = "0.12", features = ["stream"] }
regex = "1"
indicatif = "0.17"
tokio-stream = "0.1"
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

use chromiumoxide::browser::{Browser, BrowserConfig};
use futures::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use regex::Regex;
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

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
        while let Some(_) = handler.next().await {}
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
    for i in 0..60 {
        tokio::time::sleep(Duration::from_secs(1)).await;

        html = page
            .content()
            .await
            .map_err(|e| format!("Failed to get page content: {}", e))?;

        if html.contains("hlsUrl") {
            println!("      Verification passed!");
            break;
        }

        // Still on Cloudflare challenge page
        if html.contains("Just a moment") || html.contains("challenge-platform") {
            if i % 10 == 0 && i > 0 {
                println!("      Waiting for Cloudflare verification... ({}s)", i);
            }
            continue;
        }

        if i == 59 {
            return Err("Timeout: Cloudflare verification failed or page structure changed".to_string());
        }
    }

    // Close browser
    drop(page);
    drop(browser);
    handle.abort();

    Ok(html)
}

/// HTTP client with proper headers for jable.tv CDN
fn create_client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Safari/537.36")
        .timeout(Duration::from_secs(60))
        .build()
        .expect("Failed to create HTTP client")
}

/// Fetch and parse m3u8 playlist
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

/// Parse m3u8 content and extract .ts segment names
fn parse_m3u8(content: &str) -> Vec<String> {
    content
        .lines()
        .filter(|line| line.ends_with(".ts"))
        .map(String::from)
        .collect()
}

/// Build full URL for a .ts segment
fn build_ts_url(m3u8_url: &str, ts_name: &str) -> String {
    let base = m3u8_url.rsplit_once('/').map(|(b, _)| b).unwrap_or("");
    format!("{}/{}", base, ts_name)
}

/// Download a single .ts segment with retry
async fn download_segment(
    client: &reqwest::Client,
    url: &str,
    output_path: &Path,
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
                    if let Ok(mut file) = File::create(output_path) {
                        if file.write_all(&bytes).is_ok() {
                            return Ok(());
                        }
                    }
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
            output_abs.to_str().unwrap(),
        ])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| format!("Failed to run ffmpeg: {}. Is ffmpeg installed?", e))?;

    if !status.success() {
        return Err("ffmpeg failed to merge segments".to_string());
    }

    Ok(())
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
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let video_url = &args[1];

    // Validate URL
    if !video_url.contains("jable.tv/videos/") {
        return Err("Invalid URL. Must be a jable.tv video URL".to_string());
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
    let segments = parse_m3u8(&m3u8_content);

    if segments.is_empty() {
        return Err("No video segments found in m3u8".to_string());
    }

    println!("      Found {} video segments", segments.len());

    // Create temp directory in system temp location
    let temp_dir = env::temp_dir().join(format!("jable_download_{}", std::process::id()));
    if temp_dir.exists() {
        fs::remove_dir_all(&temp_dir).map_err(|e| format!("Failed to clean temp dir: {}", e))?;
    }
    fs::create_dir_all(&temp_dir).map_err(|e| format!("Failed to create temp dir: {}", e))?;
    println!("      Temp directory: {}", temp_dir.display());

    // Step 3: Download all segments with progress bar
    println!("[3/4] Downloading video segments...");

    let pb = ProgressBar::new(segments.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("      [{bar:40.cyan/blue}] {pos}/{len} ({percent}%)")
            .unwrap()
            .progress_chars("█▓░"),
    );

    let failed_count = Arc::new(AtomicUsize::new(0));
    let client = Arc::new(client);

    // Download segments with limited concurrency
    let semaphore = Arc::new(tokio::sync::Semaphore::new(8));

    let download_tasks: Vec<_> = segments
        .iter()
        .map(|seg| {
            let url = build_ts_url(&m3u8_url, seg);
            let seg_output_path = temp_dir.join(seg);
            let client = Arc::clone(&client);
            let failed_count = Arc::clone(&failed_count);
            let pb = pb.clone();
            let semaphore = Arc::clone(&semaphore);

            tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                if download_segment(&client, &url, &seg_output_path, 3)
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
        let _ = task.await;
    }

    pb.finish();

    let failed = failed_count.load(Ordering::SeqCst);
    if failed > 0 {
        eprintln!("      Warning: {} segments failed to download", failed);
        if failed > segments.len() / 10 {
            return Err("Too many segments failed to download".to_string());
        }
    }

    // Step 4: Merge segments
    println!("[4/4] Merging video...");
    let output_path = output_dir.join(&output_filename);
    merge_segments(&temp_dir, &segments, &output_path)?;

    // Cleanup
    fs::remove_dir_all(&temp_dir).ok();

    println!("      Output: {}", output_path.display());
    println!("Done!");

    Ok(())
}
