#!/usr/bin/env -S cargo +nightly -Zscript
---
[dependencies]
---

//! A simple file size statistics tool

use std::env;
use std::fs;
use std::path::Path;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: file_size <file_or_directory_path>");
        process::exit(1);
    }

    let target = Path::new(&args[1]);

    if !target.exists() {
        eprintln!("Error: Path '{}' does not exist", target.display());
        process::exit(1);
    }

    if target.is_file() {
        match fs::metadata(target) {
            Ok(metadata) => {
                let size = metadata.len();
                println!(
                    "File size: {} bytes ({:.2} KB)",
                    format_with_commas(size),
                    size as f64 / 1024.0
                );
            }
            Err(e) => {
                eprintln!("Error: Cannot read file metadata: {}", e);
                process::exit(1);
            }
        }
    } else if target.is_dir() {
        match calculate_dir_size(target) {
            Ok(total_size) => {
                println!(
                    "Directory total size: {} bytes ({:.2} KB)",
                    format_with_commas(total_size),
                    total_size as f64 / 1024.0
                );
            }
            Err(e) => {
                eprintln!("Error: Cannot calculate directory size: {}", e);
                process::exit(1);
            }
        }
    } else {
        eprintln!("Error: '{}' is not a file or directory", target.display());
        process::exit(1);
    }
}

fn calculate_dir_size(path: &Path) -> Result<u64, std::io::Error> {
    let mut total_size: u64 = 0;

    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();

        if entry_path.is_file() {
            total_size += entry.metadata()?.len();
        } else if entry_path.is_dir() {
            total_size += calculate_dir_size(&entry_path)?;
        }
    }

    Ok(total_size)
}

fn format_with_commas(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();
    
    for (i, c) in chars.iter().enumerate() {
        if i > 0 && (chars.len() - i) % 3 == 0 {
            result.push(',');
        }
        result.push(*c);
    }
    
    result
}

