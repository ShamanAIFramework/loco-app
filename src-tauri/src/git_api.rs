// src-tauri/src/git_api.rs
//! Executes system Git commands to read workspace state.

use serde::Serialize;
use std::process::Command;

#[derive(Serialize)]
pub struct GitFile {
    pub path: String,
    pub status: String,
}

#[tauri::command]
pub fn get_git_status(workspace_path: &str) -> Result<Vec<GitFile>, String> {
    let output = Command::new("git")
        .current_dir(workspace_path)
        .args(["status", "--porcelain"])
        .output()
        .map_err(|e| e.to_string())?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut files = Vec::new();

    for line in stdout.lines() {
        if line.len() > 3 {
            // --porcelain format: " M path/to/file" or "?? path/to/untracked"
            let status = line[0..2].trim().to_string();
            let path = line[3..].trim().to_string();
            files.push(GitFile { path, status });
        }
    }

    Ok(files)
}

#[tauri::command]
pub fn get_git_diff(workspace_path: &str, file_path: &str) -> Result<String, String> {
    // Diff against HEAD to capture both staged and unstaged changes
    let output = Command::new("git")
        .current_dir(workspace_path)
        .args(["diff", "HEAD", "--", file_path])
        .output()
        .map_err(|e| e.to_string())?;

    let diff = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(diff)
}