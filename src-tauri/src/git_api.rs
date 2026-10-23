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
pub fn get_git_branch(workspace_path: &str) -> Result<String, String> {
    let output = Command::new("git")
        .current_dir(workspace_path)
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .output()
        .map_err(|e| e.to_string())?;

    if !output.status.success() {
        return Ok("HEAD".to_string());
    }

    let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(if branch.is_empty() { "HEAD".to_string() } else { branch })
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
    if diff.is_empty() {
        // If empty, file might be untracked or newly created. Diff against /dev/null (or NUL on Windows).
        let null_dev = if cfg!(windows) { "NUL" } else { "/dev/null" };
        let untracked_output = Command::new("git")
            .current_dir(workspace_path)
            .args(["diff", "--no-index", "--", null_dev, file_path])
            .output();

        if let Ok(uo) = untracked_output {
            let u_diff = String::from_utf8_lossy(&uo.stdout).to_string();
            if !u_diff.is_empty() {
                return Ok(u_diff);
            }
        }
    }

    Ok(diff)
}
