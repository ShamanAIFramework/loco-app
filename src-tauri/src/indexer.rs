// src-tauri/src/indexer.rs
//! Scans the local filesystem to index source files into the CodeGraph store.
//! Runs synchronously and ignores standard build and hidden directories.

use crate::codegraph::CodeGraphStore;
use anyhow::Result;
use std::fs;
use std::path::Path;
use crate::AppState;
use notify_debouncer_mini::{new_debouncer, notify::RecursiveMode, Debouncer};
use notify_debouncer_mini::notify::RecommendedWatcher;
use std::time::Duration;
use tauri::Manager;

pub fn index_workspace(store: &CodeGraphStore, project_id: &str, root_path: &Path) -> Result<()> {
    log::info!("Indexing workspace: {:?}", root_path);
    walk_and_index(store, project_id, root_path, root_path)
}

fn walk_and_index(store: &CodeGraphStore, project_id: &str, current_dir: &Path, root_path: &Path) -> Result<()> {
    let entries = match fs::read_dir(current_dir) {
        Ok(e) => e,
        Err(_) => return Ok(()), // Ignore inaccessible directories safely
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        // Skip hidden files and common massive build directories
        if file_name_str.starts_with('.') 
            || file_name_str == "node_modules" 
            || file_name_str == "target" 
            || file_name_str == "dist" 
            || file_name_str == "build" 
        {
            continue;
        }

        if path.is_dir() {
            let _ = walk_and_index(store, project_id, &path, root_path);
        } else if path.is_file() {
            if let Ok(rel_path) = path.strip_prefix(root_path) {
                let rel_str = rel_path.to_string_lossy().to_string();
                // Upsert the file metadata idempotently
                let _ = store.upsert_file(project_id, &rel_str, None);
            }
        }
    }
    Ok(())
}

/// Spawns a background thread that listens to OS filesystem events and incrementally updates SQLite.
pub fn start_watcher(
    app: tauri::AppHandle,
    project_id: String,
    root_path: std::path::PathBuf,
) -> Result<Debouncer<RecommendedWatcher>> {
    
    // Create an MPSC channel to receive debounced events
    let (tx, rx) = std::sync::mpsc::channel();
    
    // Debounce events every 500ms to ignore editor temp-file spam
    let mut debouncer = new_debouncer(Duration::from_millis(500), tx)
        .map_err(|e| anyhow::anyhow!("Failed to create watcher: {}", e))?;
        
    debouncer
        .watcher()
        .watch(&root_path, RecursiveMode::Recursive)
        .map_err(|e| anyhow::anyhow!("Failed to watch path: {}", e))?;

    let root_clone = root_path.clone();

    // Spawn the background worker thread
    std::thread::spawn(move || {
        for res in rx {
            match res {
                Ok(events) => {
                    let state = app.state::<AppState>();
                    
                    for event in events {
                        let path = event.path;
                        let path_str = path.to_string_lossy();

                        // 1. Ignore build directories and hidden files
                        if path_str.contains("/.") 
                            || path_str.contains("\\.")
                            || path_str.contains("/node_modules/") 
                            || path_str.contains("\\node_modules\\") 
                            || path_str.contains("/target/") 
                            || path_str.contains("\\target\\") 
                        {
                            continue;
                        }

                        // 2. Calculate the relative path
                        if let Ok(rel_path) = path.strip_prefix(&root_clone) {
                            let rel_str = rel_path.to_string_lossy().to_string();

                            // 3. Incrementally update SQLite based on file existence
                            if path.exists() && path.is_file() {
                                // File was created or modified
                                let _ = state.codegraph.upsert_file(&project_id, &rel_str, None);
                            } else if !path.exists() {
                                // File was deleted
                                let _ = state.codegraph.remove_file(&project_id, &rel_str);
                            }
                        }
                    }
                }
                Err(e) => log::error!("Watch error: {:?}", e),
            }
        }
    });

    // Return the debouncer so it doesn't get dropped (which would stop the watching)
    Ok(debouncer)
}