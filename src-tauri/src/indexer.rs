// src-tauri/src/indexer.rs
//! Scans the local filesystem to index source files into the CodeGraph store.
//! Runs synchronously and ignores standard build and hidden directories.

use crate::codegraph::CodeGraphStore;
use anyhow::Result;
use std::fs;
use std::path::Path;

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