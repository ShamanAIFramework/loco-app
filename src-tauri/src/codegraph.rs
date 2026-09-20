// src-tauri/src/codegraph.rs
//! Persistence layer for the project code graph.
//! Maintains source files and symbols metadata for semantic context.
//! Operates synchronously inside a dedicated SQLite connection.

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;
use uuid::Uuid;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Project {
    pub id: String,
    pub name: String,
    pub root_path: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceFile {
    pub id: String,
    pub project_id: String,
    pub path: String,
    pub checksum: Option<String>,
    pub last_indexed: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Symbol {
    pub id: String,
    pub file_id: String,
    pub name: String,
    pub kind: String,
    pub line: u32,
}

fn now_str() -> String {
    Utc::now().to_rfc3339()
}

fn parse_dt(s: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .context("Failed to parse datetime")
}

fn parse_opt_dt(s: Option<String>) -> Option<DateTime<Utc>> {
    s.and_then(|str_val| parse_dt(&str_val).ok())
}

/// Thread-safe SQLite repository for codebase topology.
pub struct CodeGraphStore {
    conn: Mutex<Connection>,
}

impl CodeGraphStore {
    /// Opens the DB and initializes schema idempotently.
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path).context("Failed to open codegraph DB")?;

        // This duplication of PRAGMAs ensures module independence
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;"
        )?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                root_path TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS files (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                path TEXT NOT NULL,
                checksum TEXT,
                last_indexed TEXT,
                UNIQUE(project_id, path)
             );
             CREATE TABLE IF NOT EXISTS symbols (
                id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                line INTEGER NOT NULL,
                UNIQUE(file_id, name, kind, line)
             );
             CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbols(file_id);
             CREATE INDEX IF NOT EXISTS idx_files_project_id ON files(project_id);"
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Registers a new project, asserting it exists, or returns the existing matching one.
    pub fn register_project(&self, name: &str, root_path: &Path) -> Result<Project> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        
        let canonical_path = std::fs::canonicalize(root_path)
            .context("Failed to resolve absolute path for project root. Does it exist?")?;
        
        let path_str = canonical_path.to_string_lossy().to_string();
        
        // Check for existing
        if let Ok(existing) = guard.query_row(
            "SELECT id, name, created_at FROM projects WHERE root_path = ?1",
            params![path_str],
            |row| {
                Ok(Project {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    root_path: path_str.clone(),
                    created_at: parse_dt(&row.get::<_, String>(2)?).unwrap_or_default(),
                })
            }
        ) {
            return Ok(existing);
        }

        let id = Uuid::new_v4().to_string();
        let now = now_str();

        guard.execute(
            "INSERT INTO projects (id, name, root_path, created_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, name, path_str, now],
        )?;

        Ok(Project {
            id,
            name: name.to_string(),
            root_path: path_str,
            created_at: parse_dt(&now)?,
        })
    }

    /// Removes a project entirely (cascades to files and symbols).
    pub fn remove_project(&self, project_id: &str) -> Result<bool> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let count = guard.execute("DELETE FROM projects WHERE id = ?1", params![project_id])?;
        Ok(count > 0)
    }

    /// Upserts a source file into the system, maintaining its stable ID.
    pub fn upsert_file(&self, project_id: &str, path: &str, checksum: Option<&str>) -> Result<SourceFile> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let generated_id = Uuid::new_v4().to_string();
        let now = now_str();

        guard.query_row(
            r#"INSERT INTO files (id, project_id, path, checksum, last_indexed)
               VALUES (?1, ?2, ?3, ?4, ?5)
               ON CONFLICT(project_id, path) DO UPDATE SET
                   checksum = excluded.checksum,
                   last_indexed = excluded.last_indexed
               RETURNING id, project_id, path, checksum, last_indexed"#,
            params![generated_id, project_id, path, checksum, now],
            |row| {
                Ok(SourceFile {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    path: row.get(2)?,
                    checksum: row.get(3)?,
                    last_indexed: parse_opt_dt(row.get(4)?),
                })
            },
        ).context("Failed to upsert source file")
    }

    /// Upserts a parsed symbol for a specific file.
    pub fn upsert_symbol(&self, file_id: &str, name: &str, kind: &str, line: u32) -> Result<Symbol> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let generated_id = Uuid::new_v4().to_string();

        guard.query_row(
            r#"INSERT INTO symbols (id, file_id, name, kind, line)
               VALUES (?1, ?2, ?3, ?4, ?5)
               ON CONFLICT(file_id, name, kind, line) DO UPDATE SET
                   id = id -- Noop update to return the existing row
               RETURNING id, file_id, name, kind, line"#,
            params![generated_id, file_id, name, kind, line],
            |row| {
                Ok(Symbol {
                    id: row.get(0)?,
                    file_id: row.get(1)?,
                    name: row.get(2)?,
                    kind: row.get(3)?,
                    line: row.get(4)?,
                })
            },
        ).context("Failed to upsert symbol")
    }

    /// Deletes all symbols for a file, allowing fresh re-indexing.
    pub fn clear_file_symbols(&self, file_id: &str) -> Result<usize> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let count = guard.execute("DELETE FROM symbols WHERE file_id = ?1", params![file_id])?;
        Ok(count)
    }

    /// Lists all projects sorted alphabetically.
    pub fn list_projects(&self) -> Result<Vec<Project>> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let mut stmt = guard.prepare("SELECT id, name, root_path, created_at FROM projects ORDER BY name ASC")?;
        
        let res = stmt.query_map([], |row| {
            Ok(Project {
                id: row.get(0)?,
                name: row.get(1)?,
                root_path: row.get(2)?,
                created_at: parse_dt(&row.get::<_, String>(3)?).unwrap_or_default(),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
        Ok(res)
    }

    /// Retrieves all registered files for a given project.
    pub fn project_files(&self, project_id: &str) -> Result<Vec<SourceFile>> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let mut stmt = guard.prepare("SELECT id, project_id, path, checksum, last_indexed FROM files WHERE project_id = ?1 ORDER BY path ASC")?;
        
        let res = stmt.query_map(params![project_id], |row| {
            Ok(SourceFile {
                id: row.get(0)?,
                project_id: row.get(1)?,
                path: row.get(2)?,
                checksum: row.get(3)?,
                last_indexed: parse_opt_dt(row.get(4)?),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
        Ok(res)
    }

    /// Retrieves parsed symbols for a given file sorted by line number.
    pub fn file_symbols(&self, file_id: &str) -> Result<Vec<Symbol>> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let mut stmt = guard.prepare("SELECT id, file_id, name, kind, line FROM symbols WHERE file_id = ?1 ORDER BY line ASC, name ASC")?;
        
        let res = stmt.query_map(params![file_id], |row| {
            Ok(Symbol {
                id: row.get(0)?,
                file_id: row.get(1)?,
                name: row.get(2)?,
                kind: row.get(3)?,
                line: row.get(4)?,
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
        Ok(res)
    }

    /// Returns files matching the query snippet (path or filename).
    pub fn search_files(&self, project_id: &str, query: &str) -> Result<Vec<SourceFile>> {
        let guard = self.conn.lock().map_err(|_| anyhow::anyhow!("Mutex poisoned"))?;
        
        // Wrap the query in wildcards for substring matching
        let like_query = format!("%{}%", query);
        let mut stmt = guard.prepare(
            "SELECT id, project_id, path, checksum, last_indexed 
             FROM files 
             WHERE project_id = ?1 AND path LIKE ?2 
             ORDER BY path ASC LIMIT 20"
        )?;
        
        let res = stmt.query_map(rusqlite::params![project_id, like_query], |row| {
            Ok(SourceFile {
                id: row.get(0)?,
                project_id: row.get(1)?,
                path: row.get(2)?,
                checksum: row.get(3)?,
                last_indexed: parse_opt_dt(row.get(4)?),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
        
        Ok(res)
    }

    /// Removes a file from the index (e.g., when it is deleted on disk).
    pub fn remove_file(&self, project_id: &str, rel_path: &str) -> anyhow::Result<()> {
        let guard = self.conn.lock().map_err(|_| anyhow::anyhow!("Mutex poisoned"))?;
        guard.execute(
            "DELETE FROM files WHERE project_id = ?1 AND path = ?2",
            rusqlite::params![project_id, rel_path],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs;

    #[test]
    fn test_codegraph_crud() {
        let db_path = env::temp_dir().join(format!("{}.db", Uuid::new_v4()));
        let store = CodeGraphStore::new(&db_path).unwrap();
        
        // Setup valid test dir
        let test_dir = env::temp_dir().join("loco_test_proj");
        fs::create_dir_all(&test_dir).unwrap();
        
        let proj = store.register_project("Test", &test_dir).unwrap();
        let proj_dup = store.register_project("Test", &test_dir).unwrap();
        assert_eq!(proj.id, proj_dup.id);

        let file = store.upsert_file(&proj.id, "src/main.rs", Some("abc")).unwrap();
        let file_dup = store.upsert_file(&proj.id, "src/main.rs", Some("def")).unwrap();
        assert_eq!(file.id, file_dup.id);
        assert_eq!(file_dup.checksum.as_deref(), Some("def"));

        let _sym = store.upsert_symbol(&file.id, "main", "fn", 1).unwrap();
        let syms = store.file_symbols(&file.id).unwrap();
        assert_eq!(syms.len(), 1);

        store.clear_file_symbols(&file.id).unwrap();
        assert!(store.file_symbols(&file.id).unwrap().is_empty());

        store.remove_project(&proj.id).unwrap();
        assert!(store.project_files(&proj.id).unwrap().is_empty());

        let _ = fs::remove_file(db_path);
        let _ = fs::remove_dir_all(test_dir);
    }
}