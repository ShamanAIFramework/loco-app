// src-tauri/src/codegraph.rs
//! Persistence layer for the project code graph.
//! Maintains source files and symbols metadata for semantic context.
//! Operates synchronously inside a dedicated SQLite connection.

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
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
    pub index_version: i64,
    pub last_indexed: Option<DateTime<Utc>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Symbol {
    pub id: String,
    pub file_id: String,
    pub name: String,
    pub kind: String,
    pub line: u32,
    pub start_line: u32,
    pub end_line: u32,
    pub content_hash: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolEdge {
    pub id: String,
    pub from_symbol_id: String,
    pub target_name: String,
    pub kind: String,
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
             PRAGMA synchronous = NORMAL;",
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
                index_version INTEGER NOT NULL DEFAULT 0,
                last_indexed TEXT,
                UNIQUE(project_id, path)
             );
             CREATE TABLE IF NOT EXISTS symbols (
                id TEXT PRIMARY KEY,
                file_id TEXT NOT NULL REFERENCES files(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                line INTEGER NOT NULL,
                start_line INTEGER NOT NULL DEFAULT 0,
                end_line INTEGER NOT NULL DEFAULT 0,
                content_hash TEXT,
                UNIQUE(file_id, name, kind, start_line, end_line)
             );
             CREATE TABLE IF NOT EXISTS symbol_edges (
                id TEXT PRIMARY KEY,
                from_symbol_id TEXT NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
                target_name TEXT NOT NULL,
                kind TEXT NOT NULL,
                UNIQUE(from_symbol_id, target_name, kind)
             );
             CREATE INDEX IF NOT EXISTS idx_symbols_file_id ON symbols(file_id);
             CREATE INDEX IF NOT EXISTS idx_symbol_edges_from ON symbol_edges(from_symbol_id);
             CREATE INDEX IF NOT EXISTS idx_symbol_edges_target ON symbol_edges(target_name);
             CREATE INDEX IF NOT EXISTS idx_files_project_id ON files(project_id);",
        )?;

        migrate_schema(&conn)?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Registers a new project, asserting it exists, or returns the existing matching one.
    pub fn register_project(&self, name: &str, root_path: &Path) -> Result<Project> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;

        let mut canonical_path = std::fs::canonicalize(root_path)
            .context("Failed to resolve absolute path for project root. Does it exist?")?;

        if canonical_path.file_name() == Some(std::ffi::OsStr::new("src-tauri")) {
            if let Some(parent) = canonical_path.parent() {
                canonical_path = parent.to_path_buf();
            }
        }

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
            },
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
    pub fn upsert_file(
        &self,
        project_id: &str,
        path: &str,
        checksum: Option<&str>,
    ) -> Result<SourceFile> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let generated_id = Uuid::new_v4().to_string();
        let now = now_str();

        guard
            .query_row(
                r#"INSERT INTO files (id, project_id, path, checksum, index_version, last_indexed)
               VALUES (?1, ?2, ?3, ?4, 1, ?5)
               ON CONFLICT(project_id, path) DO UPDATE SET
                   index_version = CASE
                       WHEN files.checksum IS NOT excluded.checksum THEN files.index_version + 1
                       ELSE files.index_version
                   END,
                   checksum = excluded.checksum,
                   last_indexed = excluded.last_indexed
               RETURNING id, project_id, path, checksum, index_version, last_indexed"#,
                params![generated_id, project_id, path, checksum, now],
                |row| {
                    Ok(SourceFile {
                        id: row.get(0)?,
                        project_id: row.get(1)?,
                        path: row.get(2)?,
                        checksum: row.get(3)?,
                        index_version: row.get(4)?,
                        last_indexed: parse_opt_dt(row.get(5)?),
                    })
                },
            )
            .context("Failed to upsert source file")
    }

    /// Upserts a parsed symbol for a specific file.
    pub fn upsert_symbol(
        &self,
        file_id: &str,
        name: &str,
        kind: &str,
        start_line: u32,
        end_line: u32,
        content_hash: Option<&str>,
    ) -> Result<Symbol> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let generated_id = Uuid::new_v4().to_string();

        guard.query_row(
            r#"INSERT INTO symbols (id, file_id, name, kind, line, start_line, end_line, content_hash)
               VALUES (?1, ?2, ?3, ?4, ?5, ?5, ?6, ?7)
               ON CONFLICT(file_id, name, kind, line) DO UPDATE SET
                   start_line = excluded.start_line,
                   end_line = excluded.end_line,
                   content_hash = excluded.content_hash
               RETURNING id, file_id, name, kind, line, start_line, end_line, content_hash"#,
            params![generated_id, file_id, name, kind, start_line, end_line, content_hash],
            |row| {
                Ok(Symbol {
                    id: row.get(0)?,
                    file_id: row.get(1)?,
                    name: row.get(2)?,
                    kind: row.get(3)?,
                    line: row.get(4)?,
                    start_line: row.get(5)?,
                    end_line: row.get(6)?,
                    content_hash: row.get(7)?,
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

    pub fn upsert_symbol_edge(
        &self,
        from_symbol_id: &str,
        target_name: &str,
        kind: &str,
    ) -> Result<SymbolEdge> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let generated_id = Uuid::new_v4().to_string();

        guard
            .query_row(
                r#"INSERT INTO symbol_edges (id, from_symbol_id, target_name, kind)
                   VALUES (?1, ?2, ?3, ?4)
                   ON CONFLICT(from_symbol_id, target_name, kind) DO UPDATE SET
                       target_name = excluded.target_name
                   RETURNING id, from_symbol_id, target_name, kind"#,
                params![generated_id, from_symbol_id, target_name, kind],
                |row| {
                    Ok(SymbolEdge {
                        id: row.get(0)?,
                        from_symbol_id: row.get(1)?,
                        target_name: row.get(2)?,
                        kind: row.get(3)?,
                    })
                },
            )
            .context("Failed to upsert symbol edge")
    }

    pub fn symbol_edges(&self, from_symbol_id: &str) -> Result<Vec<SymbolEdge>> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let mut stmt = guard.prepare(
            "SELECT id, from_symbol_id, target_name, kind
             FROM symbol_edges
             WHERE from_symbol_id = ?1
             ORDER BY target_name ASC, kind ASC",
        )?;
        let res = stmt
            .query_map(params![from_symbol_id], |row| {
                Ok(SymbolEdge {
                    id: row.get(0)?,
                    from_symbol_id: row.get(1)?,
                    target_name: row.get(2)?,
                    kind: row.get(3)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(res)
    }

    pub fn symbol_graph_distance_to_query(
        &self,
        project_id: &str,
        from_symbol_id: &str,
        query: &str,
        max_depth: usize,
    ) -> Result<Option<usize>> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let query_lower = query.to_lowercase();
        let mut symbol_stmt = guard.prepare(
            "SELECT s.id, s.name
             FROM symbols s
             JOIN files f ON f.id = s.file_id
             WHERE f.project_id = ?1",
        )?;
        let project_symbols = symbol_stmt
            .query_map(params![project_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let symbol_names = project_symbols
            .iter()
            .cloned()
            .collect::<HashMap<String, String>>();
        let mut symbols_by_name = HashMap::<String, Vec<String>>::new();
        for (id, name) in &project_symbols {
            symbols_by_name
                .entry(name.to_lowercase())
                .or_default()
                .push(id.clone());
        }

        let mut edge_stmt = guard.prepare(
            "SELECT e.from_symbol_id, e.target_name
             FROM symbol_edges e
             JOIN symbols s ON s.id = e.from_symbol_id
             JOIN files f ON f.id = s.file_id
             WHERE f.project_id = ?1",
        )?;
        let edges = edge_stmt
            .query_map(params![project_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut edges_by_source = HashMap::<String, Vec<String>>::new();
        for (from, target_name) in edges {
            edges_by_source.entry(from).or_default().push(target_name);
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::from([(from_symbol_id.to_string(), 0usize)]);
        while let Some((symbol_id, depth)) = queue.pop_front() {
            if !visited.insert(symbol_id.clone()) || depth >= max_depth {
                continue;
            }
            let Some(edges) = edges_by_source.get(&symbol_id) else {
                continue;
            };
            for target_name in edges {
                if target_name.to_lowercase().contains(&query_lower) {
                    return Ok(Some(depth + 1));
                }
                if let Some(next_symbols) = symbols_by_name.get(&target_name.to_lowercase()) {
                    for next_id in next_symbols {
                        if symbol_names.contains_key(next_id) {
                            queue.push_back((next_id.clone(), depth + 1));
                        }
                    }
                }
            }
        }

        Ok(None)
    }

    /// Lists all projects sorted alphabetically.
    pub fn list_projects(&self) -> Result<Vec<Project>> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let mut stmt = guard
            .prepare("SELECT id, name, root_path, created_at FROM projects ORDER BY name ASC")?;

        let res = stmt
            .query_map([], |row| {
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
        let mut stmt = guard.prepare("SELECT id, project_id, path, checksum, index_version, last_indexed FROM files WHERE project_id = ?1 ORDER BY path ASC")?;

        let res = stmt
            .query_map(params![project_id], |row| {
                Ok(SourceFile {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    path: row.get(2)?,
                    checksum: row.get(3)?,
                    index_version: row.get(4)?,
                    last_indexed: parse_opt_dt(row.get(5)?),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(res)
    }

    /// Retrieves parsed symbols for a given file sorted by line number.
    pub fn file_symbols(&self, file_id: &str) -> Result<Vec<Symbol>> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let mut stmt = guard.prepare("SELECT id, file_id, name, kind, line, start_line, end_line, content_hash FROM symbols WHERE file_id = ?1 ORDER BY start_line ASC, name ASC")?;

        let res = stmt
            .query_map(params![file_id], |row| {
                Ok(Symbol {
                    id: row.get(0)?,
                    file_id: row.get(1)?,
                    name: row.get(2)?,
                    kind: row.get(3)?,
                    line: row.get(4)?,
                    start_line: row.get(5)?,
                    end_line: row.get(6)?,
                    content_hash: row.get(7)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(res)
    }

    /// Retrieves parsed symbols for a given relative path in a project.
    pub fn symbols_for_path(&self, project_id: &str, rel_path: &str) -> Result<Vec<Symbol>> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let mut stmt = guard.prepare(
            "SELECT s.id, s.file_id, s.name, s.kind, s.line, s.start_line, s.end_line, s.content_hash
             FROM symbols s
             JOIN files f ON f.id = s.file_id
             WHERE f.project_id = ?1 AND f.path = ?2
             ORDER BY s.start_line ASC, s.name ASC",
        )?;

        let res = stmt
            .query_map(params![project_id, rel_path], |row| {
                Ok(Symbol {
                    id: row.get(0)?,
                    file_id: row.get(1)?,
                    name: row.get(2)?,
                    kind: row.get(3)?,
                    line: row.get(4)?,
                    start_line: row.get(5)?,
                    end_line: row.get(6)?,
                    content_hash: row.get(7)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(res)
    }

    /// Retrieves parsed symbols for all files in a project.
    pub fn project_symbols(&self, project_id: &str) -> Result<Vec<Symbol>> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let mut stmt = guard.prepare(
            "SELECT s.id, s.file_id, s.name, s.kind, s.line, s.start_line, s.end_line, s.content_hash
             FROM symbols s
             JOIN files f ON f.id = s.file_id
             WHERE f.project_id = ?1
             ORDER BY f.path ASC, s.start_line ASC, s.name ASC",
        )?;

        let res = stmt
            .query_map(params![project_id], |row| {
                Ok(Symbol {
                    id: row.get(0)?,
                    file_id: row.get(1)?,
                    name: row.get(2)?,
                    kind: row.get(3)?,
                    line: row.get(4)?,
                    start_line: row.get(5)?,
                    end_line: row.get(6)?,
                    content_hash: row.get(7)?,
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(res)
    }

    /// Returns files matching the query snippet (path or filename).
    pub fn search_files(&self, project_id: &str, query: &str) -> Result<Vec<SourceFile>> {
        let guard = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("Mutex poisoned"))?;

        // Wrap the query in wildcards for substring matching
        let like_query = format!("%{}%", query);
        let mut stmt = guard.prepare(
            "SELECT id, project_id, path, checksum, index_version, last_indexed 
             FROM files 
             WHERE project_id = ?1 AND path LIKE ?2 
             ORDER BY path ASC LIMIT 20",
        )?;

        let res = stmt
            .query_map(rusqlite::params![project_id, like_query], |row| {
                Ok(SourceFile {
                    id: row.get(0)?,
                    project_id: row.get(1)?,
                    path: row.get(2)?,
                    checksum: row.get(3)?,
                    index_version: row.get(4)?,
                    last_indexed: parse_opt_dt(row.get(5)?),
                })
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(res)
    }

    /// Returns symbols matching a name or file-path query.
    pub fn search_symbols(
        &self,
        project_id: &str,
        query: &str,
    ) -> Result<Vec<(SourceFile, Symbol)>> {
        let guard = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("Mutex poisoned"))?;
        let like_query = format!("%{}%", query);
        let mut stmt = guard.prepare(
            "SELECT
                f.id, f.project_id, f.path, f.checksum, f.index_version, f.last_indexed,
                s.id, s.file_id, s.name, s.kind, s.line, s.start_line, s.end_line, s.content_hash
             FROM symbols s
             JOIN files f ON f.id = s.file_id
             WHERE f.project_id = ?1
               AND (
                   s.name LIKE ?2
                   OR f.path LIKE ?2
                   OR EXISTS (
                       SELECT 1 FROM symbol_edges e
                       WHERE e.from_symbol_id = s.id
                         AND e.target_name LIKE ?2
                   )
                   OR EXISTS (
                       SELECT 1 FROM symbol_edges e1
                       JOIN symbols mid ON mid.name = e1.target_name
                       JOIN files mf ON mf.id = mid.file_id AND mf.project_id = ?1
                       JOIN symbol_edges e2 ON e2.from_symbol_id = mid.id
                       WHERE e1.from_symbol_id = s.id
                         AND e2.target_name LIKE ?2
                   )
               )
             ORDER BY f.path ASC, s.start_line ASC, s.name ASC
             LIMIT 50",
        )?;

        let res = stmt
            .query_map(params![project_id, like_query], |row| {
                Ok((
                    SourceFile {
                        id: row.get(0)?,
                        project_id: row.get(1)?,
                        path: row.get(2)?,
                        checksum: row.get(3)?,
                        index_version: row.get(4)?,
                        last_indexed: parse_opt_dt(row.get(5)?),
                    },
                    Symbol {
                        id: row.get(6)?,
                        file_id: row.get(7)?,
                        name: row.get(8)?,
                        kind: row.get(9)?,
                        line: row.get(10)?,
                        start_line: row.get(11)?,
                        end_line: row.get(12)?,
                        content_hash: row.get(13)?,
                    },
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;

        Ok(res)
    }

    /// Removes a file from the index (e.g., when it is deleted on disk).
    pub fn remove_file(&self, project_id: &str, rel_path: &str) -> anyhow::Result<()> {
        let guard = self
            .conn
            .lock()
            .map_err(|_| anyhow::anyhow!("Mutex poisoned"))?;
        guard.execute(
            "DELETE FROM files WHERE project_id = ?1 AND path = ?2",
            rusqlite::params![project_id, rel_path],
        )?;
        Ok(())
    }
}

fn migrate_schema(conn: &Connection) -> Result<()> {
    ensure_column(conn, "files", "index_version", "INTEGER NOT NULL DEFAULT 0")?;
    ensure_column(conn, "symbols", "start_line", "INTEGER NOT NULL DEFAULT 0")?;
    ensure_column(conn, "symbols", "end_line", "INTEGER NOT NULL DEFAULT 0")?;
    ensure_column(conn, "symbols", "content_hash", "TEXT")?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS symbol_edges (
            id TEXT PRIMARY KEY,
            from_symbol_id TEXT NOT NULL REFERENCES symbols(id) ON DELETE CASCADE,
            target_name TEXT NOT NULL,
            kind TEXT NOT NULL,
            UNIQUE(from_symbol_id, target_name, kind)
         )",
        [],
    )?;
    conn.execute(
        "UPDATE symbols SET start_line = line WHERE start_line = 0",
        [],
    )?;
    conn.execute("UPDATE symbols SET end_line = line WHERE end_line = 0", [])?;
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_symbols_legacy_identity ON symbols(file_id, name, kind, line)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_symbols_range ON symbols(file_id, start_line, end_line)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_symbol_edges_from ON symbol_edges(from_symbol_id)",
        [],
    )?;
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_symbol_edges_target ON symbol_edges(target_name)",
        [],
    )?;
    Ok(())
}

fn ensure_column(conn: &Connection, table: &str, column: &str, declaration: &str) -> Result<()> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({})", table))?;
    let columns = stmt
        .query_map([], |row| row.get::<_, String>(1))?
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if !columns.iter().any(|existing| existing == column) {
        conn.execute(
            &format!(
                "ALTER TABLE {} ADD COLUMN {} {}",
                table, column, declaration
            ),
            [],
        )?;
    }
    Ok(())
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

        let file = store
            .upsert_file(&proj.id, "src/main.rs", Some("abc"))
            .unwrap();
        let file_dup = store
            .upsert_file(&proj.id, "src/main.rs", Some("def"))
            .unwrap();
        assert_eq!(file.id, file_dup.id);
        assert_eq!(file_dup.checksum.as_deref(), Some("def"));

        assert_eq!(file.index_version, 1);
        assert_eq!(file_dup.index_version, 2);

        let _main = store
            .upsert_symbol(&file.id, "main", "fn", 1, 4, Some("hash"))
            .unwrap();
        let _helper = store
            .upsert_symbol(&file.id, "helper", "fn", 6, 8, Some("helper_hash"))
            .unwrap();
        let syms = store.file_symbols(&file.id).unwrap();
        assert_eq!(syms.len(), 2);
        assert_eq!(syms[0].start_line, 1);
        assert_eq!(syms[0].end_line, 4);
        assert_eq!(syms[0].content_hash.as_deref(), Some("hash"));
        assert_eq!(store.project_symbols(&proj.id).unwrap().len(), 2);

        let hits = store.search_symbols(&proj.id, "main").unwrap();
        assert!(hits.iter().any(|(_, symbol)| symbol.name == "main"));

        let main_symbol = syms.iter().find(|symbol| symbol.name == "main").unwrap();
        let helper_symbol = syms.iter().find(|symbol| symbol.name == "helper").unwrap();
        let edge = store
            .upsert_symbol_edge(&main_symbol.id, "helper", "call")
            .unwrap();
        assert_eq!(edge.target_name, "helper");
        store
            .upsert_symbol_edge(&helper_symbol.id, "println", "call")
            .unwrap();
        assert_eq!(store.symbol_edges(&main_symbol.id).unwrap().len(), 1);
        assert_eq!(
            store
                .symbol_graph_distance_to_query(&proj.id, &main_symbol.id, "println", 2)
                .unwrap(),
            Some(2)
        );
        let edge_hits = store.search_symbols(&proj.id, "println").unwrap();
        assert!(edge_hits.iter().any(|(_, symbol)| symbol.name == "main"));
        assert!(edge_hits.iter().any(|(_, symbol)| symbol.name == "helper"));

        let path_symbols = store.symbols_for_path(&proj.id, "src/main.rs").unwrap();
        assert_eq!(path_symbols.len(), 2);
        assert_eq!(path_symbols[0].name, "main");
        assert_eq!(path_symbols[1].name, "helper");

        store.clear_file_symbols(&file.id).unwrap();
        assert!(store.file_symbols(&file.id).unwrap().is_empty());

        store.remove_project(&proj.id).unwrap();
        assert!(store.project_files(&proj.id).unwrap().is_empty());

        let _ = fs::remove_file(db_path);
        let _ = fs::remove_dir_all(test_dir);
    }
}
