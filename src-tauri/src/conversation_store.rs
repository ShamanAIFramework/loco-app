// src-tauri/src/conversation_store.rs
//! Persistence layer for AI conversations and messaging history.
//! Uses a dedicated synchronous SQLite connection wrapped in a Mutex.
//! Async callers must use `tauri::async_runtime::spawn_blocking`.

use anyhow::{anyhow, Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::str::FromStr;
use std::sync::Mutex;
use uuid::Uuid;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl Role {
    fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool",
        }
    }
}

impl FromStr for Role {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "system" => Ok(Role::System),
            "user" => Ok(Role::User),
            "assistant" => Ok(Role::Assistant),
            "tool" => Ok(Role::Tool),
            _ => Err(anyhow!("Invalid role string: {}", s)),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub title: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub conversation_id: String,
    pub role: Role,
    pub content: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConversationDetail {
    pub conversation: Conversation,
    pub messages: Vec<Message>,
}

fn now_str() -> String {
    Utc::now().to_rfc3339()
}

fn parse_dt(s: &str) -> Result<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(s)
        .map(|dt| dt.with_timezone(&Utc))
        .context("Failed to parse datetime")
}

/// Thread-safe SQLite repository for conversations.
pub struct ConversationStore {
    conn: Mutex<Connection>,
}

impl ConversationStore {
    /// Opens the DB and initializes schema.
    pub fn new(db_path: &Path) -> Result<Self> {
        let conn = Connection::open(db_path).context("Failed to open conversation DB")?;
        
        // This duplication of PRAGMAs ensures module independence
        conn.execute_batch(
            "PRAGMA foreign_keys = ON;
             PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA synchronous = NORMAL;"
        )?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
             );
             CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('system','user','assistant','tool')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_messages_conv_created ON messages(conversation_id, created_at);
             CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
             
             -- future_embeddings reserved table (no rust API yet)
             CREATE TABLE IF NOT EXISTS future_embeddings (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                created_at TEXT NOT NULL
             );"
        )?;

        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    /// Creates a new conversation thread.
    pub fn create_conversation(&self, title: Option<&str>) -> Result<Conversation> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let id = Uuid::new_v4().to_string();
        let title = title.unwrap_or("New conversation").to_string();
        let now = now_str();

        guard.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?1, ?2, ?3, ?4)",
            params![id, title, now, now],
        )?;

        Ok(Conversation {
            id,
            title,
            created_at: parse_dt(&now)?,
            updated_at: parse_dt(&now)?,
        })
    }

    /// Renames an existing conversation.
    pub fn rename_conversation(&self, id: &str, title: &str) -> Result<Conversation> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let title = title.trim();
        if title.is_empty() {
            return Err(anyhow!("Title cannot be empty"));
        }
        
        let now = now_str();
        let updated = guard.execute(
            "UPDATE conversations SET title = ?1, updated_at = ?2 WHERE id = ?3",
            params![title, now, id],
        )?;
        
        if updated == 0 {
            return Err(anyhow!("Conversation not found"));
        }

        guard.query_row(
            "SELECT title, created_at, updated_at FROM conversations WHERE id = ?1",
            params![id],
            |row| {
                Ok(Conversation {
                    id: id.to_string(),
                    title: row.get(0)?,
                    created_at: parse_dt(&row.get::<_, String>(1)?).unwrap_or_default(),
                    updated_at: parse_dt(&row.get::<_, String>(2)?).unwrap_or_default(),
                })
            },
        ).context("Failed to fetch renamed conversation")
    }

    /// Deletes a conversation and its cascaded messages.
    pub fn delete_conversation(&self, id: &str) -> Result<bool> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let count = guard.execute("DELETE FROM conversations WHERE id = ?1", params![id])?;
        Ok(count > 0)
    }

    /// Appends a message to a conversation.
    pub fn add_message(&self, conversation_id: &str, role: Role, content: &str) -> Result<Message> {
        let mut guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let tx = guard.transaction()?;
        
        // Verify conv exists to fail fast
        let exists: i64 = tx.query_row(
            "SELECT COUNT(1) FROM conversations WHERE id = ?1",
            params![conversation_id],
            |r| r.get(0)
        )?;
        if exists == 0 {
            return Err(anyhow!("Conversation not found"));
        }

        let id = Uuid::new_v4().to_string();
        let now = now_str();

        tx.execute(
            "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![id, conversation_id, role.as_str(), content, now],
        )?;

        tx.execute(
            "UPDATE conversations SET updated_at = ?1 WHERE id = ?2",
            params![now, conversation_id],
        )?;
        
        tx.commit()?;

        Ok(Message {
            id,
            conversation_id: conversation_id.to_string(),
            role,
            content: content.to_string(),
            created_at: parse_dt(&now)?,
        })
    }

    /// Loads a full conversation with ordered messages.
    pub fn load_conversation(&self, id: &str) -> Result<Option<ConversationDetail>> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        
        let conv_res = guard.query_row(
            "SELECT title, created_at, updated_at FROM conversations WHERE id = ?1",
            params![id],
            |row| {
                Ok(Conversation {
                    id: id.to_string(),
                    title: row.get(0)?,
                    created_at: parse_dt(&row.get::<_, String>(1)?).unwrap_or_default(),
                    updated_at: parse_dt(&row.get::<_, String>(2)?).unwrap_or_default(),
                })
            },
        );

        let conv = match conv_res {
            Ok(c) => c,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        let mut stmt = guard.prepare(
            "SELECT id, role, content, created_at FROM messages 
             WHERE conversation_id = ?1 
             ORDER BY created_at ASC, rowid ASC"
        )?;
        
        let messages = stmt.query_map(params![id], |row| {
            let role_str: String = row.get(1)?;
            Ok(Message {
                id: row.get(0)?,
                conversation_id: id.to_string(),
                role: Role::from_str(&role_str).unwrap_or(Role::System),
                content: row.get(2)?,
                created_at: parse_dt(&row.get::<_, String>(3)?).unwrap_or_default(),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;

        Ok(Some(ConversationDetail {
            conversation: conv,
            messages,
        }))
    }

    /// Lists conversations ordered by most recently updated.
    pub fn list_conversations(&self) -> Result<Vec<Conversation>> {
        let guard = self.conn.lock().map_err(|_| anyhow!("Mutex poisoned"))?;
        let mut stmt = guard.prepare("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")?;
        
        let res = stmt.query_map([], |row| {
            Ok(Conversation {
                id: row.get(0)?,
                title: row.get(1)?,
                created_at: parse_dt(&row.get::<_, String>(2)?).unwrap_or_default(),
                updated_at: parse_dt(&row.get::<_, String>(3)?).unwrap_or_default(),
            })
        })?
        .collect::<Result<Vec<_>, _>>()?;
        
        Ok(res)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_conversation_crud() {
        let db_path = env::temp_dir().join(format!("{}.db", Uuid::new_v4()));
        let store = ConversationStore::new(&db_path).unwrap();
        
        let c = store.create_conversation(Some("Test")).unwrap();
        assert_eq!(c.title, "Test");

        let m = store.add_message(&c.id, Role::User, "Hello").unwrap();
        assert_eq!(m.content, "Hello");

        let loaded = store.load_conversation(&c.id).unwrap().unwrap();
        assert_eq!(loaded.messages.len(), 1);

        store.delete_conversation(&c.id).unwrap();
        let missing = store.load_conversation(&c.id).unwrap();
        assert!(missing.is_none());
        
        let _ = std::fs::remove_file(db_path);
    }
}