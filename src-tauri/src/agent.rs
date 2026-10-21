//! Shared agent IPC event types.
//!
//! The executable agent loop lives in `orchestrator.rs` and `agent-core`; this
//! module intentionally contains no model-selected tool loop.

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SourceRef {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub symbol: Option<String>,
    pub label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NotebookEntryView {
    pub id: String,
    pub kind: String,
    pub statement: String,
    pub phase: String,
    pub status: String,
    pub tags: Vec<String>,
    pub file_path: Option<String>,
    pub line_range: Option<[usize; 2]>,
    pub symbol: Option<String>,
    pub lease_id: Option<String>,
    pub depends_on: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum AgentEvent {
    Status(String),
    Chunk(String),
    Sources(Vec<SourceRef>),
    Notebook(Vec<NotebookEntryView>),
}
