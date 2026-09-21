//! Shared agent IPC event types.
//!
//! The executable agent loop lives in `orchestrator.rs` and `agent-core`; this
//! module intentionally contains no model-selected tool loop.

use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct SourceRef {
    pub path: String,
    pub start_line: usize,
    pub end_line: usize,
    pub symbol: Option<String>,
    pub label: String,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum AgentEvent {
    Status(String),
    Chunk(String),
    Sources(Vec<SourceRef>),
}
