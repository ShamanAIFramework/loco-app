use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskRequest {
    pub run_id: String,
    pub session_id: String,
    pub workspace_path: Option<PathBuf>,
    pub question: String,
    pub conversation: Vec<ConversationTurn>,
    pub budgets: Budgets,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConversationTurn {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Budgets {
    pub max_llm_calls: usize,
    pub max_extract_rounds: usize,
    pub max_search_queries: usize,
    pub max_files_to_read: usize,
    pub max_evidence_chars: usize,
    #[serde(default = "default_max_evidence_tokens")]
    pub max_evidence_tokens: usize,
}

impl Default for Budgets {
    fn default() -> Self {
        Self {
            max_llm_calls: 4,
            max_extract_rounds: 2,
            max_search_queries: 5,
            max_files_to_read: 6,
            max_evidence_chars: 18_000,
            max_evidence_tokens: default_max_evidence_tokens(),
        }
    }
}

fn default_max_evidence_tokens() -> usize {
    4_500
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Intake,
    Extract,
    Synthesize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    Explain,
    Locate,
    Unclear,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    Done,
    NotFound,
    NeedsUser,
    NoProgress,
    BudgetExhausted,
    Cancelled,
    Error,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentAnswer {
    pub content: String,
    pub stop_reason: StopReason,
    pub cited_entries: Vec<EntryId>,
    pub notebook: Notebook,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntakeOutput {
    pub task_type: TaskType,
    pub target_hints: Vec<String>,
    pub constraints: Vec<String>,
    pub search_queries: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractOutput {
    pub inferences: Vec<ModelInference>,
    pub sufficient: bool,
    pub missing: Vec<MissingEvidence>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInference {
    pub statement: String,
    pub lease_id: LeaseId,
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MissingEvidence {
    pub kind: MissingEvidenceKind,
    pub target: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum MissingEvidenceKind {
    Callers,
    Callees,
    Definition,
    References,
    FileRange,
    Search,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SynthesizeOutput {
    pub answer: String,
    pub cited_entries: Vec<EntryId>,
    pub stop_reason: StopReason,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FileHit {
    pub path: String,
    pub score: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SymbolHit {
    pub path: String,
    pub symbol: String,
    pub kind: String,
    pub start_line: u32,
    pub end_line: u32,
    pub score: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EvidenceSlice {
    pub lease: EvidenceLease,
    pub content: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeaseStatus {
    Valid,
    Expired,
}

pub use crate::notebook::{EntryId, EvidenceLease, LeaseId, Notebook};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budgets_default_missing_token_limit() {
        let raw = r#"{
            "max_llm_calls": 4,
            "max_extract_rounds": 2,
            "max_search_queries": 5,
            "max_files_to_read": 6,
            "max_evidence_chars": 18000
        }"#;

        let budgets: Budgets = serde_json::from_str(raw).unwrap();

        assert_eq!(budgets.max_evidence_tokens, 4_500);
    }
}
