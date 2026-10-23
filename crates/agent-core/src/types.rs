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
    #[serde(default)]
    pub intake_only: bool,
}

impl Default for TaskRequest {
    fn default() -> Self {
        Self {
            run_id: String::new(),
            session_id: String::new(),
            workspace_path: None,
            question: String::new(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
            intake_only: false,
        }
    }
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
            max_llm_calls: 12,
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
    IntakeCritic,
    CandidateGate,
    CandidateGateOptimist,
    CandidateGateSkeptic,
    Plan,
    PlanProposer,
    PlanCritic,
    PlanRebuttal,
    PlanReconciler,
    Extract,
    PatchProposer,
    PatchCritic,
    PatchRebuttal,
    PatchReconciler,
    Synthesize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskType {
    Explain,
    Locate,
    Edit,
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
    #[serde(default)]
    pub edit_artifact: Option<EditArtifact>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UserIntent {
    Chat,
    GeneralQuestion,
    WorkspaceSearch,
    WorkspaceEdit,
    Ambiguous,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntentItem {
    pub intent: UserIntent,
    pub summary: String,
    pub needs_workspace_search: bool,
    #[serde(default)]
    pub direct_response: Option<String>,
    #[serde(default)]
    pub search_queries: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntakeOutput {
    pub reasoning: String,
    pub needs_workspace_search: bool,
    pub intents: Vec<IntentItem>,
    #[serde(default)]
    pub target_hints: Vec<String>,
    #[serde(default)]
    pub constraints: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CriticOutput {
    pub approved: bool,
    pub critique: String,
    #[serde(default)]
    pub has_dropped_clauses: bool,
    #[serde(default)]
    pub has_contaminated_queries: bool,
    #[serde(default)]
    pub corrected_intents: Option<Vec<IntentItem>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TriadConsensus {
    pub drafter_intents: Vec<IntentItem>,
    pub critic_approved: bool,
    pub critic_critique: String,
    pub arbiter_decision: String,
    pub final_intents: Vec<IntentItem>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditPlan {
    pub objective: String,
    pub affected_files: Vec<PlannedFile>,
    pub steps: Vec<PlanStep>,
    pub risks: Vec<String>,
    pub validation_commands: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlannedFile {
    pub path: String,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanStep {
    pub title: String,
    pub detail: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanCriticOutput {
    pub critique: String,
    #[serde(default)]
    pub missing_aspects: Vec<String>,
    #[serde(default)]
    pub disputed_steps: Vec<String>,
    #[serde(default)]
    pub unverified_assumptions: Vec<String>,
    #[serde(default)]
    pub additional_risks: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanRebuttalOutput {
    pub rebuttal: String,
    #[serde(default)]
    pub dismissed_critiques: Vec<RebuttalItem>,
    #[serde(default)]
    pub accepted_critiques: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RebuttalItem {
    pub critique: String,
    pub justification: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanStepResults {
    pub proposed_plan: EditPlan,
    pub critic_output: PlanCriticOutput,
    pub rebuttal_output: PlanRebuttalOutput,
    pub reconciled_plan: EditPlan,
    pub arbiter_validated_files: Vec<PlannedFile>,
    pub arbiter_dropped_files: Vec<PlannedFile>,
    pub committed_decision_ids: Vec<EntryId>,
    pub proposer_elapsed_ms: u64,
    pub critic_elapsed_ms: u64,
    pub rebuttal_elapsed_ms: u64,
    pub reconciler_elapsed_ms: u64,
    pub total_elapsed_ms: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchProposal {
    pub target_path: String,
    pub target_symbol: Option<String>,
    pub start_line: usize,
    pub end_line: usize,
    pub lease_id: LeaseId,
    pub replacement_code: String,
    pub justification: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchCriticOutput {
    pub approved: bool,
    pub critique: String,
    #[serde(default)]
    pub disputed_changes: Vec<String>,
    #[serde(default)]
    pub unhandled_edge_cases: Vec<String>,
    #[serde(default)]
    pub convention_violations: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchRebuttalOutput {
    pub rebuttal: String,
    #[serde(default)]
    pub dismissed_points: Vec<RebuttalItem>,
    #[serde(default)]
    pub accepted_points: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PatchReconcilerOutput {
    pub hardened_code: String,
    pub reconciliation_summary: String,
    #[serde(default)]
    pub remaining_uncertainties: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PartitionedConsensus {
    pub confirmed_code: Option<String>,
    pub disputed_issue: Option<String>,
    pub is_split: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EditArtifact {
    pub patch_id: String,
    pub target_path: String,
    pub original_start_line: usize,
    pub original_end_line: usize,
    pub replacement_code: String,
    pub diff: String,
    pub critic_approved: bool,
    pub critic_notes: String,
    #[serde(default)]
    pub impacted_callers: Vec<String>,
    #[serde(default)]
    pub validation_commands: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExtractOutput {
    #[serde(default)]
    pub inferences: Vec<ModelInference>,
    pub sufficient: bool,
    #[serde(default)]
    pub missing: Vec<MissingEvidence>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInference {
    pub statement: String,
    pub lease_id: LeaseId,
    #[serde(default)]
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolRelation {
    Definition,
    References,
    Callers,
    Callees,
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
pub struct ContentHit {
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub preview: String,
    pub score: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorHit {
    pub path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub symbol: Option<String>,
    pub snippet: String,
    pub score: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateEvaluation {
    pub path: String,
    pub keep: bool,
    pub reason: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateGateOutput {
    pub evaluations: Vec<CandidateEvaluation>,
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
