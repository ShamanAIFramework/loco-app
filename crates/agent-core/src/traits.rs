use crate::prompt::PromptParts;
use crate::types::{ContentHit, EvidenceSlice, FileHit, LeaseStatus, Phase, SymbolHit};
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmRequest {
    pub call_id: String,
    pub phase: Phase,
    pub schema_name: String,
    pub schema: serde_json::Value,
    pub prompt_parts: Option<PromptParts>,
    pub prompt: String,
    pub temperature: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct LlmUsage {
    pub queue_wait_ms: Option<u64>,
    pub prompt_tokens_total: Option<u64>,
    pub prompt_tokens_evaluated: Option<u64>,
    pub prompt_tokens_cached: Option<u64>,
    pub decode_tokens: Option<u64>,
    pub prefill_ms: Option<u64>,
    pub decode_ms: Option<u64>,
    pub wall_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LlmResponse {
    pub call_id: String,
    pub raw_json: String,
    pub usage: LlmUsage,
}

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete_json(&self, request: LlmRequest) -> Result<LlmResponse>;
}

#[async_trait]
pub trait CodeSearch: Send + Sync {
    async fn repo_map(&self, max_entries: usize) -> Result<Vec<String>>;
    async fn search(&self, query: &str, limit: usize) -> Result<Vec<FileHit>>;
    async fn search_symbols(&self, query: &str, limit: usize) -> Result<Vec<SymbolHit>>;
    async fn search_content(&self, query: &str, limit: usize) -> Result<Vec<ContentHit>>;
}

#[async_trait]
pub trait FileReader: Send + Sync {
    async fn read_file(&self, path: &str, max_chars: usize) -> Result<EvidenceSlice>;
    async fn read_range(
        &self,
        path: &str,
        symbol: Option<&str>,
        start_line: u32,
        end_line: u32,
        max_chars: usize,
    ) -> Result<EvidenceSlice>;
    async fn verify_lease(&self, lease: &crate::notebook::EvidenceLease) -> Result<LeaseStatus>;
}

#[async_trait]
pub trait TokenCounter: Send + Sync {
    async fn count_tokens(&self, text: &str) -> usize;
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum AgentProgress {
    Status(String),
    Trace(String),
}

#[async_trait]
pub trait ProgressSink: Send + Sync {
    async fn emit(&self, progress: AgentProgress);
}

pub struct ApproxTokenCounter;

#[async_trait]
impl TokenCounter for ApproxTokenCounter {
    async fn count_tokens(&self, text: &str) -> usize {
        approx_token_count(text)
    }
}

pub fn approx_token_count(text: &str) -> usize {
    text.split_whitespace().count()
}

pub struct NoopProgressSink;

#[async_trait]
impl ProgressSink for NoopProgressSink {
    async fn emit(&self, _progress: AgentProgress) {}
}
