// src-tauri/src/orchestrator.rs
//! Tauri host adapter for the deterministic agent-core runtime.

use crate::agent::{AgentEvent, SourceRef};
use crate::conversation_store::Role;
use crate::engine::EngineState;
use crate::{AppError, AppState};
use agent_core::{
    AgentProgress, AgentRuntime, Budgets, CodeSearch, ContentHit, ConversationTurn, EvidenceLease,
    EvidenceSlice, FileHit, FileReader, LeaseStatus, LineRange, LlmClient, LlmRequest, LlmResponse,
    LlmUsage, ProgressSink, RuntimeMetadata, SymbolHit, TaskRequest, TokenCounter, TraceEvent,
};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use regex::RegexBuilder;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use tauri::ipc::Channel;
use tauri::Manager;
use tokio::sync::Semaphore;
use uuid::Uuid;

#[tauri::command]
pub async fn prompt_agent_v2(
    app: tauri::AppHandle,
    session_id: String,
    question: String,
    on_chunk: Channel<AgentEvent>,
) -> Result<(), AppError> {
    let engine_state = app.state::<EngineState>();
    let conn = engine_state
        .connection_details()
        .ok_or_else(|| anyhow!("Engine sidecar is not ready"))?;

    let app_clone = app.clone();
    let session_id_clone = session_id.clone();
    let question_clone = question.clone();

    let (conversation, messages) = tauri::async_runtime::spawn_blocking(move || -> Result<_> {
        let state = app_clone.state::<AppState>();
        state
            .conversations
            .add_message(&session_id_clone, Role::User, &question_clone)?;
        let detail = state
            .conversations
            .load_conversation(&session_id_clone)?
            .ok_or_else(|| anyhow!("Session not found"))?;
        Ok((detail.conversation, detail.messages))
    })
    .await
    .context("Failed to join blocking conversation task")??;

    let project_id = if let Some(workspace) = &conversation.workspace_path {
        let app_c = app.clone();
        let workspace_c = workspace.clone();
        Some(
            tauri::async_runtime::spawn_blocking(move || {
                let state = app_c.state::<AppState>();
                state
                    .codegraph
                    .register_project("Active Workspace", Path::new(&workspace_c))
                    .map(|project| project.id)
            })
            .await
            .context("Failed to join project registration task")??,
        )
    } else {
        None
    };

    let run_id = Uuid::new_v4().to_string();
    let task = TaskRequest {
        run_id: run_id.clone(),
        session_id: session_id.clone(),
        workspace_path: conversation.workspace_path.as_ref().map(PathBuf::from),
        question: question.clone(),
        conversation: messages
            .iter()
            .map(|msg| ConversationTurn {
                role: msg.role.as_str().to_string(),
                content: msg.content.clone(),
            })
            .collect(),
        budgets: load_budgets().await.unwrap_or_default(),
    };

    let llm = LlamaServerClient {
        client: reqwest::Client::new(),
        port: conn.port,
        api_key: conn.api_key,
        slot: app.state::<AppState>().llm_slot.clone(),
    };
    let metadata = llm.runtime_metadata().await;
    let search = TauriCodeSearch {
        app: app.clone(),
        project_id,
    };
    let reader = WorkspaceFileReader {
        workspace_path: conversation.workspace_path.clone().map(PathBuf::from),
    };
    let progress = TauriProgressSink {
        channel: on_chunk.clone(),
    };

    let mut result = agent_core::run_agent(
        task,
        AgentRuntime {
            llm: &llm,
            code_search: &search,
            file_reader: &reader,
            token_counter: &llm,
            progress: &progress,
        },
    )
    .await?;

    result
        .trace
        .events
        .insert(1, TraceEvent::RuntimeMetadata { metadata });
    write_trace(&run_id, &result.trace).await?;

    let final_response = result.answer.content.clone();
    let sources = collect_answer_sources(&result.answer);
    let _ = on_chunk.send(AgentEvent::Chunk(final_response.clone()));
    if !sources.is_empty() {
        let _ = on_chunk.send(AgentEvent::Sources(sources));
    }

    tauri::async_runtime::spawn_blocking(move || -> Result<_> {
        let state = app.state::<AppState>();
        state
            .conversations
            .add_message(&session_id, Role::Assistant, &final_response)?;
        Ok(())
    })
    .await
    .context("Failed to join blocking assistant persistence task")??;

    Ok(())
}

struct LlamaServerClient {
    client: reqwest::Client,
    port: u16,
    api_key: String,
    slot: Arc<Semaphore>,
}

impl LlamaServerClient {
    async fn runtime_metadata(&self) -> RuntimeMetadata {
        let endpoint = format!("http://127.0.0.1:{}", self.port);
        RuntimeMetadata {
            host: "tauri-sidecar".to_string(),
            endpoint: endpoint.clone(),
            model_id: self.fetch_model_id(&endpoint).await,
            server_version: None,
            queue: "single-slot".to_string(),
        }
    }

    async fn fetch_model_id(&self, endpoint: &str) -> Option<String> {
        let value: serde_json::Value = self
            .client
            .get(format!("{endpoint}/v1/models"))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await
            .ok()?
            .json()
            .await
            .ok()?;
        value
            .get("data")
            .and_then(|data| data.as_array())
            .and_then(|models| models.first())
            .and_then(|model| model.get("id"))
            .and_then(|id| id.as_str())
            .map(ToString::to_string)
    }
}

#[async_trait]
impl LlmClient for LlamaServerClient {
    async fn complete_json(&self, request: LlmRequest) -> Result<LlmResponse> {
        let queued = std::time::Instant::now();
        let _permit = self
            .slot
            .acquire()
            .await
            .context("LLM queue semaphore closed")?;
        let queue_wait_ms = queued.elapsed().as_millis() as u64;

        let response_format = json!({
            "type": "json_schema",
            "json_schema": {
                "name": request.schema_name,
                "schema": request.schema,
                "strict": true
            }
        });
        let req_body = json!({
            "messages": [
                {
                    "role": "user",
                    "content": request.prompt
                }
            ],
            "temperature": request.temperature,
            "stream": false,
            "max_tokens": 2048,
            "response_format": response_format
        });

        let started = std::time::Instant::now();
        let res = self
            .client
            .post(format!("http://127.0.0.1:{}/v1/chat/completions", self.port))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&req_body)
            .send()
            .await
            .context("Failed to send deterministic LLM request")?;

        let status = res.status();
        let value: serde_json::Value = res
            .json()
            .await
            .context("Failed to parse llama-server response JSON")?;
        if !status.is_success() {
            return Err(anyhow!("llama-server request failed with {}: {}", status, value));
        }

        let raw_json = value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("{}")
            .to_string();
        let usage = &value["usage"];
        Ok(LlmResponse {
            call_id: request.call_id,
            raw_json,
            usage: LlmUsage {
                queue_wait_ms: Some(queue_wait_ms),
                prompt_tokens_total: usage["prompt_tokens"].as_u64(),
                prompt_tokens_evaluated: usage["prompt_tokens_evaluated"].as_u64(),
                prompt_tokens_cached: usage["prompt_tokens_cached"].as_u64(),
                decode_tokens: usage["completion_tokens"].as_u64(),
                prefill_ms: usage["prompt_ms"].as_u64(),
                decode_ms: usage["predicted_ms"].as_u64(),
                wall_ms: Some(started.elapsed().as_millis() as u64),
            },
        })
    }
}

#[async_trait]
impl TokenCounter for LlamaServerClient {
    async fn count_tokens(&self, text: &str) -> usize {
        let value = self
            .client
            .post(format!("http://127.0.0.1:{}/tokenize", self.port))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&json!({ "content": text }))
            .send()
            .await
            .ok()
            .and_then(|response| response.error_for_status().ok());

        let Some(response) = value else {
            return agent_core::approx_token_count(text);
        };
        let Ok(value) = response.json::<serde_json::Value>().await else {
            return agent_core::approx_token_count(text);
        };

        value
            .get("tokens")
            .and_then(|tokens| tokens.as_array())
            .map(|tokens| tokens.len())
            .unwrap_or_else(|| agent_core::approx_token_count(text))
    }
}

struct TauriCodeSearch {
    app: tauri::AppHandle,
    project_id: Option<String>,
}

#[async_trait]
impl CodeSearch for TauriCodeSearch {
    async fn repo_map(&self, max_entries: usize) -> Result<Vec<String>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        let app = self.app.clone();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let mut entries = Vec::<(String, usize, usize)>::new();
            for file in state.codegraph.project_files(&project_id)? {
                let file_symbols = state.codegraph.file_symbols(&file.id)?;
                let edge_count = file_symbols
                    .iter()
                    .map(|symbol| {
                        state
                            .codegraph
                            .symbol_edges(&symbol.id)
                            .map(|edges| edges.len())
                            .unwrap_or_default()
                    })
                    .sum::<usize>();
                let symbols = file_symbols
                    .iter()
                    .take(6)
                    .map(|symbol| {
                        format!(
                            "{} {} lines {}-{}",
                            symbol.kind, symbol.name, symbol.start_line, symbol.end_line
                        )
                    })
                    .collect::<Vec<_>>();
                if symbols.is_empty() {
                    entries.push((file.path, 0, 0));
                } else {
                    let mut outline = format!("{} :: {}", file.path, symbols.join("; "));
                    if edge_count > 0 {
                        outline.push_str(&format!("; refs {}", edge_count));
                    }
                    entries.push((outline, file_symbols.len(), edge_count));
                }
            }
            entries.sort_by(|a, b| {
                b.2.cmp(&a.2)
                    .then_with(|| b.1.cmp(&a.1))
                    .then_with(|| a.0.cmp(&b.0))
            });
            entries.truncate(max_entries);
            Ok(entries
                .into_iter()
                .map(|(entry, _, _)| entry)
                .collect::<Vec<_>>())
        })
        .await
        .context("Failed to join repo map task")?
    }

    async fn search(&self, query: &str, limit: usize) -> Result<Vec<FileHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        let app = self.app.clone();
        let query = query.to_string();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let mut hits = state
                .codegraph
                .search_files(&project_id, &query)?
                .into_iter()
                .map(|file| {
                    let score = score_file_hit(&file.path, &query);
                    FileHit {
                        path: file.path,
                        score,
                    }
                })
                .collect::<Vec<_>>();
            hits.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
            hits.truncate(limit);
            Ok(hits)
        })
        .await
        .context("Failed to join code search task")?
    }

    async fn search_symbols(&self, query: &str, limit: usize) -> Result<Vec<SymbolHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        let app = self.app.clone();
        let query = query.to_string();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let mut hits = state
                .codegraph
                .search_symbols(&project_id, &query)?
                .into_iter()
                .map(|(file, symbol)| {
                    let edge_boost = state
                        .codegraph
                        .symbol_graph_distance_to_query(&project_id, &symbol.id, &query, 2)
                        .map(|distance| symbol_graph_distance_boost(distance))
                        .unwrap_or_default();
                    let score = score_symbol_hit(&file.path, &symbol.name, &query) + edge_boost;
                    SymbolHit {
                        path: file.path,
                        symbol: symbol.name,
                        kind: symbol.kind,
                        start_line: symbol.start_line,
                        end_line: symbol.end_line,
                        score,
                    }
                })
                .collect::<Vec<_>>();
            hits.sort_by(|a, b| {
                b.score
                    .cmp(&a.score)
                    .then_with(|| a.path.cmp(&b.path))
                    .then_with(|| a.start_line.cmp(&b.start_line))
                    .then_with(|| a.symbol.cmp(&b.symbol))
            });
            hits.truncate(limit);
            Ok(hits)
        })
        .await
        .context("Failed to join symbol search task")?
    }

    async fn search_content(&self, query: &str, limit: usize) -> Result<Vec<ContentHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        let app = self.app.clone();
        let query = query.to_string();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let project = state
                .codegraph
                .list_projects()?
                .into_iter()
                .find(|project| project.id == project_id)
                .ok_or_else(|| anyhow!("Project not found for content search"))?;
            let root = PathBuf::from(project.root_path);
            let terms = search_terms(&query);
            let regex = search_regex(&query);
            if terms.is_empty() && regex.is_none() {
                return Ok(Vec::new());
            }
            let rg_candidates = rg_candidate_paths(&root, &terms);

            let mut hits = Vec::new();
            for file in state.codegraph.project_files(&project_id)? {
                if !looks_searchable_text_file(&file.path) {
                    continue;
                }
                if let Some(candidates) = rg_candidates.as_ref() {
                    if !candidates.is_empty() && !candidates.contains(&file.path) {
                        continue;
                    }
                }
                let rel_path = sanitize_relative_path(&file.path)?;
                let full_path = root.join(&rel_path);
                let Ok(content) = std::fs::read_to_string(&full_path) else {
                    continue;
                };
                collect_content_hits(
                    &file.path,
                    &content,
                    &terms,
                    regex.as_ref(),
                    &query,
                    &mut hits,
                );
            }

            hits.sort_by(|a, b| {
                b.score
                    .cmp(&a.score)
                    .then_with(|| a.path.cmp(&b.path))
                    .then_with(|| a.start_line.cmp(&b.start_line))
            });
            hits.truncate(limit);
            Ok(hits)
        })
        .await
        .context("Failed to join content search task")?
    }
}

struct WorkspaceFileReader {
    workspace_path: Option<PathBuf>,
}

#[async_trait]
impl FileReader for WorkspaceFileReader {
    async fn read_file(&self, path: &str, max_chars: usize) -> Result<EvidenceSlice> {
        let workspace_path = self
            .workspace_path
            .clone()
            .ok_or_else(|| anyhow!("No active workspace path configured"))?;
        let rel_path = sanitize_relative_path(path)?;
        tauri::async_runtime::spawn_blocking(move || {
            let root = std::fs::canonicalize(&workspace_path)
                .context("Failed to canonicalize workspace path")?;
            let full_path = root.join(&rel_path);
            let canonical = std::fs::canonicalize(&full_path)
                .with_context(|| format!("Failed to canonicalize {}", rel_path.display()))?;
            if !canonical.starts_with(&root) {
                return Err(anyhow!("Refusing to read outside workspace"));
            }

            let mut content = std::fs::read_to_string(&canonical)
                .with_context(|| format!("Failed to read {}", rel_path.display()))?;
            if content.len() > max_chars {
                content.truncate(max_chars);
                content.push_str("\n...[TRUNCATED]...");
            }
            let line_count = content.lines().count().max(1);
            let rel_display = rel_path.to_string_lossy().to_string();
            Ok(EvidenceSlice {
                lease: EvidenceLease::new(
                    rel_display,
                    None,
                    LineRange {
                        start: 1,
                        end: line_count,
                    },
                    &content,
                ),
                content,
            })
        })
        .await
        .context("Failed to join file read task")?
    }

    async fn read_range(
        &self,
        path: &str,
        symbol: Option<&str>,
        start_line: u32,
        end_line: u32,
        max_chars: usize,
    ) -> Result<EvidenceSlice> {
        let workspace_path = self
            .workspace_path
            .clone()
            .ok_or_else(|| anyhow!("No active workspace path configured"))?;
        let rel_path = sanitize_relative_path(path)?;
        let symbol = symbol.map(ToString::to_string);
        tauri::async_runtime::spawn_blocking(move || {
            let root = std::fs::canonicalize(&workspace_path)
                .context("Failed to canonicalize workspace path")?;
            let full_path = root.join(&rel_path);
            let canonical = std::fs::canonicalize(&full_path)
                .with_context(|| format!("Failed to canonicalize {}", rel_path.display()))?;
            if !canonical.starts_with(&root) {
                return Err(anyhow!("Refusing to read outside workspace"));
            }

            let content = std::fs::read_to_string(&canonical)
                .with_context(|| format!("Failed to read {}", rel_path.display()))?;
            let start = start_line.max(1);
            let end = end_line.max(start);
            let mut slice = content
                .lines()
                .enumerate()
                .filter(|(idx, _)| {
                    let line_no = (*idx + 1) as u32;
                    line_no >= start && line_no <= end
                })
                .map(|(_, line)| line)
                .collect::<Vec<_>>()
                .join("\n");
            if slice.len() > max_chars {
                slice.truncate(max_chars);
                slice.push_str("\n...[TRUNCATED]...");
            }
            let rel_display = rel_path.to_string_lossy().to_string();
            Ok(EvidenceSlice {
                lease: EvidenceLease::new(
                    rel_display,
                    symbol,
                    LineRange {
                        start: start as usize,
                        end: end as usize,
                    },
                    &slice,
                ),
                content: slice,
            })
        })
        .await
        .context("Failed to join range read task")?
    }

    async fn verify_lease(&self, lease: &EvidenceLease) -> Result<LeaseStatus> {
        let workspace_path = self
            .workspace_path
            .clone()
            .ok_or_else(|| anyhow!("No active workspace path configured"))?;
        let lease = lease.clone();
        tauri::async_runtime::spawn_blocking(move || {
            let root = std::fs::canonicalize(&workspace_path)
                .context("Failed to canonicalize workspace path")?;
            let rel_path = sanitize_relative_path(&lease.path)?;
            let full_path = root.join(&rel_path);
            let canonical = match std::fs::canonicalize(&full_path) {
                Ok(path) => path,
                Err(_) => return Ok(LeaseStatus::Expired),
            };
            if !canonical.starts_with(&root) {
                return Ok(LeaseStatus::Expired);
            }
            let content = match std::fs::read_to_string(&canonical) {
                Ok(content) => content,
                Err(_) => return Ok(LeaseStatus::Expired),
            };
            let current = content
                .lines()
                .enumerate()
                .filter(|(idx, _)| {
                    let line_no = *idx + 1;
                    line_no >= lease.range.start && line_no <= lease.range.end
                })
                .map(|(_, line)| line)
                .collect::<Vec<_>>()
                .join("\n");
            if agent_core::hash_content(&current) == lease.content_hash {
                Ok(LeaseStatus::Valid)
            } else {
                Ok(LeaseStatus::Expired)
            }
        })
        .await
        .context("Failed to join lease verification task")?
    }
}

struct TauriProgressSink {
    channel: Channel<AgentEvent>,
}

#[async_trait]
impl ProgressSink for TauriProgressSink {
    async fn emit(&self, progress: AgentProgress) {
        match progress {
            AgentProgress::Status(status) => {
                let _ = self.channel.send(AgentEvent::Status(status));
            }
            AgentProgress::Trace(_) => {}
        }
    }
}

fn collect_answer_sources(answer: &agent_core::AgentAnswer) -> Vec<SourceRef> {
    let answer_lower = answer.content.to_lowercase();
    let mut sources = answer
        .cited_entries
        .iter()
        .filter_map(|entry_id| answer.notebook.get(entry_id))
        .filter_map(|entry| source_from_entry(entry))
        .collect::<Vec<_>>();

    sources.sort_by_key(|source| {
        let file_name = source
            .path
            .rsplit('/')
            .next()
            .unwrap_or(source.path.as_str())
            .to_lowercase();
        let mentioned = answer_lower.contains(&source.path.to_lowercase())
            || answer_lower.contains(&file_name);
        (
            !mentioned,
            source.end_line.saturating_sub(source.start_line),
            source.path.clone(),
            source.start_line,
        )
    });

    let mut seen = std::collections::BTreeSet::new();
    sources.retain(|source| {
        let key = (source.path.clone(), source.start_line, source.end_line);
        seen.insert(key)
    });

    if sources.iter().any(|source| {
        let file_name = source.path.rsplit('/').next().unwrap_or(source.path.as_str());
        answer_lower.contains(&source.path.to_lowercase())
            || answer_lower.contains(&file_name.to_lowercase())
    }) {
        sources.retain(|source| {
            let file_name = source.path.rsplit('/').next().unwrap_or(source.path.as_str());
            answer_lower.contains(&source.path.to_lowercase())
                || answer_lower.contains(&file_name.to_lowercase())
        });
    }

    sources.truncate(3);
    sources
}

fn source_from_entry(entry: &agent_core::NotebookEntry) -> Option<SourceRef> {
    match &entry.provenance {
        agent_core::Provenance::File {
            path,
            symbol,
            range,
            ..
        } => Some(SourceRef {
            path: path.clone(),
            start_line: range.start,
            end_line: range.end,
            symbol: symbol.clone(),
            label: source_label(path, symbol.as_deref(), &entry.statement),
        }),
        _ => None,
    }
}

fn source_label(path: &str, symbol: Option<&str>, statement: &str) -> String {
    if let Some(symbol) = symbol {
        return symbol.to_string();
    }
    let lower = statement.to_lowercase();
    if lower.contains("changed files") || lower.contains("file list") {
        return "Changed files list".to_string();
    }
    path.rsplit('/')
        .next()
        .unwrap_or(path)
        .trim_end_matches(".tsx")
        .trim_end_matches(".ts")
        .trim_end_matches(".rs")
        .to_string()
}

fn score_file_hit(path: &str, query: &str) -> i64 {
    let path_lower = path.to_lowercase();
    let query_lower = query.to_lowercase();
    if path_lower == query_lower {
        100
    } else if path_lower.ends_with(&query_lower) {
        75
    } else if path_lower.contains(&query_lower) {
        50
    } else {
        10
    }
}

fn score_symbol_hit(path: &str, symbol: &str, query: &str) -> i64 {
    let query_lower = query.to_lowercase();
    let symbol_lower = symbol.to_lowercase();
    let path_score = score_file_hit(path, query);
    let symbol_score = if symbol_lower == query_lower {
        120
    } else if symbol_lower.contains(&query_lower) {
        90
    } else {
        0
    };
    let score = path_score + symbol_score;
    if score == 0 {
        25
    } else {
        score
    }
}

fn symbol_graph_distance_boost(distance: Option<usize>) -> i64 {
    match distance {
        Some(1) => 80,
        Some(2) => 45,
        _ => 0,
    }
}

fn looks_searchable_text_file(path: &str) -> bool {
    let lower = path.to_lowercase();
    if lower.starts_with("node_modules/")
        || lower.contains("/node_modules/")
        || lower.starts_with("target/")
        || lower.contains("/target/")
        || lower.starts_with("dist/")
        || lower.contains("/dist/")
        || lower.starts_with(".git/")
        || lower.contains("/.git/")
        || lower.ends_with(".lock")
    {
        return false;
    }
    matches!(
        lower.rsplit('.').next(),
        Some(
            "rs" | "ts"
                | "tsx"
                | "js"
                | "jsx"
                | "css"
                | "scss"
                | "html"
                | "md"
                | "json"
                | "toml"
                | "yaml"
                | "yml"
                | "sql"
                | "py"
                | "go"
                | "java"
                | "kt"
                | "swift"
                | "c"
                | "cc"
                | "cpp"
                | "h"
                | "hpp"
        )
    )
}

fn rg_candidate_paths(root: &Path, terms: &[String]) -> Option<HashSet<String>> {
    if terms.is_empty() {
        return None;
    }

    let mut candidates = HashSet::new();
    for term in terms.iter().take(8) {
        let output = Command::new("rg")
            .current_dir(root)
            .args([
                "--files-with-matches",
                "--ignore-case",
                "--fixed-strings",
                "--glob",
                "!node_modules/**",
                "--glob",
                "!target/**",
                "--glob",
                "!dist/**",
                "--glob",
                "!build/**",
                "--",
                term,
            ])
            .output()
            .ok()?;

        if !output.status.success() && output.status.code() != Some(1) {
            return None;
        }

        for line in String::from_utf8_lossy(&output.stdout).lines() {
            let normalized = line.trim().replace('\\', "/");
            if !normalized.is_empty() && looks_searchable_text_file(&normalized) {
                candidates.insert(normalized);
            }
            if candidates.len() >= 512 {
                return Some(candidates);
            }
        }
    }

    Some(candidates)
}

fn search_terms(query: &str) -> Vec<String> {
    let mut terms = query
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '-')
        .map(|term| term.trim().to_lowercase())
        .filter(|term| term.len() > 2)
        .filter(|term| {
            !matches!(
                term.as_str(),
                "the" | "and" | "for" | "with" | "where" | "what" | "how" | "our" | "code"
            )
        })
        .collect::<Vec<_>>();
    terms.sort();
    terms.dedup();
    terms
}

fn search_regex(query: &str) -> Option<regex::Regex> {
    let trimmed = query.trim();
    if trimmed.len() < 3 || trimmed.len() > 200 {
        return None;
    }
    RegexBuilder::new(trimmed)
        .case_insensitive(true)
        .build()
        .ok()
}

fn collect_content_hits(
    path: &str,
    content: &str,
    terms: &[String],
    regex: Option<&regex::Regex>,
    query: &str,
    hits: &mut Vec<ContentHit>,
) {
    let lines = content.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        return;
    }
    let query_lower = query.to_lowercase();
    let path_lower = path.to_lowercase();
    for (idx, line) in lines.iter().enumerate() {
        let line_lower = line.to_lowercase();
        let mut score = 0i64;
        if !query_lower.is_empty() && line_lower.contains(&query_lower) {
            score += 80;
        }
        if regex
            .map(|regex| regex.is_match(line))
            .unwrap_or(false)
        {
            score += 90;
        }
        let term_matches = terms
            .iter()
            .filter(|term| line_lower.contains(term.as_str()))
            .count();
        if term_matches == 0 && score == 0 {
            continue;
        }
        score += (term_matches as i64) * 12;
        if !terms.is_empty() && term_matches == terms.len() {
            score += 40;
        }
        for term in terms {
            if path_lower.contains(term) {
                score += 4;
            }
        }
        if line_lower.contains("class")
            || line_lower.contains("function")
            || line_lower.contains("const")
            || line_lower.contains("fn ")
        {
            score += 3;
        }

        let start_idx = idx.saturating_sub(4);
        let end_idx = (idx + 4).min(lines.len().saturating_sub(1));
        push_or_merge_content_hit(
            ContentHit {
                path: path.to_string(),
                start_line: (start_idx + 1) as u32,
                end_line: (end_idx + 1) as u32,
                preview: line.trim().chars().take(200).collect(),
                score,
            },
            hits,
        );
    }
}

fn push_or_merge_content_hit(hit: ContentHit, hits: &mut Vec<ContentHit>) {
    if let Some(existing) = hits.iter_mut().find(|existing| {
        existing.path == hit.path
            && existing.start_line <= hit.end_line
            && hit.start_line <= existing.end_line
    }) {
        existing.start_line = existing.start_line.min(hit.start_line);
        existing.end_line = existing.end_line.max(hit.end_line);
        existing.score += hit.score;
        if existing.preview.len() < hit.preview.len() {
            existing.preview = hit.preview;
        }
    } else {
        hits.push(hit);
    }
}

fn sanitize_relative_path(path: &str) -> Result<PathBuf> {
    let trimmed = path.trim_start_matches(['/', '\\']);
    let rel = PathBuf::from(trimmed);
    if rel
        .components()
        .any(|component| matches!(component, std::path::Component::ParentDir))
    {
        return Err(anyhow!("Parent directory segments are not allowed"));
    }
    Ok(rel)
}

async fn write_trace(run_id: &str, trace: &agent_core::Trace) -> Result<()> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine home directory"))?;
    let dir = home.join(".loco").join("traces");
    let path = dir.join(format!("{}.jsonl", run_id));
    let jsonl = trace.to_jsonl();
    tauri::async_runtime::spawn_blocking(move || {
        std::fs::create_dir_all(&dir)?;
        std::fs::write(&path, jsonl)?;
        log::info!("Wrote agent trace to {}", path.display());
        Ok::<_, anyhow::Error>(())
    })
    .await
    .context("Failed to join trace write task")??;
    Ok(())
}

#[derive(Debug, Deserialize)]
struct AgentConfig {
    budgets: Option<Budgets>,
}

async fn load_budgets() -> Result<Budgets> {
    let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine home directory"))?;
    let path = home.join(".loco").join("agent.toml");
    if !path.exists() {
        return Ok(Budgets::default());
    }

    tauri::async_runtime::spawn_blocking(move || {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        let config: AgentConfig = toml::from_str(&raw)
            .with_context(|| format!("Failed to parse {}", path.display()))?;
        Ok::<_, anyhow::Error>(config.budgets.unwrap_or_default())
    })
    .await
    .context("Failed to join budget config read task")?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_search_supports_regex_and_merges_overlapping_windows() {
        let content = [
            "const idle = true;",
            "const statusPill = createMemo(() => activeStatus());",
            "const elapsed = formatElapsed(statusElapsedMs());",
            "return <div class=\"status-pill\">{statusPill()}</div>;",
            "export default statusPill;",
        ]
        .join("\n");
        let regex = search_regex("status.*pill");
        let mut hits = Vec::new();

        collect_content_hits(
            "src/AiChat.tsx",
            &content,
            &search_terms("status pill"),
            regex.as_ref(),
            "status.*pill",
            &mut hits,
        );

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "src/AiChat.tsx");
        assert_eq!(hits[0].start_line, 1);
        assert_eq!(hits[0].end_line, 5);
        assert!(hits[0].score >= 180);
    }

    #[test]
    fn repo_text_search_skips_generated_and_binary_like_paths() {
        assert!(!looks_searchable_text_file("node_modules/pkg/index.ts"));
        assert!(!looks_searchable_text_file("target/debug/main.rs"));
        assert!(!looks_searchable_text_file("Cargo.lock"));
        assert!(looks_searchable_text_file("src/App.tsx"));
        assert!(looks_searchable_text_file("src-tauri/src/orchestrator.rs"));
    }

    #[test]
    fn rg_candidate_paths_skips_when_no_terms_are_available() {
        assert!(rg_candidate_paths(Path::new("."), &[]).is_none());
    }

    #[test]
    fn symbol_graph_distance_boost_prioritizes_nearer_edges() {
        assert_eq!(symbol_graph_distance_boost(Some(1)), 80);
        assert_eq!(symbol_graph_distance_boost(Some(2)), 45);
        assert_eq!(symbol_graph_distance_boost(Some(3)), 0);
        assert_eq!(symbol_graph_distance_boost(None), 0);
    }
}
