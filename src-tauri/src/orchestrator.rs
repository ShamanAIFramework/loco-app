// src-tauri/src/orchestrator.rs
//! Tauri host adapter for the deterministic agent-core runtime.

use crate::agent::AgentEvent;
use crate::conversation_store::Role;
use crate::engine::EngineState;
use crate::{AppError, AppState};
use agent_core::{
    AgentProgress, AgentRuntime, Budgets, CodeSearch, ConversationTurn, EvidenceLease,
    EvidenceSlice, FileHit, FileReader, LeaseStatus, LineRange, LlmClient, LlmRequest, LlmResponse,
    LlmUsage, ProgressSink, RuntimeMetadata, SymbolHit, TaskRequest, TraceEvent,
};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use std::path::{Path, PathBuf};
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
            progress: &progress,
        },
    )
    .await?;

    result
        .trace
        .events
        .insert(1, TraceEvent::RuntimeMetadata { metadata });
    write_trace(&run_id, &result.trace).await?;

    let final_response = render_answer_with_citations(&result.answer);
    let _ = on_chunk.send(AgentEvent::Chunk(final_response.clone()));

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
            let mut files = state
                .codegraph
                .project_files(&project_id)?
                .into_iter()
                .map(|file| file.path)
                .collect::<Vec<_>>();
            files.sort();
            files.truncate(max_entries);
            Ok(files)
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
                    let score = score_symbol_hit(&file.path, &symbol.name, &query);
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

fn render_answer_with_citations(answer: &agent_core::AgentAnswer) -> String {
    if answer.cited_entries.is_empty() {
        return answer.content.clone();
    }

    let mut rendered = answer.content.clone();
    rendered.push_str("\n\nSources:\n");
    for entry_id in &answer.cited_entries {
        if let Some(entry) = answer.notebook.get(entry_id) {
            rendered.push_str(&format!("- `{}` {}\n", entry.id.0, entry.statement));
        }
    }
    rendered
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
