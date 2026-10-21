// src-tauri/src/orchestrator.rs
//! Tauri host adapter for the deterministic agent-core runtime.

use crate::agent::{AgentEvent, NotebookEntryView, SourceRef};
use crate::codegraph::{SourceFile, Symbol};
use crate::conversation_store::Role;
use crate::engine::EngineState;
use crate::{AppError, AppState};
use agent_core::{
    AgentProgress, AgentRuntime, Budgets, CodeSearch, ContentHit, ConversationTurn, EvidenceLease,
    EvidenceSlice, FileHit, FileReader, LeaseStatus, LineRange, LlmClient, LlmRequest, LlmResponse,
    LlmUsage, ProgressSink, RuntimeMetadata, SymbolHit, SymbolRelation, TaskRequest, TokenCounter,
    TraceEvent, VectorHit,
};
use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use regex::RegexBuilder;
use serde::Deserialize;
use serde_json::json;
use std::collections::{BTreeMap, HashMap, HashSet};
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
        intake_only: false,
    };

    let llm = LlamaServerClient {
        client: reqwest::Client::new(),
        port: conn.port,
        api_key: conn.api_key,
        slot: app.state::<AppState>().llm_slot.clone(),
    };
    let metadata = llm.runtime_metadata().await;
    let trace_logger = Arc::new(
        crate::trace_writer::TraceLogger::new(run_id.clone(), Some(question.clone()))
            .context("Failed to initialize trace logger")?,
    );
    if let Some(model_id) = &metadata.model_id {
        trace_logger.set_model_id(model_id.clone());
    }
    trace_logger.record_event(&TraceEvent::RuntimeMetadata {
        metadata: metadata.clone(),
    });

    let search = TauriCodeSearch {
        app: app.clone(),
        project_id,
    };
    let reader = WorkspaceFileReader {
        workspace_path: conversation.workspace_path.clone().map(PathBuf::from),
    };
    let progress = TauriProgressSink {
        channel: on_chunk.clone(),
        trace_logger: trace_logger.clone(),
    };

    let mut result = match agent_core::run_agent(
        task,
        AgentRuntime {
            llm: &llm,
            code_search: &search,
            file_reader: &reader,
            token_counter: &llm,
            progress: &progress,
        },
    )
    .await
    {
        Ok(res) => res,
        Err(err) => {
            trace_logger.record_error(&err);
            return Err(err.into());
        }
    };

    result
        .trace
        .events
        .insert(1, TraceEvent::RuntimeMetadata { metadata });
    write_trace(&run_id, &result.trace).await?;

    let final_response = result.answer.content.clone();
    let sources = collect_answer_sources(&result.answer);
    let notebook_views = collect_notebook_views(&result.answer.notebook);
    let _ = on_chunk.send(AgentEvent::Chunk(final_response.clone()));
    if !sources.is_empty() {
        let _ = on_chunk.send(AgentEvent::Sources(sources));
    }
    if !notebook_views.is_empty() {
        let _ = on_chunk.send(AgentEvent::Notebook(notebook_views));
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

        log::info!(
            "[llama-server] [{:?}] Starting call_id={} (schema={}, prompt_chars={})",
            request.phase,
            request.call_id,
            request.schema_name,
            request.prompt.len()
        );

        let started = std::time::Instant::now();
        let res = self
            .client
            .post(format!(
                "http://127.0.0.1:{}/v1/chat/completions",
                self.port
            ))
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
            log::error!(
                "[llama-server] [{:?}] Call {} failed with {}: {}",
                request.phase,
                request.call_id,
                status,
                value
            );
            return Err(anyhow!(
                "llama-server request failed with {}: {}",
                status,
                value
            ));
        }

        let raw_json = value["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("{}")
            .to_string();
        let usage = &value["usage"];
        let wall_ms = started.elapsed().as_millis() as u64;

        log::info!(
            "[llama-server] [{:?}] Finished call_id={} in {}ms (queue_wait={}ms, prompt_tokens={:?}, decode_tokens={:?})",
            request.phase,
            request.call_id,
            wall_ms,
            queue_wait_ms,
            usage["prompt_tokens"].as_u64(),
            usage["completion_tokens"].as_u64()
        );

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
                wall_ms: Some(wall_ms),
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

#[derive(Clone, Debug)]
struct RelationEdge {
    target_name: String,
    kind: String,
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
                if !looks_searchable_text_file(&file.path) {
                    continue;
                }
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

    async fn changed_files(&self, limit: usize) -> Result<Vec<FileHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        let app = self.app.clone();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let project = state
                .codegraph
                .list_projects()?
                .into_iter()
                .find(|project| project.id == project_id)
                .ok_or_else(|| anyhow!("Project not found for changed file search"))?;
            let output = Command::new("git")
                .current_dir(project.root_path)
                .args(["status", "--porcelain"])
                .output()
                .context("Failed to read git status")?;
            if !output.status.success() {
                return Ok(Vec::new());
            }

            let mut hits = Vec::new();
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                if let Some((status, path)) = parse_git_status_line(line) {
                    if looks_searchable_text_file(&path) {
                        hits.push(FileHit {
                            path,
                            score: score_git_status(&status),
                        });
                    }
                }
                if hits.len() >= limit {
                    break;
                }
            }
            Ok(hits)
        })
        .await
        .context("Failed to join changed file search task")?
    }

    async fn search_adjacent_files(&self, seed_paths: &[String], limit: usize) -> Result<Vec<FileHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        if seed_paths.is_empty() {
            return Ok(Vec::new());
        }
        let app = self.app.clone();
        let seed_paths = seed_paths.to_vec();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let mut hits = state
                .codegraph
                .project_files(&project_id)?
                .into_iter()
                .filter(|file| looks_searchable_text_file(&file.path))
                .filter_map(|file| {
                    let score = score_adjacent_file(&file.path, &seed_paths);
                    (score > 0).then_some(FileHit {
                        path: file.path,
                        score,
                    })
                })
                .collect::<Vec<_>>();
            hits.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
            hits.truncate(limit);
            Ok(hits)
        })
        .await
        .context("Failed to join adjacent file search task")?
    }

    async fn search_imported_files(&self, seed_paths: &[String], limit: usize) -> Result<Vec<FileHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        if seed_paths.is_empty() {
            return Ok(Vec::new());
        }
        let app = self.app.clone();
        let seed_paths = seed_paths.to_vec();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let project = state
                .codegraph
                .list_projects()?
                .into_iter()
                .find(|project| project.id == project_id)
                .ok_or_else(|| anyhow!("Project not found for imported file search"))?;
            let root = PathBuf::from(project.root_path);
            let indexed_paths = state
                .codegraph
                .project_files(&project_id)?
                .into_iter()
                .filter(|file| looks_searchable_text_file(&file.path))
                .map(|file| file.path)
                .collect::<HashSet<_>>();

            let mut hits = imported_file_hits(&root, &indexed_paths, &seed_paths);
            hits.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
            hits.truncate(limit);
            Ok(hits)
        })
        .await
        .context("Failed to join imported file search task")?
    }

    async fn search_importing_files(&self, seed_paths: &[String], limit: usize) -> Result<Vec<FileHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        if seed_paths.is_empty() {
            return Ok(Vec::new());
        }
        let app = self.app.clone();
        let seed_paths = seed_paths.to_vec();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let project = state
                .codegraph
                .list_projects()?
                .into_iter()
                .find(|project| project.id == project_id)
                .ok_or_else(|| anyhow!("Project not found for importing file search"))?;
            let root = PathBuf::from(project.root_path);
            let indexed_paths = state
                .codegraph
                .project_files(&project_id)?
                .into_iter()
                .filter(|file| looks_searchable_text_file(&file.path))
                .map(|file| file.path)
                .collect::<HashSet<_>>();

            let mut hits = importing_file_hits(&root, &indexed_paths, &seed_paths);
            hits.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
            hits.truncate(limit);
            Ok(hits)
        })
        .await
        .context("Failed to join importing file search task")?
    }

    async fn search_scope_files(&self, seed_paths: &[String], limit: usize) -> Result<Vec<FileHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        if seed_paths.is_empty() {
            return Ok(Vec::new());
        }
        let app = self.app.clone();
        let seed_paths = seed_paths.to_vec();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let indexed_paths = state
                .codegraph
                .project_files(&project_id)?
                .into_iter()
                .filter(|file| looks_searchable_text_file(&file.path))
                .map(|file| file.path)
                .collect::<HashSet<_>>();

            let mut hits = scope_file_hits(&indexed_paths, &seed_paths);
            hits.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
            hits.truncate(limit);
            Ok(hits)
        })
        .await
        .context("Failed to join scope file search task")?
    }

    async fn search_context_docs(
        &self,
        queries: &[String],
        seed_paths: &[String],
        limit: usize,
    ) -> Result<Vec<FileHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        if queries.is_empty() || seed_paths.is_empty() {
            return Ok(Vec::new());
        }
        let app = self.app.clone();
        let queries = queries.to_vec();
        let seed_paths = seed_paths.to_vec();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let indexed_paths = state
                .codegraph
                .project_files(&project_id)?
                .into_iter()
                .filter(|file| looks_searchable_text_file(&file.path))
                .map(|file| file.path)
                .collect::<HashSet<_>>();

            let mut hits = context_doc_hits(&indexed_paths, &queries, &seed_paths);
            hits.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
            hits.truncate(limit);
            Ok(hits)
        })
        .await
        .context("Failed to join context doc search task")?
    }

    async fn search_directory_files(
        &self,
        directories: &[String],
        limit: usize,
    ) -> Result<Vec<FileHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        if directories.is_empty() {
            return Ok(Vec::new());
        }
        let app = self.app.clone();
        let directories = directories.to_vec();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let indexed_paths = state
                .codegraph
                .project_files(&project_id)?
                .into_iter()
                .filter(|file| looks_searchable_text_file(&file.path))
                .map(|file| file.path)
                .collect::<HashSet<_>>();

            let mut hits = directory_file_hits(&indexed_paths, &directories);
            hits.sort_by(|a, b| b.score.cmp(&a.score).then_with(|| a.path.cmp(&b.path)));
            hits.truncate(limit);
            Ok(hits)
        })
        .await
        .context("Failed to join directory file search task")?
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
                .project_files(&project_id)?
                .into_iter()
                .filter(|file| looks_searchable_text_file(&file.path))
                .filter_map(|file| {
                    let score = score_file_hit(&file.path, &query);
                    (score > 0).then_some(FileHit {
                        path: file.path,
                        score,
                    })
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
            let mut candidates = BTreeMap::new();
            for candidate_query in symbol_candidate_queries(&query) {
                for (file, symbol) in state
                    .codegraph
                    .search_symbols(&project_id, &candidate_query)?
                {
                    candidates
                        .entry(symbol.id.clone())
                        .or_insert((file, symbol));
                }
            }

            let mut hits = candidates
                .into_values()
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

    async fn search_file_symbols(&self, path: &str) -> Result<Vec<SymbolHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        let app = self.app.clone();
        let path = path.to_string();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let symbols = state.codegraph.symbols_for_path(&project_id, &path)?;
            Ok(symbols
                .into_iter()
                .map(|s| SymbolHit {
                    path: path.clone(),
                    symbol: s.name,
                    kind: s.kind,
                    start_line: s.start_line,
                    end_line: s.end_line,
                    score: 100,
                })
                .collect())
        })
        .await
        .context("Failed to join file symbols task")?
    }

    async fn search_vector(&self, query: &str, limit: usize) -> Result<Vec<VectorHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        let app = self.app.clone();
        let query = query.to_string();
        tauri::async_runtime::spawn_blocking(move || -> Result<Vec<VectorHit>> {
            let state = app.state::<AppState>();
            let project = state
                .codegraph
                .list_projects()?
                .into_iter()
                .find(|project| project.id == project_id)
                .ok_or_else(|| anyhow!("Project not found for vector search"))?;
            let root = PathBuf::from(project.root_path);
            let files = state.codegraph.project_files(&project_id)?;
            let symbols = state.codegraph.project_symbols(&project_id)?;
            crate::vector_search::search_project_vectors(&root, &files, &symbols, &query, limit)
        })
        .await
        .context("Failed to join vector search task")?
    }

    async fn search_related_symbols(
        &self,
        relation: SymbolRelation,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SymbolHit>> {
        let Some(project_id) = self.project_id.clone() else {
            return Ok(Vec::new());
        };
        let app = self.app.clone();
        let query = query.to_string();
        tauri::async_runtime::spawn_blocking(move || {
            let state = app.state::<AppState>();
            let files = state.codegraph.project_files(&project_id)?;
            let files_by_id = files
                .into_iter()
                .map(|file| (file.id.clone(), file))
                .collect::<HashMap<_, _>>();
            let symbols = state.codegraph.project_symbols(&project_id)?;
            let mut edges_by_source = HashMap::<String, Vec<RelationEdge>>::new();
            for source in &symbols {
                for edge in state.codegraph.symbol_edges(&source.id)? {
                    edges_by_source
                        .entry(source.id.clone())
                        .or_default()
                        .push(RelationEdge {
                            target_name: edge.target_name,
                            kind: edge.kind,
                        });
                }
            }

            let hits = match relation {
                SymbolRelation::Definition => {
                    collect_definition_symbol_hits(&files_by_id, &symbols, &query, limit)
                }
                SymbolRelation::References => collect_referencing_symbol_hits(
                    &files_by_id,
                    &symbols,
                    &edges_by_source,
                    &query,
                    false,
                    limit,
                ),
                SymbolRelation::Callers => collect_referencing_symbol_hits(
                    &files_by_id,
                    &symbols,
                    &edges_by_source,
                    &query,
                    true,
                    limit,
                ),
                SymbolRelation::Callees => collect_callee_symbol_hits(
                    &files_by_id,
                    &symbols,
                    &edges_by_source,
                    &query,
                    limit,
                ),
            };
            Ok(hits)
        })
        .await
        .context("Failed to join related symbol search task")?
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

const GIT_DIFF_SYMBOL: &str = "__git_diff__";

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

            let content = std::fs::read_to_string(&canonical)
                .with_context(|| format!("Failed to read {}", rel_path.display()))?;
            let file_end = content.lines().count().max(1);
            let (content, range) = bounded_line_content(&content, 1, file_end, max_chars);
            let rel_display = rel_path.to_string_lossy().to_string();
            Ok(EvidenceSlice {
                lease: EvidenceLease::new(rel_display, None, range, &content),
                content,
            })
        })
        .await
        .context("Failed to join file read task")?
    }

    async fn read_diff(&self, path: &str, max_chars: usize) -> Result<EvidenceSlice> {
        let workspace_path = self
            .workspace_path
            .clone()
            .ok_or_else(|| anyhow!("No active workspace path configured"))?;
        let rel_path = sanitize_relative_path(path)?;
        tauri::async_runtime::spawn_blocking(move || {
            let root = std::fs::canonicalize(&workspace_path)
                .context("Failed to canonicalize workspace path")?;
            let full_path = root.join(&rel_path);
            if !full_path.starts_with(&root) {
                return Err(anyhow!("Refusing to diff outside workspace"));
            }

            let rel_display = rel_path.to_string_lossy().to_string();
            let diff = git_diff_for_path(&root, &rel_display)?;
            if diff.trim().is_empty() {
                return Err(anyhow!("No diff available for {}", rel_display));
            }
            let file_end = diff.lines().count().max(1);
            let (content, range) = bounded_line_content(&diff, 1, file_end, max_chars);
            Ok(EvidenceSlice {
                lease: EvidenceLease::new(
                    rel_display,
                    Some(GIT_DIFF_SYMBOL.to_string()),
                    range,
                    &content,
                ),
                content,
            })
        })
        .await
        .context("Failed to join diff read task")?
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
            let (start, end) = context_range_for_read(&content, start, end, symbol.is_some());
            let (slice, range) =
                bounded_line_content(&content, start as usize, end as usize, max_chars);
            let rel_display = rel_path.to_string_lossy().to_string();
            Ok(EvidenceSlice {
                lease: EvidenceLease::new(rel_display, symbol, range, &slice),
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
            if lease.symbol.as_deref() == Some(GIT_DIFF_SYMBOL) {
                let current = match git_diff_for_path(&root, &lease.path) {
                    Ok(diff) => diff
                        .lines()
                        .enumerate()
                        .filter(|(idx, _)| {
                            let line_no = *idx + 1;
                            line_no >= lease.range.start && line_no <= lease.range.end
                        })
                        .map(|(_, line)| line)
                        .collect::<Vec<_>>()
                        .join("\n"),
                    Err(_) => return Ok(LeaseStatus::Expired),
                };
                return if agent_core::hash_content(&current) == lease.content_hash {
                    Ok(LeaseStatus::Valid)
                } else {
                    Ok(LeaseStatus::Expired)
                };
            }
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
    trace_logger: Arc<crate::trace_writer::TraceLogger>,
}

#[async_trait]
impl ProgressSink for TauriProgressSink {
    async fn emit(&self, progress: AgentProgress) {
        match progress {
            AgentProgress::Status(status) => {
                let _ = self.channel.send(AgentEvent::Status(status));
            }
            AgentProgress::Trace(json_str) => {
                if let Ok(event) = serde_json::from_str::<TraceEvent>(&json_str) {
                    self.trace_logger.record_event(&event);
                }
            }
        }
    }
}

fn collect_answer_sources(answer: &agent_core::AgentAnswer) -> Vec<SourceRef> {
    let answer_lower = answer.content.to_lowercase();
    let mut sources = answer
        .cited_entries
        .iter()
        .filter_map(|entry_id| answer.notebook.get(entry_id))
        .filter_map(|entry| {
            source_from_entry(entry).or_else(|| {
                entry
                    .depends_on
                    .first()
                    .and_then(|dep_id| answer.notebook.get(dep_id))
                    .and_then(source_from_entry)
            })
        })
        .collect::<Vec<_>>();

    sources.sort_by_key(|source| {
        let file_name = source
            .path
            .rsplit('/')
            .next()
            .unwrap_or(source.path.as_str())
            .to_lowercase();
        let mentioned =
            answer_lower.contains(&source.path.to_lowercase()) || answer_lower.contains(&file_name);
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

    sources.truncate(3);
    sources
}

fn collect_notebook_views(notebook: &agent_core::Notebook) -> Vec<NotebookEntryView> {
    notebook
        .active_entries()
        .into_iter()
        .map(|entry| {
            let (file_path, line_range, symbol, lease_id) = match &entry.provenance {
                agent_core::Provenance::File {
                    path,
                    symbol,
                    range,
                    lease_id,
                } => (
                    Some(path.clone()),
                    Some([range.start, range.end]),
                    symbol.clone(),
                    Some(lease_id.0.clone()),
                ),
                _ => (None, None, None, None),
            };

            NotebookEntryView {
                id: entry.id.0.clone(),
                kind: format!("{:?}", entry.kind),
                statement: entry.statement.clone(),
                phase: format!("{:?}", entry.phase),
                status: format!("{:?}", entry.status),
                tags: entry.tags.iter().cloned().collect(),
                file_path,
                line_range,
                symbol,
                lease_id,
                depends_on: entry.depends_on.iter().map(|id| id.0.clone()).collect(),
            }
        })
        .collect()
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
        if symbol == GIT_DIFF_SYMBOL {
            return "Git diff".to_string();
        }
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

fn collect_definition_symbol_hits(
    files_by_id: &HashMap<String, SourceFile>,
    symbols: &[Symbol],
    query: &str,
    limit: usize,
) -> Vec<SymbolHit> {
    let mut exact_hits = Vec::new();
    let mut fuzzy_hits = Vec::new();
    for symbol in symbols {
        let Some(file) = files_by_id.get(&symbol.file_id) else {
            continue;
        };
        if !looks_searchable_text_file(&file.path) {
            continue;
        }
        let symbol_score = score_symbol_name(&symbol.name, query);
        let path_score = score_file_hit(&file.path, query);
        if symbol_score == 0 && path_score == 0 {
            continue;
        }
        let hit = symbol_hit(file, symbol, symbol_score + path_score);
        if exact_symbol_name_match(&symbol.name, query) {
            exact_hits.push(hit);
        } else {
            fuzzy_hits.push(hit);
        }
    }

    let mut hits = if exact_hits.is_empty() {
        fuzzy_hits
    } else {
        exact_hits
    };
    sort_and_truncate_symbol_hits(&mut hits, limit);
    hits
}

fn exact_symbol_name_match(symbol: &str, query: &str) -> bool {
    symbol.eq_ignore_ascii_case(query)
        || normalize_identifier(symbol) == normalize_identifier(query)
}

fn collect_referencing_symbol_hits(
    files_by_id: &HashMap<String, SourceFile>,
    symbols: &[Symbol],
    edges_by_source: &HashMap<String, Vec<RelationEdge>>,
    query: &str,
    calls_only: bool,
    limit: usize,
) -> Vec<SymbolHit> {
    let symbols_by_id = symbols
        .iter()
        .map(|symbol| (symbol.id.as_str(), symbol))
        .collect::<HashMap<_, _>>();
    let mut hits = BTreeMap::<String, SymbolHit>::new();

    for (source_id, edges) in edges_by_source {
        let Some(source) = symbols_by_id.get(source_id.as_str()) else {
            continue;
        };
        let Some(source_file) = files_by_id.get(&source.file_id) else {
            continue;
        };
        if !looks_searchable_text_file(&source_file.path) {
            continue;
        }
        for edge in edges {
            if calls_only && edge.kind != "call" {
                continue;
            }
            let target_score = score_symbol_name(&edge.target_name, query);
            if target_score == 0 {
                continue;
            }
            hits.entry(source.id.clone())
                .or_insert_with(|| symbol_hit(source_file, source, target_score + 90));
        }
    }

    let mut hits = hits.into_values().collect::<Vec<_>>();
    sort_and_truncate_symbol_hits(&mut hits, limit);
    hits
}

fn collect_callee_symbol_hits(
    files_by_id: &HashMap<String, SourceFile>,
    symbols: &[Symbol],
    edges_by_source: &HashMap<String, Vec<RelationEdge>>,
    query: &str,
    limit: usize,
) -> Vec<SymbolHit> {
    let mut definitions_by_name = HashMap::<String, Vec<&Symbol>>::new();
    for symbol in symbols {
        definitions_by_name
            .entry(symbol.name.to_lowercase())
            .or_default()
            .push(symbol);
    }

    let mut hits = BTreeMap::<String, SymbolHit>::new();
    for source in symbols {
        let Some(source_file) = files_by_id.get(&source.file_id) else {
            continue;
        };
        let source_score = score_symbol_name(&source.name, query);
        let path_score = score_file_hit(&source_file.path, query);
        if source_score == 0 && path_score == 0 {
            continue;
        }
        let Some(edges) = edges_by_source.get(&source.id) else {
            continue;
        };
        for edge in edges {
            if edge.kind != "call" {
                continue;
            }
            let Some(targets) = definitions_by_name.get(&edge.target_name.to_lowercase()) else {
                continue;
            };
            for target in targets {
                let Some(target_file) = files_by_id.get(&target.file_id) else {
                    continue;
                };
                if !looks_searchable_text_file(&target_file.path) {
                    continue;
                }
                let score = source_score + path_score + 90;
                hits.entry(target.id.clone())
                    .or_insert_with(|| symbol_hit(target_file, target, score));
            }
        }
    }

    let mut hits = hits.into_values().collect::<Vec<_>>();
    sort_and_truncate_symbol_hits(&mut hits, limit);
    hits
}

fn symbol_hit(file: &SourceFile, symbol: &Symbol, score: i64) -> SymbolHit {
    SymbolHit {
        path: file.path.clone(),
        symbol: symbol.name.clone(),
        kind: symbol.kind.clone(),
        start_line: symbol.start_line,
        end_line: symbol.end_line,
        score,
    }
}

fn sort_and_truncate_symbol_hits(hits: &mut Vec<SymbolHit>, limit: usize) {
    hits.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.start_line.cmp(&b.start_line))
            .then_with(|| a.symbol.cmp(&b.symbol))
    });
    hits.truncate(limit);
}

fn imported_file_hits(
    root: &Path,
    indexed_paths: &HashSet<String>,
    seed_paths: &[String],
) -> Vec<FileHit> {
    let mut hits = BTreeMap::<String, i64>::new();
    for seed in seed_paths {
        if !indexed_paths.contains(seed) || !looks_searchable_text_file(seed) {
            continue;
        }
        let Ok(rel_seed) = sanitize_relative_path(seed) else {
            continue;
        };
        let full_path = root.join(&rel_seed);
        let Ok(content) = std::fs::read_to_string(&full_path) else {
            continue;
        };

        let mut imported = Vec::new();
        for specifier in relative_import_specifiers(&content) {
            imported.extend(resolve_relative_import(seed, &specifier, indexed_paths));
        }
        for module_name in rust_module_declarations(&content) {
            imported.extend(resolve_rust_module(seed, &module_name, indexed_paths));
        }

        for path in imported {
            if path == *seed {
                continue;
            }
            let score = if is_test_or_spec_file(&path) { 105 } else { 95 };
            hits.entry(path)
                .and_modify(|existing| *existing = (*existing).max(score))
                .or_insert(score);
        }
    }

    hits.into_iter()
        .map(|(path, score)| FileHit { path, score })
        .collect()
}

fn importing_file_hits(
    root: &Path,
    indexed_paths: &HashSet<String>,
    seed_paths: &[String],
) -> Vec<FileHit> {
    let seed_set = seed_paths
        .iter()
        .filter(|path| indexed_paths.contains(*path))
        .cloned()
        .collect::<HashSet<_>>();
    if seed_set.is_empty() {
        return Vec::new();
    }

    let mut hits = BTreeMap::<String, i64>::new();
    for candidate in indexed_paths {
        if seed_set.contains(candidate) || !looks_searchable_text_file(candidate) {
            continue;
        }
        let Ok(rel_candidate) = sanitize_relative_path(candidate) else {
            continue;
        };
        let full_path = root.join(&rel_candidate);
        let Ok(content) = std::fs::read_to_string(&full_path) else {
            continue;
        };

        let mut resolved = Vec::new();
        for specifier in relative_import_specifiers(&content) {
            resolved.extend(resolve_relative_import(candidate, &specifier, indexed_paths));
        }
        for module_name in rust_module_declarations(&content) {
            resolved.extend(resolve_rust_module(candidate, &module_name, indexed_paths));
        }

        if resolved.iter().any(|path| seed_set.contains(path)) {
            let score = if is_test_or_spec_file(candidate) { 110 } else { 100 };
            hits.entry(candidate.clone())
                .and_modify(|existing| *existing = (*existing).max(score))
                .or_insert(score);
        }
    }

    hits.into_iter()
        .map(|(path, score)| FileHit { path, score })
        .collect()
}

fn scope_file_hits(indexed_paths: &HashSet<String>, seed_paths: &[String]) -> Vec<FileHit> {
    let mut hits = BTreeMap::<String, i64>::new();
    for seed in seed_paths {
        if !indexed_paths.contains(seed) {
            continue;
        }
        for dir in enclosing_directories(seed) {
            for path in indexed_paths {
                if path == seed || !scope_file_belongs_to_dir(path, &dir) {
                    continue;
                }
                let Some(base_score) = scope_file_score(path) else {
                    continue;
                };
                let score = base_score.saturating_sub((enclosing_distance(seed, &dir) as i64) * 8);
                hits.entry(path.clone())
                    .and_modify(|existing| *existing = (*existing).max(score))
                    .or_insert(score);
            }
        }
    }

    hits.into_iter()
        .map(|(path, score)| FileHit { path, score })
        .collect()
}

fn context_doc_hits(
    indexed_paths: &HashSet<String>,
    queries: &[String],
    seed_paths: &[String],
) -> Vec<FileHit> {
    let mut hits = BTreeMap::<String, i64>::new();
    for seed in seed_paths {
        if !indexed_paths.contains(seed) {
            continue;
        }
        for dir in enclosing_directories(seed) {
            for path in indexed_paths {
                if path == seed || !context_doc_belongs_to_dir(path, &dir) {
                    continue;
                }
                let Some(query_score) = context_doc_query_score(path, queries) else {
                    continue;
                };
                let distance_penalty = (enclosing_distance(seed, &dir) as i64) * 8;
                let score = query_score.saturating_sub(distance_penalty);
                if score <= 0 {
                    continue;
                }
                hits.entry(path.clone())
                    .and_modify(|existing| *existing = (*existing).max(score))
                    .or_insert(score);
            }
        }
    }

    hits.into_iter()
        .map(|(path, score)| FileHit { path, score })
        .collect()
}

fn directory_file_hits(indexed_paths: &HashSet<String>, directories: &[String]) -> Vec<FileHit> {
    let mut hits = BTreeMap::<String, i64>::new();
    for dir in directories {
        let normalized_dir = dir.trim_matches('/').replace('\\', "/");
        if normalized_dir.is_empty() || normalized_dir.contains("..") {
            continue;
        }
        for path in indexed_paths {
            let Some(score) = directory_file_score(path, &normalized_dir) else {
                continue;
            };
            hits.entry(path.clone())
                .and_modify(|existing| *existing = (*existing).max(score))
                .or_insert(score);
        }
    }

    hits.into_iter()
        .map(|(path, score)| FileHit { path, score })
        .collect()
}

fn directory_file_score(path: &str, dir: &str) -> Option<i64> {
    if path == dir || !path.starts_with(&format!("{dir}/")) {
        return None;
    }
    let rest = path.strip_prefix(&format!("{dir}/"))?;
    if rest.is_empty() {
        return None;
    }
    let depth = rest.matches('/').count();
    let mut score = if depth == 0 { 100 } else { 72 };
    let lower = path.to_lowercase();
    let file_name = lower.rsplit('/').next().unwrap_or(lower.as_str());
    if is_project_manifest_or_config(&lower) || is_project_entrypoint_file(&lower) {
        score += 25;
    } else if file_name.starts_with("readme.") || file_name == "readme" {
        score += 18;
    } else if is_repository_instruction_doc(&lower) {
        score += 18;
    } else if is_test_or_spec_file(&lower) {
        score += 8;
    }
    Some(score)
}

fn context_doc_belongs_to_dir(path: &str, dir: &str) -> bool {
    let path_lower = path.to_lowercase();
    if !is_context_doc_file(&path_lower) {
        return false;
    }
    let path_dir = path_directory(&path_lower);
    if dir.is_empty() {
        path_dir.is_empty() || path_dir == "docs" || path_dir.starts_with("docs/")
    } else {
        path_dir == dir
            || path_dir == format!("{dir}/docs")
            || path_dir.starts_with(&format!("{dir}/docs/"))
    }
}

fn context_doc_query_score(path: &str, queries: &[String]) -> Option<i64> {
    let path_lower = path.to_lowercase();
    let file_name = path_lower.rsplit('/').next().unwrap_or(path_lower.as_str());
    let path_normalized = normalize_identifier(&path_lower);
    let mut best = 0i64;

    for query in queries {
        let query_lower = query.to_lowercase();
        let query_normalized = normalize_identifier(query);
        let mut score = 0i64;
        if !query_normalized.is_empty() && path_normalized.contains(&query_normalized) {
            score += 105;
        }
        for term in search_terms(&query_lower) {
            let term_normalized = normalize_identifier(&term);
            if path_lower.contains(&format!("/{term}"))
                || path_lower.contains(&format!("{term}."))
                || path_lower.contains(&format!("-{term}"))
                || path_lower.contains(&format!("_{term}"))
            {
                score += 58;
            } else if path_lower.contains(&term) {
                score += 34;
            } else if !term_normalized.is_empty() && path_normalized.contains(&term_normalized) {
                score += 28;
            }
        }
        if is_project_overview_doc(&path_lower)
            && search_terms(&query_lower)
                .iter()
                .any(|term| matches!(term.as_str(), "architecture" | "design" | "docs" | "documentation" | "guide" | "overview" | "readme"))
        {
            score += 40;
        }
        if file_name.starts_with("readme.") || file_name == "readme" {
            score += 12;
        }
        best = best.max(score);
    }

    (best > 0).then_some(best)
}

fn enclosing_directories(path: &str) -> Vec<String> {
    let mut dirs = Vec::new();
    let mut current = Path::new(path).parent();
    while let Some(dir) = current {
        let normalized = normalized_relative_path(dir.to_path_buf()).unwrap_or_default();
        dirs.push(normalized);
        current = dir.parent();
    }
    dirs.sort_by_key(|dir| dir.matches('/').count());
    dirs.reverse();
    dirs.dedup();
    dirs
}

fn enclosing_distance(seed: &str, dir: &str) -> usize {
    let seed_dir = path_directory(&seed.to_lowercase());
    if dir.is_empty() {
        return seed_dir.split('/').filter(|part| !part.is_empty()).count();
    }
    seed_dir
        .strip_prefix(dir)
        .map(|rest| rest.split('/').filter(|part| !part.is_empty()).count())
        .unwrap_or(usize::MAX / 4)
}

fn scope_file_belongs_to_dir(path: &str, dir: &str) -> bool {
    let path_dir = path_directory(&path.to_lowercase());
    if dir.is_empty() {
        path_dir.is_empty()
    } else {
        path_dir == dir
    }
}

fn scope_file_score(path: &str) -> Option<i64> {
    let lower = path.to_lowercase();
    let file_name = lower.rsplit('/').next().unwrap_or(lower.as_str());
    if is_project_manifest_or_config(&lower) {
        Some(125)
    } else if is_repository_instruction_doc(&lower) {
        Some(115)
    } else if file_name.starts_with("readme.") || file_name == "readme" {
        Some(105)
    } else if matches!(
        file_name,
        "contributing.md" | "developers.md" | "developer-guide.md"
    ) {
        Some(95)
    } else {
        None
    }
}

fn relative_import_specifiers(content: &str) -> Vec<String> {
    let mut specifiers = Vec::new();
    for pattern in [
        r#"\b(?:import|export)\s+(?:[^'"]*?\s+from\s+)?["']([^"']+)["']"#,
        r#"\b(?:require|import)\(\s*["']([^"']+)["']\s*\)"#,
    ] {
        let Ok(regex) = RegexBuilder::new(pattern).multi_line(true).build() else {
            continue;
        };
        for capture in regex.captures_iter(content) {
            let Some(specifier) = capture.get(1).map(|capture| capture.as_str()) else {
                continue;
            };
            if specifier.starts_with('.') {
                specifiers.push(specifier.to_string());
            }
        }
    }
    specifiers.sort();
    specifiers.dedup();
    specifiers
}

fn rust_module_declarations(content: &str) -> Vec<String> {
    let Ok(regex) = RegexBuilder::new(r#"(?m)^\s*(?:pub\s+)?mod\s+([A-Za-z_][A-Za-z0-9_]*)\s*;"#)
        .build()
    else {
        return Vec::new();
    };
    let mut modules = regex
        .captures_iter(content)
        .filter_map(|capture| capture.get(1).map(|capture| capture.as_str().to_string()))
        .collect::<Vec<_>>();
    modules.sort();
    modules.dedup();
    modules
}

fn resolve_relative_import(
    seed_path: &str,
    specifier: &str,
    indexed_paths: &HashSet<String>,
) -> Vec<String> {
    let seed_dir = Path::new(seed_path).parent().unwrap_or_else(|| Path::new(""));
    let base = seed_dir.join(specifier);
    resolve_candidate_paths(base, indexed_paths)
}

fn resolve_rust_module(
    seed_path: &str,
    module_name: &str,
    indexed_paths: &HashSet<String>,
) -> Vec<String> {
    let seed_dir = Path::new(seed_path).parent().unwrap_or_else(|| Path::new(""));
    [
        seed_dir.join(format!("{module_name}.rs")),
        seed_dir.join(module_name).join("mod.rs"),
    ]
    .into_iter()
    .filter_map(normalized_relative_path)
    .filter(|path| indexed_paths.contains(path))
    .collect()
}

fn resolve_candidate_paths(base: PathBuf, indexed_paths: &HashSet<String>) -> Vec<String> {
    let mut candidates = Vec::new();
    candidates.push(base.clone());
    for extension in [
        "ts", "tsx", "js", "jsx", "rs", "css", "scss", "json", "toml",
    ] {
        candidates.push(base.with_extension(extension));
        candidates.push(base.join(format!("index.{extension}")));
        if extension == "rs" {
            candidates.push(base.join("mod.rs"));
        }
    }

    let mut resolved = candidates
        .into_iter()
        .filter_map(normalized_relative_path)
        .filter(|path| indexed_paths.contains(path))
        .collect::<Vec<_>>();
    resolved.sort();
    resolved.dedup();
    resolved
}

fn normalized_relative_path(path: PathBuf) -> Option<String> {
    let mut parts = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::Normal(part) => {
                parts.push(part.to_string_lossy().to_string());
            }
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                parts.pop()?;
            }
            std::path::Component::RootDir | std::path::Component::Prefix(_) => return None,
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

fn score_adjacent_file(path: &str, seed_paths: &[String]) -> i64 {
    let path_lower = path.to_lowercase();
    let path_key = normalized_file_stem_key(&path_lower);
    let path_dir = path_directory(&path_lower);
    let mut best = 0i64;

    for seed in seed_paths {
        let seed_lower = seed.to_lowercase();
        if path_lower == seed_lower {
            continue;
        }
        let seed_dir = path_directory(&seed_lower);
        let same_dir = path_dir == seed_dir;
        let nested_neighbor = !path_dir.is_empty()
            && !seed_dir.is_empty()
            && (path_dir.starts_with(&format!("{seed_dir}/"))
                || seed_dir.starts_with(&format!("{path_dir}/")));
        let same_or_related_name = !path_key.is_empty()
            && path_key == normalized_file_stem_key(&seed_lower)
            && (same_dir || nested_neighbor);

        let mut score = 0i64;
        if same_or_related_name {
            score += 105;
            if is_test_or_spec_file(&path_lower) {
                score += 25;
            }
            if is_style_file(&path_lower) {
                score += 10;
            }
        }
        if same_dir {
            score += 60;
        } else if nested_neighbor {
            score += 30;
        }
        best = best.max(score);
    }

    best
}

fn normalized_file_stem_key(path_lower: &str) -> String {
    let mut name = path_lower
        .rsplit('/')
        .next()
        .unwrap_or(path_lower)
        .to_string();
    while let Some((stem, ext)) = name.rsplit_once('.') {
        if ext.is_empty() || ext.len() > 5 {
            break;
        }
        name = stem.to_string();
    }
    for suffix in [".test", ".spec", ".module", "_test", "_spec", "-test", "-spec"] {
        if let Some(stripped) = name.strip_suffix(suffix) {
            name = stripped.to_string();
        }
    }
    for prefix in ["test_", "spec_", "test-", "spec-"] {
        if let Some(stripped) = name.strip_prefix(prefix) {
            name = stripped.to_string();
        }
    }
    normalize_identifier(&name)
}

fn path_directory(path_lower: &str) -> String {
    path_lower
        .rsplit_once('/')
        .map(|(dir, _)| dir.to_string())
        .unwrap_or_default()
}

fn is_style_file(path_lower: &str) -> bool {
    matches!(
        path_lower.rsplit('.').next(),
        Some("css" | "scss" | "sass" | "less")
    )
}

fn is_context_doc_file(path_lower: &str) -> bool {
    matches!(
        path_lower.rsplit('.').next(),
        Some("md" | "mdx" | "markdown" | "txt" | "rst" | "adoc")
    )
}

fn score_file_hit(path: &str, query: &str) -> i64 {
    let path_lower = path.to_lowercase();
    let query_lower = query.to_lowercase();
    let path_normalized = normalize_identifier(path);
    if path_lower == query_lower {
        return 140;
    } else if path_lower.ends_with(&query_lower) {
        return 110;
    } else if path_lower.contains(&query_lower) {
        return 80;
    }

    let mut score = 0i64;
    let terms = search_terms(query);
    if is_validation_context_query(&terms) && is_project_manifest_or_config(&path_lower) {
        score += 85;
    }
    if is_related_test_query(&terms) && is_test_or_spec_file(&path_lower) {
        score += 90;
    }
    if is_project_research_query(&terms) && is_project_overview_doc(&path_lower) {
        score += 85;
    }
    if is_repository_instruction_query(&terms) && is_repository_instruction_doc(&path_lower) {
        score += 100;
    }
    if is_project_entrypoint_query(&terms) && is_project_entrypoint_file(&path_lower) {
        score += 95;
    }
    for term in terms {
        let term_normalized = normalize_identifier(&term);
        if path_lower == term {
            score += 90;
        } else if path_lower.ends_with(&term) {
            score += 70;
        } else if path_lower.contains(&format!("/{term}"))
            || path_lower.contains(&format!("{term}."))
            || path_lower.contains(&format!("-{term}"))
            || path_lower.contains(&format!("_{term}"))
        {
            score += 55;
        } else if path_lower.contains(&term) {
            score += 30;
        } else if !term_normalized.is_empty() && path_normalized.contains(&term_normalized) {
            score += 24;
        }
    }
    score
}

fn is_validation_context_query(terms: &[String]) -> bool {
    terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "manifest"
                | "build"
                | "test"
                | "script"
                | "scripts"
                | "validation"
                | "command"
                | "workspace"
                | "config"
        )
    })
}

fn is_related_test_query(terms: &[String]) -> bool {
    let has_test_term = terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "test" | "tests" | "spec" | "specs" | "coverage" | "regression"
        )
    });
    let has_related_term = terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "related" | "nearby" | "coverage" | "regression" | "behavior"
        )
    });
    let is_script_query = terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "script" | "scripts" | "manifest" | "build" | "command" | "config" | "workspace"
        )
    });

    has_test_term && has_related_term && !is_script_query
}

fn is_project_research_query(terms: &[String]) -> bool {
    terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "overview"
                | "readme"
                | "architecture"
                | "architectural"
                | "design"
                | "docs"
                | "documentation"
                | "developer"
                | "guide"
                | "instruction"
                | "instructions"
                | "guidance"
        )
    })
}

fn is_repository_instruction_query(terms: &[String]) -> bool {
    terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "instruction"
                | "instructions"
                | "guidance"
                | "agent"
                | "agents"
                | "repository"
                | "repo"
                | "coding"
                | "convention"
                | "conventions"
        )
    })
}

fn is_project_entrypoint_query(terms: &[String]) -> bool {
    terms.iter().any(|term| {
        matches!(
            term.as_str(),
            "entrypoint"
                | "entrypoints"
                | "startup"
                | "bootstrap"
                | "boot"
                | "main"
                | "app"
                | "application"
        )
    })
}

fn is_project_overview_doc(path_lower: &str) -> bool {
    let file_name = path_lower.rsplit('/').next().unwrap_or(path_lower);
    file_name == "readme"
        || file_name.starts_with("readme.")
        || file_name == "architecture.md"
        || file_name == "design.md"
        || file_name == "contributing.md"
        || file_name == "developers.md"
        || file_name == "developer-guide.md"
        || path_lower.starts_with("docs/")
        || path_lower.contains("/docs/")
        || path_lower.starts_with("documentation/")
        || path_lower.contains("/documentation/")
        || path_lower.contains("architecture")
        || path_lower.contains("design")
}

fn is_project_entrypoint_file(path_lower: &str) -> bool {
    let file_name = path_lower.rsplit('/').next().unwrap_or(path_lower);
    matches!(
        file_name,
        "main.rs"
            | "main.ts"
            | "main.tsx"
            | "main.js"
            | "main.jsx"
            | "app.ts"
            | "app.tsx"
            | "app.js"
            | "app.jsx"
            | "index.ts"
            | "index.tsx"
            | "index.js"
            | "index.jsx"
            | "lib.rs"
            | "mod.rs"
            | "server.ts"
            | "server.js"
            | "startup.rs"
            | "bootstrap.ts"
            | "bootstrap.js"
    ) || path_lower.starts_with("src/main.")
        || path_lower.contains("/src/main.")
        || path_lower.starts_with("src-tauri/src/main.")
        || path_lower.contains("/src-tauri/src/main.")
        || path_lower.starts_with("src/app.")
        || path_lower.contains("/src/app.")
        || path_lower.starts_with("app/")
        || path_lower.contains("/app/")
}

fn is_repository_instruction_doc(path_lower: &str) -> bool {
    let file_name = path_lower.rsplit('/').next().unwrap_or(path_lower);
    matches!(
        file_name,
        "agents.md"
            | "agent.md"
            | "claude.md"
            | "cursor.md"
            | ".cursorrules"
            | ".cursor-rules"
            | "copilot-instructions.md"
            | "instructions.md"
            | "conventions.md"
            | "coding-guidelines.md"
    ) || path_lower.starts_with(".github/instructions/")
        || path_lower.contains("/.github/instructions/")
        || path_lower.starts_with(".cursor/rules/")
        || path_lower.contains("/.cursor/rules/")
}

fn is_test_or_spec_file(path_lower: &str) -> bool {
    let file_name = path_lower.rsplit('/').next().unwrap_or(path_lower);
    file_name.contains(".test.")
        || file_name.contains(".spec.")
        || file_name.ends_with("_test.rs")
        || file_name.ends_with("_test.go")
        || (file_name.starts_with("test_") && file_name.ends_with(".py"))
        || path_lower.starts_with("test/")
        || path_lower.starts_with("tests/")
        || path_lower.starts_with("__tests__/")
        || path_lower.contains("/test/")
        || path_lower.contains("/tests/")
        || path_lower.contains("/__tests__/")
}

fn is_project_manifest_or_config(path_lower: &str) -> bool {
    let file_name = path_lower.rsplit('/').next().unwrap_or(path_lower);
    matches!(
        file_name,
        "package.json"
            | "cargo.toml"
            | "pyproject.toml"
            | "go.mod"
            | "pom.xml"
            | "build.gradle"
            | "build.gradle.kts"
            | "settings.gradle"
            | "settings.gradle.kts"
            | "requirements.txt"
            | "makefile"
            | "justfile"
            | "taskfile.yml"
            | "taskfile.yaml"
            | "deno.json"
            | "bunfig.toml"
            | "vite.config.ts"
            | "vite.config.js"
            | "tsconfig.json"
    ) || file_name.ends_with(".config.js")
        || file_name.ends_with(".config.ts")
        || file_name.ends_with(".config.cjs")
        || file_name.ends_with(".config.mjs")
}

fn score_symbol_hit(path: &str, symbol: &str, query: &str) -> i64 {
    let path_score = score_file_hit(path, query);
    let symbol_score = score_symbol_name(symbol, query);
    let score = path_score + symbol_score;
    if score == 0 {
        25
    } else {
        score
    }
}

fn score_symbol_name(symbol: &str, query: &str) -> i64 {
    let query_lower = query.to_lowercase();
    let symbol_lower = symbol.to_lowercase();
    let query_normalized = normalize_identifier(query);
    let symbol_normalized = normalize_identifier(symbol);
    let terms = search_terms(query);
    if symbol_lower == query_lower {
        120
    } else if symbol_lower.contains(&query_lower) {
        90
    } else if !query_normalized.is_empty() && symbol_normalized.contains(&query_normalized) {
        88
    } else if !terms.is_empty()
        && terms
            .iter()
            .all(|term| symbol_normalized.contains(&normalize_identifier(term)))
    {
        75 + (terms.len() as i64 * 8)
    } else if terms
        .iter()
        .any(|term| symbol_normalized.contains(&normalize_identifier(term)))
    {
        35
    } else {
        0
    }
}

fn symbol_candidate_queries(query: &str) -> Vec<String> {
    let mut queries = vec![query.trim().to_string()];
    queries.extend(search_terms(query));
    queries.retain(|query| !query.is_empty());
    queries.sort();
    queries.dedup();
    queries
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

fn parse_git_status_line(line: &str) -> Option<(String, String)> {
    if line.len() < 4 {
        return None;
    }
    let status = line.get(0..2)?.trim().to_string();
    let mut path = line.get(3..)?.trim().replace('\\', "/");
    if let Some((_, to)) = path.rsplit_once(" -> ") {
        path = to.trim().to_string();
    }
    (!status.is_empty() && !path.is_empty()).then_some((status, path))
}

fn git_diff_for_path(root: &Path, path: &str) -> Result<String> {
    let output = Command::new("git")
        .current_dir(root)
        .args(["diff", "HEAD", "--", path])
        .output()
        .context("Failed to read git diff")?;
    if !output.status.success() {
        return Err(anyhow!("git diff failed for {}", path));
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

fn score_git_status(status: &str) -> i64 {
    if status == "??" {
        120
    } else if status.contains('M') || status.contains('A') || status.contains('R') {
        115
    } else {
        100
    }
}

const RG_CANDIDATE_PREFILTER_LIMIT: usize = 512;

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
            if candidates.len() >= RG_CANDIDATE_PREFILTER_LIMIT {
                return None;
            }
        }
    }

    Some(candidates)
}

fn search_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    for segment in query.split(|ch: char| !ch.is_alphanumeric()) {
        let segment = segment.trim();
        if segment.is_empty() {
            continue;
        }
        terms.push(segment.to_lowercase());
        terms.extend(identifier_words(segment));
    }
    terms.sort();
    terms.dedup();
    terms.retain(|term| {
        term.len() > 2
            && !matches!(
                term.as_str(),
                "the" | "and" | "for" | "with" | "where" | "what" | "how" | "our" | "code"
            )
    });
    expand_software_concept_terms(&mut terms);
    terms.sort();
    terms.dedup();
    terms
}

fn identifier_words(value: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    let mut prev_lower_or_digit = false;

    for ch in value.chars() {
        if !ch.is_alphanumeric() {
            if !current.is_empty() {
                words.push(current.to_lowercase());
                current.clear();
            }
            prev_lower_or_digit = false;
            continue;
        }

        if ch.is_uppercase() && prev_lower_or_digit && !current.is_empty() {
            words.push(current.to_lowercase());
            current.clear();
        }
        prev_lower_or_digit = ch.is_lowercase() || ch.is_ascii_digit();
        current.push(ch);
    }

    if !current.is_empty() {
        words.push(current.to_lowercase());
    }

    let compact = normalize_identifier(value);
    if compact.len() > 2 {
        words.push(compact);
    }
    words
}

fn expand_software_concept_terms(terms: &mut Vec<String>) {
    let originals = terms.clone();
    for term in originals {
        terms.extend(
            software_concept_aliases(&term)
                .into_iter()
                .map(ToString::to_string),
        );
    }
}

fn software_concept_aliases(term: &str) -> Vec<&'static str> {
    match term {
        "action" | "actions" | "tool" | "tools" | "usage" => {
            vec!["command", "commands", "operation", "operations"]
        }
        "api" | "endpoint" | "endpoints" | "service" | "services" => {
            vec!["route", "handler", "request", "response", "client"]
        }
        "context" | "contexts" => vec!["evidence", "source", "sources", "citation", "citations"],
        "data" => vec!["state", "model", "schema", "storage"],
        "discovery" | "retrieve" | "retrieval" | "search" => {
            vec![
                "index",
                "indexes",
                "symbol",
                "symbols",
                "reference",
                "references",
            ]
        }
        "flow" | "lifecycle" | "loop" | "loops" | "workflow" => {
            vec!["pipeline", "phase", "phases", "runtime", "orchestration"]
        }
        "persistence" | "persisted" | "storage" => {
            vec!["store", "database", "cache", "state"]
        }
        "prompt" | "prompts" | "prompting" => {
            vec!["instruction", "instructions", "schema", "schemas", "system"]
        }
        "research" | "overview" | "architecture" | "design" => {
            vec!["docs", "documentation", "readme", "guide"]
        }
        "screen" | "screens" | "view" | "views" | "ui" => {
            vec!["component", "components", "widget", "widgets", "template"]
        }
        "state" => vec!["store", "model", "cache", "session"],
        "validation" | "validate" | "verify" => vec!["test", "tests", "build", "check"],
        _ => Vec::new(),
    }
}

fn normalize_identifier(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
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
    let query_normalized = normalize_identifier(query);
    let path_lower = path.to_lowercase();
    for (idx, line) in lines.iter().enumerate() {
        let line_lower = line.to_lowercase();
        let line_normalized = normalize_identifier(line);
        let mut score = 0i64;
        if !query_lower.is_empty() && line_lower.contains(&query_lower) {
            score += 80;
        }
        if !query_normalized.is_empty() && line_normalized.contains(&query_normalized) {
            score += 55;
        }
        if regex.map(|regex| regex.is_match(line)).unwrap_or(false) {
            score += 90;
        }
        let term_matches = terms
            .iter()
            .filter(|term| {
                line_lower.contains(term.as_str())
                    || line_normalized.contains(&normalize_identifier(term))
            })
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

fn bounded_line_content(
    content: &str,
    start_line: usize,
    end_line: usize,
    max_chars: usize,
) -> (String, LineRange) {
    let start = start_line.max(1);
    let end = end_line.max(start);
    let mut lines = Vec::new();
    let mut used_chars = 0usize;
    let mut actual_end = start;

    for (idx, line) in content.lines().enumerate() {
        let line_no = idx + 1;
        if line_no < start || line_no > end {
            continue;
        }

        let additional = line.len() + usize::from(!lines.is_empty());
        if !lines.is_empty() && used_chars.saturating_add(additional) > max_chars {
            break;
        }

        used_chars = used_chars.saturating_add(additional);
        actual_end = line_no;
        lines.push(line);
    }

    (
        lines.join("\n"),
        LineRange {
            start,
            end: actual_end,
        },
    )
}

fn context_range_for_read(
    content: &str,
    start_line: u32,
    end_line: u32,
    expand_symbol_context: bool,
) -> (u32, u32) {
    let start = start_line.max(1);
    let end = end_line.max(start);
    if !expand_symbol_context {
        return (start, end);
    }

    let file_end = content.lines().count().max(1) as u32;
    let context_lines = 3;
    (
        start.saturating_sub(context_lines).max(1),
        end.saturating_add(context_lines).min(file_end),
    )
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
    let latest_path = dir.join("latest.jsonl");
    let jsonl = trace.to_jsonl();
    tauri::async_runtime::spawn_blocking(move || {
        std::fs::create_dir_all(&dir)?;
        std::fs::write(&path, &jsonl)?;
        std::fs::write(&latest_path, &jsonl)?;
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
        let config: AgentConfig =
            toml::from_str(&raw).with_context(|| format!("Failed to parse {}", path.display()))?;
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
    fn content_search_matches_identifier_variants() {
        let content = "const statusPill = createMemo(() => activeStatus());";
        let mut hits = Vec::new();

        collect_content_hits(
            "src/AiChat.tsx",
            content,
            &search_terms("status_pill"),
            None,
            "status_pill",
            &mut hits,
        );

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "src/AiChat.tsx");
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
    fn git_status_parser_handles_modified_untracked_and_renamed_files() {
        assert_eq!(
            parse_git_status_line(" M src/main.rs"),
            Some(("M".to_string(), "src/main.rs".to_string()))
        );
        assert_eq!(
            parse_git_status_line("?? src/new.rs"),
            Some(("??".to_string(), "src/new.rs".to_string()))
        );
        assert_eq!(
            parse_git_status_line("R  src/old.rs -> src/new.rs"),
            Some(("R".to_string(), "src/new.rs".to_string()))
        );
    }

    #[test]
    fn rg_candidate_paths_disables_prefilter_when_matches_are_too_broad() {
        let root = std::env::temp_dir().join(format!("loco-rg-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).expect("temp rg test dir should be created");

        for index in 0..=RG_CANDIDATE_PREFILTER_LIMIT {
            std::fs::write(root.join(format!("file-{index}.rs")), "fn needle() {}")
                .expect("temp rg test file should be written");
        }

        let candidates = rg_candidate_paths(&root, &["needle".to_string()]);
        let _ = std::fs::remove_dir_all(&root);

        assert!(candidates.is_none());
    }

    #[test]
    fn bounded_line_content_preserves_verifiable_line_ranges() {
        let (content, range) = bounded_line_content("one\ntwo\nthree\nfour", 2, 4, 7);

        assert_eq!(content, "two");
        assert_eq!(range.start, 2);
        assert_eq!(range.end, 2);
    }

    #[test]
    fn symbol_context_range_expands_only_symbol_reads() {
        let content = (1..=20)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");

        assert_eq!(context_range_for_read(&content, 10, 12, true), (7, 15));
        assert_eq!(context_range_for_read(&content, 10, 12, false), (10, 12));
        assert_eq!(context_range_for_read(&content, 1, 2, true), (1, 5));
    }

    #[test]
    fn file_scoring_handles_multi_word_queries_by_terms() {
        assert!(score_file_hit("src/App.tsx", "app startup sequence") > 0);
        assert!(score_file_hit("src-tauri/src/main.rs", "main startup") > 0);
        assert_eq!(score_file_hit("src/GitPanel.tsx", "splash startup"), 0);
    }

    #[test]
    fn search_terms_split_identifier_styles() {
        let terms = search_terms("status_pill statusPill status-panel");

        assert!(terms.contains(&"status".to_string()));
        assert!(terms.contains(&"pill".to_string()));
        assert!(terms.contains(&"statuspill".to_string()));
    }

    #[test]
    fn search_terms_expand_common_software_research_concepts() {
        let terms = search_terms("agentic loops tool usage context filling");

        assert!(terms.contains(&"pipeline".to_string()));
        assert!(terms.contains(&"phase".to_string()));
        assert!(terms.contains(&"command".to_string()));
        assert!(terms.contains(&"evidence".to_string()));
        assert!(terms.contains(&"source".to_string()));
    }

    #[test]
    fn concept_expansion_improves_project_research_scoring() {
        assert!(score_file_hit("crates/agent-core/src/pipeline.rs", "agentic loops") > 0);
        assert!(score_file_hit("src/commands/mod.rs", "tool usage") > 0);
        assert!(score_file_hit("crates/agent-core/src/evidence.rs", "context filling") > 0);
    }

    #[test]
    fn validation_context_queries_surface_project_manifests() {
        assert!(score_file_hit("package.json", "project manifest") > 0);
        assert!(score_file_hit("Cargo.toml", "build script") > 0);
        assert!(score_file_hit("vite.config.ts", "workspace config") > 0);
        assert_eq!(score_file_hit("src/GitPanel.tsx", "project manifest"), 0);
    }

    #[test]
    fn directory_file_search_surfaces_ranked_files_under_requested_directory() {
        let indexed_paths = [
            "packages/web/src/App.tsx",
            "packages/web/src/main.tsx",
            "packages/web/src/components/Button.tsx",
            "packages/web/src/README.md",
            "packages/web/package.json",
            "packages/api/src/App.tsx",
        ]
        .into_iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();

        let hits = directory_file_hits(&indexed_paths, &["packages/web/src".to_string()]);
        let paths = hits
            .iter()
            .map(|hit| hit.path.as_str())
            .collect::<HashSet<_>>();
        let score = |path: &str| {
            hits.iter()
                .find(|hit| hit.path == path)
                .map(|hit| hit.score)
                .unwrap_or_default()
        };

        assert!(paths.contains("packages/web/src/App.tsx"));
        assert!(paths.contains("packages/web/src/main.tsx"));
        assert!(paths.contains("packages/web/src/components/Button.tsx"));
        assert!(paths.contains("packages/web/src/README.md"));
        assert!(!paths.contains("packages/web/package.json"));
        assert!(!paths.contains("packages/api/src/App.tsx"));
        assert!(score("packages/web/src/main.tsx") > score("packages/web/src/components/Button.tsx"));
        assert!(score("packages/web/src/README.md") > score("packages/web/src/components/Button.tsx"));
    }

    #[test]
    fn related_test_queries_surface_test_files_without_noisy_script_matches() {
        let related_score = score_file_hit("src/status-panel.test.tsx", "related tests");
        let script_score = score_file_hit("src/status-panel.test.tsx", "test script");

        assert!(related_score > 0);
        assert!(score_file_hit("tests/status_panel.rs", "test coverage") > 0);
        assert!(score_file_hit("src/__tests__/status-panel.tsx", "nearby specs") > 0);
        assert_eq!(score_file_hit("src/status-panel.tsx", "related tests"), 0);
        assert!(related_score > script_score);
    }

    #[test]
    fn adjacent_file_scoring_finds_siblings_without_cross_project_noise() {
        let seeds = vec!["src/features/status-panel.tsx".to_string()];

        assert!(
            score_adjacent_file("src/features/status-panel.test.tsx", &seeds)
                > score_adjacent_file("src/features/status-panel.css", &seeds)
        );
        assert!(score_adjacent_file("src/features/status-panel.css", &seeds) > 0);
        assert!(score_adjacent_file("src/features/status-utils.ts", &seeds) > 0);
        assert_eq!(score_adjacent_file("src/features/status-panel.tsx", &seeds), 0);
        assert_eq!(score_adjacent_file("src/other/status-panel.test.tsx", &seeds), 0);
        assert_eq!(score_adjacent_file("src/other/unrelated.ts", &seeds), 0);
    }

    #[test]
    fn scope_file_search_finds_nearest_package_context_without_sibling_noise() {
        let indexed_paths = [
            "package.json",
            "README.md",
            "packages/web/package.json",
            "packages/web/vite.config.ts",
            "packages/web/README.md",
            "packages/web/AGENTS.md",
            "packages/web/src/status-panel.tsx",
            "packages/api/package.json",
        ]
        .into_iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();

        let hits = scope_file_hits(
            &indexed_paths,
            &["packages/web/src/status-panel.tsx".to_string()],
        );
        let paths = hits
            .iter()
            .map(|hit| hit.path.as_str())
            .collect::<HashSet<_>>();
        let score = |path: &str| {
            hits.iter()
                .find(|hit| hit.path == path)
                .map(|hit| hit.score)
                .unwrap_or_default()
        };

        assert!(paths.contains("packages/web/package.json"));
        assert!(paths.contains("packages/web/vite.config.ts"));
        assert!(paths.contains("packages/web/README.md"));
        assert!(paths.contains("packages/web/AGENTS.md"));
        assert!(paths.contains("package.json"));
        assert!(paths.contains("README.md"));
        assert!(!paths.contains("packages/api/package.json"));
        assert!(score("packages/web/package.json") > score("package.json"));
        assert!(score("packages/web/AGENTS.md") > score("README.md"));
    }

    #[test]
    fn context_doc_search_finds_nearby_domain_docs_without_sibling_noise() {
        let indexed_paths = [
            "README.md",
            "docs/status-panel.md",
            "packages/web/src/status-panel.tsx",
            "packages/web/docs/status-panel.md",
            "packages/web/docs/routing.md",
            "packages/web/README.md",
            "packages/api/docs/status-panel.md",
        ]
        .into_iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();

        let hits = context_doc_hits(
            &indexed_paths,
            &["status panel".to_string()],
            &["packages/web/src/status-panel.tsx".to_string()],
        );
        let paths = hits
            .iter()
            .map(|hit| hit.path.as_str())
            .collect::<HashSet<_>>();
        let score = |path: &str| {
            hits.iter()
                .find(|hit| hit.path == path)
                .map(|hit| hit.score)
                .unwrap_or_default()
        };

        assert!(paths.contains("packages/web/docs/status-panel.md"));
        assert!(paths.contains("docs/status-panel.md"));
        assert!(!paths.contains("packages/api/docs/status-panel.md"));
        assert!(!paths.contains("packages/web/docs/routing.md"));
        assert!(score("packages/web/docs/status-panel.md") > score("docs/status-panel.md"));
    }

    #[test]
    fn imported_file_search_resolves_relative_js_and_rust_modules() {
        let root = std::env::temp_dir().join(format!("loco-imports-{}", uuid::Uuid::new_v4()));
        let src = root.join("src");
        std::fs::create_dir_all(src.join("nested")).expect("temp import dirs should be created");
        std::fs::write(
            src.join("status-panel.tsx"),
            [
                "import { helper } from './status-utils';",
                "import './status-panel.css';",
                "const lazy = () => import('./lazy');",
                "const legacy = require('./legacy');",
                "export { exported } from './exported';",
            ]
            .join("\n"),
        )
        .expect("seed tsx file should be written");
        std::fs::write(src.join("lib.rs"), "pub mod child;\nmod nested;")
            .expect("seed rust file should be written");
        for path in [
            "status-utils.ts",
            "status-panel.css",
            "lazy.tsx",
            "legacy.js",
            "exported.ts",
            "child.rs",
            "nested/mod.rs",
        ] {
            std::fs::write(src.join(path), "// imported").expect("imported file should be written");
        }

        let indexed_paths = [
            "src/status-panel.tsx",
            "src/status-utils.ts",
            "src/status-panel.css",
            "src/lazy.tsx",
            "src/legacy.js",
            "src/exported.ts",
            "src/lib.rs",
            "src/child.rs",
            "src/nested/mod.rs",
        ]
        .into_iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();

        let hits = imported_file_hits(
            &root,
            &indexed_paths,
            &[
                "src/status-panel.tsx".to_string(),
                "src/lib.rs".to_string(),
            ],
        );
        let paths = hits
            .into_iter()
            .map(|hit| hit.path)
            .collect::<HashSet<_>>();
        let _ = std::fs::remove_dir_all(&root);

        assert!(paths.contains("src/status-utils.ts"));
        assert!(paths.contains("src/status-panel.css"));
        assert!(paths.contains("src/lazy.tsx"));
        assert!(paths.contains("src/legacy.js"));
        assert!(paths.contains("src/exported.ts"));
        assert!(paths.contains("src/child.rs"));
        assert!(paths.contains("src/nested/mod.rs"));
        assert!(!paths.contains("src/status-panel.tsx"));
        assert!(!paths.contains("src/lib.rs"));
    }

    #[test]
    fn importing_file_search_finds_reverse_relative_js_and_rust_modules() {
        let root = std::env::temp_dir().join(format!("loco-importers-{}", uuid::Uuid::new_v4()));
        let src = root.join("src");
        std::fs::create_dir_all(src.join("nested")).expect("temp importer dirs should be created");
        std::fs::write(
            src.join("status-panel.tsx"),
            "import { helper } from './status-utils';",
        )
        .expect("importer tsx file should be written");
        std::fs::write(src.join("status-utils.ts"), "export const helper = 1;")
            .expect("imported ts file should be written");
        std::fs::write(src.join("lib.rs"), "pub mod child;\nmod nested;")
            .expect("rust parent file should be written");
        std::fs::write(src.join("child.rs"), "// child")
            .expect("rust child file should be written");
        std::fs::write(src.join("nested").join("mod.rs"), "// nested")
            .expect("rust nested module should be written");
        std::fs::write(src.join("unrelated.ts"), "import './other';")
            .expect("unrelated file should be written");

        let indexed_paths = [
            "src/status-panel.tsx",
            "src/status-utils.ts",
            "src/lib.rs",
            "src/child.rs",
            "src/nested/mod.rs",
            "src/unrelated.ts",
        ]
        .into_iter()
        .map(ToString::to_string)
        .collect::<HashSet<_>>();

        let hits = importing_file_hits(
            &root,
            &indexed_paths,
            &[
                "src/status-utils.ts".to_string(),
                "src/child.rs".to_string(),
                "src/nested/mod.rs".to_string(),
            ],
        );
        let paths = hits
            .into_iter()
            .map(|hit| hit.path)
            .collect::<HashSet<_>>();
        let _ = std::fs::remove_dir_all(&root);

        assert!(paths.contains("src/status-panel.tsx"));
        assert!(paths.contains("src/lib.rs"));
        assert!(!paths.contains("src/status-utils.ts"));
        assert!(!paths.contains("src/child.rs"));
        assert!(!paths.contains("src/nested/mod.rs"));
        assert!(!paths.contains("src/unrelated.ts"));
    }

    #[test]
    fn project_research_queries_surface_overview_docs() {
        assert!(score_file_hit("README.md", "project overview") > 0);
        assert!(score_file_hit("docs/architecture.md", "architecture docs") > 0);
        assert!(score_file_hit("CONTRIBUTING.md", "developer guide") > 0);
        assert_eq!(score_file_hit("src/GitPanel.tsx", "project overview"), 0);
    }

    #[test]
    fn repository_instruction_queries_surface_agent_guidance_files() {
        assert!(score_file_hit("AGENTS.md", "repository instructions") > 0);
        assert!(score_file_hit("CLAUDE.md", "agent guidance") > 0);
        assert!(score_file_hit(".cursorrules", "coding conventions") > 0);
        assert!(score_file_hit(".github/instructions/rust.md", "repository instructions") > 0);
        assert_eq!(score_file_hit("src/GitPanel.tsx", "repository instructions"), 0);
    }

    #[test]
    fn project_entrypoint_queries_surface_startup_files() {
        assert!(score_file_hit("src/main.tsx", "project entrypoints") > 0);
        assert!(score_file_hit("src-tauri/src/main.rs", "project startup") > 0);
        assert!(score_file_hit("src/App.tsx", "application bootstrap") > 0);
        assert_eq!(score_file_hit("docs/architecture.md", "project entrypoints"), 0);
    }

    #[test]
    fn symbol_scoring_handles_multi_word_queries() {
        let candidates = symbol_candidate_queries("status panel");

        assert!(candidates.contains(&"status panel".to_string()));
        assert!(candidates.contains(&"status".to_string()));
        assert!(candidates.contains(&"panel".to_string()));
        assert!(score_symbol_hit("src/status.tsx", "StatusPanel", "status panel") > 90);
        assert!(score_symbol_hit("src/status.tsx", "statusPill", "status_pill") > 90);
    }

    #[test]
    fn callee_collection_returns_definitions_called_by_matching_symbol() {
        let (files, symbols, edges) = relation_fixture();

        let hits = collect_callee_symbol_hits(&files, &symbols, &edges, "main", 5);

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "src/helper.rs");
        assert_eq!(hits[0].symbol, "helper");
        assert_eq!(hits[0].start_line, 10);
        assert_eq!(hits[0].end_line, 14);
    }

    #[test]
    fn reference_collection_returns_symbols_that_target_query() {
        let (files, symbols, edges) = relation_fixture();

        let hits = collect_referencing_symbol_hits(&files, &symbols, &edges, "helper", false, 5);

        assert_eq!(hits.len(), 2);
        assert!(hits
            .iter()
            .any(|hit| hit.path == "src/main.rs" && hit.symbol == "main"));
        assert!(hits
            .iter()
            .any(|hit| hit.path == "src/reader.rs" && hit.symbol == "readHelper"));
    }

    #[test]
    fn caller_collection_excludes_non_call_references() {
        let (files, symbols, edges) = relation_fixture();

        let hits = collect_referencing_symbol_hits(&files, &symbols, &edges, "helper", true, 5);

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "src/main.rs");
        assert_eq!(hits[0].symbol, "main");
    }

    #[test]
    fn definition_collection_returns_matching_symbol_definitions() {
        let (files, symbols, _) = relation_fixture();

        let hits = collect_definition_symbol_hits(&files, &symbols, "helper", 5);

        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].path, "src/helper.rs");
        assert_eq!(hits[0].symbol, "helper");
        assert_eq!(hits[0].start_line, 10);
        assert_eq!(hits[0].end_line, 14);
    }

    fn relation_fixture() -> (
        HashMap<String, SourceFile>,
        Vec<Symbol>,
        HashMap<String, Vec<RelationEdge>>,
    ) {
        let files = HashMap::from([
            (
                "file-main".to_string(),
                SourceFile {
                    id: "file-main".to_string(),
                    project_id: "project".to_string(),
                    path: "src/main.rs".to_string(),
                    checksum: None,
                    index_version: 1,
                    last_indexed: None,
                },
            ),
            (
                "file-helper".to_string(),
                SourceFile {
                    id: "file-helper".to_string(),
                    project_id: "project".to_string(),
                    path: "src/helper.rs".to_string(),
                    checksum: None,
                    index_version: 1,
                    last_indexed: None,
                },
            ),
            (
                "file-reader".to_string(),
                SourceFile {
                    id: "file-reader".to_string(),
                    project_id: "project".to_string(),
                    path: "src/reader.rs".to_string(),
                    checksum: None,
                    index_version: 1,
                    last_indexed: None,
                },
            ),
        ]);
        let symbols = vec![
            Symbol {
                id: "main".to_string(),
                file_id: "file-main".to_string(),
                name: "main".to_string(),
                kind: "fn".to_string(),
                line: 1,
                start_line: 1,
                end_line: 6,
                content_hash: None,
            },
            Symbol {
                id: "helper".to_string(),
                file_id: "file-helper".to_string(),
                name: "helper".to_string(),
                kind: "fn".to_string(),
                line: 10,
                start_line: 10,
                end_line: 14,
                content_hash: None,
            },
            Symbol {
                id: "read-helper".to_string(),
                file_id: "file-reader".to_string(),
                name: "readHelper".to_string(),
                kind: "fn".to_string(),
                line: 20,
                start_line: 20,
                end_line: 25,
                content_hash: None,
            },
        ];
        let edges = HashMap::from([
            (
                "main".to_string(),
                vec![RelationEdge {
                    target_name: "helper".to_string(),
                    kind: "call".to_string(),
                }],
            ),
            (
                "read-helper".to_string(),
                vec![RelationEdge {
                    target_name: "helper".to_string(),
                    kind: "reference".to_string(),
                }],
            ),
        ]);
        (files, symbols, edges)
    }

    #[test]
    fn repo_search_filter_skips_generated_dist_entries() {
        assert!(!looks_searchable_text_file("dist/assets/index.js"));
        assert!(!looks_searchable_text_file("dist/index.html"));
        assert!(looks_searchable_text_file("src/App.tsx"));
    }

    #[test]
    fn symbol_graph_distance_boost_prioritizes_nearer_edges() {
        assert_eq!(symbol_graph_distance_boost(Some(1)), 80);
        assert_eq!(symbol_graph_distance_boost(Some(2)), 45);
        assert_eq!(symbol_graph_distance_boost(Some(3)), 0);
        assert_eq!(symbol_graph_distance_boost(None), 0);
    }
}
