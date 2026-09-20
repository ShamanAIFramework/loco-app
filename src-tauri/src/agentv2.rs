// src-tauri/src/agentv2.rs
//! Low-level agent primitives, tool definitions, and LLM communication wrappers.

use crate::AppState;
use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::HashMap;
use tauri::Manager;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanTask {
    pub id: String,
    pub description: String,
    #[serde(default = "default_pending")]
    pub status: String,
}

fn default_pending() -> String {
    "pending".to_string()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PlanMutation {
    AddTask { description: String },
    CompleteTask { id: String },
}

#[derive(Default, Debug, Clone)]
pub struct ExtractedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// Declares tools available to discovery and planning workers.
pub fn get_worker_tools() -> serde_json::Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "search_files",
                "description": "Search the local codebase for files matching a path or filename query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "A partial file name or folder path (e.g. 'App.tsx', 'src/components')."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the exact contents of a file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The exact relative path of the file (e.g. 'src/App.tsx')."
                        }
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "modify_plan",
                "description": "Add a new discovery subtask if new dependencies or files need investigation.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add_task"],
                            "description": "The mutation action to perform."
                        },
                        "description": {
                            "type": "string",
                            "description": "Description of the new subtask to execute."
                        }
                    },
                    "required": ["action", "description"]
                }
            }
        }
    ])
}

/// Executes an inference call against llama-server, capturing streamed chunks or tool calls.
pub async fn call_model_stream(
    client: &reqwest::Client,
    port: u16,
    api_key: &str,
    messages: &[serde_json::Value],
    tools: Option<serde_json::Value>,
    mut chunk_callback: impl FnMut(String),
) -> Result<(String, Vec<ExtractedToolCall>)> {
    let mut req_body = json!({
        "messages": messages,
        "temperature": 0.1,
        "stream": true,
        "max_tokens": -1,
        "stop": ["Observation:", "<|tool_output|>", "```output"]
    });

    if let Some(t) = tools {
        req_body["tools"] = t;
        req_body["tool_choice"] = json!("auto");
    }

    let mut res = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&req_body)
        .send()
        .await
        .context("Failed to send request to inference engine")?;

    let mut active_tool_calls: HashMap<usize, ExtractedToolCall> = HashMap::new();
    let mut raw_response = String::new();

    while let Some(chunk) = res.chunk().await.context("Failed to read stream chunk")? {
        let text = String::from_utf8_lossy(&chunk);
        for line in text.lines() {
            let line = line.trim();
            if !line.starts_with("data: ") {
                continue;
            }
            let data = &line[6..];
            if data == "[DONE]" {
                continue;
            }

            if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                let choice = &val["choices"][0];

                if let Some(content) = choice["delta"]["content"].as_str() {
                    raw_response.push_str(content);

                    let is_leaking = raw_response.contains("{\"name\"") || raw_response.contains("```json");
                    if active_tool_calls.is_empty() && !is_leaking {
                        chunk_callback(content.to_string());
                    }
                }

                if let Some(tc_array) = choice["delta"]["tool_calls"].as_array() {
                    for tc in tc_array {
                        if let Some(idx) = tc["index"].as_u64() {
                            let entry = active_tool_calls.entry(idx as usize).or_default();
                            if let Some(id) = tc["id"].as_str() {
                                entry.id = id.to_string();
                            }
                            if let Some(f) = tc.get("function") {
                                if let Some(n) = f.get("name").and_then(|n| n.as_str()) {
                                    entry.name.push_str(n);
                                }
                                if let Some(a) = f.get("arguments").and_then(|a| a.as_str()) {
                                    entry.arguments.push_str(a);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Aggressive JSON tool call interceptor for local model stutters
    if active_tool_calls.is_empty() && raw_response.contains("\"name\"") {
        for (start_idx, _) in raw_response.match_indices('{') {
            if let Some(end_offset) = raw_response[start_idx..].rfind('}') {
                let end_idx = start_idx + end_offset;
                let slice = &raw_response[start_idx..=end_idx];
                let clean_slice = slice.replace("```json", "").replace("```", "");

                if let Ok(val) = serde_json::from_str::<serde_json::Value>(&clean_slice) {
                    if let (Some(name), Some(args)) = (val["name"].as_str(), val.get("arguments")) {
                        active_tool_calls.insert(
                            0,
                            ExtractedToolCall {
                                id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                                name: name.to_string(),
                                arguments: args.to_string(),
                            },
                        );
                        let first_brace = raw_response.find('{').unwrap_or(0);
                        raw_response.truncate(first_brace);
                        break;
                    }
                }
            }
        }
    }

    let calls = active_tool_calls.into_values().collect();
    Ok((raw_response, calls))
}

/// Executes individual tool operations safely against CodeGraph or the local filesystem.
pub async fn execute_tool(
    app: &tauri::AppHandle,
    workspace_path: Option<&str>,
    project_id: Option<&str>,
    tool_name: &str,
    args_json: &str,
) -> Result<String> {
    let args: serde_json::Value = serde_json::from_str(args_json).unwrap_or(json!({}));

    match tool_name {
        "search_files" => {
            let query = args["query"].as_str().unwrap_or("").to_string();
            if let Some(pid) = project_id {
                let app_c = app.clone();
                let pid_c = pid.to_string();
                tauri::async_runtime::spawn_blocking(move || {
                    let state = app_c.state::<AppState>();
                    match state.codegraph.search_files(&pid_c, &query) {
                        Ok(files) => {
                            let paths: Vec<String> = files.into_iter().map(|f| f.path).collect();
                            if paths.is_empty() {
                                "No files found.".to_string()
                            } else {
                                paths.join("\n")
                            }
                        }
                        Err(e) => format!("Error searching files: {}", e),
                    }
                })
                .await
                .map_err(|e| anyhow!("Task spawn failure: {}", e))
            } else {
                Ok("No active workspace context to search.".to_string())
            }
        }
        "read_file" => {
            let raw_path = args["path"].as_str().unwrap_or("");
            let clean_path = raw_path.trim_start_matches(|c| c == '/' || c == '\\');

            if let Some(wp) = workspace_path {
                let full_path = std::path::Path::new(wp).join(clean_path);
                match std::fs::read_to_string(&full_path) {
                    Ok(mut content) => {
                        if content.len() > 12000 {
                            content.truncate(12000);
                            content.push_str("\n\n...[TRUNCATED FOR LOCAL CONTEXT]...");
                        }
                        Ok(content)
                    }
                    Err(e) => Ok(format!("ERROR: Cannot read '{}': {}", clean_path, e)),
                }
            } else {
                Ok("No active workspace path configured.".to_string())
            }
        }
        _ => Ok(format!("Error: Tool '{}' unrecognized.", tool_name)),
    }
}

/// Headroom-style context reduction step: strips excessive blank space,
/// normalizes line breaks, and caps individual segment length to conserve local tokens.
pub fn compress_context(raw: &str) -> String {
    let mut compressed = String::with_capacity(raw.len());
    let mut prev_was_empty = false;

    for line in raw.lines() {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            if !prev_was_empty {
                compressed.push('\n');
                prev_was_empty = true;
            }
        } else {
            compressed.push_str(trimmed);
            compressed.push('\n');
            prev_was_empty = false;
        }
    }

    if compressed.len() > 8000 {
        compressed.truncate(8000);
        compressed.push_str("\n...[HEADROOM COMPRESSED]...\n");
    }

    compressed
}