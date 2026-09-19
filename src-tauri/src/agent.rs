// src-tauri/src/agent.rs
//! Orchestrates the AI request lifecycle.
//! Manages tool calling, conversation history, and streams LLM responses via Tauri IPC.

use crate::conversation_store::Role;
use crate::engine::EngineState;
use crate::{AppError, AppState};
use anyhow::{anyhow, Context, Result};
use serde_json::json;
use std::collections::HashMap;
use tauri::ipc::Channel;
use tauri::Manager;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", content = "data")]
pub enum AgentEvent {
    Status(String),
    Chunk(String),
}

#[derive(Default, Debug)]
struct ActiveToolCall {
    id: String,
    name: String,
    arguments: String,
}

#[tauri::command]
pub async fn prompt_agent(
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
        state.conversations.add_message(&session_id_clone, Role::User, &question_clone)?;
        
        let detail = state.conversations
            .load_conversation(&session_id_clone)?
            .ok_or_else(|| anyhow!("Session not found"))?;
            
        Ok((detail.conversation, detail.messages))
    })
    .await
    .context("Failed to join blocking task")??;

    // Fetch the project ID for tool calling
    let mut project_id_opt = None;
    if let Some(wp) = &conversation.workspace_path {
        let app_c = app.clone();
        let wp_c = wp.clone();
        let project = tauri::async_runtime::spawn_blocking(move || {
            let state = app_c.state::<AppState>();
            state.codegraph.register_project("Active Workspace", std::path::Path::new(&wp_c))
        })
        .await
        .context("Failed to join project register thread")??;
        project_id_opt = Some(project.id);
    }

    let system_context = format!(
        "You are Loco, a local-first AI coding assistant.\n\
         Workspace Path: {}\n\
         \n\
         CRITICAL RULES:\n\
         1. You MUST use the 'search_files' tool to find a file path.\n\
         2. You MUST use the 'read_file' tool to read a file's exact contents before answering ANY questions about it.\n\
         3. NEVER guess, assume, or hallucinate code. If you haven't used 'read_file', you DO NOT know the contents.\n\
         4. When outputting a tool call, output ONLY the JSON and immediately stop generating.",
        conversation.workspace_path.as_deref().unwrap_or("Unknown")
    );

    let mut llm_messages = vec![json!({
        "role": "system",
        "content": system_context
    })];

    for msg in messages {
        llm_messages.push(json!({
            "role": msg.role.as_str(),
            "content": msg.content
        }));
    }

    // Define available tools
    let tools_json = json!([{
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search the local codebase for files by path or filename.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A partial file name or folder path (e.g. 'main.rs', 'src/components')."
                    }
                },
                "required": ["query"]
            }
        }
    },{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the exact contents of a file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The exact relative file path returned by search_files (e.g. 'src/App.tsx')."
                        }
                    },
                    "required": ["path"]
                }
            }
        }]);

    let client = reqwest::Client::new();
    let mut full_response = String::new(); 
    let mut max_turns = 3; // Safety bound to prevent infinite tool loops

    // Outer inference loop to handle chained tool calls
    while max_turns > 0 {
        max_turns -= 1;
        
        let req_body = json!({
            "messages": llm_messages,
            "tools": tools_json,
            "tool_choice": "auto",
            "temperature": 0.1,
            "stream": true,
            "max_tokens": -1,
            "stop": ["Observation:", "<|tool_output|>", "```output"]
        });
        
        let mut res = client
            .post(format!("http://127.0.0.1:{}/v1/chat/completions", conn.port))
            .header("Authorization", format!("Bearer {}", conn.api_key))
            .json(&req_body)
            .send()
            .await
            .context("Failed to send request")?;
            
        let mut active_tool_calls: HashMap<usize, ActiveToolCall> = HashMap::new();
        let mut turn_response = String::new();

        // Buffer the stream deltas for this turn
        while let Some(chunk) = res.chunk().await.context("Failed to read stream chunk")? {
            let text = String::from_utf8_lossy(&chunk);
            for line in text.lines() {
                let line = line.trim();
                if line.starts_with("data: ") {
                    let data = &line[6..];
                    if data == "[DONE]" { continue; }
                    
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                        let choice = &val["choices"][0];
                        
                        if let Some(content) = choice["delta"]["content"].as_str() {
                            turn_response.push_str(content);
                            
                            // Suspend UI streaming if it looks like a tool call is leaking, even if malformed
                            let is_leaking = turn_response.contains("{\"name\"") || turn_response.contains("```json");
                            
                            if active_tool_calls.is_empty() && !is_leaking {
                                let _ = on_chunk.send(AgentEvent::Chunk(content.to_string()));
                            }
                        }
                        
                        if let Some(tc_array) = choice["delta"]["tool_calls"].as_array() {
                            for tc in tc_array {
                                if let Some(idx) = tc["index"].as_u64() {
                                    let entry = active_tool_calls.entry(idx as usize).or_default();
                                    if let Some(id) = tc["id"].as_str() { entry.id = id.to_string(); }
                                    if let Some(f) = tc.get("function") {
                                        if let Some(n) = f.get("name").and_then(|n| n.as_str()) { entry.name.push_str(n); }
                                        if let Some(a) = f.get("arguments").and_then(|a| a.as_str()) { entry.arguments.push_str(a); }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // --- LOCAL LLM QUIRK FIX: AGGRESSIVE TOOL LEAK INTERCEPTOR ---
        if active_tool_calls.is_empty() && turn_response.contains("\"name\"") {
            let mut parsed_tool = None;
            
            // Iterate over every '{' in the string
            for (start_idx, _) in turn_response.match_indices('{') {
                // Find the matching or last '}'
                if let Some(end_offset) = turn_response[start_idx..].rfind('}') {
                    let end_idx = start_idx + end_offset;
                    
                    // Grab the block and strip any markdown the LLM might have injected inside it
                    let slice = &turn_response[start_idx..=end_idx];
                    let clean_slice = slice.replace("```json", "").replace("```", "");
                    
                    if let Ok(val) = serde_json::from_str::<serde_json::Value>(&clean_slice) {
                        if let (Some(name), Some(args)) = (val["name"].as_str(), val.get("arguments")) {
                            parsed_tool = Some((name.to_string(), args.to_string()));
                            break;
                        }
                    }
                }
            }

            // If we successfully salvaged a tool call from the garbage text:
            if let Some((name, args)) = parsed_tool {
                active_tool_calls.insert(0, ActiveToolCall {
                    id: format!("call_{}", uuid::Uuid::new_v4().simple()),
                    name,
                    arguments: args,
                });
                
                // Truncate at the very FIRST '{' in the response to erase the stuttering prefix entirely
                let first_brace = turn_response.find('{').unwrap_or(0);
                turn_response.truncate(first_brace);
            }
        }
        // ----------------------------------------------------

        // Tool Execution Phase
        if !active_tool_calls.is_empty() {
            let mut calls_for_llm = Vec::new();
            let mut tool_results = Vec::new();
            
            for (_, tc) in active_tool_calls.iter() {
                calls_for_llm.push(json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments
                    }
                }));
                
                let mut result_str = String::new();

                if tc.name == "search_files" {
                    let args: serde_json::Value = serde_json::from_str(&tc.arguments).unwrap_or(json!({}));
                    let query = args["query"].as_str().unwrap_or("");
                    
                    // Emit UI status event
                    let _ = on_chunk.send(AgentEvent::Status(format!("Searching workspace for `{}`...", query)));

                    result_str = if let Some(pid) = &project_id_opt {
                        let app_c = app.clone();
                        let q = query.to_string();
                        let pid_c = pid.clone();
                        tauri::async_runtime::spawn_blocking(move || {
                            let state = app_c.state::<AppState>();
                            match state.codegraph.search_files(&pid_c, &q) {
                                Ok(files) => {
                                    let paths: Vec<String> = files.into_iter().map(|f| f.path).collect();
                                    if paths.is_empty() { "No files found.".to_string() } else { paths.join("\n") }
                                },
                                Err(e) => format!("Error searching files: {}", e)
                            }
                        }).await.unwrap_or_else(|_| "Thread crash".to_string())
                    } else {
                        "No active workspace context to search.".to_string()
                    };
                    
                } else if tc.name == "read_file" {
                    let args: serde_json::Value = serde_json::from_str(&tc.arguments).unwrap_or(json!({}));
                    let raw_path = args["path"].as_str().unwrap_or("");
                    let clean_path = raw_path.trim_start_matches(|c| c == '/' || c == '\\');
                    
                    // Emit UI status event
                    let _ = on_chunk.send(AgentEvent::Status(format!("Reading `{}`...", clean_path)));

                    if let Some(wp) = &conversation.workspace_path {
                        let full_path = std::path::Path::new(wp).join(clean_path);
                        match std::fs::read_to_string(&full_path) {
                            Ok(mut content) => {
                                if content.len() > 16000 {
                                    content.truncate(16000);
                                    content.push_str("\n\n...[FILE TRUNCATED FOR LENGTH]...");
                                }
                                result_str = content;
                            },
                            Err(e) => {
                                result_str = format!("ERROR: Cannot read '{}': {}", clean_path, e);
                            }
                        }
                    } else {
                        result_str = "No active workspace context to read from.".to_string();
                    }
                } else {
                    result_str = format!("Error: Tool '{}' does not exist.", tc.name);
                }
                
                tool_results.push(json!({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str
                }));
            }
            
            llm_messages.push(json!({
                "role": "assistant",
                "content": if turn_response.trim().is_empty() { serde_json::Value::Null } else { json!(turn_response.trim()) },
                "tool_calls": calls_for_llm
            }));
            
            llm_messages.extend(tool_results);
        } else {
            // No tools were called: this turn produced the final answer!
            full_response = turn_response;
            break;
        }
    }

    // Save only the final answer to SQLite
    tauri::async_runtime::spawn_blocking(move || -> Result<_> {
        let state = app.state::<AppState>();
        state.conversations.add_message(&session_id, Role::Assistant, &full_response)?;
        Ok(())
    })
    .await
    .context("Failed to join blocking task")??;

    Ok(())
}