// src-tauri/src/agent.rs
//! Orchestrates the AI request lifecycle.
//! Retrieves workspace context, manages conversation history, and streams LLM responses via Tauri IPC.

use crate::conversation_store::Role;
use crate::engine::EngineState;
use crate::{AppError, AppState};
use anyhow::{anyhow, Context, Result}; // FIX: Added Result to the anyhow imports
use serde_json::json;
use tauri::ipc::Channel;
use tauri::Manager;

/// Intercepts a chat request, builds context, and streams the AI's response back to the UI.
#[tauri::command]
pub async fn prompt_agent(
    app: tauri::AppHandle,
    session_id: String,
    question: String,
    on_chunk: Channel<String>,
) -> Result<(), AppError> {
    let engine_state = app.state::<EngineState>();
    
    // 1. Verify engine is alive
    let conn = engine_state
        .connection_details()
        .ok_or_else(|| anyhow!("Engine sidecar is not ready"))?;

    let app_clone = app.clone();
    let session_id_clone = session_id.clone();
    let question_clone = question.clone();

    // 2. Synchronous Database Operations (Spawned to avoid blocking the async runtime)
    let (conversation, messages) = tauri::async_runtime::spawn_blocking(move || -> Result<_> {
        let state = app_clone.state::<AppState>();
        
        // Save the user's question
        state.conversations.add_message(&session_id_clone, Role::User, &question_clone)?;
        
        // Load the full conversation payload including workspace path
        let detail = state.conversations
            .load_conversation(&session_id_clone)?
            .ok_or_else(|| anyhow!("Session not found"))?;
            
        Ok((detail.conversation, detail.messages))
    })
    .await
    .context("Failed to join blocking task")??;

    // 3. Build the System Context
    // TODO: In the future, use `conversation.workspace_path` to query the `codegraph` store here.
    let system_context = format!(
        "You are Loco, a local-first AI coding assistant.\n\
         Workspace Path: {}\n\
         Provide clear, concise code.",
        conversation.workspace_path.unwrap_or_else(|| "Unknown".to_string())
    );

    let mut llm_messages = vec![json!({
        "role": "system",
        "content": system_context
    })];

    // Append conversation history
    for msg in messages {
        llm_messages.push(json!({
            "role": msg.role.as_str(),
            "content": msg.content
        }));
    }

    // 4. Send HTTP Request to local llama-server
    let client = reqwest::Client::new();
    let mut res = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", conn.port))
        .header("Authorization", format!("Bearer {}", conn.api_key))
        .json(&json!({
            "messages": llm_messages,
            "temperature": 0.1,
            "stream": true,
            "max_tokens": -1
        }))
        .send()
        .await
        .context("Failed to send request to engine")?;

    // 5. Parse the SSE stream and pipe chunks directly to the UI
    let mut full_response = String::new();
    
    while let Some(chunk) = res.chunk().await.context("Failed to read stream chunk")? {
        let text = String::from_utf8_lossy(&chunk);
        
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("data: ") {
                let data = &line[6..];
                if data == "[DONE]" {
                    continue;
                }
                
                // Parse llama.cpp delta JSON safely
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(content) = val["choices"][0]["delta"]["content"].as_str() {
                        full_response.push_str(content);
                        // Send clean string to frontend via Tauri Channel
                        let _ = on_chunk.send(content.to_string());
                    }
                }
            }
        }
    }

    // 6. Save the final assistant response to the database
    tauri::async_runtime::spawn_blocking(move || -> Result<_> {
        let state = app.state::<AppState>();
        state.conversations.add_message(&session_id, Role::Assistant, &full_response)?;
        Ok(())
    })
    .await
    .context("Failed to join blocking task")??;

    Ok(())
}