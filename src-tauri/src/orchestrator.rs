// src-tauri/src/orchestrator.rs
//! Orchestrates the multi-turn speculative planning, parallel worker dispatch,
//! and final synthesis pipeline for Loco.

use crate::agent::AgentEvent;
use crate::agentv2::{
    call_model_stream, compress_context, execute_tool, get_worker_tools, PlanMutation, PlanTask,
};
use crate::conversation_store::Role;
use crate::engine::EngineState;
use crate::{AppError, AppState};
use anyhow::{anyhow, Context, Result};
use serde_json::json;
use std::sync::Arc;
use tauri::ipc::Channel;
use tauri::Manager;
use tokio::sync::{mpsc, Mutex};

/// Events sent from parallel worker tasks back to the central orchestrator loop.
enum OrchestratorEvent {
    WorkerStatus(String),
    PlanUpdate(PlanMutation),
    TaskFinished { task_id: String, gathered_context: String },
}

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

    // Persist user question and load context
    let (conversation, messages) = tauri::async_runtime::spawn_blocking(move || -> Result<_> {
        let state = app_clone.state::<AppState>();
        state.conversations.add_message(&session_id_clone, Role::User, &question_clone)?;
        let detail = state
            .conversations
            .load_conversation(&session_id_clone)?
            .ok_or_else(|| anyhow!("Session not found"))?;
        Ok((detail.conversation, detail.messages))
    })
    .await
    .context("Failed to join blocking task")??;

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

    let client = reqwest::Client::new();
    let _ = on_chunk.send(AgentEvent::Status("Formulating plan...".to_string()));

    // -------------------------------------------------------------
    // PHASE 1: SPECULATIVE INTENT & HIGH-LEVEL PLAN (LLM)
    // -------------------------------------------------------------
    let planner_system = format!(
        "You are the planner for Loco, a local coding assistant.\n\
         Workspace: {}\n\
         Analyze the request and return a JSON array containing 1 to 3 targeted discovery tasks.\n\
         Output ONLY raw JSON format.",
        conversation.workspace_path.as_deref().unwrap_or("Unknown")
    );

    let planner_prompt = vec![
        json!({ "role": "system", "content": planner_system }),
        json!({ "role": "user", "content": format!("Create discovery plan for: {}", question) }),
    ];

    // Define the strict JSON schema that maps to our PlanTask struct
    let plan_schema = json!({
        "type": "json_schema",
        "json_schema": {
            "name": "plan_array",
            "schema": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "string" },
                        "description": { "type": "string" },
                        "status": { "type": "string" }
                    },
                    "required": ["id", "description"],
                    "additionalProperties": false
                }
            },
            "strict": true
        }
    });

    // Pass `Some(plan_schema)` as the response_format
    let (plan_raw, _) = call_model_stream(
        &client, 
        conn.port, 
        &conn.api_key, 
        &planner_prompt, 
        None, 
        Some(plan_schema), // <-- ENFORCE GRAMMAR HERE
        |_| {}
    ).await?;

    let mut initial_tasks: Vec<PlanTask> = Vec::new();
    if let Some(start) = plan_raw.find('[') {
        if let Some(end) = plan_raw.rfind(']') {
            let slice = &plan_raw[start..=end];
            if let Ok(parsed) = serde_json::from_str::<Vec<PlanTask>>(slice) {
                initial_tasks = parsed;
            }
        }
    }

    if initial_tasks.is_empty() {
        initial_tasks.push(PlanTask {
            id: "1".to_string(),
            description: format!("Locate and inspect relevant files for: {}", question),
            status: "pending".to_string(),
        });
    }

    // -------------------------------------------------------------
    // PHASE 2: CONCURRENT MPSC WORKER EXECUTION
    // -------------------------------------------------------------
    let (tx, mut rx) = mpsc::channel::<OrchestratorEvent>(32);
    let plan_queue = Arc::new(Mutex::new(initial_tasks));
    let accumulated_context = Arc::new(Mutex::new(Vec::<String>::new()));

    let num_workers = 2;
    for worker_idx in 0..num_workers {
        let tx_c = tx.clone();
        let plan_c = Arc::clone(&plan_queue);
        let client_c = client.clone();
        
        // Properly isolate connection primitives for the thread
        let port_c = conn.port;
        let api_key_c = conn.api_key.clone();
        
        let app_c = app.clone();
        let ws_c = conversation.workspace_path.clone();
        let pid_c = project_id_opt.clone();

        tokio::spawn(async move {
            loop {
                let next_task = {
                    let mut lock = plan_c.lock().await;
                    let task_pos = lock.iter().position(|t| t.status == "pending");
                    if let Some(pos) = task_pos {
                        lock[pos].status = "in_progress".to_string();
                        Some(lock[pos].clone())
                    } else {
                        None
                    }
                };

                let task = match next_task {
                    Some(t) => t,
                    None => break, // Queue exhausted, terminate worker
                };

                let _ = tx_c
                    .send(OrchestratorEvent::WorkerStatus(format!(
                        "Worker {}: {}",
                        worker_idx + 1,
                        task.description
                    )))
                    .await;

                let worker_system = "You are a code investigator. Use search_files, read_file, or modify_plan. When finished, summarize findings concisely.";
                let mut worker_messages = vec![
                    json!({ "role": "system", "content": worker_system }),
                    json!({ "role": "user", "content": format!("Investigate task: {}", task.description) }),
                ];

                let mut findings = String::new();
                for _ in 0..2 {
                    let (turn_response, tools_called) = match call_model_stream(
                        &client_c,
                        port_c,
                        &api_key_c,
                        &worker_messages,
                        Some(get_worker_tools()),
                        None,
                        |_| {},
                    )
                    .await
                    {
                        Ok(res) => res,
                        Err(_) => break,
                    };

                    if tools_called.is_empty() {
                        break;
                    }

                    // --- CORRECTED: Append Assistant's tool_calls payload to history first ---
                    let mut calls_for_llm = Vec::new();
                    for tc in &tools_called {
                        calls_for_llm.push(json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": tc.arguments
                            }
                        }));
                    }
                    worker_messages.push(json!({
                        "role": "assistant",
                        "content": if turn_response.trim().is_empty() { serde_json::Value::Null } else { json!(turn_response.trim()) },
                        "tool_calls": calls_for_llm
                    }));
                    // -------------------------------------------------------------------------

                    // Execute tools and append results
                    for tc in tools_called {
                        let result = if tc.name == "modify_plan" {
                            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&tc.arguments) {
                                if let Some(desc) = val["description"].as_str() {
                                    let _ = tx_c
                                        .send(OrchestratorEvent::PlanUpdate(PlanMutation::AddTask {
                                            description: desc.to_string(),
                                        }))
                                        .await;
                                }
                            }
                            "Plan updated successfully.".to_string()
                        } else {
                            execute_tool(
                                &app_c,
                                ws_c.as_deref(),
                                pid_c.as_deref(),
                                &tc.name,
                                &tc.arguments,
                            )
                            .await
                            .unwrap_or_else(|e| format!("Execution error: {}", e))
                        };

                        findings.push_str(&format!("\n[Tool: {}]\n{}\n", tc.name, result));
                        worker_messages.push(json!({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result
                        }));
                    }
                }

                let _ = tx_c
                    .send(OrchestratorEvent::TaskFinished {
                        task_id: task.id,
                        gathered_context: findings,
                    })
                    .await;
            }
        });
    }

    drop(tx); // Drop the main thread's copy of the Sender so the channel closes when workers finish

    // Central Single-Consumer Loop
    while let Some(event) = rx.recv().await {
        match event {
            OrchestratorEvent::WorkerStatus(status) => {
                let _ = on_chunk.send(AgentEvent::Status(status));
            }
            OrchestratorEvent::PlanUpdate(mutation) => match mutation {
                PlanMutation::AddTask { description } => {
                    let mut lock = plan_queue.lock().await;
                    let next_id = (lock.len() + 1).to_string();
                    lock.push(PlanTask {
                        id: next_id,
                        description,
                        status: "pending".to_string(),
                    });
                }
                PlanMutation::CompleteTask { id } => {
                    let mut lock = plan_queue.lock().await;
                    if let Some(t) = lock.iter_mut().find(|t| t.id == id) {
                        t.status = "completed".to_string();
                    }
                }
            },
            OrchestratorEvent::TaskFinished { task_id, gathered_context } => {
                {
                    let mut lock = plan_queue.lock().await;
                    if let Some(t) = lock.iter_mut().find(|t| t.id == task_id) {
                        t.status = "completed".to_string();
                    }
                }
                if !gathered_context.trim().is_empty() {
                    let mut ctx_lock = accumulated_context.lock().await;
                    ctx_lock.push(compress_context(&gathered_context));
                }
            }
        }
    }

    // -------------------------------------------------------------
    // PHASE 3: FINAL STREAMING SYNTHESIS (LLM)
    // -------------------------------------------------------------
    let _ = on_chunk.send(AgentEvent::Status("Synthesizing solution...".to_string()));

    let final_context = {
        let lock = accumulated_context.lock().await;
        lock.join("\n---\n")
    };

    let synthesis_system = format!(
        "You are Loco, an expert local software engineer.\n\
         Workspace: {}\n\
         Below is verified codebase context gathered for this request:\n\
         {}\n\
         Use this verified context to author an accurate, well-structured solution. Do not guess.",
        conversation.workspace_path.as_deref().unwrap_or("Unknown"),
        if final_context.is_empty() { "No specific file context retrieved." } else { &final_context }
    );

    let mut synthesis_messages = vec![json!({ "role": "system", "content": synthesis_system })];
    for msg in messages {
        synthesis_messages.push(json!({ "role": msg.role.as_str(), "content": msg.content }));
    }

    let on_chunk_c = on_chunk.clone();
    let (final_response, _) = call_model_stream(
        &client,
        conn.port,
        &conn.api_key,
        &synthesis_messages,
        None,
        None,
        move |chunk| {
            let _ = on_chunk_c.send(AgentEvent::Chunk(chunk));
        },
    )
    .await?;

    tauri::async_runtime::spawn_blocking(move || -> Result<_> {
        let state = app.state::<AppState>();
        state.conversations.add_message(&session_id, Role::Assistant, &final_response)?;
        Ok(())
    })
    .await
    .context("Failed to join blocking task")??;

    Ok(())
}