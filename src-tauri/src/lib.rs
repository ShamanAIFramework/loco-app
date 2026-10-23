// src-tauri/src/lib.rs
//! Main application module connecting the AI engine and local persistence stores.

use anyhow::Context;
use notify_debouncer_mini::notify::RecommendedWatcher;
use notify_debouncer_mini::Debouncer;
use std::sync::Arc;
use std::sync::Mutex;
use tauri::{Emitter, Manager};
use tokio::sync::Semaphore;

mod agent;
mod codegraph;
mod conversation_store;
mod engine;
mod git_api;
mod indexer;
mod orchestrator;
mod phoenix;
mod trace_writer;
mod vector_search;
pub mod worktree;

#[derive(Debug)]
pub struct AppError(anyhow::Error);

impl serde::Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(&format!("{:#}", self.0))
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        Self(err)
    }
}

pub struct AppState {
    pub conversations: conversation_store::ConversationStore,
    pub codegraph: codegraph::CodeGraphStore,
    pub active_watcher: Mutex<Option<Debouncer<RecommendedWatcher>>>,
    pub llm_slot: Arc<Semaphore>,
}

#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

#[tauri::command]
fn get_ai_connection(
    state: tauri::State<'_, engine::EngineState>,
) -> Option<engine::AiConnectionDetails> {
    state.connection_details()
}

pub fn resolve_workspace_root(requested: Option<String>) -> std::path::PathBuf {
    if let Some(req) = requested {
        let p = std::path::PathBuf::from(req.trim());
        if !req.trim().is_empty() && p.exists() {
            if p.file_name() == Some(std::ffi::OsStr::new("src-tauri")) {
                if let Some(parent) = p.parent() {
                    return parent.to_path_buf();
                }
            }
            return p;
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        if cwd.file_name() == Some(std::ffi::OsStr::new("src-tauri")) {
            if let Some(parent) = cwd.parent() {
                return parent.to_path_buf();
            }
        }

        let mut probe = cwd.clone();
        while let Some(parent) = probe.parent() {
            if probe.join("src-tauri").exists() {
                return probe;
            }
            if parent.join("src-tauri").exists() {
                return parent.to_path_buf();
            }
            probe = parent.to_path_buf();
        }

        return cwd;
    }

    std::path::PathBuf::from(".")
}

#[tauri::command]
async fn create_session(
    app: tauri::AppHandle,
    workspace_path: Option<String>,
) -> Result<conversation_store::Conversation, AppError> {
    let resolved = resolve_workspace_root(workspace_path);
    let path_str = resolved.to_string_lossy().to_string();
    let app_clone = app.clone();
    let path_clone = path_str.clone();

    // 1. Fast path: register project and create conversation in SQLite immediately (<5ms)
    let (conv, project_id) = tauri::async_runtime::spawn_blocking(move || -> anyhow::Result<_> {
        let state = app_clone.state::<AppState>();
        let path_obj = std::path::Path::new(&path_clone);
        let project = state.codegraph.register_project("Active Workspace", path_obj)?;

        let conv = state.conversations.create_conversation(
            Some("Active Workspace"),
            Some(&path_clone),
            Some("Qwen2.5-Coder"),
        )?;

        Ok((conv, project.id))
    })
    .await
    .context("Failed to join session creation task")??;

    // 2. Launch workspace indexing in the background so window loads instantly
    let bg_app = app.clone();
    let bg_path = path_str.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let state = bg_app.state::<AppState>();
        let path_obj = std::path::Path::new(&bg_path);

        let _ = bg_app.emit("workspace-index-status", "indexing");
        log::info!("Starting background indexing for workspace: {:?}", path_obj);

        if let Err(e) = crate::indexer::index_workspace(&state.codegraph, &project_id, path_obj) {
            log::warn!("Background indexing encountered an error: {:?}", e);
        }

        if let Ok(watcher) = crate::indexer::start_watcher(
            bg_app.clone(),
            project_id,
            path_obj.to_path_buf(),
        ) {
            let mut guard = state.active_watcher.lock().unwrap();
            *guard = Some(watcher);
        }

        let _ = bg_app.emit("workspace-index-status", "ready");
        log::info!("Background indexing complete for: {:?}", path_obj);
    });

    Ok(conv)
}

#[tauri::command]
async fn list_conversations(
    app: tauri::AppHandle,
) -> Result<Vec<conversation_store::Conversation>, AppError> {
    let app_clone = app.clone();
    let list = tauri::async_runtime::spawn_blocking(move || {
        let state = app_clone.state::<AppState>();
        state.conversations.list_conversations()
    })
    .await
    .context("Failed to join list conversations task")??;

    Ok(list)
}

#[tauri::command]
async fn trigger_index_workspace(
    app: tauri::AppHandle,
    workspace_path: Option<String>,
) -> Result<(), AppError> {
    let resolved = resolve_workspace_root(workspace_path);
    let path_str = resolved.to_string_lossy().to_string();
    let bg_app = app.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let state = bg_app.state::<AppState>();
        let path_obj = std::path::Path::new(&path_str);
        let project = match state.codegraph.register_project("Active Workspace", path_obj) {
            Ok(p) => p,
            Err(e) => {
                log::error!("Failed to register project for indexing: {:?}", e);
                return;
            }
        };

        let _ = bg_app.emit("workspace-index-status", "indexing");
        log::info!("Triggering background indexing for workspace: {:?}", path_obj);

        if let Err(e) = crate::indexer::index_workspace(&state.codegraph, &project.id, path_obj) {
            log::warn!("Background indexing encountered an error: {:?}", e);
        }

        if let Ok(watcher) = crate::indexer::start_watcher(
            bg_app.clone(),
            project.id,
            path_obj.to_path_buf(),
        ) {
            let mut guard = state.active_watcher.lock().unwrap();
            *guard = Some(watcher);
        }

        let _ = bg_app.emit("workspace-index-status", "ready");
        log::info!("Background indexing complete for: {:?}", path_obj);
    });

    Ok(())
}

#[tauri::command]
async fn get_session_details(
    app: tauri::AppHandle,
    session_id: String,
) -> Result<Option<conversation_store::ConversationDetail>, AppError> {
    let app_clone = app.clone();
    let detail = tauri::async_runtime::spawn_blocking(move || {
        let state = app_clone.state::<AppState>();
        state.conversations.load_conversation(&session_id)
    })
    .await
    .context("Failed to join blocking task")??;

    Ok(detail)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::new()
                .targets([
                    tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::Stdout),
                    tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::LogDir {
                        file_name: None,
                    }),
                    tauri_plugin_log::Target::new(tauri_plugin_log::TargetKind::Webview),
                ])
                .level(log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let app_handle = app.handle().clone();

            let home_dir = dirs::home_dir().context("Could not determine home directory")?;
            let db_dir = home_dir.join(".loco");
            std::fs::create_dir_all(&db_dir)
                .context("Failed to create application data directory")?;
            let db_path = db_dir.join("loco.db");

            let conversations = conversation_store::ConversationStore::new(&db_path)
                .context("Failed to initialize conversation store")?;
            let codegraph = codegraph::CodeGraphStore::new(&db_path)
                .context("Failed to initialize codegraph store")?;

            app.manage(AppState {
                conversations,
                codegraph,
                active_watcher: Mutex::new(None),
                llm_slot: Arc::new(Semaphore::new(1)),
            });

            app.manage(engine::EngineState::new());
            engine::initialize(app_handle);

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            get_ai_connection,
            create_session,
            list_conversations,
            get_session_details,
            trigger_index_workspace,
            orchestrator::prompt_agent_v2,
            git_api::get_git_branch,
            git_api::get_git_status,
            git_api::get_git_diff
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                engine::shutdown(app);
            }
        });
}
