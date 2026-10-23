// src-tauri/src/lib.rs
//! Main application module connecting the AI engine and local persistence stores.

use anyhow::Context;
use notify_debouncer_mini::notify::RecommendedWatcher;
use notify_debouncer_mini::Debouncer;
use std::sync::Arc;
use std::sync::Mutex;
use tauri::Manager;
use tokio::sync::Semaphore;

mod agent;
mod agentv2;
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

#[tauri::command]
async fn create_session(
    app: tauri::AppHandle,
    workspace_path: Option<String>,
) -> Result<conversation_store::Conversation, AppError> {
    // Resolve path, falling back to current working dir if path doesn't exist
    let path = workspace_path
        .filter(|p| !p.trim().is_empty() && std::path::Path::new(p).exists())
        .or_else(|| std::env::current_dir().ok().map(|d| d.to_string_lossy().to_string()));
    let app_clone = app.clone();

    let conv = tauri::async_runtime::spawn_blocking(move || {
        let state = app_clone.state::<AppState>();

        // If a path is provided, register the project and trigger the indexer immediately
        if let Some(p) = &path {
            let path_obj = std::path::Path::new(p);
            if let Ok(project) = state
                .codegraph
                .register_project("Active Workspace", path_obj)
            {
                // 1. Do the initial full index
                let _ = crate::indexer::index_workspace(&state.codegraph, &project.id, path_obj);

                // 2. Start the incremental watcher
                if let Ok(watcher) = crate::indexer::start_watcher(
                    app_clone.clone(),
                    project.id,
                    path_obj.to_path_buf(),
                ) {
                    let mut guard = state.active_watcher.lock().unwrap();
                    *guard = Some(watcher);
                }
            }
        }

        state.conversations.create_conversation(
            Some("Active Workspace"),
            path.as_deref(),
            Some("Qwen2.5-Coder"),
        )
    })
    .await
    .context("Failed to join blocking task")??;

    Ok(conv)
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
            get_session_details,
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
