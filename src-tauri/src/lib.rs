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
    pub indexing_coordinator: WorkspaceIndexCoordinator,
}

pub struct WorkspaceIndexCoordinator {
    in_flight: Mutex<std::collections::HashMap<std::path::PathBuf, tokio::sync::watch::Receiver<bool>>>,
    pub completed: Mutex<std::collections::HashSet<std::path::PathBuf>>,
}

impl WorkspaceIndexCoordinator {
    pub fn new() -> Self {
        Self {
            in_flight: Mutex::new(std::collections::HashMap::new()),
            completed: Mutex::new(std::collections::HashSet::new()),
        }
    }

    pub fn is_indexing(&self, root: &std::path::Path) -> bool {
        let guard = self.in_flight.lock().unwrap();
        guard.contains_key(root)
    }

    pub fn is_completed(&self, root: &std::path::Path) -> bool {
        let guard = self.completed.lock().unwrap();
        guard.contains(root)
    }

    pub fn begin_indexing(
        &self,
        root: &std::path::Path,
    ) -> (Option<tokio::sync::watch::Sender<bool>>, tokio::sync::watch::Receiver<bool>) {
        let mut guard = self.in_flight.lock().unwrap();
        if let Some(existing) = guard.get(root) {
            (None, existing.clone())
        } else {
            let (tx, rx) = tokio::sync::watch::channel(false);
            guard.insert(root.to_path_buf(), rx.clone());
            (Some(tx), rx)
        }
    }

    pub fn finish_indexing(
        &self,
        root: &std::path::Path,
        tx: Option<tokio::sync::watch::Sender<bool>>,
    ) {
        {
            let mut guard = self.in_flight.lock().unwrap();
            guard.remove(root);
        }
        {
            let mut comp = self.completed.lock().unwrap();
            comp.insert(root.to_path_buf());
        }
        if let Some(sender) = tx {
            let _ = sender.send(true);
        }
    }

    pub fn get_receiver(&self, root: &std::path::Path) -> Option<tokio::sync::watch::Receiver<bool>> {
        let guard = self.in_flight.lock().unwrap();
        guard.get(root).cloned()
    }
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

pub fn start_workspace_indexing(
    app: tauri::AppHandle,
    workspace_root: std::path::PathBuf,
) -> tokio::sync::watch::Receiver<bool> {
    let state = app.state::<AppState>();
    let (tx_opt, rx) = state.indexing_coordinator.begin_indexing(&workspace_root);

    let Some(tx) = tx_opt else {
        return rx;
    };

    let bg_app = app.clone();
    let bg_path = workspace_root.clone();

    tauri::async_runtime::spawn_blocking(move || {
        let state = bg_app.state::<AppState>();
        let project = match state.codegraph.register_project("Active Workspace", &bg_path) {
            Ok(p) => p,
            Err(e) => {
                log::error!("Failed to register project for indexing: {:?}", e);
                state.indexing_coordinator.finish_indexing(&bg_path, Some(tx));
                return;
            }
        };

        let _ = bg_app.emit("workspace-index-status", "indexing");
        log::info!("Starting background indexing for workspace: {:?}", bg_path);

        if let Err(e) = crate::indexer::index_workspace(&state.codegraph, &project.id, &bg_path) {
            log::warn!("Background indexing encountered an error: {:?}", e);
        }

        if let Ok(watcher) = crate::indexer::start_watcher(
            bg_app.clone(),
            project.id,
            bg_path.clone(),
        ) {
            let mut guard = state.active_watcher.lock().unwrap();
            *guard = Some(watcher);
        }

        state.indexing_coordinator.finish_indexing(&bg_path, Some(tx));
        let _ = bg_app.emit("workspace-index-status", "ready");
        log::info!("Background indexing complete for: {:?}", bg_path);
    });

    rx
}

pub async fn ensure_workspace_indexed(
    app: &tauri::AppHandle,
    workspace_root: &std::path::Path,
    on_chunk: Option<&tauri::ipc::Channel<crate::agent::AgentEvent>>,
) -> Result<(), anyhow::Error> {
    let state = app.state::<AppState>();

    // 1. If indexing is currently in-flight, notify UI and await completion
    if let Some(mut rx) = state.indexing_coordinator.get_receiver(workspace_root) {
        if !*rx.borrow_and_update() {
            if let Some(channel) = on_chunk {
                let _ = channel.send(crate::agent::AgentEvent::Status(
                    "Indexing codebase before processing prompt...".to_string(),
                ));
            }
            log::info!("Waiting for in-flight indexing on: {:?}", workspace_root);
            while !*rx.borrow_and_update() {
                if rx.changed().await.is_err() {
                    break;
                }
            }
            return Ok(());
        }
    }

    // 2. If already completed in this process run, proceed immediately
    if state.indexing_coordinator.is_completed(workspace_root) {
        return Ok(());
    }

    // 3. Check if SQLite codegraph already has indexed files from previous runs
    let root_c = workspace_root.to_path_buf();
    let app_c = app.clone();
    let has_indexed_files = tauri::async_runtime::spawn_blocking(move || -> anyhow::Result<bool> {
        let state = app_c.state::<AppState>();
        let project = state.codegraph.register_project("Active Workspace", &root_c)?;
        let files = state.codegraph.project_files(&project.id)?;
        Ok(!files.is_empty())
    })
    .await??;

    if has_indexed_files {
        state.indexing_coordinator.completed.lock().unwrap().insert(workspace_root.to_path_buf());
        return Ok(());
    }

    // 4. Not yet indexed: start indexing and await completion
    if let Some(channel) = on_chunk {
        let _ = channel.send(crate::agent::AgentEvent::Status(
            "Indexing codebase before processing prompt...".to_string(),
        ));
    }
    log::info!("Starting required workspace indexing before prompt: {:?}", workspace_root);
    let mut rx = start_workspace_indexing(app.clone(), workspace_root.to_path_buf());
    while !*rx.borrow_and_update() {
        if rx.changed().await.is_err() {
            break;
        }
    }

    Ok(())
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
    let conv = tauri::async_runtime::spawn_blocking(move || -> anyhow::Result<_> {
        let state = app_clone.state::<AppState>();
        let path_obj = std::path::Path::new(&path_clone);
        let _ = state.codegraph.register_project("Active Workspace", path_obj)?;

        let conv = state.conversations.create_conversation(
            Some("Active Workspace"),
            Some(&path_clone),
            Some("Qwen2.5-Coder"),
        )?;

        Ok(conv)
    })
    .await
    .context("Failed to join session creation task")??;

    // 2. Launch workspace indexing in the background so window loads instantly
    start_workspace_indexing(app, resolved);

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
    start_workspace_indexing(app, resolved);
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
                indexing_coordinator: WorkspaceIndexCoordinator::new(),
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
