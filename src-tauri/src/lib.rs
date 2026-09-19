// src-tauri/src/lib.rs
//! Main application module connecting the AI engine and local persistence stores.
//! Coordinates initialization, state management, and exposes commands to the frontend.

use anyhow::Context;
use tauri::Manager;

mod codegraph;
mod conversation_store;
mod engine;
mod agent;

/// Wrapper for anyhow::Error to allow returning it via Tauri commands.
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

// FIX: Direct implementation avoids E0119 trait conflict errors
impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        Self(err)
    }
}

/// Global application state holding synchronous store instances.
pub struct AppState {
    pub conversations: conversation_store::ConversationStore,
    pub codegraph: codegraph::CodeGraphStore,
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

// FIX: Use AppHandle instead of State to allow moving into a 'static blocking thread
#[tauri::command]
async fn create_session(
    app: tauri::AppHandle,
    workspace_path: Option<String>,
) -> Result<conversation_store::Conversation, AppError> {
    let path = workspace_path.clone();
    
    let app_clone = app.clone();
    let conv = tauri::async_runtime::spawn_blocking(move || {
        let state = app_clone.state::<AppState>();
        state.conversations.create_conversation(
            Some("Active Workspace"), 
            path.as_deref(), 
            Some("Qwen2.5-Coder")
        )
    })
    .await
    .context("Failed to join blocking task")??; // FIX: Bubble up thread JoinErrors securely
    
    Ok(conv)
}

// FIX: Use AppHandle instead of State
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

/// Main entry point for the Tauri application.
#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_log::Builder::new().build())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_http::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let app_handle = app.handle().clone();

            // Set up local database directory
            let home_dir = dirs::home_dir().context("Could not determine home directory")?;
            let db_dir = home_dir.join(".loco");
            std::fs::create_dir_all(&db_dir)
                .context("Failed to create application data directory")?;
            let db_path = db_dir.join("loco.db");

            // Initialize stores
            let conversations = conversation_store::ConversationStore::new(&db_path)
                .context("Failed to initialize conversation store")?;
            let codegraph = codegraph::CodeGraphStore::new(&db_path)
                .context("Failed to initialize codegraph store")?;

            app.manage(AppState {
                conversations,
                codegraph,
            });

            // Initialize Engine
            app.manage(engine::EngineState::new());
            engine::initialize(app_handle);

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            greet,
            get_ai_connection,
            create_session,
            get_session_details,
            agent::prompt_agent
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app, event| {
            if let tauri::RunEvent::Exit = event {
                engine::shutdown(app);
            }
        });
}