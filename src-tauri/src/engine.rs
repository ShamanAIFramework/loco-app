// src-tauri/src/engine.rs
//! Manages the local llama.cpp server sidecar.
//! Spawns the server as a background process, allocates dynamic ports,
//! tracks its startup health, and exposes connection parameters.

use anyhow::{anyhow, Context, Result};
use serde::Serialize;
use std::net::TcpListener;
use std::sync::Mutex;
use std::time::Duration;
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::{process::CommandEvent, ShellExt};

const SIDECAR_NAME: &str = "llama-server";
const MODEL_RESOURCE_PATH: &str = "resources/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf";
const HOST: &str = "127.0.0.1";
const CTX_SIZE: &str = "32768";
const POLL_INTERVAL: Duration = Duration::from_millis(500);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(2);
const STARTUP_TIMEOUT: Duration = Duration::from_secs(180);

/// Current health/lifecycle state of the AI engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum AiState {
    Starting,
    Ready,
    Failed,
}

/// Active connection details for API consumers.
#[derive(Clone, Debug, Serialize)]
pub struct AiConnectionDetails {
    pub port: u16,
    pub api_key: String,
    pub state: AiState,
}

struct EngineStateInner {
    port: Option<u16>,
    api_key: Option<String>,
    state: AiState,
    child: Option<tauri_plugin_shell::process::CommandChild>,
}

/// Thread-safe wrapper for the engine's lifecycle state.
pub struct EngineState {
    inner: Mutex<EngineStateInner>,
}

impl EngineState {
    /// Creates a new EngineState starting in the Starting condition.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(EngineStateInner {
                port: None,
                api_key: None,
                state: AiState::Starting,
                child: None,
            }),
        }
    }

    /// True if the engine is ready for inference.
    #[allow(dead_code)]
    pub fn is_ready(&self) -> bool {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        guard.state == AiState::Ready
    }

    /// Retrieves active connection details if successfully bound.
    pub fn connection_details(&self) -> Option<AiConnectionDetails> {
        let guard = self.inner.lock().ok()?;
        if let (Some(port), Some(api_key)) = (guard.port, guard.api_key.as_ref()) {
            Some(AiConnectionDetails {
                port,
                api_key: api_key.clone(),
                state: guard.state,
            })
        } else {
            None
        }
    }

    fn set_state(&self, state: AiState) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.state = state;
        }
    }

    fn update_details(
        &self,
        port: u16,
        api_key: String,
        child: tauri_plugin_shell::process::CommandChild,
    ) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.port = Some(port);
            guard.api_key = Some(api_key);
            guard.child = Some(child);
        }
    }

    fn take_child(&self) -> Option<tauri_plugin_shell::process::CommandChild> {
        if let Ok(mut guard) = self.inner.lock() {
            guard.child.take()
        } else {
            None
        }
    }
}

/// Starts the engine sidecar asynchronously.
pub fn initialize(app_handle: AppHandle) {
    tauri::async_runtime::spawn(async move {
        if let Err(e) = initialize_inner(&app_handle).await {
            log::error!("AI engine failed to initialize: {:#}", e);
            let state = app_handle.state::<EngineState>();
            state.set_state(AiState::Failed);
            let _ = app_handle.emit("ai-state", "failed");
        }
    });
}

async fn initialize_inner(app: &AppHandle) -> Result<()> {
    let engine_state = app.state::<EngineState>();

    // 1. Emit starting
    let _ = app.emit("ai-state", "starting");

    // 2. Pick a free port. There is a small race condition here between dropping the
    // listener and the sidecar claiming it, but it is standard for local orchestration.
    let listener = TcpListener::bind((HOST, 0)).context("Failed to bind to dynamic port")?;
    let port = listener
        .local_addr()
        .context("Failed to get local addr")?
        .port();
    drop(listener);

    // 3. Generate session API key
    let api_key = format!(
        "{}{}",
        uuid::Uuid::new_v4().simple(),
        uuid::Uuid::new_v4().simple()
    );

    // 4. Resolve the model
    let model_path = app
        .path()
        .resolve(MODEL_RESOURCE_PATH, BaseDirectory::Resource)
        .context("Failed to resolve model resource path")?;

    if !model_path.exists() {
        return Err(anyhow!("Model file missing at {:?}", model_path));
    }

    // 5. Spawn the sidecar
    let (mut rx, child) = app
        .shell()
        .sidecar(SIDECAR_NAME)
        .context("Failed to locate sidecar binary")?
        .args([
            "serve",
            "--host",
            HOST,
            "--port",
            &port.to_string(),
            "--api-key",
            &api_key,
            "--no-ui",
            "--model",
            &model_path.to_string_lossy(),
            "--ctx-size",
            CTX_SIZE,
        ])
        .spawn()
        .context("Failed to spawn llama-server sidecar")?;

    engine_state.update_details(port, api_key.clone(), child);

    // 6. Drain CommandEvent receiver
    let app_handle_for_task = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    log::info!(target: "llama", "{}", String::from_utf8_lossy(&line));
                }
                CommandEvent::Stderr(line) => {
                    log::info!(target: "llama", "{}", String::from_utf8_lossy(&line));
                }
                CommandEvent::Error(err) => {
                    log::error!("llama-server error: {}", err);
                }
                CommandEvent::Terminated(payload) => {
                    let state = app_handle_for_task.state::<EngineState>();
                    let is_failed = {
                        if let Ok(guard) = state.inner.lock() {
                            guard.state == AiState::Starting || guard.state == AiState::Ready
                        } else {
                            false
                        }
                    };
                    if is_failed {
                        log::error!(
                            "llama-server terminated unexpectedly with code: {:?}",
                            payload.code
                        );
                        state.set_state(AiState::Failed);
                        let _ = app_handle_for_task.emit("ai-state", "failed");
                    }
                    break;
                }
                _ => {}
            }
        }
    });

    // 7. Poll HTTP endpoint
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("Failed to build HTTP client")?;

    let url = format!("http://{}:{}/v1/models", HOST, port);
    let start = tokio::time::Instant::now();

    loop {
        if start.elapsed() > STARTUP_TIMEOUT {
            if let Some(child) = engine_state.take_child() {
                let _ = child.kill();
            }
            return Err(anyhow!(
                "Timed out waiting for llama-server to become ready"
            ));
        }

        // Fast abort if process crashed
        let current_state = {
            let guard = engine_state
                .inner
                .lock()
                .map_err(|_| anyhow!("Mutex poisoned"))?;
            guard.state
        };
        if current_state == AiState::Failed {
            return Err(anyhow!("Engine process failed before becoming ready"));
        }

        match client
            .get(&url)
            .header("Authorization", format!("Bearer {}", api_key))
            .send()
            .await
        {
            Ok(res) if res.status().is_success() => {
                engine_state.set_state(AiState::Ready);
                let _ = app.emit("ai-state", "ready");
                log::info!("AI Engine ready");
                break;
            }
            _ => {
                tokio::time::sleep(POLL_INTERVAL).await;
            }
        }
    }

    Ok(())
}

/// Synchronously stops the sidecar process if running.
pub fn shutdown(app_handle: &AppHandle) {
    let state = app_handle.state::<EngineState>();
    if let Some(child) = state.take_child() {
        log::info!("Shutting down engine sidecar");
        if let Err(e) = child.kill() {
            log::warn!("Failed to cleanly kill sidecar process: {}", e);
        }
    }
}
