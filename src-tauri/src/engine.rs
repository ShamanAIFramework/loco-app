// src-tauri/src/engine.rs
//! Manages the local llama.cpp server sidecar.
//! Spawns the server as a background process, allocates dynamic ports,
//! tracks its startup health, and exposes connection parameters.

use anyhow::{anyhow, Context, Result};
use crate::agent::PerformanceMetrics;
use serde::Serialize;
use std::net::TcpListener;
use std::sync::Mutex;
use std::time::Duration;
use tauri::path::BaseDirectory;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_shell::{process::CommandEvent, ShellExt};

const SIDECAR_NAME: &str = "llama-server";
const MODEL_RESOURCE_PATH: &str = "resources/models/default.gguf";
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
    pub is_cpu_bound: bool,
    pub offload_info: Option<String>,
}

struct EngineStateInner {
    port: Option<u16>,
    api_key: Option<String>,
    state: AiState,
    child: Option<tauri_plugin_shell::process::CommandChild>,
    is_cpu_bound: bool,
    offload_info: Option<String>,
    current_tps: f64,
    average_tps: f64,
    total_tokens: u64,
    recent_tps: Vec<f64>,
    model_size_bytes: u64,
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
                is_cpu_bound: false,
                offload_info: None,
                current_tps: 0.0,
                average_tps: 0.0,
                total_tokens: 0,
                recent_tps: Vec::new(),
                model_size_bytes: 0,
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

    /// True if the engine is running in CPU-bound mode.
    #[allow(dead_code)]
    pub fn is_cpu_bound(&self) -> bool {
        let guard = match self.inner.lock() {
            Ok(g) => g,
            Err(_) => return false,
        };
        guard.is_cpu_bound
    }

    /// Retrieves active connection details if successfully bound.
    pub fn connection_details(&self) -> Option<AiConnectionDetails> {
        let guard = self.inner.lock().ok()?;
        if let (Some(port), Some(api_key)) = (guard.port, guard.api_key.as_ref()) {
            Some(AiConnectionDetails {
                port,
                api_key: api_key.clone(),
                state: guard.state,
                is_cpu_bound: guard.is_cpu_bound,
                offload_info: guard.offload_info.clone(),
            })
        } else {
            None
        }
    }

    pub fn performance_metrics(&self) -> PerformanceMetrics {
        if let Ok(guard) = self.inner.lock() {
            PerformanceMetrics {
                current_tps: guard.current_tps,
                average_tps: guard.average_tps,
                total_tokens: guard.total_tokens,
                is_cpu_bound: guard.is_cpu_bound,
                offload_info: guard.offload_info.clone(),
            }
        } else {
            PerformanceMetrics {
                current_tps: 0.0,
                average_tps: 0.0,
                total_tokens: 0,
                is_cpu_bound: false,
                offload_info: None,
            }
        }
    }

    pub fn set_offload_status(&self, is_cpu: bool, offload_info: Option<String>) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.is_cpu_bound = is_cpu;
            if offload_info.is_some() {
                guard.offload_info = offload_info;
            }
        }
    }

    pub fn record_inference(
        &self,
        tokens: u64,
        wall_ms: u64,
        call_tps: Option<f64>,
    ) -> PerformanceMetrics {
        if let Ok(mut guard) = self.inner.lock() {
            let tps = if let Some(rate) = call_tps {
                if rate > 0.0 {
                    rate
                } else if wall_ms > 0 && tokens > 0 {
                    (tokens as f64) / (wall_ms as f64 / 1000.0)
                } else {
                    0.0
                }
            } else if wall_ms > 0 && tokens > 0 {
                (tokens as f64) / (wall_ms as f64 / 1000.0)
            } else {
                0.0
            };

            guard.current_tps = tps;
            guard.total_tokens += tokens;

            if tps > 0.0 {
                guard.recent_tps.push(tps);
                if guard.recent_tps.len() > 15 {
                    guard.recent_tps.remove(0);
                }
                guard.average_tps =
                    guard.recent_tps.iter().sum::<f64>() / guard.recent_tps.len() as f64;

                // Model-size normalized effective memory bandwidth heuristic:
                // Generation speed in autoregressive models is bound by streaming weights:
                // Effective Bandwidth (GB/s) = tokens/sec * model_size (GB).
                // On Apple Silicon, CPU memory bus achieves only ~20-30 GB/s for generation,
                // whereas Metal GPU achieves ~150-320 GB/s.
                let model_gb = if guard.model_size_bytes > 0 {
                    guard.model_size_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                } else {
                    10.0
                };
                let effective_bandwidth_gbps = tps * model_gb;

                if guard
                    .offload_info
                    .as_deref()
                    .map(|s| s.to_lowercase().contains("cpu") || s.contains("BLAS"))
                    .unwrap_or(false)
                {
                    guard.is_cpu_bound = true;
                } else if tokens >= 12 && effective_bandwidth_gbps < 45.0 {
                    guard.is_cpu_bound = true;
                    if guard.offload_info.is_none() {
                        guard.offload_info = Some(format!(
                            "Low throughput ({:.1} GB/s bandwidth, likely CPU)",
                            effective_bandwidth_gbps
                        ));
                    }
                } else {
                    guard.is_cpu_bound = false;
                }
            }

            PerformanceMetrics {
                current_tps: guard.current_tps,
                average_tps: guard.average_tps,
                total_tokens: guard.total_tokens,
                is_cpu_bound: guard.is_cpu_bound,
                offload_info: guard.offload_info.clone(),
            }
        } else {
            PerformanceMetrics {
                current_tps: 0.0,
                average_tps: 0.0,
                total_tokens: 0,
                is_cpu_bound: false,
                offload_info: None,
            }
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
        model_size_bytes: u64,
    ) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.port = Some(port);
            guard.api_key = Some(api_key);
            guard.child = Some(child);
            guard.model_size_bytes = model_size_bytes;
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
            "--parallel",
            "1",
            "--device",
            "MTL0",
            "--n-gpu-layers",
            "99",
            "--flash-attn",
            "on",
        ])
        .spawn()
        .context("Failed to spawn llama-server sidecar")?;

    let model_size_bytes = std::fs::metadata(&model_path).map(|m| m.len()).unwrap_or(0);
    engine_state.update_details(port, api_key.clone(), child, model_size_bytes);
    engine_state.set_offload_status(false, Some("Metal GPU active (MTL0)".to_string()));

fn inspect_engine_line(line: &str, state: &EngineState, app: &AppHandle) {
    let lower = line.to_lowercase();
    if lower.contains("offloaded") && lower.contains("layers") {
        if let Some(pos) = lower.find("offloaded") {
            let slice = &lower[pos..];
            let words: Vec<&str> = slice.split_whitespace().collect();
            if let Some(ratio) = words.get(1) {
                if let Some((off, tot)) = ratio.split_once('/') {
                    if let (Ok(o), Ok(t)) = (off.parse::<u32>(), tot.parse::<u32>()) {
                        let is_cpu = o == 0 || (t > 0 && o < t / 2);
                        let info = if lower.contains("metal") || lower.contains("gpu") {
                            format!("Metal GPU ({}/{} layers)", o, t)
                        } else if is_cpu {
                            format!("CPU-bound ({}/{} GPU layers)", o, t)
                        } else {
                            format!("GPU ({}/{} layers)", o, t)
                        };
                        state.set_offload_status(is_cpu, Some(info));
                        let _ = app.emit("ai-performance", state.performance_metrics());
                    }
                }
            }
        }
    } else if lower.contains("using device blas") || lower.contains("loaded cpu backend") || lower.contains("backend cpu") {
        state.set_offload_status(true, Some("CPU backend active (Accelerate/BLAS)".to_string()));
        let _ = app.emit("ai-performance", state.performance_metrics());
    } else if lower.contains("using device mtl") || lower.contains("device mtl0") || lower.contains("loaded metal backend") || lower.contains("ggml_metal_init") || lower.contains("libggml-metal") {
        state.set_offload_status(false, Some("Metal GPU active (MTL0)".to_string()));
        let _ = app.emit("ai-performance", state.performance_metrics());
    }
}

    // 6. Drain CommandEvent receiver
    let app_handle_for_task = app.clone();
    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    let text = String::from_utf8_lossy(&line);
                    log::info!(target: "llama", "{}", text);
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/loco-llama.log") {
                        use std::io::Write;
                        let _ = writeln!(f, "{}", text);
                    }
                    let state = app_handle_for_task.state::<EngineState>();
                    inspect_engine_line(&text, &state, &app_handle_for_task);
                }
                CommandEvent::Stderr(line) => {
                    let text = String::from_utf8_lossy(&line);
                    log::info!(target: "llama", "{}", text);
                    if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/loco-llama.log") {
                        use std::io::Write;
                        let _ = writeln!(f, "{}", text);
                    }
                    let state = app_handle_for_task.state::<EngineState>();
                    inspect_engine_line(&text, &state, &app_handle_for_task);
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
