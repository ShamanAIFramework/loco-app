// src-tauri/src/trace_writer.rs
//! Real-time trace logger and file manager for Loco agent runs.
//!
//! Maintains real-time tail-able trace files in `~/.loco/traces/`:
//! - `live.log`: Formatted human-readable stream of all agent steps, LLM prompts, completions, and tools.
//!   Can be monitored with: `tail -f ~/.loco/traces/live.log`
//! - `live.jsonl`: Raw JSONL stream of all `TraceEvent` objects.
//!   Can be monitored with: `tail -f ~/.loco/traces/live.jsonl`
//! - `<run_id>.jsonl`: JSONL trace for the specific run, flushed incrementally.
//! - `latest.log` & `latest.jsonl`: Current / most recent run traces.
//!
//! Also integrates with Arize Phoenix when running locally.

use crate::phoenix::{self, PhoenixConfig};
use agent_core::TraceEvent;
use anyhow::{anyhow, Result};
use chrono::Local;
use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Clone)]
pub struct TraceLogger {
    run_id: String,
    question: Option<String>,
    model_id: ArcSwapOptionString,
    traces_dir: PathBuf,
    phoenix_config: PhoenixConfig,
    http_client: reqwest::Client,
    file_lock: std::sync::Arc<Mutex<()>>,
}

// Simple thread-safe option string for model_id updates
#[derive(Clone, Default)]
struct ArcSwapOptionString(std::sync::Arc<Mutex<Option<String>>>);

impl ArcSwapOptionString {
    fn set(&self, val: String) {
        if let Ok(mut lock) = self.0.lock() {
            *lock = Some(val);
        }
    }
    fn get(&self) -> Option<String> {
        self.0.lock().ok().and_then(|lock| lock.clone())
    }
}

impl TraceLogger {
    pub fn new(run_id: String, question: Option<String>) -> Result<Self> {
        let home = dirs::home_dir().ok_or_else(|| anyhow!("Could not determine home directory"))?;
        let traces_dir = home.join(".loco").join("traces");
        create_dir_all(&traces_dir)?;

        // Clear or initialize latest.log and latest.jsonl for this new run
        let latest_log = traces_dir.join("latest.log");
        let latest_jsonl = traces_dir.join("latest.jsonl");
        let _ = std::fs::write(&latest_log, "");
        let _ = std::fs::write(&latest_jsonl, "");

        Ok(Self {
            run_id,
            question,
            model_id: ArcSwapOptionString::default(),
            traces_dir,
            phoenix_config: PhoenixConfig::default(),
            http_client: reqwest::Client::new(),
            file_lock: std::sync::Arc::new(Mutex::new(())),
        })
    }

    pub fn set_model_id(&self, model: String) {
        self.model_id.set(model);
    }

    /// Record a trace event in real-time.
    pub fn record_event(&self, event: &TraceEvent) {
        // Update model_id if RuntimeMetadata is seen
        if let TraceEvent::RuntimeMetadata { metadata } = event {
            if let Some(id) = &metadata.model_id {
                self.set_model_id(id.clone());
            }
        }

        // 1. Emit to system log (visible in stdout / terminal)
        log::info!("[loco-trace] {}", event.summary());

        // 2. Prepare serialized and formatted representations
        let now_str = Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string();
        let json_line = serde_json::to_string(event).unwrap_or_default();
        let pretty_entry = format!("[{now_str}] {}\n", event.format_pretty());

        // 3. Append to files safely under lock
        if let Ok(_guard) = self.file_lock.lock() {
            // Append to live.jsonl
            append_line(&self.traces_dir.join("live.jsonl"), &json_line);
            // Append to latest.jsonl
            append_line(&self.traces_dir.join("latest.jsonl"), &json_line);
            // Append to <run_id>.jsonl
            append_line(
                &self.traces_dir.join(format!("{}.jsonl", self.run_id)),
                &json_line,
            );

            // Append to live.log
            append_text(&self.traces_dir.join("live.log"), &pretty_entry);
            // Append to latest.log
            append_text(&self.traces_dir.join("latest.log"), &pretty_entry);
        }

        // 4. Export to Phoenix Arize asynchronously if enabled
        phoenix::export_trace_event_async(
            self.http_client.clone(),
            self.phoenix_config.clone(),
            self.run_id.clone(),
            self.question.clone(),
            self.model_id.get(),
            event.clone(),
        );
    }

    /// Record an unexpected error during the agent run.
    pub fn record_error(&self, err: &anyhow::Error) {
        log::error!("[loco-trace] Agent run failed: {:#}", err);

        let now_str = Local::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string();
        let entry = format!(
            "[{now_str}] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\
             [{now_str}] AGENT ERROR: {:#}\n\
             [{now_str}] !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n",
            err
        );

        if let Ok(_guard) = self.file_lock.lock() {
            append_text(&self.traces_dir.join("live.log"), &entry);
            append_text(&self.traces_dir.join("latest.log"), &entry);
        }
    }
}

fn append_line(path: &Path, line: &str) {
    if line.is_empty() {
        return;
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(f, "{}", line);
    }
}

fn append_text(path: &Path, text: &str) {
    if text.is_empty() {
        return;
    }
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = write!(f, "{}", text);
    }
}
