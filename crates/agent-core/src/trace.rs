
use crate::traits::{LlmRequest, LlmResponse};
use crate::types::{Phase, StopReason};
use crate::{LeaseId, Notebook};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeMetadata {
    pub host: String,
    pub endpoint: String,
    pub model_id: Option<String>,
    pub server_version: Option<String>,
    pub queue: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum TraceEvent {
    RunStarted {
        run_id: String,
        question: String,
    },
    RuntimeMetadata {
        metadata: RuntimeMetadata,
    },
    PhaseStarted {
        phase: Phase,
    },
    LlmCall {
        request: LlmRequest,
        response: LlmResponse,
    },
    ToolCall {
        name: String,
        args: serde_json::Value,
        result_count: usize,
    },
    NotebookSnapshot {
        phase: Phase,
        notebook: Notebook,
    },
    LeaseInvalidated {
        lease_id: LeaseId,
    },
    RunFinished {
        stop_reason: StopReason,
    },
}

impl TraceEvent {
    /// Concise single-line summary of this trace event.
    pub fn summary(&self) -> String {
        match self {
            TraceEvent::RunStarted { run_id, question } => {
                let preview: String = question.chars().take(60).collect();
                format!("Run started [{run_id}]: \"{preview}\"")
            }
            TraceEvent::RuntimeMetadata { metadata } => {
                format!(
                    "Runtime: host={}, endpoint={}, model={}",
                    metadata.host,
                    metadata.endpoint,
                    metadata.model_id.as_deref().unwrap_or("unknown")
                )
            }
            TraceEvent::PhaseStarted { phase } => {
                format!(">>> Phase started: {:?}", phase)
            }
            TraceEvent::LlmCall { request, response } => {
                let wall = response.usage.wall_ms.unwrap_or(0);
                let prompt_tok = response
                    .usage
                    .prompt_tokens_total
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "?".to_string());
                let decode_tok = response
                    .usage
                    .decode_tokens
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "?".to_string());
                format!(
                    "LLM Call [{:?}] (schema: {}) in {}ms (prompt: {} tok, decode: {} tok)",
                    request.phase, request.schema_name, wall, prompt_tok, decode_tok
                )
            }
            TraceEvent::ToolCall {
                name,
                args,
                result_count,
            } => {
                format!("Tool call [{name}] args={args} -> {result_count} items")
            }
            TraceEvent::NotebookSnapshot { phase, notebook } => {
                format!(
                    "Notebook snapshot [{:?}]: {} entries",
                    phase,
                    notebook.iter_in_append_order().count()
                )
            }
            TraceEvent::LeaseInvalidated { lease_id } => {
                format!("Lease invalidated: {}", lease_id.0)
            }
            TraceEvent::RunFinished { stop_reason } => {
                format!("Run finished: stop_reason={:?}", stop_reason)
            }
        }
    }

    /// Detailed multi-line human-readable representation suitable for live logs and debugging.
    pub fn format_pretty(&self) -> String {
        match self {
            TraceEvent::RunStarted { run_id, question } => {
                format!(
                    "==================================================\n\
                     AGENT RUN STARTED\n\
                     Run ID:   {run_id}\n\
                     Question: {question}\n\
                     =================================================="
                )
            }
            TraceEvent::RuntimeMetadata { metadata } => {
                format!(
                    "[RUNTIME METADATA]\n\
                     Host:     {}\n\
                     Endpoint: {}\n\
                     Model:    {}\n\
                     Queue:    {}",
                    metadata.host,
                    metadata.endpoint,
                    metadata.model_id.as_deref().unwrap_or("unknown"),
                    metadata.queue
                )
            }
            TraceEvent::PhaseStarted { phase } => {
                format!("\n>>> PHASE: {:?} <<<", phase)
            }
            TraceEvent::LlmCall { request, response } => {
                let wall = response.usage.wall_ms.unwrap_or(0);
                let prompt_tok = response
                    .usage
                    .prompt_tokens_total
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "?".to_string());
                let decode_tok = response
                    .usage
                    .decode_tokens
                    .map(|t| t.to_string())
                    .unwrap_or_else(|| "?".to_string());
                format!(
                    "--------------------------------------------------\n\
                     LLM CALL [{:?}]\n\
                     Call ID:     {}\n\
                     Schema:      {}\n\
                     Latency:     {}ms\n\
                     Tokens:      prompt={}, decode={}\n\
                     Temperature: {}\n\
                     --- PROMPT ---\n\
                     {}\n\
                     --- RESPONSE (RAW JSON) ---\n\
                     {}\n\
                     --------------------------------------------------",
                    request.phase,
                    request.call_id,
                    request.schema_name,
                    wall,
                    prompt_tok,
                    decode_tok,
                    request.temperature,
                    request.prompt.trim(),
                    response.raw_json.trim()
                )
            }
            TraceEvent::ToolCall {
                name,
                args,
                result_count,
            } => {
                format!(
                    "[TOOL CALL] {}\n\
                     Arguments: {}\n\
                     Results:   {} items",
                    name, args, result_count
                )
            }
            TraceEvent::NotebookSnapshot { phase, notebook } => {
                format!(
                    "[NOTEBOOK SNAPSHOT] Phase: {:?}, Active entries: {}",
                    phase,
                    notebook.iter_in_append_order().count()
                )
            }
            TraceEvent::LeaseInvalidated { lease_id } => {
                format!("[LEASE INVALIDATED] {}", lease_id.0)
            }
            TraceEvent::RunFinished { stop_reason } => {
                format!(
                    "==================================================\n\
                     AGENT RUN FINISHED\n\
                     Stop Reason: {:?}\n\
                     ==================================================",
                    stop_reason
                )
            }
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Trace {
    pub events: Vec<TraceEvent>,
}

impl Trace {
    pub fn push(&mut self, event: TraceEvent) {
        log::info!("[agent-core] {}", event.summary());

        if std::env::var_os("LOCO_TRACE_STDERR").is_some()
            || std::env::var_os("LOCO_DEBUG").is_some()
        {
            eprintln!("[agent-core] {}", event.format_pretty());
        }

        if let Ok(path) = std::env::var("LOCO_TRACE_FILE") {
            if let Ok(line) = serde_json::to_string(&event) {
                use std::io::Write;
                if let Ok(mut file) = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    let _ = writeln!(file, "{}", line);
                }
            }
        }

        self.events.push(event);
    }

    pub fn to_jsonl(&self) -> String {
        self.events
            .iter()
            .filter_map(|event| serde_json::to_string(event).ok())
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{LlmRequest, LlmResponse, LlmUsage};
    use crate::types::Phase;

    #[test]
    fn trace_event_formatting_and_summary_work() {
        let event = TraceEvent::LlmCall {
            request: LlmRequest {
                call_id: "intake-1".to_string(),
                phase: Phase::Intake,
                schema_name: "IntakeOutput".to_string(),
                schema: serde_json::json!({}),
                prompt_parts: None,
                prompt: "Please classify the request.".to_string(),
                temperature: 0.2,
            },
            response: LlmResponse {
                call_id: "intake-1".to_string(),
                raw_json: r#"{"task_type":"Locate"}"#.to_string(),
                usage: LlmUsage {
                    queue_wait_ms: Some(15),
                    prompt_tokens_total: Some(250),
                    prompt_tokens_evaluated: None,
                    prompt_tokens_cached: None,
                    decode_tokens: Some(40),
                    prefill_ms: None,
                    decode_ms: None,
                    wall_ms: Some(350),
                },
            },
        };

        let summary = event.summary();
        assert!(summary.contains("LLM Call [Intake]"));
        assert!(summary.contains("350ms"));
        assert!(summary.contains("prompt: 250 tok"));

        let pretty = event.format_pretty();
        assert!(pretty.contains("LLM CALL [Intake]"));
        assert!(pretty.contains("Please classify the request."));
        assert!(pretty.contains(r#"{"task_type":"Locate"}"#));

        let mut trace = Trace::default();
        trace.push(event);
        assert_eq!(trace.events.len(), 1);
        let jsonl = trace.to_jsonl();
        assert!(jsonl.contains("intake-1"));
    }
}
