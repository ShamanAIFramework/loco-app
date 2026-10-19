// src-tauri/src/phoenix.rs
//! Arize Phoenix OpenInference trace exporter for Loco agent traces.
//!
//! Posts traces directly to Arize Phoenix's REST API at `/v1/projects/{project_name}/spans`.
//! Automatically creates the project in Phoenix on the first trace, and builds
//! hierarchical trace trees (Agent Root -> LLM calls and Tool calls).

use agent_core::TraceEvent;
use chrono::{Duration, Utc};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::time::Duration as StdDuration;

#[derive(Clone, Debug)]
pub struct PhoenixConfig {
    pub endpoint: String,
    pub project_name: String,
    pub enabled: bool,
}

impl Default for PhoenixConfig {
    fn default() -> Self {
        let endpoint = std::env::var("PHOENIX_COLLECTOR_ENDPOINT")
            .or_else(|_| std::env::var("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"))
            .or_else(|_| std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT"))
            .unwrap_or_else(|_| "http://127.0.0.1:6006".to_string());

        let project_name =
            std::env::var("PHOENIX_PROJECT_NAME").unwrap_or_else(|_| "loco".to_string());

        let enabled = std::env::var("PHOENIX_ENABLED")
            .map(|val| val != "0" && val.to_lowercase() != "false")
            .unwrap_or(true);

        Self {
            endpoint,
            project_name,
            enabled,
        }
    }
}

impl PhoenixConfig {
    /// Return the spans REST API URL: `/v1/projects/{project_name}/spans`.
    pub fn spans_url(&self) -> String {
        let base = self.endpoint.trim_end_matches('/');
        if base.contains("/v1/projects/") {
            base.to_string()
        } else if let Some(stripped) = base.strip_suffix("/v1/traces") {
            format!("{}/v1/projects/{}/spans", stripped.trim_end_matches('/'), self.project_name)
        } else {
            format!("{}/v1/projects/{}/spans", base, self.project_name)
        }
    }
}

/// Convert a run_id into a valid 32-character hex trace ID.
pub fn to_trace_id(run_id: &str) -> String {
    let clean = run_id.replace('-', "").to_lowercase();
    if clean.len() == 32 && clean.chars().all(|c| c.is_ascii_hexdigit()) {
        clean
    } else {
        let hash = Sha256::digest(run_id.as_bytes());
        format!("{:032x}", &hash[..16].iter().fold(0u128, |acc, &b| (acc << 8) | (b as u128)))
    }
}

/// Generate a deterministic root span ID for a run.
pub fn root_span_id(run_id: &str) -> String {
    let hash = Sha256::digest(format!("root:{}", run_id).as_bytes());
    format!("{:016x}", &hash[..8].iter().fold(0u64, |acc, &b| (acc << 8) | (b as u64)))
}

/// Generate a new 16-character hex span ID.
pub fn new_span_id() -> String {
    let id = uuid::Uuid::new_v4().as_u128() as u64;
    format!("{:016x}", id)
}

/// Send spans to Phoenix REST API.
pub async fn send_spans(
    client: &reqwest::Client,
    config: &PhoenixConfig,
    spans: Vec<Value>,
) {
    if !config.enabled || spans.is_empty() {
        return;
    }

    let url = config.spans_url();
    let body = json!({ "data": spans });

    let res = client
        .post(&url)
        .header("Content-Type", "application/json")
        .timeout(StdDuration::from_secs(3))
        .json(&body)
        .send()
        .await;

    match res {
        Ok(resp) => {
            let status = resp.status();
            if status.is_success() {
                log::info!(
                    "[phoenix] Successfully exported {} span(s) to project '{}'",
                    spans.len(),
                    config.project_name
                );
            } else {
                let err_text = resp.text().await.unwrap_or_default();
                log::warn!(
                    "[phoenix] Phoenix rejected spans with status {}: {}",
                    status,
                    err_text
                );
            }
        }
        Err(err) => {
            log::debug!(
                "[phoenix] Could not send trace to Phoenix at {} (is Phoenix running?): {}",
                url,
                err
            );
        }
    }
}

/// Convert a `TraceEvent` into an OpenInference span and export to Phoenix.
pub fn export_trace_event_async(
    client: reqwest::Client,
    config: PhoenixConfig,
    run_id: String,
    question: Option<String>,
    model_id: Option<String>,
    event: TraceEvent,
) {
    if !config.enabled {
        return;
    }

    tauri::async_runtime::spawn(async move {
        let trace_id = to_trace_id(&run_id);
        let root_id = root_span_id(&run_id);
        let now = Utc::now();

        let span = match event {
            TraceEvent::RunStarted { question, .. } => {
                let q_preview: String = question.chars().take(50).collect();
                Some(json!({
                    "name": format!("Loco Agent: {}", q_preview),
                    "span_kind": "CHAIN",
                    "start_time": now.to_rfc3339(),
                    "end_time": (now + Duration::milliseconds(50)).to_rfc3339(),
                    "status_code": "OK",
                    "status_message": "",
                    "parent_id": null,
                    "context": {
                        "trace_id": trace_id,
                        "span_id": root_id
                    },
                    "attributes": {
                        "openinference.span.kind": "CHAIN",
                        "input.value": question,
                        "loco.run_id": run_id
                    },
                    "events": []
                }))
            }
            TraceEvent::LlmCall { request, response } => {
                let wall_ms = response.usage.wall_ms.unwrap_or(0);
                let start_time = now - Duration::milliseconds(wall_ms as i64);

                let mut attrs = json!({
                    "openinference.span.kind": "LLM",
                    "llm.model_name": model_id.unwrap_or_else(|| "default".to_string()),
                    "input.value": request.prompt,
                    "output.value": response.raw_json,
                    "llm.invocation_parameters": json!({
                        "schema_name": request.schema_name,
                        "temperature": request.temperature,
                    }).to_string(),
                    "loco.phase": format!("{:?}", request.phase),
                    "loco.call_id": request.call_id,
                    "llm.temperature": request.temperature,
                });

                if let Some(prompt_tokens) = response.usage.prompt_tokens_total {
                    attrs["llm.token_count.prompt"] = json!(prompt_tokens);
                }
                if let Some(decode_tokens) = response.usage.decode_tokens {
                    attrs["llm.token_count.completion"] = json!(decode_tokens);
                }
                if let (Some(p), Some(d)) = (
                    response.usage.prompt_tokens_total,
                    response.usage.decode_tokens,
                ) {
                    attrs["llm.token_count.total"] = json!(p + d);
                }

                Some(json!({
                    "name": format!("LLM [{:?}] {}", request.phase, request.schema_name),
                    "span_kind": "LLM",
                    "start_time": start_time.to_rfc3339(),
                    "end_time": now.to_rfc3339(),
                    "status_code": "OK",
                    "status_message": "",
                    "parent_id": root_id,
                    "context": {
                        "trace_id": trace_id,
                        "span_id": new_span_id()
                    },
                    "attributes": attrs,
                    "events": []
                }))
            }
            TraceEvent::ToolCall {
                name,
                args,
                result_count,
            } => {
                Some(json!({
                    "name": format!("Tool: {}", name),
                    "span_kind": "TOOL",
                    "start_time": now.to_rfc3339(),
                    "end_time": now.to_rfc3339(),
                    "status_code": "OK",
                    "status_message": "",
                    "parent_id": root_id,
                    "context": {
                        "trace_id": trace_id,
                        "span_id": new_span_id()
                    },
                    "attributes": {
                        "openinference.span.kind": "TOOL",
                        "tool.name": name,
                        "tool.parameters": args.to_string(),
                        "tool.output": format!("{} results", result_count),
                        "input.value": args.to_string(),
                        "output.value": format!("{} results", result_count),
                        "tool.result_count": result_count
                    },
                    "events": []
                }))
            }
            TraceEvent::RunFinished { stop_reason } => {
                let q_str = question.unwrap_or_else(|| "Coding task".to_string());
                let q_preview: String = q_str.chars().take(50).collect();
                Some(json!({
                    "name": format!("Loco Agent: {}", q_preview),
                    "span_kind": "CHAIN",
                    "start_time": now.to_rfc3339(),
                    "end_time": now.to_rfc3339(),
                    "status_code": "OK",
                    "status_message": "",
                    "parent_id": null,
                    "context": {
                        "trace_id": trace_id,
                        "span_id": root_id
                    },
                    "attributes": {
                        "openinference.span.kind": "CHAIN",
                        "input.value": q_str,
                        "output.value": format!("{:?}", stop_reason),
                        "loco.stop_reason": format!("{:?}", stop_reason)
                    },
                    "events": []
                }))
            }
            _ => None,
        };

        if let Some(span) = span {
            send_spans(&client, &config, vec![span]).await;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_id_is_always_32_hex_chars() {
        let uuid_str = uuid::Uuid::new_v4().to_string();
        let trace_id = to_trace_id(&uuid_str);
        assert_eq!(trace_id.len(), 32);
        assert!(trace_id.chars().all(|c| c.is_ascii_hexdigit()));

        let custom_str = "my-custom-run-123";
        let trace_id_2 = to_trace_id(custom_str);
        assert_eq!(trace_id_2.len(), 32);
        assert!(trace_id_2.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn root_and_span_ids_are_16_hex_chars() {
        let span_id = new_span_id();
        assert_eq!(span_id.len(), 16);
        assert!(span_id.chars().all(|c| c.is_ascii_hexdigit()));

        let root_id = root_span_id("test-run-1");
        assert_eq!(root_id.len(), 16);
        assert!(root_id.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn spans_url_formats_correctly() {
        let config = PhoenixConfig {
            endpoint: "http://127.0.0.1:6006".to_string(),
            project_name: "loco".to_string(),
            enabled: true,
        };
        assert_eq!(config.spans_url(), "http://127.0.0.1:6006/v1/projects/loco/spans");

        let config_with_traces = PhoenixConfig {
            endpoint: "http://localhost:6006/v1/traces".to_string(),
            project_name: "my-project".to_string(),
            enabled: true,
        };
        assert_eq!(
            config_with_traces.spans_url(),
            "http://localhost:6006/v1/projects/my-project/spans"
        );
    }
}
