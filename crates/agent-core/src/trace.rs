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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Trace {
    pub events: Vec<TraceEvent>,
}

impl Trace {
    pub fn push(&mut self, event: TraceEvent) {
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
