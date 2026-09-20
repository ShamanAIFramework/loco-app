use crate::notebook::{Notebook, Provenance};
use crate::types::{EvidenceSlice, Phase, TaskRequest};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromptParts {
    pub global_prefix: String,
    pub phase_prefix: String,
    pub task: String,
    pub notebook: String,
    pub evidence: String,
    pub question: String,
}

impl PromptParts {
    pub fn render(&self) -> String {
        [
            self.global_prefix.as_str(),
            self.phase_prefix.as_str(),
            self.task.as_str(),
            self.notebook.as_str(),
            self.evidence.as_str(),
            self.question.as_str(),
        ]
        .join("\n\n")
    }
}

pub fn global_prefix(repo_map: &[String]) -> String {
    let repo = if repo_map.is_empty() {
        "No indexed files are available.".to_string()
    } else {
        repo_map.join("\n")
    };
    format!(
        "SYSTEM\nYou are Loco's bounded reasoning coprocessor. Deterministic Rust code controls workflow, tools, budgets, and stopping. You return only JSON matching the requested schema. The codebase is the source of truth.\n\nREPO MAP SNAPSHOT\n{}",
        repo
    )
}

pub fn phase_prefix(phase: Phase, schema_name: &str) -> String {
    format!(
        "PHASE {:?}\nReturn JSON for schema `{}` only. Do not call tools. Do not include markdown.",
        phase, schema_name
    )
}

pub fn task_block(task: &TaskRequest) -> String {
    format!(
        "TASK\nWorkspace: {}\nUser request: {}",
        task.workspace_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        task.question
    )
}

pub fn notebook_block(notebook: &Notebook, target_tags: &[String], limit: usize) -> String {
    let entries = notebook.active_entries_for_tags(target_tags, limit);
    if entries.is_empty() {
        return "NOTEBOOK\nNo active notebook entries selected.".to_string();
    }

    let mut lines = Vec::new();
    for entry in entries {
        lines.push(format!(
            "- {} {:?} {:?} {}: {}",
            entry.id.0,
            entry.kind,
            entry.status,
            provenance_label(&entry.provenance),
            entry.statement
        ));
    }
    format!("NOTEBOOK\n{}", lines.join("\n"))
}

pub fn evidence_block(evidence: &[EvidenceSlice]) -> String {
    if evidence.is_empty() {
        return "EPHEMERAL EVIDENCE\nNo current code slices.".to_string();
    }

    let mut chunks = Vec::new();
    for slice in evidence {
        chunks.push(format!(
            "LEASE {}\nFILE {}\nLINES {}-{}\nHASH {}\n```text\n{}\n```",
            slice.lease.id.0,
            slice.lease.path,
            slice.lease.range.start,
            slice.lease.range.end,
            slice.lease.content_hash,
            line_numbered_content(slice)
        ));
    }
    format!("EPHEMERAL EVIDENCE\n{}", chunks.join("\n\n"))
}

fn line_numbered_content(slice: &EvidenceSlice) -> String {
    slice
        .content
        .lines()
        .enumerate()
        .map(|(idx, line)| format!("{:>5}: {}", slice.lease.range.start + idx, line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn provenance_label(provenance: &Provenance) -> String {
    match provenance {
        Provenance::File {
            path,
            symbol,
            range,
            ..
        } => {
            let symbol = symbol
                .as_ref()
                .map(|value| format!(" symbol={}", value))
                .unwrap_or_default();
            format!("[{} lines {}-{}{}]", path, range.start, range.end, symbol)
        }
        Provenance::CodeGraph { query, .. } => format!("[codegraph query={}]", query),
        Provenance::Model { call_id } => format!("[model call={}]", call_id),
        Provenance::Tool { name, .. } => format!("[tool={}]", name),
    }
}

pub fn schema_intake() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "task_type": { "type": "string", "enum": ["Explain", "Locate", "Unclear"] },
            "target_hints": { "type": "array", "items": { "type": "string" } },
            "constraints": { "type": "array", "items": { "type": "string" } },
            "search_queries": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["task_type", "target_hints", "constraints", "search_queries"],
        "additionalProperties": false
    })
}

pub fn schema_extract() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "inferences": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": { "type": "string" },
                        "lease_id": { "type": "string" },
                        "tags": { "type": "array", "items": { "type": "string" } }
                    },
                    "required": ["statement", "lease_id", "tags"],
                    "additionalProperties": false
                }
            },
            "sufficient": { "type": "boolean" },
            "missing": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": { "type": "string", "enum": ["Callers", "Callees", "Definition", "References", "FileRange", "Search"] },
                        "target": { "type": "string" },
                        "reason": { "type": "string" }
                    },
                    "required": ["kind", "target", "reason"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["inferences", "sufficient", "missing"],
        "additionalProperties": false
    })
}

pub fn schema_synthesize() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "answer": { "type": "string" },
            "cited_entries": { "type": "array", "items": { "type": "string" } },
            "stop_reason": { "type": "string", "enum": ["Done", "NotFound", "NeedsUser", "NoProgress", "BudgetExhausted", "Cancelled", "Error"] }
        },
        "required": ["answer", "cited_entries", "stop_reason"],
        "additionalProperties": false
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notebook::Provenance;

    #[test]
    fn consecutive_prompts_share_stable_prefix() {
        let global = global_prefix(&["src/main.rs".to_string()]);
        let phase = phase_prefix(Phase::Extract, "ExtractOutput");
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "Explain main".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
        };
        let mut notebook = Notebook::default();
        notebook.append_tool_fact(
            "Read src/main.rs",
            Provenance::Tool {
                name: "read_file".to_string(),
                call_id: "read-1".to_string(),
            },
            Phase::Extract,
            vec!["src/main.rs".to_string()],
        );

        let task_rendered = task_block(&task);
        let notebook_rendered = notebook_block(&notebook, &["src/main.rs".to_string()], 16);
        let stable = [
            global.as_str(),
            phase.as_str(),
            task_rendered.as_str(),
            notebook_rendered.as_str(),
        ]
        .join("\n\n");

        let first = PromptParts {
            global_prefix: global.clone(),
            phase_prefix: phase.clone(),
            task: task_block(&task),
            notebook: notebook_block(&notebook, &["src/main.rs".to_string()], 16),
            evidence: "EPHEMERAL EVIDENCE\nfirst".to_string(),
            question: "QUESTION\nfirst".to_string(),
        }
        .render();
        let second = PromptParts {
            global_prefix: global,
            phase_prefix: phase,
            task: task_block(&task),
            notebook: notebook_block(&notebook, &["src/main.rs".to_string()], 16),
            evidence: "EPHEMERAL EVIDENCE\nsecond".to_string(),
            question: "QUESTION\nsecond".to_string(),
        }
        .render();

        assert!(first.starts_with(&stable));
        assert!(second.starts_with(&stable));
        assert!(common_prefix_len(&first, &second) >= stable.len());
    }

    #[test]
    fn evidence_block_adds_line_numbers() {
        let slice = EvidenceSlice {
            lease: crate::notebook::EvidenceLease::new(
                "src/GitPanel.tsx".to_string(),
                None,
                crate::notebook::LineRange {
                    start: 106,
                    end: 108,
                },
                "first\nsecond\nthird",
            ),
            content: "first\nsecond\nthird".to_string(),
        };

        let rendered = evidence_block(&[slice]);

        assert!(rendered.contains("LINES 106-108"));
        assert!(rendered.contains("  106: first"));
        assert!(rendered.contains("  108: third"));
    }

    fn common_prefix_len(left: &str, right: &str) -> usize {
        left.bytes()
            .zip(right.bytes())
            .take_while(|(a, b)| a == b)
            .count()
    }
}
