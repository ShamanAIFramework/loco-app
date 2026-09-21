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
        "SYSTEM\nYou are Loco's local coding reasoning coprocessor. Deterministic Rust host code controls workflow, tools, search, file reads, budgets, leases, source display, and stopping. You are in a separated harness phase, not a ReAct loop. You never edit files, never call tools, never claim to have run tools, and never invent code facts. Treat the repo map as navigation only; leased evidence and active notebook entries are the source of truth. Return only JSON matching the requested schema.\n\nREPO MAP SNAPSHOT\n{}",
        repo
    )
}

pub fn phase_prefix(phase: Phase, schema_name: &str) -> String {
    format!(
        "PHASE {:?}\nComplete only this phase. Return one JSON object for schema `{}` only. Do not call tools, plan future tool calls, edit files, or include markdown, prose, comments, or code fences. Use exact enum strings from the schema.",
        phase, schema_name
    )
}

pub fn task_block(task: &TaskRequest) -> String {
    let conversation = recent_conversation_block(task);
    format!(
        "TASK\nWorkspace: {}\n{}\nCurrent user request: {}",
        task.workspace_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        conversation,
        task.question
    )
}

fn recent_conversation_block(task: &TaskRequest) -> String {
    let mut turns = task
        .conversation
        .iter()
        .filter(|turn| turn.role != "system")
        .collect::<Vec<_>>();

    if matches!(turns.last(), Some(last) if last.role == "user" && last.content.trim() == task.question.trim())
    {
        turns.pop();
    }

    let mut rendered = Vec::new();
    let mut used_chars = 0usize;
    for turn in turns
        .into_iter()
        .rev()
        .take(6)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
    {
        let content = compact_text(&turn.content, 320);
        let line = format!("{}: {}", turn.role, content);
        used_chars += line.len();
        if used_chars > 1_500 {
            break;
        }
        rendered.push(line);
    }

    if rendered.is_empty() {
        "Recent conversation: none".to_string()
    } else {
        format!("Recent conversation:\n{}", rendered.join("\n"))
    }
}

fn compact_text(value: &str, max_chars: usize) -> String {
    let compacted = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if compacted.chars().count() <= max_chars {
        compacted
    } else {
        format!(
            "{}...[truncated]",
            compacted.chars().take(max_chars).collect::<String>()
        )
    }
}

pub fn intake_instruction() -> String {
    [
        "QUESTION",
        "Classify the local coding request and propose deterministic code-search queries for the host to run.",
        "Use `Edit` for requests to add, update, fix, remove, refactor, style, or otherwise change code. Use `Locate` for where/which-file questions, `Explain` for how/what/why/update-analysis questions, and `Unclear` only when the request has no actionable code target.",
        "Prefer to proceed on short local-code questions if they mention UI, git, components, files, functions, panels, workspace, Tauri, Rust, TypeScript, or concrete change intent.",
        "Produce 3-6 short search queries likely to match paths, symbols, UI labels, component names, or domain nouns. Include exact identifiers and useful variants; omit filler words.",
        "Do not answer the user here.",
    ]
    .join("\n")
}

pub fn plan_instruction() -> String {
    [
        "QUESTION",
        "Draft a safe implementation plan for this local coding change. Do not propose an edit artifact yet and do not claim files were changed.",
        "Use only active notebook entries and leased evidence. If evidence is incomplete, make the plan narrow and name the uncertainty in risks.",
        "Affected files should include only files that current evidence supports as likely touch points.",
        "Steps should be concrete implementation actions the host could later turn into edit artifacts.",
        "Validation commands should be commands the user or host can run in this workspace after edits, such as package scripts, cargo checks, or focused tests inferred from evidence.",
    ]
    .join("\n")
}

pub fn extract_instruction() -> String {
    [
        "QUESTION",
        "Extract only atomic claims supported by leased evidence. Do not propose edits.",
        "Every inference must cite exactly one provided lease_id. Copy lease ids exactly.",
        "Each statement should preserve the most specific file path, line range, symbol/block name, and user-facing UI/code purpose available.",
        "For UI location questions, identify the exact JSX/render block, list/map/conditional, or component section when present.",
        "For implementation questions, distinguish definitions, callers, state, IPC commands, and rendering code.",
        "Set sufficient=true when the evidence directly answers the request. Use missing only for concrete additional evidence the host should search for; do not ask the user for missing code evidence.",
    ]
    .join("\n")
}

pub fn synthesize_instruction() -> String {
    [
        "QUESTION",
        "Write a concise final answer for a local coding assistant.",
        "For location questions, lead with the exact file and line range, then explain what that block does.",
        "Do not include a markdown Sources section, notebook ids, lease ids, hashes, or internal tracing details; the UI renders structured sources separately.",
        "Cite only active notebook entry ids that directly support claims in the answer. Prefer the narrowest relevant file range and avoid background citations.",
        "If evidence is weak or incomplete, say exactly what is uncertain and why.",
    ]
    .join("\n")
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
            "task_type": {
                "type": "string",
                "enum": ["Explain", "Locate", "Edit", "Unclear"],
                "description": "Edit for requested code changes, Locate for where/which-file questions, Explain for code behavior or change analysis, Unclear only when no actionable local-code target exists."
            },
            "target_hints": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Short identifiers, UI labels, filenames, symbols, and domain nouns relevant to the request."
            },
            "constraints": {
                "type": "array",
                "items": { "type": "string" },
                "description": "User constraints such as no edits, UI-only, backend-only, or exact behavior to inspect."
            },
            "search_queries": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Three to six concise queries likely to match indexed paths or symbols."
            }
        },
        "required": ["task_type", "target_hints", "constraints", "search_queries"],
        "additionalProperties": false
    })
}

pub fn schema_plan() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "objective": {
                "type": "string",
                "description": "One sentence describing the requested code change."
            },
            "affected_files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "reason": { "type": "string" }
                    },
                    "required": ["path", "reason"],
                    "additionalProperties": false
                },
                "description": "Files likely to need edits based on current evidence only."
            },
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": { "type": "string" },
                        "detail": { "type": "string" }
                    },
                    "required": ["title", "detail"],
                    "additionalProperties": false
                },
                "description": "Concrete implementation steps for a later edit-artifact phase."
            },
            "risks": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Uncertainties, likely regression risks, or missing evidence."
            },
            "validation_commands": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Commands to run after edits are eventually applied."
            }
        },
        "required": ["objective", "affected_files", "steps", "risks", "validation_commands"],
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
                        "statement": {
                            "type": "string",
                            "description": "One evidence-backed claim with exact file path, line range, symbol/block, and coding/UI purpose when available."
                        },
                        "lease_id": {
                            "type": "string",
                            "description": "The exact lease id from the evidence that supports this inference."
                        },
                        "tags": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Relevant paths, symbols, and user-facing concepts for notebook retrieval."
                        }
                    },
                    "required": ["statement", "lease_id", "tags"],
                    "additionalProperties": false
                }
            },
            "sufficient": {
                "type": "boolean",
                "description": "True when current evidence directly answers the user request."
            },
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
            "answer": {
                "type": "string",
                "description": "Concise user-facing answer. Do not include Sources, notebook ids, lease ids, or hashes."
            },
            "cited_entries": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Only notebook entry ids directly used by the answer; prefer narrow source ranges."
            },
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
    fn local_coding_instructions_are_explicit() {
        let global = global_prefix(&["src/GitPanel.tsx".to_string()]);
        let intake = intake_instruction();
        let plan = plan_instruction();
        let extract = extract_instruction();
        let synthesize = synthesize_instruction();

        assert!(global.contains("never edit files"));
        assert!(global.contains("repo map as navigation only"));
        assert!(intake.contains("Prefer to proceed on short local-code questions"));
        assert!(intake.contains("Use `Edit`"));
        assert!(intake.contains("UI"));
        assert!(plan.contains("safe implementation plan"));
        assert!(plan.contains("Do not propose an edit artifact yet"));
        assert!(extract.contains("exact JSX/render block"));
        assert!(extract.contains("Copy lease ids exactly"));
        assert!(synthesize.contains("Do not include a markdown Sources section"));
        assert!(synthesize.contains("structured sources separately"));
    }

    #[test]
    fn schemas_describe_local_code_outputs() {
        let intake = schema_intake();
        let plan = schema_plan();
        let extract = schema_extract();
        let synthesize = schema_synthesize();

        assert!(intake["properties"]["task_type"]["description"]
            .as_str()
            .unwrap()
            .contains("requested code changes"));
        assert!(plan["properties"]["affected_files"]["description"]
            .as_str()
            .unwrap()
            .contains("current evidence"));
        assert!(
            extract["properties"]["inferences"]["items"]["properties"]["statement"]["description"]
                .as_str()
                .unwrap()
                .contains("line range")
        );
        assert!(synthesize["properties"]["answer"]["description"]
            .as_str()
            .unwrap()
            .contains("Do not include Sources"));
    }

    #[test]
    fn task_block_includes_recent_conversation_without_repeating_current_request() {
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "Where is the git panel rendered?".to_string(),
            conversation: vec![
                crate::types::ConversationTurn {
                    role: "user".to_string(),
                    content: "We were looking at source control UI.".to_string(),
                },
                crate::types::ConversationTurn {
                    role: "assistant".to_string(),
                    content: "The source control UI is mounted in App.tsx.".to_string(),
                },
                crate::types::ConversationTurn {
                    role: "user".to_string(),
                    content: "Where is the git panel rendered?".to_string(),
                },
            ],
            budgets: crate::types::Budgets::default(),
        };

        let rendered = task_block(&task);

        assert!(rendered.contains("Recent conversation:"));
        assert!(rendered.contains("source control UI"));
        assert_eq!(
            rendered.matches("Where is the git panel rendered?").count(),
            1
        );
        assert!(rendered.contains("Current user request: Where is the git panel rendered?"));
    }

    #[test]
    fn recent_conversation_is_compacted_for_local_context() {
        let long = "word ".repeat(500);
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "Summarize the prior topic.".to_string(),
            conversation: vec![crate::types::ConversationTurn {
                role: "assistant".to_string(),
                content: long,
            }],
            budgets: crate::types::Budgets::default(),
        };

        let rendered = task_block(&task);

        assert!(rendered.contains("[truncated]"));
        assert!(rendered.len() < 2_000);
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
