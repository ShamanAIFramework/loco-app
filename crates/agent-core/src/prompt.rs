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
        let parts = [
            self.global_prefix.as_str(),
            self.phase_prefix.as_str(),
            self.task.as_str(),
            self.notebook.as_str(),
            self.evidence.as_str(),
            self.question.as_str(),
        ];
        parts
            .iter()
            .copied()
            .filter(|p| !p.trim().is_empty())
            .collect::<Vec<_>>()
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
        "SYSTEM\nYou are Loco's local coding reasoning coprocessor. Deterministic host code controls workflow, tools, search, file reads, budgets, leases, source display, and stopping. You are in a separated harness phase, not a ReAct loop. You never edit files, never call tools, never claim to have run tools, and never invent code facts. Treat the repo map as navigation only; leased evidence and active notebook entries are the source of truth. Return only JSON matching the requested schema.\n\nREPO MAP SNAPSHOT\n{}",
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
        "TASK\nWorkspace: {}\n{}\nCurrent user request: {}\n(Classify ONLY this current request; do NOT carry over greetings or tasks from prior conversation history)",
        task.workspace_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "unknown".to_string()),
        conversation,
        task.question.trim()
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
        format!(
            "Recent conversation:\n(PRIOR HISTORY ONLY - DO NOT extract intents or greetings from this history)\n{}",
            rendered.join("\n")
        )
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
        "Analyze the user's request and decompose it into one or more distinct intents in `intents`.",
        "",
        "CRITICAL RULES:",
        "1. MULTI-INTENT DECOMPOSITION: If the user request contains multiple questions, greetings, or tasks (often joined by 'and', 'also', 'then', or multiple '?'), you MUST produce a separate item in `intents` for EACH distinct part. Never collapse multiple questions into one item.",
        "2. NO GHOST GREETINGS: Classify ONLY the current user request. Do NOT include a `Chat` / greeting intent unless the current request itself explicitly contains a greeting (such as 'hi', 'hello', 'hey', 'good morning', etc.). If the current request begins directly with a question (e.g., 'where...', 'how...', 'what...'), there is NO greeting intent. Never copy greetings from prior turns in the conversation history.",
        "3. NO CODEBASE GUESSING IN DIRECT RESPONSE: `direct_response` is ONLY for general concepts or greetings. You must NEVER answer or guess where code is in this repository in `direct_response`.",
        "4. REPOSITORY QUESTIONS: Any question asking about this codebase (e.g., 'where is', 'where do we', 'where does our', 'how do we', 'how does our', 'in our codebase', 'in this project', finding files, or editing code) MUST be classified as `WorkspaceSearch` or `WorkspaceEdit` with `needs_workspace_search: true` and 1-4 targeted `search_queries`.",
        "5. INTENT CATEGORIES:",
        "   - `Chat`: Greetings, acknowledgments, or pleasantries (ONLY if present in the CURRENT request).",
        "   - `GeneralQuestion`: Programming concepts, language syntax, algorithms, or tools that do NOT refer to this local repository.",
        "   - `WorkspaceSearch`: Questions asking to locate, inspect, explain, or trace code in this local repository.",
        "   - `WorkspaceEdit`: Requests to modify, implement, refactor, style, or fix code in this local repository.",
        "   - `Ambiguous`: Requests that are too vague to act on without clarification.",
        "",
        "6. SEARCH QUERY GENERATION & ARCHITECTURAL HYPOTHESES:",
        "   - For `search_queries`, formulate 2-4 concise, targeted search terms (symbols, paths, or concepts).",
        "   - Formulate both literal domain terms AND architectural hypotheses derived from the REPOSITORY CONTEXT / REPO MAP.",
        "   - When the user asks about an architectural or cross-cutting concern (e.g. routing/views, lifecycle, configuration, auth, persistence) and no dedicated module is evident in the repo map, hypothesize candidate entrypoints, shells, root modules, or dispatchers visible in the repo outline.",
        "   - Do not guess full file paths; use short identifier or concept queries.",
        "",
        "7. SEPARATING CODE LOCATION FROM BEST PRACTICES / CONVENTIONS:",
        "   - When the user asks where code is located AND asks whether it follows best practices, conventions, or design patterns (e.g. 'where is X, and is it aligned with best practices?'), you MUST produce TWO distinct intents:",
        "     1. `WorkspaceSearch`: To locate the relevant code. `search_queries` must contain ONLY terms targeting code files (e.g. entrypoint, view, component, shell). NEVER include 'best practices', 'conventions', 'guidelines', or framework evaluation in `search_queries`.",
        "     2. `GeneralQuestion`: To explain the best practice or evaluate the pattern. `needs_workspace_search: false`. Provide a direct answer in `direct_response`.",
        "",
        "EXAMPLES:",
        "",
        "Example 1 (No greeting - compound question):",
        "User: \"Where do we validate authentication tokens in the API, and what is the difference between symmetric and asymmetric JWT signing?\"",
        "Output:",
        "{\"reasoning\":\"The user asks to locate API authentication token validation in the local codebase, and asks a general conceptual question about JWT signing.\",\"needs_workspace_search\":true,\"intents\":[{\"intent\":\"WorkspaceSearch\",\"summary\":\"Locate authentication token validation in API\",\"needs_workspace_search\":true,\"direct_response\":null,\"search_queries\":[\"auth\", \"token\", \"jwt\", \"validate\"]},{\"intent\":\"GeneralQuestion\",\"summary\":\"Explain symmetric vs asymmetric JWT signing\",\"needs_workspace_search\":false,\"direct_response\":\"Symmetric signing uses a single shared secret key...\",\"search_queries\":[]}],\"target_hints\":[\"auth\",\"token\"],\"constraints\":[]}",
        "",
        "Example 2 (With greeting - compound question):",
        "User: \"Hello! Where do we define database connection pools, and can you also explain connection pool sizing?\"",
        "Output:",
        "{\"reasoning\":\"The user greets, asks to find database connection pool configuration in the codebase, and asks a general conceptual question about connection pool sizing.\",\"needs_workspace_search\":true,\"intents\":[{\"intent\":\"Chat\",\"summary\":\"Greeting\",\"needs_workspace_search\":false,\"direct_response\":\"Hello! Happy to help.\",\"search_queries\":[]},{\"intent\":\"WorkspaceSearch\",\"summary\":\"Locate database connection pool configuration in codebase\",\"needs_workspace_search\":true,\"direct_response\":null,\"search_queries\":[\"connection\", \"pool\", \"database\", \"datasource\"]},{\"intent\":\"GeneralQuestion\",\"summary\":\"Explain connection pool sizing\",\"needs_workspace_search\":false,\"direct_response\":\"Connection pool sizing balances concurrency against resource limits...\",\"search_queries\":[]}],\"target_hints\":[\"connection\",\"pool\"],\"constraints\":[]}",
        "",
        "Example 3 (Code location + Best practice evaluation):",
        "User: \"Where does our UI configure the route settings, and is it aligned with modern architecture best practices?\"",
        "Output:",
        "{\"reasoning\":\"The user asks to locate UI route/view configuration in the local codebase, and asks a general question about modern architecture best practices.\",\"needs_workspace_search\":true,\"intents\":[{\"intent\":\"WorkspaceSearch\",\"summary\":\"Locate UI route and view configuration in codebase\",\"needs_workspace_search\":true,\"direct_response\":null,\"search_queries\":[\"route\", \"view\", \"navigation\", \"App\"]},{\"intent\":\"GeneralQuestion\",\"summary\":\"Explain modern UI routing and navigation best practices\",\"needs_workspace_search\":false,\"direct_response\":\"Modern UI architecture typically favors declarative route definitions for multi-page flows or state-driven view switching for desktop single-window apps...\",\"search_queries\":[]}],\"target_hints\":[\"route\",\"view\",\"App\"],\"constraints\":[]}",
    ]
    .join("\n")
}

pub fn plan_instruction() -> String {
    [
        "QUESTION",
        "Draft a safe implementation plan for this local coding change. Do not propose an edit artifact yet and do not claim files were changed.",
        "Use only active notebook entries and leased evidence. If evidence is incomplete, make the plan narrow and name the uncertainty in risks.",
        "For edit requests: propose concrete implementation actions, affected files, risks, and validation commands.",
        "For location, search, or explanatory requests: propose concrete verification and synthesis steps, key files to cite, potential ambiguities or risks in the explanation, and validation checks.",
        "Affected files should include only files that current evidence supports as likely touch points or primary citations.",
        "For user-facing, startup, loading, layout, styling, interaction, or display changes, prefer evidence-backed entrypoints, screens/views, components/widgets, templates, styles, routing/navigation, app shell, and app/window/display configuration. Do not involve unrelated orchestration, search, service, build, or model/AI plumbing files unless cited evidence shows they are directly required.",
        "Validation commands should be commands the user or host can run in this workspace. Infer them from evidence such as project manifests, scripts, build files, test files, or existing repository conventions; do not assume a language or framework.",
    ]
    .join("\n")
}

pub fn plan_critic_instruction(plan_json: &str) -> String {
    [
        "QUESTION",
        "EVALUATION PERSPECTIVE: ADVERSARIAL PLAN CRITIC (ANTITHESIS)",
        "Role: You are an adversarial code and plan auditor challenging the proposed plan.",
        "Mission: Scrutinize every detail of the proposed plan. Challenge missing dependencies, uninspected files, unverified assumptions, and regression risks.",
        "",
        "Proposed Plan to Evaluate:",
        "```json",
        plan_json.trim(),
        "```",
        "",
        "Audit Guidelines:",
        "1. Missing Aspects: Identify any critical dependencies, caller/callee context, edge cases, configuration, or structural patterns that the plan failed to account for.",
        "2. Disputed Steps: Challenge any step that assumes too much, makes unverified changes, lacks leased evidence backing, or is logically out of sequence.",
        "3. Unverified Assumptions: Highlight any unverified hypothesis or dependency (e.g. 'assumes service X exists or function Y handles Z without evidence').",
        "4. Additional Risks: List concrete regression hazards, breaking changes, or gaps in validation.",
        "",
        "Return one JSON object matching schema `PlanCriticOutput`.",
    ]
    .join("\n")
}

pub fn plan_rebuttal_instruction(plan_json: &str, critic_json: &str) -> String {
    [
        "QUESTION",
        "EVALUATION PERSPECTIVE: PRAGMATIC SCOPE ARBITER / REBUTTAL (CRITIC OF THE CRITIC)",
        "Role: You are an appellate defender and scope auditor evaluating the critic's objections.",
        "Mission: Distinguish between genuine blocking hazards and pedantic analysis paralysis / overblown hedges.",
        "",
        "Proposed Plan:",
        "```json",
        plan_json.trim(),
        "```",
        "",
        "Adversarial Critique:",
        "```json",
        critic_json.trim(),
        "```",
        "",
        "Rebuttal Guidelines:",
        "1. Eliminate Analysis Paralysis: If the critic demands unnecessary context (e.g., 'we don't know enough about Y'), evaluate whether knowing Y is strictly required to fulfill the user request. If current leased evidence and notebook entries are sufficient, dismiss the critique with clear technical justification.",
        "2. Dismissed Critiques: Put overblown hedges, speculative unknowns, and out-of-scope concerns into `dismissed_critiques` with a precise justification explaining why the existing evidence is sufficient.",
        "3. Accepted Critiques: Concede valid risks, missed edge cases, or genuine regressions into `accepted_critiques` so they can be incorporated into the final plan.",
        "",
        "Return one JSON object matching schema `PlanRebuttalOutput`.",
    ]
    .join("\n")
}

pub fn plan_reconciler_instruction(plan_json: &str, critic_json: &str, rebuttal_json: &str) -> String {
    [
        "QUESTION",
        "EVALUATION PERSPECTIVE: CONVERGENCE RECONCILER (SYNTHESIS)",
        "Role: You are the final planning reconciler harmonizing the proposal, critique, and rebuttal.",
        "Mission: Produce a lean, verified, actionable final EditPlan.",
        "",
        "Initial Plan:",
        "```json",
        plan_json.trim(),
        "```",
        "",
        "Adversarial Critique:",
        "```json",
        critic_json.trim(),
        "```",
        "",
        "Pragmatic Rebuttal:",
        "```json",
        rebuttal_json.trim(),
        "```",
        "",
        "Reconciliation Guidelines:",
        "1. Adopt all `accepted_critiques` to harden the plan against real bugs, edge cases, and regressions.",
        "2. Exclude all `dismissed_critiques` to avoid analysis paralysis, unnecessary rabbit holes, or out-of-scope files.",
        "3. Ensure all affected files are supported by active leased evidence.",
        "4. Output the final consolidated plan.",
        "",
        "Return one JSON object matching schema `EditPlan`.",
    ]
    .join("\n")
}

pub fn patch_proposer_instruction(
    target_path: &str,
    target_symbol: Option<&str>,
    start_line: usize,
    end_line: usize,
    objective: &str,
) -> String {
    let symbol_info = match target_symbol {
        Some(sym) => format!(" (symbol `{}`)", sym),
        None => String::new(),
    };
    [
        "QUESTION",
        "EVALUATION PERSPECTIVE: PATCH PROPOSER (THESIS)",
        "Role: You are an expert software engineer drafting a precise code replacement.",
        &format!("Objective: {}", objective),
        &format!("Target: `{}` lines {}-{}{}", target_path, start_line, end_line, symbol_info),
        "",
        "Requirements for Replacement Code:",
        "1. Complete & Valid: Provide the exact, drop-in replacement code for the targeted block. It must be syntactically complete and valid.",
        "2. No Placeholders: Do not write lazy placeholders or comments like '// ... rest of code unchanged' or '/* keep existing */'.",
        "3. Match Idioms: Strictly match existing repository style, indentation, type conventions, and patterns shown in the leased evidence and sibling exemplars.",
        "4. Minimal Blast Radius: Change only what is strictly required to fulfill the plan objective.",
        "",
        "Return one JSON object matching schema `PatchProposal`.",
    ]
    .join("\n")
}

pub fn patch_critic_instruction(proposed_patch_json: &str) -> String {
    [
        "QUESTION",
        "EVALUATION PERSPECTIVE: ADVERSARIAL PATCH CRITIC (ANTITHESIS)",
        "Role: You are an uncompromising senior code reviewer auditing the proposed patch for defects, regressions, and style mismatches.",
        "",
        "Proposed Patch:",
        "```json",
        proposed_patch_json.trim(),
        "```",
        "",
        "Audit Criteria:",
        "1. Interface Regressions: Does the patch break callers, change public function/method signatures, or alter expected return types?",
        "2. Unhandled Edge Cases: Does the code omit null/empty checks, error path handling, or unexpected state transitions?",
        "3. Convention Violations: Does the code deviate from the idioms and styling shown in the sibling exemplar or active knowledge rubrics?",
        "4. Fragility & Leaks: Does it introduce unhandled unwraps, resource leaks, or missing teardown logic?",
        "",
        "If the patch is fully sound, safe, and ready to apply, set approved=true.",
        "Return one JSON object matching schema `PatchCriticOutput`.",
    ]
    .join("\n")
}

pub fn patch_rebuttal_instruction(proposed_patch_json: &str, critic_json: &str) -> String {
    [
        "QUESTION",
        "EVALUATION PERSPECTIVE: PRAGMATIC SCOPE ARBITER / PATCH REBUTTAL (DEFENSE)",
        "Role: You are an appellate defender and scope auditor evaluating the critic's objections.",
        "Mission: Distinguish between genuine blocking bugs and pedantic nitpicks or speculative hedges.",
        "",
        "Proposed Patch:",
        "```json",
        proposed_patch_json.trim(),
        "```",
        "",
        "Adversarial Critique:",
        "```json",
        critic_json.trim(),
        "```",
        "",
        "Rebuttal Guidelines:",
        "1. Protect Functional Progress: If the critic demands out-of-scope refactoring, speculative abstractions, or raises concerns contradicted by active evidence, dismiss the critique with clear technical justification in `dismissed_points`.",
        "2. Concede Real Flaws: Accept genuine bugs, syntax hazards, or broken interfaces into `accepted_points` so the reconciler can surgically fix them.",
        "",
        "Return one JSON object matching schema `PatchRebuttalOutput`.",
    ]
    .join("\n")
}

pub fn patch_reconciler_instruction(
    proposed_patch_json: &str,
    critic_json: &str,
    rebuttal_json: &str,
) -> String {
    [
        "QUESTION",
        "EVALUATION PERSPECTIVE: CONVERGENCE RECONCILER (SYNTHESIS)",
        "Role: You are the final patch reconciler harmonizing the proposal, critique, and rebuttal.",
        "Mission: Produce the finalized, hardened replacement code.",
        "",
        "Initial Patch:",
        "```json",
        proposed_patch_json.trim(),
        "```",
        "",
        "Adversarial Critique:",
        "```json",
        critic_json.trim(),
        "```",
        "",
        "Pragmatic Rebuttal:",
        "```json",
        rebuttal_json.trim(),
        "```",
        "",
        "Reconciliation Guidelines:",
        "1. Incorporate all `accepted_points` to fix genuine bugs, edge cases, and interface mismatches.",
        "2. Exclude `dismissed_points` to avoid over-engineering or unnecessary scope creep.",
        "3. Output the finalized replacement code, ready to drop directly into the target file.",
        "",
        "Return one JSON object matching schema `PatchReconcilerOutput`.",
    ]
    .join("\n")
}

pub fn extract_instruction() -> String {
    [
        "QUESTION",
        "Extract 2 to 5 high-value, non-redundant atomic claims directly addressing the user request from leased evidence. Do not propose edits.",
        "Quality over quantity: Provide only the most informative claims that directly inform or answer the user request. Never repeat similar or overlapping statements.",
        "Every inference must cite exactly one provided lease_id. Copy lease ids exactly.",
        "Each statement should preserve the most specific file path, line range, symbol/block name, and user-facing UI/code purpose available.",
        "For user-facing location questions, identify the exact render/output block, screen/view section, list/map/loop/conditional, template, component/widget, or display code when present.",
        "Set sufficient=true when the evidence directly answers the request. Use missing only for concrete additional evidence the host should search for; do not ask the user for missing code evidence.",
    ]
    .join("\n")
}

pub fn synthesize_instruction() -> String {
    [
        "QUESTION",
        "Write a direct, accurate, and concise final answer for a local coding assistant addressing the user's request.",
        "Address all parts and intents of the request:",
        "- For code location questions, lead with the exact file and line range, then explain what that block does based on active evidence and notebook inferences.",
        "- For questions asking whether code aligns with best practices or conventions, evaluate the observed codebase patterns authoritatively against standard software engineering practices.",
        "- Do not defensively hedge (e.g. do not say 'without further context it is uncertain' or 'requires more detailed review') when evaluating visible architecture against standard design patterns.",
        "- If the request mentions a specific library or pattern that differs from the observed code (such as a multi-page routing framework when the app uses state-driven view switching), clearly explain what the codebase actually does and compare the trade-offs.",
        "- Ground code-level facts in active notebook entry ids. Prefer the narrowest relevant file range and avoid background citations.",
        "- If code evidence was genuinely not found in the repository, state that clearly instead of guessing.",
        "Do not include a markdown Sources section, notebook ids, lease ids, hashes, or internal tracing details; the UI renders structured sources separately.",
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
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of how the user's prompt was broken down into intents."
            },
            "needs_workspace_search": {
                "type": "boolean",
                "description": "True if any intent item requires searching or reading files in the local workspace repository; false if all intents can be satisfied directly."
            },
            "intents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": ["Chat", "GeneralQuestion", "WorkspaceSearch", "WorkspaceEdit", "Ambiguous"],
                            "description": "Intent classification for this part of the user request."
                        },
                        "summary": {
                            "type": "string",
                            "description": "Short summary of what this specific intent requires."
                        },
                        "needs_workspace_search": {
                            "type": "boolean",
                            "description": "True if this specific intent requires searching or reading files in the local workspace."
                        },
                        "direct_response": {
                            "type": ["string", "null"],
                            "description": "Direct answer or reply if this intent does not need workspace search (greeting, general explanation, or clarification question). Null if workspace search is required."
                        },
                        "search_queries": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "1 to 4 targeted search queries (exact symbols, paths, function names) for this specific intent. Empty array if needs_workspace_search is false."
                        }
                    },
                    "required": ["intent", "summary", "needs_workspace_search", "search_queries"],
                    "additionalProperties": false
                },
                "description": "Decomposed list of distinct intents identified in the user prompt (1 or more)."
            },
            "target_hints": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Key identifiers, filenames, component names, or symbols identified across the request."
            },
            "constraints": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Any user-specified constraints mentioned in the request."
            }
        },
        "required": ["reasoning", "needs_workspace_search", "intents"],
        "additionalProperties": false
    })
}

pub fn intake_critic_instruction(candidate_draft_json: &str) -> String {
    [
        "QUESTION",
        "You are an Adversarial Intent Auditor (Node 2 of the Triad Consensus).",
        "Your mission is to audit the candidate intent decomposition against the user's verbatim request and identify any failures.",
        "",
        "CANDIDATE DRAFT TO AUDIT:",
        candidate_draft_json,
        "",
        "AUDIT CHECKLIST:",
        "1. CLAUSE COVERAGE: Did the candidate draft drop any sub-question, condition, greeting, or clause? (Check for conjunctions like 'and', 'also', 'then', or multiple '?'). If the user asked both a codebase question AND a conceptual/best-practice question, they MUST be separated.",
        "2. QUERY CONTAMINATION: Do any `search_queries` contain conversational words, non-code tokens, or meta-concepts (e.g. 'best practices', 'conventions', 'guidelines', 'how to', 'is it')? Search queries must ONLY be terms targeting code files/symbols.",
        "3. INTENT BOUNDARIES: Is any general conceptual question improperly marked with `needs_workspace_search: true`?",
        "",
        "VERDICT:",
        "- If the candidate draft is 100% complete, fully covers all clauses, and has clean code search queries, set `approved: true` with `has_dropped_clauses: false` and `has_contaminated_queries: false`.",
        "- If ANY clause was missed, collapsed, or queries contaminated, set `approved: false` and supply the complete, corrected decomposition in `corrected_intents`.",
    ]
    .join("\n")
}

pub fn schema_intake_critic() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "approved": {
                "type": "boolean",
                "description": "True if the draft decomposition is 100% complete, captures all clauses, and has clean code-only search queries. False if any clause was dropped, contaminated, or misclassified."
            },
            "critique": {
                "type": "string",
                "description": "Adversarial critique explaining what was missed, dropped, contaminated with non-code words, or misclassified."
            },
            "has_dropped_clauses": {
                "type": "boolean",
                "description": "True if any sub-question, condition, or clause from the user request was dropped or collapsed."
            },
            "has_contaminated_queries": {
                "type": "boolean",
                "description": "True if search queries contain non-code meta words like 'best practices', 'conventions', 'guidelines', or 'is it'."
            },
            "corrected_intents": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "intent": {
                            "type": "string",
                            "enum": ["Chat", "GeneralQuestion", "WorkspaceSearch", "WorkspaceEdit", "Ambiguous"],
                            "description": "Intent classification for this part of the user request."
                        },
                        "summary": {
                            "type": "string",
                            "description": "Short summary of what this specific intent requires."
                        },
                        "needs_workspace_search": {
                            "type": "boolean",
                            "description": "True if this specific intent requires searching or reading files in the local workspace."
                        },
                        "direct_response": {
                            "type": ["string", "null"],
                            "description": "Direct answer or reply if this intent does not need workspace search. Null if workspace search is required."
                        },
                        "search_queries": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "1 to 4 clean search queries targeting code files. Empty if needs_workspace_search is false."
                        }
                    },
                    "required": ["intent", "summary", "needs_workspace_search", "search_queries"],
                    "additionalProperties": false
                },
                "description": "The complete, corrected list of intents covering the entire user prompt without omissions or query contamination."
            }
        },
        "required": ["approved", "critique", "has_dropped_clauses", "has_contaminated_queries", "corrected_intents"],
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

pub fn schema_plan_critic() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "critique": {
                "type": "string",
                "description": "Thorough critique challenging the proposed plan, dependencies, and assumptions."
            },
            "missing_aspects": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Critical dependencies, files, edge cases, or contexts overlooked by the plan."
            },
            "disputed_steps": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Proposed steps that are premature, risky, unsupported by evidence, or flawed."
            },
            "unverified_assumptions": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Assumptions in the plan that lack verification in leased evidence."
            },
            "additional_risks": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Potential regression or architectural failure risks."
            }
        },
        "required": ["critique", "missing_aspects", "disputed_steps", "unverified_assumptions", "additional_risks"],
        "additionalProperties": false
    })
}

pub fn schema_plan_rebuttal() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "rebuttal": {
                "type": "string",
                "description": "Evaluation of the critique, defending viable paths and explaining why certain objections are unnecessary hedges."
            },
            "dismissed_critiques": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "critique": { "type": "string" },
                        "justification": { "type": "string" }
                    },
                    "required": ["critique", "justification"],
                    "additionalProperties": false
                },
                "description": "Critiques dismissed as overblown hedges, pedantic blockers, or out-of-scope concerns."
            },
            "accepted_critiques": {
                "type": "array",
                "items": { "type": "string" },
                "description": "Valid critiques that must be addressed in the reconciled plan."
            }
        },
        "required": ["rebuttal", "dismissed_critiques", "accepted_critiques"],
        "additionalProperties": false
    })
}

pub fn schema_patch_proposer() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "replacement_code": {
                "type": "string",
                "description": "Complete, valid replacement code for the targeted block. Do not include placeholders or omission comments."
            },
            "justification": {
                "type": "string",
                "description": "Technical justification of how the replacement fulfills the objective."
            }
        },
        "required": ["replacement_code", "justification"],
        "additionalProperties": false
    })
}

pub fn schema_patch_critic() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "approved": { "type": "boolean", "description": "True if the patch is safe, clean, and ready to apply." },
            "critique": { "type": "string", "description": "Technical critique evaluating interface stability and edge cases." },
            "disputed_changes": { "type": "array", "items": { "type": "string" }, "description": "Specific code changes that introduce bugs or regressions." },
            "unhandled_edge_cases": { "type": "array", "items": { "type": "string" }, "description": "Edge cases or error conditions not handled." },
            "convention_violations": { "type": "array", "items": { "type": "string" }, "description": "Deviations from project styling or conventions." }
        },
        "required": ["approved", "critique", "disputed_changes", "unhandled_edge_cases", "convention_violations"],
        "additionalProperties": false
    })
}

pub fn schema_patch_rebuttal() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "rebuttal": { "type": "string", "description": "Appellate evaluation distinguishing real flaws from speculative nitpicks." },
            "accepted_points": { "type": "array", "items": { "type": "string" }, "description": "Genuine issues that must be addressed." },
            "dismissed_points": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "critique": { "type": "string" },
                        "justification": { "type": "string" }
                    },
                    "required": ["critique", "justification"],
                    "additionalProperties": false
                },
                "description": "Objections dismissed with technical justification."
            }
        },
        "required": ["rebuttal", "accepted_points", "dismissed_points"],
        "additionalProperties": false
    })
}

pub fn schema_patch_reconciler() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "hardened_code": { "type": "string", "description": "Finalized replacement code ready for in-memory patch application." },
            "reconciliation_summary": { "type": "string", "description": "Summary of adjustments made to resolve accepted critiques." },
            "remaining_uncertainties": { "type": "array", "items": { "type": "string" }, "description": "Any unresolved ambiguities." }
        },
        "required": ["hardened_code", "reconciliation_summary", "remaining_uncertainties"],
        "additionalProperties": false
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateGateItem {
    pub path: String,
    pub symbol: Option<String>,
    pub start_line: u32,
    pub end_line: u32,
    pub source: String,
    pub snippet: String,
}

pub fn schema_candidate_gate() -> serde_json::Value {
    json!({
        "type": "object",
        "properties": {
            "evaluations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "The exact relative path of the candidate file."
                        },
                        "keep": {
                            "type": "boolean",
                            "description": "True if the snippet directly implements, configures, or answers the intent; false if it is irrelevant, false positive, or secondary clutter."
                        },
                        "reason": {
                            "type": "string",
                            "description": "One concise sentence explaining why the candidate was retained or rejected."
                        }
                    },
                    "required": ["path", "keep", "reason"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["evaluations"],
        "additionalProperties": false
    })
}

pub fn prompt_candidate_gate(
    task_question: &str,
    intent_summary: &str,
    candidates: &[CandidateGateItem],
) -> String {
    prompt_candidate_gate_optimist(task_question, intent_summary, candidates)
}

pub fn prompt_candidate_gate_prefix(
    task_question: &str,
    intent_summary: &str,
    candidates: &[CandidateGateItem],
) -> String {
    let mut out = String::new();
    out.push_str("CANDIDATE CODE EVALUATION\n");
    out.push_str(&format!("User Request: {}\n", task_question.trim()));
    out.push_str(&format!("Target Intent: {}\n\n", intent_summary.trim()));
    out.push_str("Candidates to evaluate:\n");

    for (i, c) in candidates.iter().enumerate() {
        out.push_str(&format!(
            "--- Candidate {} ---\nPath: {}\nLines: {}-{}\nSymbol: {}\nSource: {}\n```\n{}\n```\n\n",
            i + 1,
            c.path,
            c.start_line,
            c.end_line,
            c.symbol.as_deref().unwrap_or("none"),
            c.source,
            c.snippet.trim()
        ));
    }
    out
}

pub fn prompt_candidate_gate_optimist(
    task_question: &str,
    intent_summary: &str,
    candidates: &[CandidateGateItem],
) -> String {
    let mut out = prompt_candidate_gate_prefix(task_question, intent_summary, candidates);
    out.push_str("EVALUATION PERSPECTIVE: OPTIMIST (ARCHITECTURAL SCOUT)\n");
    out.push_str("Role: You are an inclusion-focused architectural scout maximizing retrieval recall.\n");
    out.push_str("Mission: Identify every code candidate that provides valuable direct implementation, structural context, or architectural clarity for the user request.\n");
    out.push_str("Inclusion Guidelines:\n");
    out.push_str("1. Entrypoint & Root Mount Context: Keep application entrypoints and mount files that bootstrap the primary UI hierarchy. Even if an entrypoint only renders a root component or sets up top-level providers, it provides vital evidence regarding framework selection, top-level providers, and top-level architecture.\n");
    out.push_str("2. Root Shells & View Switching: Keep main application shells and screen components. Modern UI applications frequently implement navigation and routing via conditional view switching, screen state signals, or panel containers rather than dedicated routing packages. These view-switching components are primary routing configurations.\n");
    out.push_str("3. Structural Context: Keep candidates that define state signals, lifecycle hooks, or data contracts that govern the requested feature.\n");
    out.push_str("4. Alternative & Negative Evidence: When a request asks where an architectural capability (such as routing, navigation, or views) is configured, and the codebase handles it via conditional view switching or state signals rather than a standalone router library, the root component shell demonstrating this alternative IS primary evidence and must be kept.\n");
    out.push_str("If there is a valid architectural justification to keep a candidate for downstream synthesis, advocate for it with keep: true.\n\n");
    out.push_str("Return a JSON object matching schema `CandidateGateOutput` with an evaluation for each candidate.\n");
    out
}

pub fn prompt_candidate_gate_skeptic(
    task_question: &str,
    intent_summary: &str,
    candidates: &[CandidateGateItem],
) -> String {
    let mut out = prompt_candidate_gate_prefix(task_question, intent_summary, candidates);
    out.push_str("EVALUATION PERSPECTIVE: SKEPTIC (PRECISION CRITIC)\n");
    out.push_str("Role: You are a precision-focused code critic eliminating false positives and noise.\n");
    out.push_str("Mission: Scrutinize every code candidate. Challenge whether each slice directly answers the target intent or is merely an unneeded distraction.\n");
    out.push_str("Pruning Guidelines:\n");
    out.push_str("1. Coincidental Matches: Drop candidates that only match keywords coincidentally without implementing or configuring the requested capability.\n");
    out.push_str("2. Generic Utilities: Drop isolated utility helpers (e.g. DOM manipulation, syntax highlighting, text formatting, or generic handlers) that do not configure core views or logic.\n");
    out.push_str("3. Stylesheets & Pure Presentation: Drop CSS/styling files unless the user explicitly requested styling rules.\n");
    out.push_str("4. False Positives: If a candidate does not directly implement or define the target feature, reject it with keep: false. Note: Do not reject main application component shells that house the view rendering, screen state, or layout for the application.\n\n");
    out.push_str("Return a JSON object matching schema `CandidateGateOutput` with an evaluation for each candidate.\n");
    out
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
                            "description": "One evidence-backed claim with exact file path, line range, symbol/block, and software purpose when available."
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
                "description": "Concise user-facing answer. Ground code facts in leased evidence and architectural evaluation in standard conventions without defensive hedging. Do not include Sources, notebook ids, lease ids, or hashes."
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
            intake_only: false,
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
    fn stereoscopic_prompts_share_stable_candidate_prefix() {
        let items = vec![CandidateGateItem {
            path: "src/App.tsx".to_string(),
            symbol: Some("App".to_string()),
            start_line: 1,
            end_line: 50,
            source: "CodeGraph".to_string(),
            snippet: "export default function App() {}".to_string(),
        }];
        let opt = prompt_candidate_gate_optimist("Where is routing", "Locate routing", &items);
        let skep = prompt_candidate_gate_skeptic("Where is routing", "Locate routing", &items);
        let common = prompt_candidate_gate_prefix("Where is routing", "Locate routing", &items);

        assert!(opt.starts_with(&common));
        assert!(skep.starts_with(&common));
        assert!(common.contains("src/App.tsx"));
        assert!(common_prefix_len(&opt, &skep) >= common.len());
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
        assert!(intake.contains("decompose it into one or more distinct intents"));
        assert!(intake.contains("needs_workspace_search"));
        assert!(intake.contains("direct_response"));
        assert!(plan.contains("safe implementation plan"));
        assert!(plan.contains("Do not propose an edit artifact yet"));
        assert!(plan.contains("Do not involve unrelated orchestration"));
        assert!(plan.contains("unless cited evidence shows"));
        assert!(extract.contains("exact render/output block"));
        assert!(extract.contains("Copy lease ids exactly"));
        assert!(synthesize.contains("Do not include a markdown Sources section"));
        assert!(synthesize.contains("structured sources separately"));
        assert!(synthesize.contains("Do not defensively hedge"));
    }

    #[test]
    fn reusable_prompts_do_not_assume_a_specific_stack() {
        let rendered = [
            global_prefix(&[]),
            intake_instruction(),
            intake_critic_instruction(""),
            plan_instruction(),
            extract_instruction(),
            synthesize_instruction(),
            schema_intake().to_string(),
            schema_intake_critic().to_string(),
            schema_candidate_gate().to_string(),
            prompt_candidate_gate("", "", &[]),
            prompt_candidate_gate_optimist("", "", &[]),
            prompt_candidate_gate_skeptic("", "", &[]),
            schema_plan().to_string(),
            schema_extract().to_string(),
            schema_synthesize().to_string(),
            patch_proposer_instruction("", None, 1, 1, ""),
            patch_critic_instruction(""),
            patch_rebuttal_instruction("", ""),
            patch_reconciler_instruction("", "", ""),
            schema_patch_proposer().to_string(),
            schema_patch_critic().to_string(),
            schema_patch_rebuttal().to_string(),
            schema_patch_reconciler().to_string(),
        ]
        .join("\n");

        for forbidden in [
            "Tauri",
            "Solid",
            "React",
            "Vue",
            "Svelte",
            "Rust",
            "TypeScript",
            "JavaScript",
            "JSX",
            "TSX",
            "Cargo",
            "npm",
            "src/App",
            "tauri.conf",
            "orchestrator.rs",
        ] {
            assert!(
                !rendered.contains(forbidden),
                "reusable prompts should not contain stack-specific term `{forbidden}`"
            );
        }
    }

    #[test]
    fn schemas_describe_local_code_outputs() {
        let intake = schema_intake();
        let plan = schema_plan();
        let extract = schema_extract();
        let synthesize = schema_synthesize();

        assert!(intake["properties"]["intents"]["items"]["properties"]["intent"]["description"]
            .as_str()
            .unwrap()
            .contains("Intent classification"));
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
            intake_only: false,
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
            intake_only: false,
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
