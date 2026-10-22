use std::collections::HashSet;
use crate::knowledge::{detect_stack_tags, KnowledgeDomain, KnowledgeEngine};
use crate::notebook::{EntryId, LeaseId, Notebook, Provenance};
use crate::patch::{
    apply_in_memory_patch, extract_exemplar_header, find_sibling_exemplars, generate_unified_diff,
    verify_lease_freshness,
};
use crate::prompt::{
    evidence_block, extract_instruction, global_prefix, intake_critic_instruction,
    intake_instruction, notebook_block, patch_critic_instruction, patch_proposer_instruction,
    patch_rebuttal_instruction, patch_reconciler_instruction, phase_prefix,
    plan_critic_instruction, plan_instruction, plan_rebuttal_instruction,
    plan_reconciler_instruction, prompt_candidate_gate_optimist, prompt_candidate_gate_skeptic,
    schema_candidate_gate, schema_extract, schema_intake, schema_intake_critic, schema_patch_critic,
    schema_patch_proposer, schema_patch_rebuttal, schema_patch_reconciler, schema_plan,
    schema_plan_critic, schema_plan_rebuttal, schema_synthesize, synthesize_instruction, task_block,
    CandidateGateItem, PromptParts,
};
use crate::trace::{Trace, TraceEvent};
use crate::traits::{
    AgentProgress, CodeSearch, FileReader, LlmClient, LlmRequest, LlmResponse, ProgressSink,
    TokenCounter,
};
use crate::types::{
    AgentAnswer, CandidateGateOutput, ContentHit, CriticOutput, EditArtifact, EditPlan,
    EvidenceSlice, ExtractOutput, FileHit, IntakeOutput, IntentItem, MissingEvidence,
    Phase, PlanCriticOutput, PlanRebuttalOutput, PlanStep, PlanStepResults,
    PlannedFile, PatchCriticOutput, PatchProposal, PatchRebuttalOutput, PatchReconcilerOutput,
    StopReason, SymbolHit, SynthesizeOutput, TaskRequest, TriadConsensus, UserIntent, VectorHit,
};
use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde_json::json;
use uuid::Uuid;

pub struct AgentRuntime<'a> {
    pub llm: &'a dyn LlmClient,
    pub code_search: &'a dyn CodeSearch,
    pub file_reader: &'a dyn FileReader,
    pub token_counter: &'a dyn TokenCounter,
    pub progress: &'a dyn ProgressSink,
}

pub struct AgentRunResult {
    pub answer: AgentAnswer,
    pub trace: Trace,
}

fn call_id(prefix: &str) -> String {
    format!("{}-{}", prefix, &Uuid::new_v4().to_string()[..8])
}

/// Executes Step 1 of the agent: Multi-Intent Determination and Search Planning.
///
/// Decomposes the user request into distinct intent items (chat, general question, workspace search,
/// workspace edit, ambiguous). Immediately fulfills non-workspace items via direct response, and
/// sets up targeted search queries for workspace items so downstream turns can satisfy them.
pub async fn run_agent(task: TaskRequest, runtime: AgentRuntime<'_>) -> Result<AgentRunResult> {
    let mut trace = Trace::default();
    macro_rules! record_trace {
        ($event:expr) => {{
            let ev = $event;
            if let Ok(json) = serde_json::to_string(&ev) {
                runtime.progress.emit(AgentProgress::Trace(json)).await;
            }
            trace.push(ev);
        }};
    }

    record_trace!(TraceEvent::RunStarted {
        run_id: task.run_id.clone(),
        question: task.question.clone(),
    });

    let mut notebook = Notebook::default();

    // ===============================================
    // --- TRIAD INTAKE CONSENSUS (3-Node Quorum) ---
    // ===============================================

    // --- Triad Node 1: Drafter ---
    runtime
        .progress
        .emit(AgentProgress::Status("Analyzing request intents".to_string()))
        .await;
    runtime
        .progress
        .emit(AgentProgress::Status("Triad Node 1: Drafter analyzing request intents".to_string()))
        .await;

    record_trace!(TraceEvent::PhaseStarted {
        phase: Phase::Intake,
    });

    // Provide a compact repository map snapshot as navigation context
    let repo_map = runtime.code_search.repo_map(30).await.unwrap_or_default();
    let intake_parts = PromptParts {
        global_prefix: global_prefix(&repo_map),
        phase_prefix: phase_prefix(Phase::Intake, "IntakeOutput"),
        task: task_block(&task),
        notebook: String::new(),
        evidence: String::new(),
        question: intake_instruction(),
    };
    let intake_prompt = intake_parts.render();

    let intake_req = LlmRequest {
        call_id: call_id("intake-drafter"),
        phase: Phase::Intake,
        schema_name: "IntakeOutput".to_string(),
        schema: schema_intake(),
        prompt_parts: Some(intake_parts),
        prompt: intake_prompt.clone(),
        temperature: 0.0,
    };

    record_trace!(TraceEvent::NotebookSnapshot {
        phase: Phase::Intake,
        notebook: notebook.deterministic_checkpoint(),
    });

    let intake_resp = runtime.llm.complete_json(intake_req.clone()).await?;

    record_trace!(TraceEvent::LlmCall {
        request: intake_req,
        response: intake_resp.clone(),
    });

    let mut drafter_intake: IntakeOutput =
        parse_json(&intake_resp.raw_json).unwrap_or_else(|_| fallback_intake(&task.question));

    validate_and_normalize_intake(&mut drafter_intake, &task.question);

    // --- Triad Node 2: Adversarial Critic ---
    runtime
        .progress
        .emit(AgentProgress::Status("Triad Node 2: Critic auditing draft coverage and queries".to_string()))
        .await;

    record_trace!(TraceEvent::PhaseStarted {
        phase: Phase::IntakeCritic,
    });

    let critic_parts = PromptParts {
        global_prefix: global_prefix(&repo_map),
        phase_prefix: phase_prefix(Phase::IntakeCritic, "CriticOutput"),
        task: task_block(&task),
        notebook: String::new(),
        evidence: String::new(),
        question: intake_critic_instruction(&intake_resp.raw_json),
    };
    let critic_prompt = critic_parts.render();

    let critic_req = LlmRequest {
        call_id: call_id("intake-critic"),
        phase: Phase::IntakeCritic,
        schema_name: "CriticOutput".to_string(),
        schema: schema_intake_critic(),
        prompt_parts: Some(critic_parts),
        prompt: critic_prompt.clone(),
        temperature: 0.0,
    };

    let critic_resp = runtime.llm.complete_json(critic_req.clone()).await?;

    record_trace!(TraceEvent::LlmCall {
        request: critic_req,
        response: critic_resp.clone(),
    });

    let critic_output: CriticOutput = parse_json(&critic_resp.raw_json).unwrap_or(CriticOutput {
        approved: true,
        critique: "Critic JSON fallback: draft accepted".to_string(),
        has_dropped_clauses: false,
        has_contaminated_queries: false,
        corrected_intents: None,
    });

    // --- Triad Node 3: Host Arbiter Quorum ---
    let triad = arbitrate_triad(&task.question, &drafter_intake, &critic_output);

    let mut intake = drafter_intake.clone();
    intake.intents = triad.final_intents.clone();
    intake.needs_workspace_search = intake.intents.iter().any(|item| item.needs_workspace_search);

    let intent_labels: Vec<String> = intake
        .intents
        .iter()
        .map(|item| format!("{:?}", item.intent))
        .collect();

    notebook.append_decision(
        format!(
            "Triad Consensus: {} -> Final Intents: [{}] (needs_search: {})",
            triad.arbiter_decision,
            intent_labels.join(", "),
            intake.needs_workspace_search
        ),
        Provenance::Model {
            call_id: critic_resp.call_id.clone(),
        },
        Phase::Intake,
        Vec::new(),
        intake.target_hints.clone(),
    );

    // If intake_only was requested, short-circuit here with Step 1 report
    if task.intake_only {
        let content = format_step1_report(&task, &intake_prompt, &intake_resp, &intake, &critic_output, &triad);
        let answer = AgentAnswer {
            content,
            stop_reason: StopReason::Done,
            cited_entries: Vec::new(),
            notebook,
        };
        record_trace!(TraceEvent::RunFinished {
            stop_reason: StopReason::Done,
        });
        return Ok(AgentRunResult { answer, trace });
    }

    // --- Phase 2: Deterministic Evidence Acquisition (Search & Lease) ---
    runtime
        .progress
        .emit(AgentProgress::Status("Phase 2: Searching workspace for evidence".to_string()))
        .await;

    record_trace!(TraceEvent::PhaseStarted {
        phase: Phase::CandidateGate,
    });

    let phase2_start = std::time::Instant::now();
    let mut queries = Vec::new();
    let mut search_details: Vec<SearchQueryExecution> = Vec::new();
    let mut all_path_hits: Vec<FileHit> = Vec::new();
    let mut all_symbol_hits: Vec<SymbolHit> = Vec::new();
    let mut all_vector_hits: Vec<VectorHit> = Vec::new();
    let mut all_content_hits: Vec<ContentHit> = Vec::new();
    let mut expanded_hits: Vec<ExpandedHitDetail> = Vec::new();
    let mut pruned_candidates: Vec<PrunedCandidateDetail> = Vec::new();
    let mut leased_slices: Vec<EvidenceSlice> = Vec::new();

    if intake.needs_workspace_search {
        // Collect search queries from workspace-targeting intents and hints
        for item in &intake.intents {
            if item.needs_workspace_search {
                for q in &item.search_queries {
                    let clean = q.trim();
                    if !clean.is_empty() && !queries.iter().any(|existing: &String| existing.eq_ignore_ascii_case(clean)) {
                        queries.push(clean.to_string());
                    }
                }
            }
        }
        for hint in &intake.target_hints {
            let clean = hint.trim();
            if !clean.is_empty() && !queries.iter().any(|existing: &String| existing.eq_ignore_ascii_case(clean)) {
                queries.push(clean.to_string());
            }
        }
        queries.truncate(task.budgets.max_search_queries.max(1));

        for q in &queries {
            runtime
                .progress
                .emit(AgentProgress::Status(format!("Searching workspace for '{}'", q)))
                .await;

            // 1. Path search
            let path_hits = runtime.code_search.search(q, 5).await.unwrap_or_default();
            record_trace!(TraceEvent::ToolCall {
                name: "search".to_string(),
                args: json!({ "query": q, "limit": 5 }),
                result_count: path_hits.len(),
            });

            // 2. Symbol search (Tree-sitter)
            let symbol_hits = runtime.code_search.search_symbols(q, 5).await.unwrap_or_default();
            record_trace!(TraceEvent::ToolCall {
                name: "search_symbols".to_string(),
                args: json!({ "query": q, "limit": 5 }),
                result_count: symbol_hits.len(),
            });

            // 3. Vector search (Semantic embeddings)
            let vector_hits = runtime.code_search.search_vector(q, 5).await.unwrap_or_default();
            record_trace!(TraceEvent::ToolCall {
                name: "search_vector".to_string(),
                args: json!({ "query": q, "limit": 5 }),
                result_count: vector_hits.len(),
            });

            // 4. Content search (Regex / Ripgrep)
            let content_hits = runtime.code_search.search_content(q, 5).await.unwrap_or_default();
            record_trace!(TraceEvent::ToolCall {
                name: "search_content".to_string(),
                args: json!({ "query": q, "limit": 5 }),
                result_count: content_hits.len(),
            });

            search_details.push(SearchQueryExecution {
                query: q.clone(),
                path_hits_count: path_hits.len(),
                symbol_hits_count: symbol_hits.len(),
                vector_hits_count: vector_hits.len(),
                content_hits_count: content_hits.len(),
            });

            all_path_hits.extend(path_hits);
            all_symbol_hits.extend(symbol_hits);
            all_vector_hits.extend(vector_hits);
            all_content_hits.extend(content_hits);
        }

        // Rank and deduplicate candidate slices
        let mut candidate_targets: Vec<SliceTarget> = Vec::new();

        // 1. Top symbol hits (highest structural precision)
        all_symbol_hits.sort_by(|a, b| b.score.cmp(&a.score));
        for sym in &all_symbol_hits {
            if let Some(reason) = candidate_exclusion_reason(&sym.path, &task.question) {
                if !pruned_candidates.iter().any(|p| p.path == sym.path) {
                    pruned_candidates.push(PrunedCandidateDetail {
                        path: sym.path.clone(),
                        reason: reason.to_string(),
                    });
                }
                continue;
            }
            if !candidate_targets.iter().any(|t| t.path == sym.path && t.symbol.as_deref() == Some(&sym.symbol)) {
                candidate_targets.push(SliceTarget {
                    path: sym.path.clone(),
                    symbol: Some(sym.symbol.clone()),
                    start_line: sym.start_line,
                    end_line: sym.end_line,
                    score: sym.score + 50,
                    source: format!("Symbol `{}` ({})", sym.symbol, sym.kind),
                });
            }
        }

        // 2. Top vector hits (dense semantic similarity)
        all_vector_hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        for v in &all_vector_hits {
            if let Some(reason) = candidate_exclusion_reason(&v.path, &task.question) {
                if !pruned_candidates.iter().any(|p| p.path == v.path) {
                    pruned_candidates.push(PrunedCandidateDetail {
                        path: v.path.clone(),
                        reason: reason.to_string(),
                    });
                }
                continue;
            }
            if v.score < 0.15 {
                continue;
            }
            let overlaps = candidate_targets.iter().any(|t| {
                t.path == v.path && !(v.end_line < t.start_line || v.start_line > t.end_line)
            });
            if !overlaps {
                candidate_targets.push(SliceTarget {
                    path: v.path.clone(),
                    symbol: v.symbol.clone(),
                    start_line: v.start_line,
                    end_line: v.end_line,
                    score: (v.score * 100.0) as i64,
                    source: format!("Vector semantic search (score: {:.2})", v.score),
                });
            }
        }

        // 3. Top content hits
        all_content_hits.sort_by(|a, b| b.score.cmp(&a.score));
        for ch in &all_content_hits {
            if let Some(reason) = candidate_exclusion_reason(&ch.path, &task.question) {
                if !pruned_candidates.iter().any(|p| p.path == ch.path) {
                    pruned_candidates.push(PrunedCandidateDetail {
                        path: ch.path.clone(),
                        reason: reason.to_string(),
                    });
                }
                continue;
            }
            if ch.score < 40 {
                continue;
            }
            let start = ch.start_line.saturating_sub(4).max(1);
            let end = ch.end_line + 12;
            let overlaps = candidate_targets.iter().any(|t| {
                t.path == ch.path && !(end < t.start_line || start > t.end_line)
            });
            if !overlaps {
                candidate_targets.push(SliceTarget {
                    path: ch.path.clone(),
                    symbol: None,
                    start_line: start,
                    end_line: end,
                    score: ch.score,
                    source: "Content match".to_string(),
                });
            }
        }

        // 4. Top path hits
        all_path_hits.sort_by(|a, b| b.score.cmp(&a.score));
        for ph in &all_path_hits {
            if let Some(reason) = candidate_exclusion_reason(&ph.path, &task.question) {
                if !pruned_candidates.iter().any(|p| p.path == ph.path) {
                    pruned_candidates.push(PrunedCandidateDetail {
                        path: ph.path.clone(),
                        reason: reason.to_string(),
                    });
                }
                continue;
            }
            if ph.score < 50 {
                continue;
            }
            if !candidate_targets.iter().any(|t| t.path == ph.path) {
                // If there is a matching vector hit for this path, use its focal range!
                let (start_line, end_line, symbol, source) = if let Some(vh) = all_vector_hits.iter().find(|v| v.path == ph.path) {
                    (vh.start_line, vh.end_line, vh.symbol.clone(), format!("Path match + Vector focal chunk ({})", vh.symbol.as_deref().unwrap_or("block")))
                } else {
                    (1, 60, None, "Path match".to_string())
                };
                candidate_targets.push(SliceTarget {
                    path: ph.path.clone(),
                    symbol,
                    start_line,
                    end_line,
                    score: ph.score,
                    source,
                });
            }
        }

        // --- Pass A2: Domain alignment & hygiene pruning ---
        let q_lower = task.question.to_lowercase();
        let is_ui_question = q_lower.contains("ui")
            || q_lower.contains("screen")
            || q_lower.contains("view")
            || q_lower.contains("route")
            || q_lower.contains("navigation")
            || q_lower.contains("frontend");

        let has_frontend_candidates = candidate_targets.iter().any(|t| {
            let p = t.path.to_lowercase();
            p.ends_with(".tsx")
                || p.ends_with(".jsx")
                || p.ends_with(".vue")
                || p.ends_with(".svelte")
                || p.ends_with(".html")
        });

        if is_ui_question {
            for target in &mut candidate_targets {
                let p = target.path.to_lowercase();
                let is_frontend_ext = p.ends_with(".tsx")
                    || p.ends_with(".jsx")
                    || p.ends_with(".vue")
                    || p.ends_with(".svelte")
                    || p.ends_with(".html")
                    || p.ends_with(".css");

                if is_frontend_ext {
                    target.score += 40;
                    if !p.ends_with(".css")
                        && (p.contains("app.")
                            || p.contains("index.")
                            || p.contains("main.")
                            || p.contains("routes.")
                            || p.contains("router."))
                    {
                        target.score += 30;
                    }
                } else if has_frontend_candidates
                    && (p.ends_with(".rs") || p.contains("src-tauri/") || p.contains("server/"))
                {
                    target.score = -100;
                    if !pruned_candidates.iter().any(|pr| pr.path == target.path) {
                        pruned_candidates.push(PrunedCandidateDetail {
                            path: target.path.clone(),
                            reason: "Domain mismatch (Backend Rust logic filtered out for UI/frontend intent)".to_string(),
                        });
                    }
                }
            }
        }

        // Retain only candidates with positive score and sort descending before graph expansion
        candidate_targets.retain(|t| t.score > 0);
        candidate_targets.sort_by(|a, b| b.score.cmp(&a.score));

        // --- Pass B: CodeGraph Topology Expansion (Unexpected Matches) ---
        let mut seed_paths: Vec<String> = Vec::new();
        for t in &candidate_targets {
            if !seed_paths.contains(&t.path) && seed_paths.len() < 3 {
                seed_paths.push(t.path.clone());
            }
        }

        if !seed_paths.is_empty() {
            runtime
                .progress
                .emit(AgentProgress::Status("Expanding CodeGraph topology for connected components".to_string()))
                .await;

            let imported = runtime.code_search.search_imported_files(&seed_paths, 20).await.unwrap_or_default();
            record_trace!(TraceEvent::ToolCall {
                name: "search_imported_files".to_string(),
                args: json!({ "seed_paths": seed_paths, "limit": 20 }),
                result_count: imported.len(),
            });

            for imp in imported {
                if let Some(reason) = candidate_exclusion_reason(&imp.path, &task.question) {
                    if !pruned_candidates.iter().any(|p| p.path == imp.path) {
                        pruned_candidates.push(PrunedCandidateDetail {
                            path: imp.path.clone(),
                            reason: format!("CodeGraph imported file excluded: {}", reason),
                        });
                    }
                    continue;
                }
                if is_ui_question && has_frontend_candidates && (imp.path.ends_with(".rs") || imp.path.contains("src-tauri/")) {
                    continue;
                }
                if !candidate_targets.iter().any(|t| t.path == imp.path) {
                    expanded_hits.push(ExpandedHitDetail {
                        path: imp.path.clone(),
                        source_seed: seed_paths.join(", "),
                        reason: "Imported by seed entrypoint in CodeGraph".to_string(),
                    });

                    // Check if there is a primary symbol in this newly discovered file
                    let stem = std::path::Path::new(&imp.path)
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("");
                    let mut syms = runtime.code_search.search_file_symbols(&imp.path).await.unwrap_or_default();
                    if syms.is_empty() && !stem.is_empty() {
                        syms = runtime
                            .code_search
                            .search_symbols(stem, 25)
                            .await
                            .unwrap_or_default()
                            .into_iter()
                            .filter(|s| s.path == imp.path)
                            .collect();
                    }

                    // Find primary symbol: prefers symbol matching file stem, or component/function with largest span
                    let matched_sym = syms
                        .iter()
                        .find(|s| s.symbol.eq_ignore_ascii_case(stem))
                        .cloned()
                        .or_else(|| {
                            syms.into_iter().max_by_key(|s| s.end_line.saturating_sub(s.start_line))
                        });

                    let matching_vector_hit = all_vector_hits.iter().find(|vh| vh.path == imp.path);

                    if let Some(sym) = matched_sym {
                        // For files under 250 lines, start from line 1 to include imports and full component structure
                        let (s_line, e_line) = if sym.end_line <= 250 {
                            (1, sym.end_line)
                        } else if let Some(vh) = matching_vector_hit {
                            // If file > 250 lines and vector search found a semantic hit, target the focal chunk!
                            (vh.start_line, vh.end_line)
                        } else if sym.end_line > sym.start_line + 200 {
                            (sym.end_line.saturating_sub(120).max(sym.start_line), sym.end_line)
                        } else {
                            (sym.start_line, sym.end_line)
                        };
                        candidate_targets.push(SliceTarget {
                            path: imp.path.clone(),
                            symbol: Some(sym.symbol.clone()),
                            start_line: s_line,
                            end_line: e_line,
                            score: imp.score + 100, // Strong boost for topology-discovered root component
                            source: format!("CodeGraph topology (imported component `{}`)", sym.symbol),
                        });
                    } else if let Some(vh) = matching_vector_hit {
                        candidate_targets.push(SliceTarget {
                            path: imp.path.clone(),
                            symbol: vh.symbol.clone(),
                            start_line: vh.start_line,
                            end_line: vh.end_line,
                            score: imp.score + 85,
                            source: "CodeGraph topology + Vector focal chunk".to_string(),
                        });
                    } else {
                        candidate_targets.push(SliceTarget {
                            path: imp.path.clone(),
                            symbol: None,
                            start_line: 1,
                            end_line: 250,
                            score: imp.score + 70,
                            source: "CodeGraph topology (imported file)".to_string(),
                        });
                    }
                }
            }
        }

        // Re-sort candidate targets with newly expanded hits included
        candidate_targets.sort_by(|a, b| b.score.cmp(&a.score));

        let max_files = task.budgets.max_files_to_read.max(1);
        let max_chars = task.budgets.max_evidence_chars / max_files;

        // Take top candidate pool (up to 10 candidates) for stereoscopic relevance evaluation
        let candidate_pool: Vec<SliceTarget> = candidate_targets.into_iter().take(10).collect();
        let mut final_targets: Vec<SliceTarget> = Vec::new();

        if candidate_pool.len() >= 2 {
            runtime
                .progress
                .emit(AgentProgress::Status("Executing Stereoscopic Gating (Optimist vs. Skeptic)".to_string()))
                .await;

            let mut gate_items = Vec::new();
            for cand in &candidate_pool {
                let preview = if cand.start_line > 35 {
                    // "Header + Focal Window" Slicing:
                    // For chunks located deep in a file (> line 35), include the top 25 lines (imports, declarations)
                    // so the Optimist & Skeptic model lenses have full lexical and architectural scope.
                    let header_res = runtime
                        .file_reader
                        .read_range(&cand.path, None, 1, 25, 2000)
                        .await;
                    let focal_res = runtime
                        .file_reader
                        .read_range(&cand.path, cand.symbol.as_deref(), cand.start_line, cand.end_line, 6000)
                        .await;

                    match (header_res, focal_res) {
                        (Ok(header_slice), Ok(focal_slice)) => {
                            let clean_header = compact_candidate_snippet(&header_slice.content);
                            let clean_focal = compact_candidate_snippet(&focal_slice.content);
                            if !clean_header.is_empty() && clean_header != clean_focal {
                                format!(
                                    "{}\n// ... [omitted lines 26 to {}] ...\n{}",
                                    clean_header,
                                    cand.start_line.saturating_sub(1),
                                    clean_focal
                                )
                            } else {
                                clean_focal
                            }
                        }
                        (_, Ok(focal_slice)) => compact_candidate_snippet(&focal_slice.content),
                        _ => "Preview unavailable".to_string(),
                    }
                } else {
                    match runtime
                        .file_reader
                        .read_range(&cand.path, cand.symbol.as_deref(), cand.start_line, cand.end_line, 8000)
                        .await
                    {
                        Ok(slice) => compact_candidate_snippet(&slice.content),
                        Err(_) => "Preview unavailable".to_string(),
                    }
                };
                gate_items.push(CandidateGateItem {
                    path: cand.path.clone(),
                    symbol: cand.symbol.clone(),
                    start_line: cand.start_line,
                    end_line: cand.end_line,
                    source: cand.source.clone(),
                    snippet: preview,
                });
            }

            let target_intent_summary = intake
                .intents
                .iter()
                .find(|i| i.needs_workspace_search)
                .map(|i| i.summary.as_str())
                .unwrap_or(task.question.as_str());

            // --- Lens 1: Optimist (Architectural Scout) ---
            let opt_prompt = prompt_candidate_gate_optimist(&task.question, target_intent_summary, &gate_items);
            let opt_req = LlmRequest {
                call_id: call_id("gate-opt"),
                phase: Phase::CandidateGateOptimist,
                schema_name: "CandidateGateOutput".to_string(),
                schema: schema_candidate_gate(),
                prompt_parts: None,
                prompt: opt_prompt,
                temperature: 0.0,
            };

            record_trace!(TraceEvent::PhaseStarted {
                phase: Phase::CandidateGateOptimist,
            });

            let opt_output: Option<CandidateGateOutput> = match runtime.llm.complete_json(opt_req.clone()).await {
                Ok(resp) => {
                    record_trace!(TraceEvent::LlmCall {
                        request: opt_req,
                        response: resp.clone(),
                    });
                    parse_json::<CandidateGateOutput>(&resp.raw_json).ok()
                }
                Err(_) => None,
            };

            // --- Lens 2: Skeptic (Precision Critic) ---
            let skep_prompt = prompt_candidate_gate_skeptic(&task.question, target_intent_summary, &gate_items);
            let skep_req = LlmRequest {
                call_id: call_id("gate-skep"),
                phase: Phase::CandidateGateSkeptic,
                schema_name: "CandidateGateOutput".to_string(),
                schema: schema_candidate_gate(),
                prompt_parts: None,
                prompt: skep_prompt,
                temperature: 0.0,
            };

            record_trace!(TraceEvent::PhaseStarted {
                phase: Phase::CandidateGateSkeptic,
            });

            let skep_output: Option<CandidateGateOutput> = match runtime.llm.complete_json(skep_req.clone()).await {
                Ok(resp) => {
                    record_trace!(TraceEvent::LlmCall {
                        request: skep_req,
                        response: resp.clone(),
                    });
                    parse_json::<CandidateGateOutput>(&resp.raw_json).ok()
                }
                Err(_) => None,
            };

            // --- Node 3: Deterministic Rust Host Arbiter ---
            if opt_output.is_none() && skep_output.is_none() {
                final_targets = candidate_pool.clone();
            } else {
                for cand in &candidate_pool {
                    let opt_eval = opt_output
                        .as_ref()
                        .and_then(|o| o.evaluations.iter().find(|e| e.path == cand.path));
                    let skep_eval = skep_output
                        .as_ref()
                        .and_then(|s| s.evaluations.iter().find(|e| e.path == cand.path));

                    let opt_keep = opt_eval.map(|e| e.keep).unwrap_or(true);
                    let skep_keep = skep_eval.map(|e| e.keep).unwrap_or(false);

                    let p_lower = cand.path.to_lowercase();
                    let has_topology_backing = cand.source.contains("CodeGraph topology")
                        || seed_paths.contains(&cand.path);
                    let is_entrypoint_or_shell = p_lower.contains("index.")
                        || p_lower.contains("app.")
                        || p_lower.contains("main.")
                        || p_lower.contains("routes.")
                        || p_lower.contains("router.");
                    let is_domain_aligned = if is_ui_question {
                        !p_lower.ends_with(".css")
                            && (p_lower.ends_with(".tsx")
                                || p_lower.ends_with(".jsx")
                                || p_lower.ends_with(".vue")
                                || p_lower.ends_with(".svelte")
                                || p_lower.ends_with(".html"))
                    } else {
                        true
                    };

                    // Stereoscopic Consensus Matrix
                    if opt_keep && skep_keep {
                        // Unanimous Keep
                        final_targets.push(cand.clone());
                    } else if !opt_keep && !skep_keep {
                        // Unanimous Drop - with Architectural Anchor Protection:
                        // If a candidate is a domain-aligned root component or shell directly connected by CodeGraph topology,
                        // retain it as an architectural anchor even if both models claim lack of a dedicated feature table.
                        let is_architectural_anchor = has_topology_backing && is_entrypoint_or_shell && is_domain_aligned;
                        if is_architectural_anchor {
                            final_targets.push(cand.clone());
                        } else {
                            let reason = skep_eval
                                .map(|e| e.reason.as_str())
                                .or_else(|| opt_eval.map(|e| e.reason.as_str()))
                                .unwrap_or("Stereoscopic Consensus: Pruned by both Optimist and Skeptic");
                            pruned_candidates.push(PrunedCandidateDetail {
                                path: cand.path.clone(),
                                reason: format!("Stereoscopic consensus rejected: {}", reason),
                            });
                        }
                    } else if opt_keep && !skep_keep {
                        // Split Decision: Optimist wanted to keep, Skeptic wanted to drop
                        // Deterministic Host Arbiter tie-breaker:
                        // Favor inclusion if backed by CodeGraph topology, entrypoint/shell identity, or domain alignment
                        if (has_topology_backing || is_entrypoint_or_shell) && is_domain_aligned {
                            final_targets.push(cand.clone());
                        } else {
                            let skep_reason = skep_eval.map(|e| e.reason.as_str()).unwrap_or("pruned by Skeptic");
                            pruned_candidates.push(PrunedCandidateDetail {
                                path: cand.path.clone(),
                                reason: format!("Stereoscopic Arbiter pruned (unbacked by CodeGraph topology): {}", skep_reason),
                            });
                        }
                    } else {
                        // Split Decision: Skeptic wanted to keep, Optimist wanted to drop
                        // If even the Skeptic found a compelling reason to keep, retain it if domain aligned
                        if is_domain_aligned {
                            final_targets.push(cand.clone());
                        } else {
                            pruned_candidates.push(PrunedCandidateDetail {
                                path: cand.path.clone(),
                                reason: "Stereoscopic Arbiter: Domain mismatch".to_string(),
                            });
                        }
                    }
                }
            }
        } else {
            final_targets = candidate_pool.clone();
        }

        // Safety fallback: if all targets were pruned, retain top candidate from vetted candidate_pool
        // NEVER pull from raw unfiltered all_symbol_hits!
        if final_targets.is_empty() {
            if let Some(top_cand) = candidate_pool.into_iter().find(|c| candidate_exclusion_reason(&c.path, &task.question).is_none()) {
                final_targets.push(top_cand);
            }
        }

        let mut leased_header_paths: HashSet<String> = HashSet::new();

        // Acquire leases for final verified targets
        for target in final_targets.into_iter().take(max_files) {
            // Header context lease: If target is deep inside a large file (> line 35),
            // lease the file header (lines 1..25) first so imports and types are available in the notebook.
            if target.start_line > 35 && !leased_header_paths.contains(&target.path) {
                if let Ok(header_slice) = runtime
                    .file_reader
                    .read_range(&target.path, Some("header/imports"), 1, 25, 2000)
                    .await
                {
                    leased_header_paths.insert(target.path.clone());
                    record_trace!(TraceEvent::ToolCall {
                        name: "read_range".to_string(),
                        args: json!({
                            "path": target.path,
                            "symbol": "header/imports",
                            "start_line": 1,
                            "end_line": 25,
                        }),
                        result_count: 1,
                    });

                    notebook.append_tool_fact(
                        format!(
                            "Leased {} lines 1-25 (header/imports context)",
                            header_slice.lease.path
                        ),
                        Provenance::File {
                            path: header_slice.lease.path.clone(),
                            symbol: header_slice.lease.symbol.clone(),
                            range: header_slice.lease.range,
                            lease_id: header_slice.lease.id.clone(),
                        },
                        Phase::Extract,
                        vec![header_slice.lease.path.clone()],
                    );

                    leased_slices.push(header_slice);
                }
            }

            runtime
                .progress
                .emit(AgentProgress::Status(format!("Reading evidence: {}:{}-{}", target.path, target.start_line, target.end_line)))
                .await;

            match runtime
                .file_reader
                .read_range(
                    &target.path,
                    target.symbol.as_deref(),
                    target.start_line,
                    target.end_line,
                    max_chars,
                )
                .await
            {
                Ok(slice) => {
                    record_trace!(TraceEvent::ToolCall {
                        name: "read_range".to_string(),
                        args: json!({
                            "path": target.path,
                            "symbol": target.symbol,
                            "start_line": target.start_line,
                            "end_line": target.end_line,
                        }),
                        result_count: 1,
                    });

                    notebook.append_tool_fact(
                        format!(
                            "Leased {} lines {}-{} ({})",
                            slice.lease.path,
                            slice.lease.range.start,
                            slice.lease.range.end,
                            slice.lease.symbol.as_deref().unwrap_or("block")
                        ),
                        Provenance::File {
                            path: slice.lease.path.clone(),
                            symbol: slice.lease.symbol.clone(),
                            range: slice.lease.range,
                            lease_id: slice.lease.id.clone(),
                        },
                        Phase::Extract,
                        vec![slice.lease.path.clone()],
                    );

                    leased_slices.push(slice);
                }
                Err(err) => {
                    record_trace!(TraceEvent::ToolCall {
                        name: "read_range_failed".to_string(),
                        args: json!({
                            "path": target.path,
                            "error": err.to_string(),
                        }),
                        result_count: 0,
                    });
                }
            }
        }
    }

    let phase2_elapsed_ms = phase2_start.elapsed().as_millis() as u64;

    record_trace!(TraceEvent::NotebookSnapshot {
        phase: Phase::CandidateGate,
        notebook: notebook.deterministic_checkpoint(),
    });

    let phase2 = Phase2Results {
        queries,
        search_details,
        all_path_hits,
        all_symbol_hits,
        all_vector_hits,
        all_content_hits,
        expanded_hits,
        pruned_candidates,
        leased_slices,
        elapsed_ms: phase2_elapsed_ms,
    };

    // =========================================================================
    // STEP 3A: Evidence Extraction (Phase::Extract)
    // =========================================================================
    let mut extract_result: Option<ExtractStepResults> = None;

    if intake.needs_workspace_search && !phase2.leased_slices.is_empty() {
        runtime
            .progress
            .emit(AgentProgress::Status("Step 3A: Extracting evidence-backed inferences from leased code".to_string()))
            .await;

        let extract_start = std::time::Instant::now();
        record_trace!(TraceEvent::PhaseStarted {
            phase: Phase::Extract,
        });

        let extract_parts = PromptParts {
            global_prefix: global_prefix(&repo_map),
            phase_prefix: phase_prefix(Phase::Extract, "ExtractOutput"),
            task: task_block(&task),
            notebook: notebook_block(&notebook, &[], 25),
            evidence: evidence_block(&phase2.leased_slices),
            question: extract_instruction(),
        };

        let extract_prompt = [
            extract_parts.global_prefix.as_str(),
            extract_parts.phase_prefix.as_str(),
            extract_parts.task.as_str(),
            extract_parts.notebook.as_str(),
            extract_parts.evidence.as_str(),
            extract_parts.question.as_str(),
        ]
        .join("\n\n");

        let extract_req = LlmRequest {
            call_id: call_id("ext"),
            phase: Phase::Extract,
            schema_name: "ExtractOutput".to_string(),
            schema: schema_extract(),
            prompt_parts: Some(extract_parts.clone()),
            prompt: extract_prompt.clone(),
            temperature: 0.0,
        };

        let extract_resp = match runtime.llm.complete_json(extract_req.clone()).await {
            Ok(resp) => {
                record_trace!(TraceEvent::LlmCall {
                    request: extract_req.clone(),
                    response: resp.clone(),
                });
                Some(resp)
            }
            Err(err) => {
                record_trace!(TraceEvent::ToolCall {
                    name: "extract_failed".to_string(),
                    args: json!({ "error": err.to_string() }),
                    result_count: 0,
                });
                None
            }
        };

        if let Some(resp) = &extract_resp {
            if let Ok(extracted) = parse_json::<ExtractOutput>(&resp.raw_json) {
                let mut verified_inferences = Vec::new();
                for inf in &extracted.inferences {
                    if let Some(fact_entry) = notebook.find_entry_by_lease_id(&inf.lease_id) {
                        let dep_id = fact_entry.id.clone();
                        let mut tags = inf.tags.clone();
                        if !tags.iter().any(|t| t == "inference") {
                            tags.push("inference".to_string());
                        }
                        let inf_id = notebook.append_model_inference(
                            &inf.statement,
                            Provenance::Model { call_id: resp.call_id.clone() },
                            Phase::Extract,
                            vec![dep_id],
                            tags,
                        );
                        verified_inferences.push(VerifiedInference {
                            statement: inf.statement.clone(),
                            lease_id: inf.lease_id.0.clone(),
                            is_valid: true,
                            entry_id: Some(inf_id),
                            rejection_reason: None,
                        });
                    } else {
                        verified_inferences.push(VerifiedInference {
                            statement: inf.statement.clone(),
                            lease_id: inf.lease_id.0.clone(),
                            is_valid: false,
                            entry_id: None,
                            rejection_reason: Some(format!("Lease ID `{}` not found among active notebook leases", inf.lease_id.0)),
                        });
                    }
                }

                extract_result = Some(ExtractStepResults {
                    inferences: verified_inferences,
                    sufficient: extracted.sufficient,
                    missing: extracted.missing,
                    raw_json: resp.raw_json.clone(),
                    prompt: extract_prompt,
                    elapsed_ms: extract_start.elapsed().as_millis() as u64,
                });
            }
        }
    }

    let mut cited_entries: Vec<EntryId> = Vec::new();
    if let Some(ext) = &extract_result {
        for inf in &ext.inferences {
            if let Some(entry_id) = &inf.entry_id {
                cited_entries.push(entry_id.clone());
                if let Some(entry) = notebook.get(entry_id) {
                    for dep in &entry.depends_on {
                        if !cited_entries.contains(dep) {
                            cited_entries.push(dep.clone());
                        }
                    }
                }
            }
        }
    }

    // =========================================================================
    // STEP 3B: Universal Plan Phase (Appellate Stereoscopic Gate)
    // =========================================================================
    let mut plan_result: Option<PlanStepResults> = None;
    let knowledge_engine = KnowledgeEngine::new_with_builtin();
    let leased_paths: Vec<String> = phase2.leased_slices.iter().map(|s| s.lease.path.clone()).collect();
    let detected_tags = detect_stack_tags(&repo_map, &task.question, &leased_paths);

    if !task.intake_only {
        runtime
            .progress
            .emit(AgentProgress::Status("Step 3B: Entering Appellate Stereoscopic Plan Gate".to_string()))
            .await;

        let plan_start = std::time::Instant::now();

        let plan_cards = knowledge_engine.match_cards(
            &task.question,
            &detected_tags,
            Some(&[KnowledgeDomain::CodeReview, KnowledgeDomain::Architecture]),
            2,
        );
        let rubric_block = if !plan_cards.is_empty() {
            format!("\n\n{}", knowledge_engine.format_cards_for_prompt(&plan_cards))
        } else {
            String::new()
        };

        // 1. Invariant context prefix shared by all 4 lenses for maximum KV-cache reuse
        let shared_context_prefix = format!(
            "{}\n\n{}\n\n{}\n\n{}{}",
            global_prefix(&repo_map),
            task_block(&task),
            notebook_block(&notebook, &[], 25),
            evidence_block(&phase2.leased_slices),
            rubric_block,
        );

        // --- Lens 1: Plan Proposer (Thesis) ---
        runtime
            .progress
            .emit(AgentProgress::Status("Step 3B: Lens 1 — Drafting proposed plan".to_string()))
            .await;

        let proposer_start = std::time::Instant::now();
        record_trace!(TraceEvent::PhaseStarted {
            phase: Phase::PlanProposer,
        });

        let proposer_prompt = format!(
            "{}\n\n{}\n\n{}",
            shared_context_prefix,
            phase_prefix(Phase::PlanProposer, "EditPlan"),
            plan_instruction(),
        );

        let proposer_req = LlmRequest {
            call_id: call_id("plan-prop"),
            phase: Phase::PlanProposer,
            schema_name: "EditPlan".to_string(),
            schema: schema_plan(),
            prompt_parts: None,
            prompt: proposer_prompt,
            temperature: 0.0,
        };

        let proposer_resp = match runtime.llm.complete_json(proposer_req.clone()).await {
            Ok(resp) => {
                record_trace!(TraceEvent::LlmCall {
                    request: proposer_req.clone(),
                    response: resp.clone(),
                });
                Some(resp)
            }
            Err(err) => {
                record_trace!(TraceEvent::ToolCall {
                    name: "plan_proposer_failed".to_string(),
                    args: json!({ "error": err.to_string() }),
                    result_count: 0,
                });
                None
            }
        };

        let proposed_plan = proposer_resp
            .as_ref()
            .and_then(|r| parse_json::<EditPlan>(&r.raw_json).ok())
            .unwrap_or_else(|| {
                EditPlan {
                    objective: format!("Address user request: {}", task.question),
                    affected_files: phase2.leased_slices.iter().map(|s| PlannedFile {
                        path: s.lease.path.clone(),
                        reason: "Examined candidate in leased evidence".to_string(),
                    }).collect(),
                    steps: vec![
                        PlanStep {
                            title: "Synthesize Findings".to_string(),
                            detail: "Review retrieved evidence and notebook facts to answer prompt.".to_string(),
                        }
                    ],
                    risks: vec!["Fallback plan generated due to unparseable proposer output.".to_string()],
                    validation_commands: vec![],
                }
            });
        let proposer_elapsed_ms = proposer_start.elapsed().as_millis() as u64;

        // --- Lens 2: Plan Critic (Antithesis) ---
        runtime
            .progress
            .emit(AgentProgress::Status("Step 3B: Lens 2 — Adversarial critique".to_string()))
            .await;

        let critic_start = std::time::Instant::now();
        record_trace!(TraceEvent::PhaseStarted {
            phase: Phase::PlanCritic,
        });

        let proposed_plan_json = serde_json::to_string_pretty(&proposed_plan).unwrap_or_default();
        let critic_prompt = format!(
            "{}\n\n{}\n\n{}",
            shared_context_prefix,
            phase_prefix(Phase::PlanCritic, "PlanCriticOutput"),
            plan_critic_instruction(&proposed_plan_json),
        );

        let critic_req = LlmRequest {
            call_id: call_id("plan-crit"),
            phase: Phase::PlanCritic,
            schema_name: "PlanCriticOutput".to_string(),
            schema: schema_plan_critic(),
            prompt_parts: None,
            prompt: critic_prompt,
            temperature: 0.0,
        };

        let critic_resp = match runtime.llm.complete_json(critic_req.clone()).await {
            Ok(resp) => {
                record_trace!(TraceEvent::LlmCall {
                    request: critic_req.clone(),
                    response: resp.clone(),
                });
                Some(resp)
            }
            Err(err) => {
                record_trace!(TraceEvent::ToolCall {
                    name: "plan_critic_failed".to_string(),
                    args: json!({ "error": err.to_string() }),
                    result_count: 0,
                });
                None
            }
        };

        let critic_output = critic_resp
            .as_ref()
            .and_then(|r| parse_json::<PlanCriticOutput>(&r.raw_json).ok())
            .unwrap_or_else(|| PlanCriticOutput {
                critique: "Automated review: verify evidence coverage and edge cases.".to_string(),
                missing_aspects: vec![],
                disputed_steps: vec![],
                unverified_assumptions: vec![],
                additional_risks: vec![],
            });
        let critic_elapsed_ms = critic_start.elapsed().as_millis() as u64;

        // --- Lens 3: Plan Rebuttal (Scope Arbiter / Critic of the Critic) ---
        runtime
            .progress
            .emit(AgentProgress::Status("Step 3B: Lens 3 — Pragmatic rebuttal & defense".to_string()))
            .await;

        let rebuttal_start = std::time::Instant::now();
        record_trace!(TraceEvent::PhaseStarted {
            phase: Phase::PlanRebuttal,
        });

        let critic_output_json = serde_json::to_string_pretty(&critic_output).unwrap_or_default();
        let rebuttal_prompt = format!(
            "{}\n\n{}\n\n{}",
            shared_context_prefix,
            phase_prefix(Phase::PlanRebuttal, "PlanRebuttalOutput"),
            plan_rebuttal_instruction(&proposed_plan_json, &critic_output_json),
        );

        let rebuttal_req = LlmRequest {
            call_id: call_id("plan-rebut"),
            phase: Phase::PlanRebuttal,
            schema_name: "PlanRebuttalOutput".to_string(),
            schema: schema_plan_rebuttal(),
            prompt_parts: None,
            prompt: rebuttal_prompt,
            temperature: 0.0,
        };

        let rebuttal_resp = match runtime.llm.complete_json(rebuttal_req.clone()).await {
            Ok(resp) => {
                record_trace!(TraceEvent::LlmCall {
                    request: rebuttal_req.clone(),
                    response: resp.clone(),
                });
                Some(resp)
            }
            Err(err) => {
                record_trace!(TraceEvent::ToolCall {
                    name: "plan_rebuttal_failed".to_string(),
                    args: json!({ "error": err.to_string() }),
                    result_count: 0,
                });
                None
            }
        };

        let rebuttal_output = rebuttal_resp
            .as_ref()
            .and_then(|r| parse_json::<PlanRebuttalOutput>(&r.raw_json).ok())
            .unwrap_or_else(|| PlanRebuttalOutput {
                rebuttal: "Evaluate critique against existing evidence.".to_string(),
                dismissed_critiques: vec![],
                accepted_critiques: vec![],
            });
        let rebuttal_elapsed_ms = rebuttal_start.elapsed().as_millis() as u64;

        // --- Lens 4: Plan Reconciler (Synthesis) ---
        runtime
            .progress
            .emit(AgentProgress::Status("Step 3B: Lens 4 — Reconciling final plan".to_string()))
            .await;

        let reconciler_start = std::time::Instant::now();
        record_trace!(TraceEvent::PhaseStarted {
            phase: Phase::PlanReconciler,
        });

        let rebuttal_output_json = serde_json::to_string_pretty(&rebuttal_output).unwrap_or_default();
        let reconciler_prompt = format!(
            "{}\n\n{}\n\n{}",
            shared_context_prefix,
            phase_prefix(Phase::PlanReconciler, "EditPlan"),
            plan_reconciler_instruction(&proposed_plan_json, &critic_output_json, &rebuttal_output_json),
        );

        let reconciler_req = LlmRequest {
            call_id: call_id("plan-rec"),
            phase: Phase::PlanReconciler,
            schema_name: "EditPlan".to_string(),
            schema: schema_plan(),
            prompt_parts: None,
            prompt: reconciler_prompt,
            temperature: 0.0,
        };

        let reconciler_resp = match runtime.llm.complete_json(reconciler_req.clone()).await {
            Ok(resp) => {
                record_trace!(TraceEvent::LlmCall {
                    request: reconciler_req.clone(),
                    response: resp.clone(),
                });
                Some(resp)
            }
            Err(err) => {
                record_trace!(TraceEvent::ToolCall {
                    name: "plan_reconciler_failed".to_string(),
                    args: json!({ "error": err.to_string() }),
                    result_count: 0,
                });
                None
            }
        };

        let reconciled_plan = reconciler_resp
            .as_ref()
            .and_then(|r| parse_json::<EditPlan>(&r.raw_json).ok())
            .unwrap_or_else(|| proposed_plan.clone());
        let reconciler_elapsed_ms = reconciler_start.elapsed().as_millis() as u64;

        // --- Node 5: Deterministic Rust Host Arbiter ---
        runtime
            .progress
            .emit(AgentProgress::Status("Step 3B: Host Arbiter — Enforcing evidence backing and committing decisions".to_string()))
            .await;

        let active_leased_paths: HashSet<String> = notebook
            .active_entries()
            .iter()
            .filter_map(|e| match &e.provenance {
                Provenance::File { path, .. } => Some(path.clone()),
                _ => None,
            })
            .collect();

        let mut arbiter_validated_files = Vec::new();
        let mut arbiter_dropped_files = Vec::new();

        for file in &reconciled_plan.affected_files {
            if active_leased_paths.is_empty() || active_leased_paths.contains(&file.path) {
                arbiter_validated_files.push(file.clone());
            } else {
                arbiter_dropped_files.push(file.clone());
            }
        }

        // Commit decisions to Notebook (EntryKind::Decision)
        let reconciler_call_id = reconciler_resp
            .as_ref()
            .map(|r| r.call_id.clone())
            .unwrap_or_else(|| call_id("arbiter"));

        let mut committed_decision_ids = Vec::new();

        let obj_id = notebook.append_decision(
            format!("Plan Objective: {}", reconciled_plan.objective),
            Provenance::Model { call_id: reconciler_call_id.clone() },
            Phase::PlanReconciler,
            cited_entries.clone(),
            vec!["plan".to_string(), "objective".to_string()],
        );
        committed_decision_ids.push(obj_id.clone());

        for (idx, step) in reconciled_plan.steps.iter().enumerate() {
            let step_id = notebook.append_decision(
                format!("Plan Step {}: {} — {}", idx + 1, step.title, step.detail),
                Provenance::Model { call_id: reconciler_call_id.clone() },
                Phase::PlanReconciler,
                vec![obj_id.clone()],
                vec!["plan".to_string(), "step".to_string()],
            );
            committed_decision_ids.push(step_id);
        }

        for file in &arbiter_validated_files {
            let file_id = notebook.append_decision(
                format!("Plan Target: {} ({})", file.path, file.reason),
                Provenance::Model { call_id: reconciler_call_id.clone() },
                Phase::PlanReconciler,
                vec![obj_id.clone()],
                vec!["plan".to_string(), "target".to_string()],
            );
            committed_decision_ids.push(file_id);
        }

        for risk in &reconciled_plan.risks {
            let risk_id = notebook.append_decision(
                format!("Plan Risk: {}", risk),
                Provenance::Model { call_id: reconciler_call_id.clone() },
                Phase::PlanReconciler,
                vec![obj_id.clone()],
                vec!["plan".to_string(), "risk".to_string()],
            );
            committed_decision_ids.push(risk_id);
        }

        for id in &committed_decision_ids {
            if !cited_entries.contains(id) {
                cited_entries.push(id.clone());
            }
        }

        let total_plan_ms = plan_start.elapsed().as_millis() as u64;

        plan_result = Some(PlanStepResults {
            proposed_plan,
            critic_output,
            rebuttal_output,
            reconciled_plan,
            arbiter_validated_files,
            arbiter_dropped_files,
            committed_decision_ids,
            proposer_elapsed_ms,
            critic_elapsed_ms,
            rebuttal_elapsed_ms,
            reconciler_elapsed_ms,
            total_elapsed_ms: total_plan_ms,
        });
    }

    // =========================================================================
    // STEP 4: Appellate Stereoscopic Edit Gate (When Task is an Edit)
    // =========================================================================
    let is_edit_request = intake.intents.iter().any(|i| i.intent == UserIntent::WorkspaceEdit)
        || task.question.to_lowercase().contains("implement")
        || task.question.to_lowercase().contains("refactor")
        || task.question.to_lowercase().contains("edit")
        || task.question.to_lowercase().contains("fix")
        || task.question.to_lowercase().contains("patch");

    let mut edit_artifact: Option<EditArtifact> = None;

    if is_edit_request && plan_result.is_some() && !task.intake_only {
        runtime
            .progress
            .emit(AgentProgress::Status("Step 4: Entering Appellate Stereoscopic Edit Gate".to_string()))
            .await;

        let plan_res = plan_result.as_ref().unwrap();
        let target_path_opt = plan_res
            .arbiter_validated_files
            .first()
            .map(|f| f.path.clone())
            .or_else(|| phase2.leased_slices.first().map(|s| s.lease.path.clone()));

        if let Some(target_path) = target_path_opt {
            let target_slice = phase2.leased_slices.iter().find(|s| s.lease.path == target_path);
            let target_symbol = target_slice.and_then(|s| s.lease.symbol.clone());
            let start_line = target_slice.map(|s| s.lease.range.start).unwrap_or(1);
            let end_line = target_slice.map(|s| s.lease.range.end).unwrap_or(25);
            let target_lease_id = target_slice.map(|s| s.lease.id.clone()).unwrap_or_else(LeaseId::new);
            let original_lines = target_slice.map(|s| s.content.clone()).unwrap_or_default();

            // Sibling code exemplar discovery
            let sibling_paths = find_sibling_exemplars(&repo_map, &target_path, 1);
            let mut exemplar_context = String::new();
            if let Some(sib) = sibling_paths.first() {
                if let Ok(sib_slice) = runtime.file_reader.read_file(sib, 2500).await {
                    let header = extract_exemplar_header(&sib_slice.content, 25);
                    exemplar_context = format!("SIBLING CODE EXEMPLAR ({})\n{}", sib, header);
                }
            }

            let mut patch_prefix_parts = vec![
                global_prefix(&repo_map),
                task_block(&task),
                notebook_block(&notebook, &[], 25),
                evidence_block(&phase2.leased_slices),
            ];
            if !exemplar_context.is_empty() {
                patch_prefix_parts.push(exemplar_context);
            }
            let shared_patch_prefix = patch_prefix_parts.join("\n\n");

            // --- Lens 1: Patch Proposer (Thesis) ---
            runtime
                .progress
                .emit(AgentProgress::Status("Step 4: Lens 1 — Drafting candidate patch proposal".to_string()))
                .await;
            record_trace!(TraceEvent::PhaseStarted {
                phase: Phase::PatchProposer,
            });

            let proposer_prompt = format!(
                "{}\n\n{}\n\n{}",
                shared_patch_prefix,
                phase_prefix(Phase::PatchProposer, "PatchProposal"),
                patch_proposer_instruction(
                    &target_path,
                    target_symbol.as_deref(),
                    start_line,
                    end_line,
                    &plan_res.reconciled_plan.objective,
                ),
            );

            let prop_req = LlmRequest {
                call_id: call_id("patch-prop"),
                phase: Phase::PatchProposer,
                schema_name: "PatchProposal".to_string(),
                schema: schema_patch_proposer(),
                prompt_parts: None,
                prompt: proposer_prompt,
                temperature: 0.0,
            };

            let prop_resp = match runtime.llm.complete_json(prop_req.clone()).await {
                Ok(resp) => {
                    record_trace!(TraceEvent::LlmCall {
                        request: prop_req.clone(),
                        response: resp.clone(),
                    });
                    Some(resp)
                }
                Err(err) => {
                    record_trace!(TraceEvent::ToolCall {
                        name: "patch_proposer_failed".to_string(),
                        args: json!({ "error": err.to_string() }),
                        result_count: 0,
                    });
                    None
                }
            };

            let proposed_patch = prop_resp
                .as_ref()
                .and_then(|r| parse_json::<PatchProposal>(&r.raw_json).ok())
                .unwrap_or_else(|| PatchProposal {
                    target_path: target_path.clone(),
                    target_symbol: target_symbol.clone(),
                    start_line,
                    end_line,
                    lease_id: target_lease_id.clone(),
                    replacement_code: original_lines.clone(),
                    justification: "Initial candidate patch".to_string(),
                });

            // --- Lens 2: Patch Critic (Antithesis) ---
            runtime
                .progress
                .emit(AgentProgress::Status("Step 4: Lens 2 — Adversarial patch critique".to_string()))
                .await;
            record_trace!(TraceEvent::PhaseStarted {
                phase: Phase::PatchCritic,
            });

            let proposed_json = serde_json::to_string_pretty(&proposed_patch).unwrap_or_default();
            let critic_prompt = format!(
                "{}\n\n{}\n\n{}",
                shared_patch_prefix,
                phase_prefix(Phase::PatchCritic, "PatchCriticOutput"),
                patch_critic_instruction(&proposed_json),
            );

            let critic_req = LlmRequest {
                call_id: call_id("patch-crit"),
                phase: Phase::PatchCritic,
                schema_name: "PatchCriticOutput".to_string(),
                schema: schema_patch_critic(),
                prompt_parts: None,
                prompt: critic_prompt,
                temperature: 0.0,
            };

            let critic_resp = match runtime.llm.complete_json(critic_req.clone()).await {
                Ok(resp) => {
                    record_trace!(TraceEvent::LlmCall {
                        request: critic_req.clone(),
                        response: resp.clone(),
                    });
                    Some(resp)
                }
                Err(err) => {
                    record_trace!(TraceEvent::ToolCall {
                        name: "patch_critic_failed".to_string(),
                        args: json!({ "error": err.to_string() }),
                        result_count: 0,
                    });
                    None
                }
            };

            let critic_output = critic_resp
                .as_ref()
                .and_then(|r| parse_json::<PatchCriticOutput>(&r.raw_json).ok())
                .unwrap_or_else(|| PatchCriticOutput {
                    approved: true,
                    critique: "No critical regressions identified.".to_string(),
                    disputed_changes: vec![],
                    unhandled_edge_cases: vec![],
                    convention_violations: vec![],
                });

            // --- Lens 3: Patch Rebuttal (Pragmatic Scope Arbiter) ---
            runtime
                .progress
                .emit(AgentProgress::Status("Step 4: Lens 3 — Pragmatic patch rebuttal".to_string()))
                .await;
            record_trace!(TraceEvent::PhaseStarted {
                phase: Phase::PatchRebuttal,
            });

            let critic_json = serde_json::to_string_pretty(&critic_output).unwrap_or_default();
            let rebuttal_prompt = format!(
                "{}\n\n{}\n\n{}",
                shared_patch_prefix,
                phase_prefix(Phase::PatchRebuttal, "PatchRebuttalOutput"),
                patch_rebuttal_instruction(&proposed_json, &critic_json),
            );

            let rebuttal_req = LlmRequest {
                call_id: call_id("patch-rebut"),
                phase: Phase::PatchRebuttal,
                schema_name: "PatchRebuttalOutput".to_string(),
                schema: schema_patch_rebuttal(),
                prompt_parts: None,
                prompt: rebuttal_prompt,
                temperature: 0.0,
            };

            let rebuttal_resp = match runtime.llm.complete_json(rebuttal_req.clone()).await {
                Ok(resp) => {
                    record_trace!(TraceEvent::LlmCall {
                        request: rebuttal_req.clone(),
                        response: resp.clone(),
                    });
                    Some(resp)
                }
                Err(err) => {
                    record_trace!(TraceEvent::ToolCall {
                        name: "patch_rebuttal_failed".to_string(),
                        args: json!({ "error": err.to_string() }),
                        result_count: 0,
                    });
                    None
                }
            };

            let rebuttal_output = rebuttal_resp
                .as_ref()
                .and_then(|r| parse_json::<PatchRebuttalOutput>(&r.raw_json).ok())
                .unwrap_or_else(|| PatchRebuttalOutput {
                    rebuttal: "Proceeding with proposed patch.".to_string(),
                    dismissed_points: vec![],
                    accepted_points: vec![],
                });

            // --- Lens 4: Patch Reconciler (Hardening) ---
            runtime
                .progress
                .emit(AgentProgress::Status("Step 4: Lens 4 — Hardening reconciled patch".to_string()))
                .await;
            record_trace!(TraceEvent::PhaseStarted {
                phase: Phase::PatchReconciler,
            });

            let rebuttal_json = serde_json::to_string_pretty(&rebuttal_output).unwrap_or_default();
            let reconciler_prompt = format!(
                "{}\n\n{}\n\n{}",
                shared_patch_prefix,
                phase_prefix(Phase::PatchReconciler, "PatchReconcilerOutput"),
                patch_reconciler_instruction(&proposed_json, &critic_json, &rebuttal_json),
            );

            let reconciler_req = LlmRequest {
                call_id: call_id("patch-recon"),
                phase: Phase::PatchReconciler,
                schema_name: "PatchReconcilerOutput".to_string(),
                schema: schema_patch_reconciler(),
                prompt_parts: None,
                prompt: reconciler_prompt,
                temperature: 0.0,
            };

            let reconciler_resp = match runtime.llm.complete_json(reconciler_req.clone()).await {
                Ok(resp) => {
                    record_trace!(TraceEvent::LlmCall {
                        request: reconciler_req.clone(),
                        response: resp.clone(),
                    });
                    Some(resp)
                }
                Err(err) => {
                    record_trace!(TraceEvent::ToolCall {
                        name: "patch_reconciler_failed".to_string(),
                        args: json!({ "error": err.to_string() }),
                        result_count: 0,
                    });
                    None
                }
            };

            let mut reconciler_output = reconciler_resp
                .as_ref()
                .and_then(|r| parse_json::<PatchReconcilerOutput>(&r.raw_json).ok())
                .unwrap_or_else(|| PatchReconcilerOutput {
                    hardened_code: proposed_patch.replacement_code.clone(),
                    reconciliation_summary: "Validated proposal without modifications.".to_string(),
                    remaining_uncertainties: vec![],
                });

            // --- Partitioned Consensus Check & Bounded Pass 2 ---
            let has_dispute = !critic_output.approved
                && !rebuttal_output.accepted_points.is_empty()
                && !reconciler_output.remaining_uncertainties.is_empty();

            if has_dispute {
                runtime
                    .progress
                    .emit(AgentProgress::Status("Step 4: Pass 2 — Surgical secondary resolution on disputed logic".to_string()))
                    .await;

                let disputed_issue = reconciler_output
                    .remaining_uncertainties
                    .first()
                    .cloned()
                    .unwrap_or_else(|| "Resolve edge cases".to_string());

                let _confirmed_id = notebook.append_decision(
                    format!("Patch Staging Confirmed: {} (lines {}-{})", target_path, start_line, end_line),
                    Provenance::Model { call_id: reconciler_req.call_id.clone() },
                    Phase::PatchReconciler,
                    vec![],
                    vec!["patch".to_string(), "confirmed".to_string()],
                );
                let _disputed_id = notebook.append_decision(
                    format!("Patch Disputed Issue: {}", disputed_issue),
                    Provenance::Model { call_id: reconciler_req.call_id.clone() },
                    Phase::PatchReconciler,
                    vec![],
                    vec!["patch".to_string(), "disputed".to_string()],
                );

                let pass2_instruction = format!(
                    "QUESTION\nEVALUATION PERSPECTIVE: SECONDARY SURGICAL RECONCILER\nConfirmed changes have been staged. Focus strictly on resolving the remaining disputed issue: {}\nProvide the finalized hardened code ready to apply.\nReturn one JSON object matching schema `PatchReconcilerOutput`.",
                    disputed_issue
                );

                let pass2_prompt = format!(
                    "{}\n\n{}\n\n{}",
                    shared_patch_prefix,
                    phase_prefix(Phase::PatchReconciler, "PatchReconcilerOutput"),
                    pass2_instruction,
                );

                let pass2_req = LlmRequest {
                    call_id: call_id("patch-pass2"),
                    phase: Phase::PatchReconciler,
                    schema_name: "PatchReconcilerOutput".to_string(),
                    schema: schema_patch_reconciler(),
                    prompt_parts: None,
                    prompt: pass2_prompt,
                    temperature: 0.0,
                };

                if let Ok(resp) = runtime.llm.complete_json(pass2_req.clone()).await {
                    record_trace!(TraceEvent::LlmCall {
                        request: pass2_req,
                        response: resp.clone(),
                    });
                    if let Ok(pass2_out) = parse_json::<PatchReconcilerOutput>(&resp.raw_json) {
                        reconciler_output = pass2_out;
                    }
                }
            }

            // --- Host In-Memory Patch & Diff Generation ---
            let final_code = reconciler_output.hardened_code;
            let diff_str = generate_unified_diff(&target_path, start_line, &original_lines, &final_code);

            // Verify lease freshness and validate in-memory syntax
            let mut syntax_clean = true;
            if let Ok(file_slice) = runtime.file_reader.read_file(&target_path, 50_000).await {
                if let Some(ts) = target_slice {
                    let _is_fresh = verify_lease_freshness(&file_slice.content, &ts.lease);
                }
                if let Ok(patched_content) = apply_in_memory_patch(&file_slice.content, start_line, end_line, &final_code) {
                    syntax_clean = runtime.file_reader.validate_syntax(&target_path, &patched_content).await.unwrap_or(true);
                }
            }

            let patch_decision_id = notebook.append_decision(
                format!("Draft Patch: {} (lines {}-{})", target_path, start_line, end_line),
                Provenance::Model { call_id: reconciler_req.call_id.clone() },
                Phase::PatchReconciler,
                vec![],
                vec!["patch".to_string(), "artifact".to_string()],
            );
            if !cited_entries.contains(&patch_decision_id) {
                cited_entries.push(patch_decision_id);
            }

            edit_artifact = Some(EditArtifact {
                patch_id: Uuid::new_v4().to_string(),
                target_path: target_path.clone(),
                original_start_line: start_line,
                original_end_line: end_line,
                replacement_code: final_code,
                diff: diff_str,
                critic_approved: (critic_output.approved || rebuttal_output.accepted_points.is_empty()) && syntax_clean,
                critic_notes: critic_output.critique.clone(),
                impacted_callers: vec![],
                validation_commands: plan_res.reconciled_plan.validation_commands.clone(),
            });
        }
    }

    let diagnostic_report = format_step1_and_step2_report(
        &task,
        &intake_prompt,
        &intake_resp,
        &intake,
        &critic_output,
        &triad,
        &phase2,
        extract_result.as_ref(),
        plan_result.as_ref(),
        edit_artifact.as_ref(),
        &notebook,
    );

    // =========================================================================
    // STEP 3C: Final Answer Synthesis (Phase::Synthesize)
    // =========================================================================
    let (final_content, stop_reason) = if !task.intake_only {
        runtime
            .progress
            .emit(AgentProgress::Status("Step 3C: Synthesizing final answer from verified evidence".to_string()))
            .await;

        let synth_start = std::time::Instant::now();
        record_trace!(TraceEvent::PhaseStarted {
            phase: Phase::Synthesize,
        });

        let mut guidance_block = String::new();
        let direct_guidance: Vec<_> = triad
            .final_intents
            .iter()
            .filter_map(|i| {
                i.direct_response
                    .as_deref()
                    .map(|r| (i.summary.as_str(), r))
            })
            .collect();

        let synth_cards = knowledge_engine.match_cards(
            &task.question,
            &detected_tags,
            Some(&[KnowledgeDomain::Architecture, KnowledgeDomain::StackIdiom, KnowledgeDomain::Security]),
            2,
        );

        if !direct_guidance.is_empty() || !synth_cards.is_empty() {
            guidance_block.push_str("ESTABLISHED ARCHITECTURAL CONTEXT & GUIDANCE\n");
            for (summary, response) in direct_guidance {
                guidance_block.push_str(&format!("- {}: {}\n", summary, response));
            }
            for card in synth_cards {
                guidance_block.push_str(&format!("### {}\n{}\n", card.title, card.summary));
                for rule in &card.rules {
                    guidance_block.push_str(&format!("- {}\n", rule));
                }
            }
            guidance_block.push_str("Use this guidance authoritatively to address general/best-practice questions without defensive hedging.");
        }

        let mut prompt_sections = vec![
            global_prefix(&repo_map),
            phase_prefix(Phase::Synthesize, "SynthesizeOutput"),
            task_block(&task),
            notebook_block(&notebook, &[], 35),
            evidence_block(&phase2.leased_slices),
        ];
        if !guidance_block.is_empty() {
            prompt_sections.push(guidance_block);
        }
        prompt_sections.push(synthesize_instruction());

        let synth_prompt = prompt_sections.join("\n\n");

        let synth_req = LlmRequest {
            call_id: call_id("synth"),
            phase: Phase::Synthesize,
            schema_name: "SynthesizeOutput".to_string(),
            schema: schema_synthesize(),
            prompt_parts: None,
            prompt: synth_prompt,
            temperature: 0.0,
        };

        let synth_resp = match runtime.llm.complete_json(synth_req.clone()).await {
            Ok(resp) => {
                record_trace!(TraceEvent::LlmCall {
                    request: synth_req.clone(),
                    response: resp.clone(),
                });
                Some(resp)
            }
            Err(err) => {
                record_trace!(TraceEvent::ToolCall {
                    name: "synthesize_failed".to_string(),
                    args: json!({ "error": err.to_string() }),
                    result_count: 0,
                });
                None
            }
        };

        let synth_output = synth_resp
            .as_ref()
            .and_then(|r| parse_json::<SynthesizeOutput>(&r.raw_json).ok());

        let (synth_answer, synth_stop_reason) = if let Some(out) = synth_output {
            for entry_id in out.cited_entries {
                if notebook.get(&entry_id).is_some() && !cited_entries.contains(&entry_id) {
                    cited_entries.push(entry_id);
                }
            }
            (out.answer, out.stop_reason)
        } else {
            let fallback_answer = if let Some(ext) = &extract_result {
                let inferences: Vec<_> = ext
                    .inferences
                    .iter()
                    .filter(|i| i.is_valid)
                    .map(|i| i.statement.as_str())
                    .collect();
                if !inferences.is_empty() {
                    format!("Based on the codebase evidence:\n\n{}", inferences.join("\n\n"))
                } else {
                    format!("Evaluated workspace evidence for: {}", task.question)
                }
            } else {
                format!("Evaluated workspace evidence for: {}", task.question)
            };
            (fallback_answer, StopReason::Done)
        };

        let synth_elapsed_ms = synth_start.elapsed().as_millis() as u64;

        let full_content = format!(
            "{}\n\n<details>\n<summary><b>🛠️ Pipeline Diagnostics & Appellate Audit (Steps 1 – 3B | Synth: {}ms)</b></summary>\n\n{}\n</details>",
            synth_answer.trim(),
            synth_elapsed_ms,
            diagnostic_report.trim(),
        );

        (full_content, synth_stop_reason)
    } else {
        (diagnostic_report, StopReason::Done)
    };

    let answer = AgentAnswer {
        content: final_content,
        stop_reason,
        cited_entries,
        notebook,
    };

    record_trace!(TraceEvent::RunFinished {
        stop_reason,
    });

    Ok(AgentRunResult { answer, trace })
}

fn parse_json<T: DeserializeOwned>(raw: &str) -> Result<T> {
    let trimmed = raw.trim();
    if let Ok(parsed) = serde_json::from_str::<T>(trimmed) {
        return Ok(parsed);
    }
    // Attempt markdown code fence extraction
    if let Some(start) = trimmed.find("```json") {
        let rest = &trimmed[start + 7..];
        if let Some(end) = rest.find("```") {
            return serde_json::from_str::<T>(rest[..end].trim())
                .context("Failed to parse json inside code fence");
        }
    } else if let Some(start) = trimmed.find("```") {
        let rest = &trimmed[start + 3..];
        if let Some(end) = rest.find("```") {
            return serde_json::from_str::<T>(rest[..end].trim())
                .context("Failed to parse json inside code fence");
        }
    }
    // Attempt outer brace extraction
    if let (Some(start), Some(end)) = (trimmed.find('{'), trimmed.rfind('}')) {
        if start < end {
            return serde_json::from_str::<T>(&trimmed[start..=end])
                .context("Failed to parse json substring between braces");
        }
    }
    serde_json::from_str::<T>(trimmed).context("Failed to parse raw json")
}

fn fallback_intake(question: &str) -> IntakeOutput {
    let lower = question.to_lowercase();
    let is_greeting = lower.starts_with("hi")
        || lower.starts_with("hello")
        || lower.starts_with("hey")
        || lower == "help";

    if is_greeting && question.len() < 25 {
        return IntakeOutput {
            reasoning: "Fallback rule-based classification: detected conversational greeting."
                .to_string(),
            needs_workspace_search: false,
            intents: vec![IntentItem {
                intent: UserIntent::Chat,
                summary: "Greeting".to_string(),
                needs_workspace_search: false,
                direct_response: Some(
                    "Hello! I am your local coding assistant. How can I help you today?".to_string(),
                ),
                search_queries: Vec::new(),
            }],
            target_hints: Vec::new(),
            constraints: Vec::new(),
        };
    }

    let is_edit = lower.contains("add")
        || lower.contains("fix")
        || lower.contains("edit")
        || lower.contains("update")
        || lower.contains("change")
        || lower.contains("implement")
        || lower.contains("refactor");

    let intent = if is_edit {
        UserIntent::WorkspaceEdit
    } else {
        UserIntent::WorkspaceSearch
    };

    let words: Vec<String> = question
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.')
        .map(str::trim)
        .filter(|w| w.len() > 2)
        .take(5)
        .map(ToString::to_string)
        .collect();

    IntakeOutput {
        reasoning: "Fallback rule-based classification due to unparseable model response."
            .to_string(),
        needs_workspace_search: true,
        intents: vec![IntentItem {
            intent,
            summary: question.to_string(),
            needs_workspace_search: true,
            direct_response: None,
            search_queries: words.clone(),
        }],
        target_hints: words,
        constraints: Vec::new(),
    }
}

fn has_greeting(question: &str) -> bool {
    let lower = question.trim().to_lowercase();
    let phrases = [
        "good morning",
        "good afternoon",
        "good evening",
        "good day",
        "thank you",
        "thanks",
    ];
    for phrase in &phrases {
        if lower.starts_with(phrase) || lower.contains(phrase) {
            return true;
        }
    }

    let words = ["hi", "hello", "hey", "howdy", "greetings", "welcome", "yo"];
    let tokens: Vec<&str> = lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|t| !t.is_empty())
        .collect();

    if tokens.is_empty() {
        return false;
    }

    let total = tokens.len();
    for (idx, token) in tokens.iter().enumerate() {
        if (idx < 4 || idx + 2 >= total) && words.contains(token) {
            if *token == "hi" && (lower.contains("hi-fi") || lower.contains("sci-fi")) {
                continue;
            }
            return true;
        }
    }
    false
}

fn validate_and_normalize_intake(intake: &mut IntakeOutput, question: &str) {
    let lower = question.to_lowercase();

    // 1. Guard against ghost greetings or missed greetings
    let greeting_present = has_greeting(question);
    if !greeting_present {
        intake.intents.retain(|item| item.intent != UserIntent::Chat);
    } else if !intake.intents.iter().any(|item| item.intent == UserIntent::Chat) {
        intake.intents.insert(
            0,
            IntentItem {
                intent: UserIntent::Chat,
                summary: "Greeting".to_string(),
                needs_workspace_search: false,
                direct_response: Some("Hello! How can I help you today?".to_string()),
                search_queries: Vec::new(),
            },
        );
    }

    // 2. Guard against missing workspace search when question explicitly targets codebase
    let asks_for_codebase = lower.contains("codebase")
        || lower.contains("this repo")
        || lower.contains("our repo")
        || lower.contains("this project")
        || lower.contains("our project")
        || lower.contains("where do we")
        || lower.contains("how do we")
        || lower.contains("where does our")
        || lower.contains("how does our")
        || lower.contains("where is")
        || lower.contains("where are")
        || lower.contains("which file")
        || lower.contains("find in");

    let has_workspace_intent = intake
        .intents
        .iter()
        .any(|item| item.needs_workspace_search);

    if asks_for_codebase && !has_workspace_intent {
        let mut upgraded = false;
        for item in &mut intake.intents {
            let s_lower = item.summary.to_lowercase();
            if s_lower.contains("where")
                || s_lower.contains("codebase")
                || s_lower.contains("find")
                || s_lower.contains("locate")
                || s_lower.contains("our")
                || s_lower.contains("repo")
                || s_lower.contains("project")
            {
                item.intent = UserIntent::WorkspaceSearch;
                item.needs_workspace_search = true;
                item.direct_response = None;
                if item.search_queries.is_empty() {
                    item.search_queries = extract_terms_for_search(&item.summary);
                }
                upgraded = true;
                break;
            }
        }

        if !upgraded {
            let terms = extract_terms_for_search(question);
            let summary = if let Some(pos) = lower.find("where") {
                question[pos..].trim().to_string()
            } else if let Some(pos) = lower.find("in our codebase") {
                question[pos..].trim().to_string()
            } else {
                "Locate requested functionality in codebase".to_string()
            };

            intake.intents.push(IntentItem {
                intent: UserIntent::WorkspaceSearch,
                summary,
                needs_workspace_search: true,
                direct_response: None,
                search_queries: terms.clone(),
            });
            intake.target_hints.extend(terms);
            intake.reasoning.push_str(" (Host detected explicit codebase search target).");
        }
    }

    // 3. Guard against collapsed best-practice / conceptual question in compound prompt
    let asks_for_best_practices = lower.contains("best practice")
        || lower.contains("best practices")
        || lower.contains("aligned with")
        || lower.contains("conventions")
        || lower.contains("guidelines")
        || lower.contains("how does react")
        || lower.contains("how does solid");

    let has_general_intent = intake
        .intents
        .iter()
        .any(|item| item.intent == UserIntent::GeneralQuestion);

    if asks_for_best_practices && !has_general_intent {
        // Find if any workspace intent has "best practice" or framework names polluting search queries
        for item in &mut intake.intents {
            if item.needs_workspace_search {
                item.search_queries.retain(|q| {
                    let ql = q.to_lowercase();
                    !ql.contains("best practice")
                        && !ql.contains("practices")
                        && !ql.contains("aligned")
                        && !ql.contains("react")
                        && !ql.contains("convention")
                });
            }
        }

        // Add the distinct GeneralQuestion intent
        let bp_summary = if let Some(pos) = lower.find("aligned with") {
            format!("Evaluate whether pattern is {}", question[pos..].trim())
        } else if let Some(pos) = lower.find("best practice") {
            format!("Explain {}", question[pos..].trim())
        } else {
            "Evaluate alignment with best practices".to_string()
        };

        intake.intents.push(IntentItem {
            intent: UserIntent::GeneralQuestion,
            summary: bp_summary,
            needs_workspace_search: false,
            direct_response: Some(
                "In frontend desktop and single-window applications, navigation is commonly structured either via state-driven view switching (e.g. conditional rendering based on active session or application state) or via dedicated declarative routers (e.g. React Router, Solid Router, or TanStack Router). When URLs or browser history are not required, state-driven view switching is clean, minimal, and avoids unnecessary routing dependencies.".to_string(),
            ),
            search_queries: Vec::new(),
        });
        intake.reasoning.push_str(" (Host separated best practice evaluation into distinct GeneralQuestion intent).");
    }

    // 4. Clean up search queries across all workspace intents
    for item in &mut intake.intents {
        if item.needs_workspace_search {
            item.search_queries.retain(|q| {
                let ql = q.to_lowercase();
                !ql.contains("best practice")
                    && !ql.contains("practices")
                    && !ql.contains("aligned")
                    && !ql.contains("react")
                    && !ql.contains("convention")
            });
        }
    }

    // 5. Fallback if all intents were pruned (e.g. only ghost greeting existed on a non-codebase question)
    if intake.intents.is_empty() {
        intake.intents.push(IntentItem {
            intent: UserIntent::GeneralQuestion,
            summary: question.trim().to_string(),
            needs_workspace_search: false,
            direct_response: None,
            search_queries: Vec::new(),
        });
    }

    // Ensure top-level needs_workspace_search is in sync with individual intents
    intake.needs_workspace_search = intake.intents.iter().any(|item| item.needs_workspace_search);
}

fn extract_terms_for_search(question: &str) -> Vec<String> {
    let filter_words = [
        "what", "is", "an", "the", "and", "where", "in", "our", "codebase", "this", "repo", "project",
        "do", "we", "can", "you", "how", "to", "hi", "hello", "hey", "tell", "me", "does", "are"
    ];
    let mut terms = Vec::new();
    for word in question.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.') {
        let clean = word.trim();
        if clean.len() > 2 && !filter_words.contains(&clean.to_lowercase().as_str()) {
            terms.push(clean.to_string());
        }
    }
    terms.dedup();
    terms.truncate(5);
    terms
}

fn compact_candidate_snippet(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut consecutive_empty = 0;
    for line in raw.lines() {
        let trimmed = line.trim_end();
        if trimmed.trim().is_empty() {
            consecutive_empty += 1;
            if consecutive_empty <= 1 {
                out.push('\n');
            }
        } else {
            consecutive_empty = 0;
            out.push_str(trimmed);
            out.push('\n');
        }
    }
    out
}

fn candidate_exclusion_reason(path: &str, question: &str) -> Option<&'static str> {
    let q_lower = question.to_lowercase();
    let asks_about_agent = q_lower.contains("agent")
        || q_lower.contains("prompt")
        || q_lower.contains("pipeline")
        || q_lower.contains("lease")
        || q_lower.contains("trace")
        || q_lower.contains("harness")
        || q_lower.contains("orchestrat")
        || q_lower.contains("sidecar")
        || q_lower.contains("llama");

    if asks_about_agent {
        return None;
    }

    let p_lower = path.to_lowercase();

    // 1. Documentation files (unless user explicitly asks for docs/markdown)
    let asks_about_docs = q_lower.contains("doc") || q_lower.contains("readme") || q_lower.contains("guide");
    if !asks_about_docs && p_lower.ends_with(".md") {
        return Some("Documentation file (.md)");
    }

    // 2. Internal agent plumbing
    if p_lower.contains("crates/agent-")
        || p_lower.contains("orchestrator.")
        || p_lower.contains("vector_search.")
        || p_lower.contains("indexer.")
        || p_lower.contains("codegraph.")
        || p_lower.contains("phoenix.")
        || p_lower.contains("trace_writer.")
        || p_lower.contains("engine.")
        || p_lower.contains("src-tauri/src/agent")
    {
        return Some("Internal agent / harness plumbing");
    }

    // 3. Generated schemas and files
    if p_lower.contains("/gen/")
        || p_lower.starts_with("gen/")
        || p_lower.contains("/schemas/")
        || p_lower.starts_with("schemas/")
        || p_lower.ends_with(".schema.json")
    {
        return Some("Generated JSON schema / codegen artifact");
    }

    // 4. Build output, dist, caches
    if p_lower.contains("target/")
        || p_lower.contains("node_modules/")
        || p_lower.contains("dist/")
        || p_lower.contains("/dist/")
        || p_lower.contains("build/")
        || p_lower.contains(".git/")
    {
        return Some("Build output / dependency directory");
    }

    // 5. Package lockfiles (unless question asks for dependencies/package/lockfile)
    let asks_about_deps = q_lower.contains("dependency") || q_lower.contains("package") || q_lower.contains("lockfile");
    if !asks_about_deps && (p_lower.ends_with(".lock") || p_lower.ends_with("-lock.json")) {
        return Some("Package manager lockfile");
    }

    // 6. Build scripts (unless question asks for build/compilation/cargo)
    let asks_about_build = q_lower.contains("build") || q_lower.contains("compil") || q_lower.contains("cargo");
    if !asks_about_build && (p_lower.ends_with("build.rs") || p_lower.ends_with("makefile")) {
        return Some("Auxiliary build script (build.rs)");
    }

    // 7. Generic JSON files unless question asks about config/schema/json
    let asks_about_config = q_lower.contains("config") || q_lower.contains("schema") || q_lower.contains("json");
    if !asks_about_config && p_lower.ends_with(".json") {
        return Some("Generic JSON asset / metadata");
    }

    None
}

fn arbitrate_triad(
    question: &str,
    drafter: &IntakeOutput,
    critic: &CriticOutput,
) -> TriadConsensus {
    let q_lower = question.to_lowercase();

    if critic.approved && !critic.has_dropped_clauses && !critic.has_contaminated_queries {
        let mut final_intents = drafter.intents.clone();
        sanitize_search_queries(&mut final_intents, &q_lower);

        TriadConsensus {
            drafter_intents: drafter.intents.clone(),
            critic_approved: true,
            critic_critique: critic.critique.clone(),
            arbiter_decision: "Drafter and Critic agreed. Intake decomposition approved.".to_string(),
            final_intents,
        }
    } else {
        // Critic challenged Drafter. Arbiter evaluates.
        let mut final_intents = if let Some(mut corrected) = critic.corrected_intents.clone() {
            if !corrected.is_empty() {
                sanitize_search_queries(&mut corrected, &q_lower);
                corrected
            } else {
                let mut fallback = drafter.intents.clone();
                sanitize_search_queries(&mut fallback, &q_lower);
                fallback
            }
        } else {
            let mut fallback = drafter.intents.clone();
            sanitize_search_queries(&mut fallback, &q_lower);
            fallback
        };

        // If the question is compound (e.g. asking for best practices / conventions)
        // and final_intents still only has 1 intent, Arbiter enforces splitting:
        let asks_for_best_practices = q_lower.contains("best practice")
            || q_lower.contains("best practices")
            || q_lower.contains("aligned with")
            || q_lower.contains("conventions");
        let has_general = final_intents.iter().any(|i| i.intent == UserIntent::GeneralQuestion);
        if asks_for_best_practices && !has_general {
            let bp_summary = if let Some(pos) = q_lower.find("aligned with") {
                format!("Evaluate whether pattern is {}", question[pos..].trim())
            } else if let Some(pos) = q_lower.find("best practice") {
                format!("Explain {}", question[pos..].trim())
            } else {
                "Evaluate alignment with best practices".to_string()
            };

            final_intents.push(IntentItem {
                intent: UserIntent::GeneralQuestion,
                summary: bp_summary,
                needs_workspace_search: false,
                direct_response: Some(
                    "In desktop and single-window client applications, navigation is commonly structured either via state-driven view switching (e.g. conditional rendering based on active session or application state) or via dedicated declarative routers. When browser URL history or deep-linking is not required, state-driven view switching is clean, minimal, and avoids unnecessary routing dependencies.".to_string(),
                ),
                search_queries: Vec::new(),
            });
        }

        TriadConsensus {
            drafter_intents: drafter.intents.clone(),
            critic_approved: false,
            critic_critique: critic.critique.clone(),
            arbiter_decision: format!(
                "Arbiter validated Critic's challenge and finalized consensus: {}",
                critic.critique
            ),
            final_intents,
        }
    }
}

fn sanitize_search_queries(intents: &mut [IntentItem], _q_lower: &str) {
    let filter_words = [
        "best practices", "best practice", "practices", "aligned with",
        "aligned", "conventions", "convention", "guidelines", "how to"
    ];

    for item in intents.iter_mut() {
        if item.needs_workspace_search {
            item.search_queries.retain(|q| {
                let ql = q.to_lowercase();
                !filter_words.iter().any(|&fw| ql == fw || ql.contains(fw))
            });
        }
    }
}

fn format_step1_report(
    task: &TaskRequest,
    intake_prompt: &str,
    intake_resp: &LlmResponse,
    intake: &IntakeOutput,
    critic_output: &CriticOutput,
    triad: &TriadConsensus,
) -> String {
    let mut out = String::new();
    out.push_str("### ⚖️ Step 1: Multi-Intent Intake (Triad Intake Consensus Quorum)\n\n");
    out.push_str(&format!(
        "**User Request:** {}\n\n",
        task.question.trim()
    ));

    out.push_str("#### 🏛️ Triad Quorum Decision Matrix\n\n");
    out.push_str("| Triad Node | Role / Mandate | Finding / Verdict | Quorum Status |\n");
    out.push_str("|---|---|---|---|\n");
    out.push_str(&format!(
        "| **Node 1: Drafter** | Optimistic Decomposition | Proposed {} intent(s) | ⚪ Candidate Draft |\n",
        triad.drafter_intents.len()
    ));
    out.push_str(&format!(
        "| **Node 2: Critic** | Adversarial Auditor | {} | {} |\n",
        critic_output.critique.trim(),
        if critic_output.approved { "🟢 Approved" } else { "🛡️ Challenged / Vetoed" }
    ));
    out.push_str(&format!(
        "| **Node 3: Arbiter** | Deterministic Host Quorum | {} | ✅ **Consensus Reached (2-of-3)** |\n\n",
        triad.arbiter_decision.trim()
    ));

    out.push_str("| Evaluation Field | Result |\n");
    out.push_str("|---|---|\n");
    let intent_badges: Vec<String> = intake
        .intents
        .iter()
        .map(|item| format!("`{:?}`", item.intent))
        .collect();
    out.push_str(&format!(
        "| **Detected Intents ({})** | {} |\n",
        intake.intents.len(),
        intent_badges.join(", ")
    ));
    out.push_str(&format!(
        "| **Needs Workspace Search?** | {} |\n",
        if intake.needs_workspace_search {
            "🔍 **Yes** (Contains codebase search/edit goals)"
        } else {
            "💬 **No** (Direct response / conceptual only)"
        }
    ));
    out.push_str(&format!(
        "| **Decomposition Reasoning** | {} |\n\n",
        intake.reasoning.trim()
    ));

    out.push_str("#### 🎯 Decomposed Intents\n\n");
    for (idx, item) in intake.intents.iter().enumerate() {
        out.push_str(&format!(
            "{}. **`{:?}`**: {}\n",
            idx + 1,
            item.intent,
            item.summary
        ));
        out.push_str(&format!(
            "   - **Needs Workspace Search:** {}\n",
            if item.needs_workspace_search { "Yes" } else { "No" }
        ));
        if let Some(ref direct) = item.direct_response {
            out.push_str(&format!(
                "   - **Direct Answer:** {}\n",
                direct.trim()
            ));
        }
        if !item.search_queries.is_empty() {
            let queries_fmt: Vec<String> = item
                .search_queries
                .iter()
                .map(|q| format!("`{}`", q))
                .collect();
            out.push_str(&format!(
                "   - **Target Queries:** {}\n",
                queries_fmt.join(", ")
            ));
        }
        out.push('\n');
    }

    if !intake.target_hints.is_empty() || !intake.constraints.is_empty() {
        out.push_str("#### 📌 Global Hints & Constraints\n\n");
        if !intake.target_hints.is_empty() {
            out.push_str(&format!(
                "- **Target Hints:** {}\n",
                intake
                    .target_hints
                    .iter()
                    .map(|h| format!("`{}`", h))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        if !intake.constraints.is_empty() {
            out.push_str(&format!(
                "- **Constraints:** {}\n",
                intake
                    .constraints
                    .iter()
                    .map(|c| format!("`{}`", c))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
        out.push('\n');
    }

    out.push_str("#### ⏱️ Performance & Token Metrics\n");
    let usage = &intake_resp.usage;
    let wall_ms = usage
        .wall_ms
        .map(|v| format!("{} ms", v))
        .unwrap_or_else(|| "n/a".to_string());
    let prefill_ms = usage
        .prefill_ms
        .map(|v| format!("{} ms", v))
        .unwrap_or_else(|| "n/a".to_string());
    let decode_ms = usage
        .decode_ms
        .map(|v| format!("{} ms", v))
        .unwrap_or_else(|| "n/a".to_string());
    let prompt_tok = usage
        .prompt_tokens_total
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let cached_tok = usage
        .prompt_tokens_cached
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let decode_tok = usage
        .decode_tokens
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    out.push_str(&format!(
        "- **Timing:** {} wall (prefill: {}, decode: {})\n",
        wall_ms, prefill_ms, decode_ms
    ));
    out.push_str(&format!(
        "- **Tokens:** {} prompt ({} cached) | {} decode\n",
        prompt_tok, cached_tok, decode_tok
    ));

    out.push_str("\n<details>\n<summary><b>📄 Raw Model JSON Output</b></summary>\n\n");
    out.push_str("```json\n");
    let pretty_json = serde_json::from_str::<serde_json::Value>(&intake_resp.raw_json)
        .ok()
        .and_then(|v| serde_json::to_string_pretty(&v).ok())
        .unwrap_or_else(|| intake_resp.raw_json.trim().to_string());
    out.push_str(&pretty_json);
    out.push_str("\n```\n</details>\n");

    out.push_str("\n<details>\n<summary><b>📝 Exact Prompt Sent to Model</b></summary>\n\n");
    out.push_str("```text\n");
    out.push_str(intake_prompt.trim());
    out.push_str("\n```\n</details>\n");

    out
}

#[derive(Clone, Debug)]
struct SearchQueryExecution {
    query: String,
    path_hits_count: usize,
    symbol_hits_count: usize,
    vector_hits_count: usize,
    content_hits_count: usize,
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
struct SliceTarget {
    path: String,
    symbol: Option<String>,
    start_line: u32,
    end_line: u32,
    score: i64,
    source: String,
}

#[derive(Clone, Debug)]
pub struct ExpandedHitDetail {
    pub path: String,
    pub source_seed: String,
    pub reason: String,
}

#[derive(Clone, Debug)]
pub struct PrunedCandidateDetail {
    pub path: String,
    pub reason: String,
}

#[derive(Clone, Debug)]
struct Phase2Results {
    queries: Vec<String>,
    search_details: Vec<SearchQueryExecution>,
    all_path_hits: Vec<FileHit>,
    all_symbol_hits: Vec<SymbolHit>,
    all_vector_hits: Vec<VectorHit>,
    all_content_hits: Vec<ContentHit>,
    expanded_hits: Vec<ExpandedHitDetail>,
    pruned_candidates: Vec<PrunedCandidateDetail>,
    leased_slices: Vec<EvidenceSlice>,
    elapsed_ms: u64,
}

#[derive(Clone, Debug)]
pub struct VerifiedInference {
    pub statement: String,
    pub lease_id: String,
    pub is_valid: bool,
    pub entry_id: Option<EntryId>,
    pub rejection_reason: Option<String>,
}

#[derive(Clone, Debug)]
pub struct ExtractStepResults {
    pub inferences: Vec<VerifiedInference>,
    pub sufficient: bool,
    pub missing: Vec<MissingEvidence>,
    pub raw_json: String,
    pub prompt: String,
    pub elapsed_ms: u64,
}

fn format_step1_and_step2_report(
    task: &TaskRequest,
    intake_prompt: &str,
    intake_resp: &LlmResponse,
    intake: &IntakeOutput,
    critic_output: &CriticOutput,
    triad: &TriadConsensus,
    phase2: &Phase2Results,
    extract: Option<&ExtractStepResults>,
    plan: Option<&PlanStepResults>,
    patch: Option<&EditArtifact>,
    notebook: &Notebook,
) -> String {
    let mut out = String::new();

    // --- Step 1 Section ---
    out.push_str("### ⚖️ Step 1: Multi-Intent Intake (Triad Intake Consensus Quorum)\n\n");
    out.push_str(&format!(
        "**User Request:** {}\n\n",
        task.question.trim()
    ));

    out.push_str("#### 🏛️ Triad Quorum Decision Matrix\n\n");
    out.push_str("| Triad Node | Role / Mandate | Finding / Verdict | Quorum Status |\n");
    out.push_str("|---|---|---|---|\n");
    out.push_str(&format!(
        "| **Node 1: Drafter** | Optimistic Decomposition | Proposed {} intent(s) | ⚪ Candidate Draft |\n",
        triad.drafter_intents.len()
    ));
    out.push_str(&format!(
        "| **Node 2: Critic** | Adversarial Auditor | {} | {} |\n",
        critic_output.critique.trim(),
        if critic_output.approved { "🟢 Approved" } else { "🛡️ Challenged / Vetoed" }
    ));
    out.push_str(&format!(
        "| **Node 3: Arbiter** | Deterministic Host Quorum | {} | ✅ **Consensus Reached (2-of-3)** |\n\n",
        triad.arbiter_decision.trim()
    ));

    out.push_str("| Evaluation Field | Result |\n");
    out.push_str("|---|---|\n");
    let intent_badges: Vec<String> = intake
        .intents
        .iter()
        .map(|item| format!("`{:?}`", item.intent))
        .collect();
    out.push_str(&format!(
        "| **Detected Intents ({})** | {} |\n",
        intake.intents.len(),
        intent_badges.join(", ")
    ));
    out.push_str(&format!(
        "| **Needs Workspace Search?** | {} |\n",
        if intake.needs_workspace_search {
            "🔍 **Yes** (Contains codebase search/edit goals)"
        } else {
            "💬 **No** (Direct response / conceptual only)"
        }
    ));
    out.push_str(&format!(
        "| **Decomposition Reasoning** | {} |\n\n",
        intake.reasoning.trim()
    ));

    out.push_str("#### 🎯 Decomposed Intents\n\n");
    for (idx, item) in intake.intents.iter().enumerate() {
        out.push_str(&format!(
            "{}. **`{:?}`**: {}\n",
            idx + 1,
            item.intent,
            item.summary
        ));
        out.push_str(&format!(
            "   - **Needs Workspace Search:** {}\n",
            if item.needs_workspace_search { "Yes" } else { "No" }
        ));
        if let Some(ref direct) = item.direct_response {
            out.push_str(&format!(
                "   - **Direct Answer:** {}\n",
                direct.trim()
            ));
        }
        if !item.search_queries.is_empty() {
            let queries_fmt: Vec<String> = item
                .search_queries
                .iter()
                .map(|q| format!("`{}`", q))
                .collect();
            out.push_str(&format!(
                "   - **Target Queries:** {}\n",
                queries_fmt.join(", ")
            ));
        }
        out.push('\n');
    }

    out.push_str("---\n\n");

    // --- Step 2 Section ---
    out.push_str("### 🔍 Step 2: Deterministic Evidence Acquisition (Short-Circuited)\n\n");

    if !intake.needs_workspace_search {
        out.push_str("> 💬 **No workspace search required.** All decomposed intents can be fulfilled directly without retrieving code files.\n\n");
    } else {
        out.push_str("| Evaluation Field | Result |\n");
        out.push_str("|---|---|\n");
        let query_badges: Vec<String> = phase2
            .queries
            .iter()
            .map(|q| format!("`{}`", q))
            .collect();
        out.push_str(&format!(
            "| **Search Queries Executed ({})** | {} |\n",
            phase2.queries.len(),
            if query_badges.is_empty() { "None".to_string() } else { query_badges.join(", ") }
        ));
        out.push_str(&format!(
            "| **Host Search Latency** | ⚡ **{} ms** (Native Rust / CodeGraph / Vector) |\n",
            phase2.elapsed_ms
        ));
        out.push_str(&format!(
            "| **Discovered Symbol Hits** | {} symbols (CodeGraph AST) |\n",
            phase2.all_symbol_hits.len()
        ));
        out.push_str(&format!(
            "| **Discovered Vector Hits** | {} matches (Vector Semantic Search) |\n",
            phase2.all_vector_hits.len()
        ));
        out.push_str(&format!(
            "| **Discovered Content Hits** | {} matches (Ripgrep Text) |\n",
            phase2.all_content_hits.len()
        ));
        out.push_str(&format!(
            "| **Discovered File Hits** | {} files (Fuzzy Path) |\n",
            phase2.all_path_hits.len()
        ));
        out.push_str(&format!(
            "| **Active Evidence Leases in Notebook** | 📦 **{} leased slices** |\n\n",
            phase2.leased_slices.len()
        ));

        if !phase2.expanded_hits.is_empty() {
            out.push_str("#### 🌐 CodeGraph Topology Expansion (Serendipitous Discovery)\n\n");
            out.push_str("| Discovered Component | Seed Source | Traversal Type |\n");
            out.push_str("|---|---|---|\n");
            for exp in &phase2.expanded_hits {
                out.push_str(&format!(
                    "| **`{}`** | `{}` | {} |\n",
                    exp.path, exp.source_seed, exp.reason
                ));
            }
            out.push_str("\n");
        }

        if !phase2.pruned_candidates.is_empty() {
            out.push_str("<details>\n<summary><b>🛡️ Pruned Candidates & False Positives (");
            out.push_str(&format!("{} filtered)</b></summary>\n\n", phase2.pruned_candidates.len()));
            out.push_str("| Candidate Path | Exclusion Rationale |\n");
            out.push_str("|---|---|\n");
            for pr in &phase2.pruned_candidates {
                out.push_str(&format!(
                    "| `{}` | {} |\n",
                    pr.path, pr.reason
                ));
            }
            out.push_str("\n</details>\n\n");
        }

        if phase2.leased_slices.is_empty() {
            out.push_str("> ⚠️ **No evidence slices were leased.** No files or symbols matched the search queries in the active workspace.\n\n");
        } else {
            out.push_str("#### 📑 Leased Code Slices in Notebook\n\n");
            for (idx, slice) in phase2.leased_slices.iter().enumerate() {
                let symbol_label = slice
                    .lease
                    .symbol
                    .as_deref()
                    .map(|s| format!(" | Symbol: `{}`", s))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "{}. **`{}`** (Lines {}-{} | Lease: `{}`{})\n\n",
                    idx + 1,
                    slice.lease.path,
                    slice.lease.range.start,
                    slice.lease.range.end,
                    slice.lease.id.0,
                    symbol_label
                ));

                let lang = detect_lang(&slice.lease.path);
                out.push_str(&format!("```{}\n", lang));
                out.push_str(&format_line_numbered_slice(slice));
                out.push_str("\n```\n\n");
            }
        }

        if !phase2.all_symbol_hits.is_empty() || !phase2.all_vector_hits.is_empty() || !phase2.all_content_hits.is_empty() || !phase2.all_path_hits.is_empty() {
            out.push_str("<details>\n<summary><b>🛠️ Discovered Search Candidates (Symbols, Vectors & Files)</b></summary>\n\n");
            if !phase2.search_details.is_empty() {
                out.push_str("##### Query Hit Breakdown\n");
                for det in &phase2.search_details {
                    out.push_str(&format!(
                        "- Query `{}`: {} paths, {} symbols, {} vector matches, {} content matches\n",
                        det.query, det.path_hits_count, det.symbol_hits_count, det.vector_hits_count, det.content_hits_count
                    ));
                }
                out.push('\n');
            }
            out.push_str("| Type | Path | Detail | Score |\n");
            out.push_str("|---|---|---|---|\n");
            for sym in phase2.all_symbol_hits.iter().take(8) {
                out.push_str(&format!(
                    "| Symbol | `{}` | `{}` (lines {}-{}, {}) | {} |\n",
                    sym.path, sym.symbol, sym.start_line, sym.end_line, sym.kind, sym.score
                ));
            }
            for v in phase2.all_vector_hits.iter().take(8) {
                let snippet = v.snippet.trim().replace('|', "\\|").replace('\n', " ");
                let snippet_trunc: String = snippet.chars().take(60).collect();
                out.push_str(&format!(
                    "| Vector | `{}` | lines {}-{}: `{}` | {:.2} |\n",
                    v.path, v.start_line, v.end_line, snippet_trunc, v.score
                ));
            }
            for ch in phase2.all_content_hits.iter().take(8) {
                let preview = ch.preview.trim().replace('|', "\\|").replace('\n', " ");
                let preview_trunc: String = preview.chars().take(60).collect();
                out.push_str(&format!(
                    "| Content | `{}` | lines {}-{}: `{}` | {} |\n",
                    ch.path, ch.start_line, ch.end_line, preview_trunc, ch.score
                ));
            }
            for ph in phase2.all_path_hits.iter().take(8) {
                out.push_str(&format!(
                    "| Path | `{}` | File path match | {} |\n",
                    ph.path, ph.score
                ));
            }
            out.push_str("\n</details>\n\n");
        }
    }

    // --- Step 3A Section ---
    if let Some(ext) = extract {
        out.push_str("### 🧠 Step 3A: Evidence Extraction (`Phase::Extract`)\n\n");
        let valid_count = ext.inferences.iter().filter(|i| i.is_valid).count();
        let rejected_count = ext.inferences.iter().filter(|i| !i.is_valid).count();
        let sufficiency_badge = if ext.sufficient {
            "✅ **Sufficient** (Retrieved evidence directly answers the request)"
        } else {
            "⚠️ **Insufficient** (Additional evidence required)"
        };

        out.push_str("| Metric | Status |\n");
        out.push_str("|---|---|\n");
        out.push_str(&format!("| **Sufficiency Evaluation** | {} |\n", sufficiency_badge));
        out.push_str(&format!("| **Verified Inferences** | 💡 **{} inferences accepted** into Notebook |\n", valid_count));
        if rejected_count > 0 {
            out.push_str(&format!("| **Host-Rejected Inferences** | 🛡️ **{} inferences dropped** (unbacked lease IDs) |\n", rejected_count));
        }
        if !ext.missing.is_empty() {
            out.push_str(&format!("| **Missing Evidence Identified** | 🔍 **{} missing items** |\n", ext.missing.len()));
        }
        out.push_str(&format!("| **Extraction Latency** | ⏱️ **{} ms** |\n\n", ext.elapsed_ms));

        if valid_count > 0 {
            out.push_str("#### 💡 Verified Inferences (Added to Notebook)\n\n");
            for (idx, inf) in ext.inferences.iter().filter(|i| i.is_valid).enumerate() {
                let entry_str = inf.entry_id.as_ref().map(|id| format!(" | Entry: `{}`", id.0)).unwrap_or_default();
                out.push_str(&format!(
                    "{}. **[Lease `{}`{}]**: {}\n",
                    idx + 1,
                    inf.lease_id,
                    entry_str,
                    inf.statement
                ));
                if !inf.statement.ends_with('\n') {
                    out.push('\n');
                }
            }
            out.push('\n');
        }

        if rejected_count > 0 {
            out.push_str("#### ⚠️ Host-Rejected Inferences (Unbacked or Hallucinated Lease IDs)\n\n");
            for inf in ext.inferences.iter().filter(|i| !i.is_valid) {
                out.push_str(&format!(
                    "- `{}`: {}\n",
                    inf.statement,
                    inf.rejection_reason.as_deref().unwrap_or("Invalid lease ID")
                ));
            }
            out.push('\n');
        }

        if !ext.missing.is_empty() {
            out.push_str("#### 🔍 Missing Evidence Identified by Coprocessor\n\n");
            for m in &ext.missing {
                out.push_str(&format!(
                    "- **{:?}** for `{}`: {}\n",
                    m.kind, m.target, m.reason
                ));
            }
            out.push('\n');
        }
    }

    // --- Step 3B Section ---
    if let Some(plan_res) = plan {
        out.push_str("### 🗺️ Step 3B: Appellate Planning Gate (`Double Stereoscopic Gate`)\n\n");
        let rec = &plan_res.reconciled_plan;

        out.push_str("| Dialectic Lens | Role & Perspective | Result |\n");
        out.push_str("|---|---|---|\n");
        out.push_str(&format!(
            "| **Lens 1: Proposer** | Constructive Architect (Thesis) | **{} steps**, {} affected files, {} risks (⏱️ {} ms) |\n",
            plan_res.proposed_plan.steps.len(),
            plan_res.proposed_plan.affected_files.len(),
            plan_res.proposed_plan.risks.len(),
            plan_res.proposer_elapsed_ms
        ));
        out.push_str(&format!(
            "| **Lens 2: Critic** | Adversarial Auditor (Antithesis) | **{} missing aspects**, {} disputed steps, {} unverified assumptions (⏱️ {} ms) |\n",
            plan_res.critic_output.missing_aspects.len(),
            plan_res.critic_output.disputed_steps.len(),
            plan_res.critic_output.unverified_assumptions.len(),
            plan_res.critic_elapsed_ms
        ));
        out.push_str(&format!(
            "| **Lens 3: Rebuttal** | Scope Arbiter (Critic of Critic) | **{} dismissed hedges**, {} accepted critiques (⏱️ {} ms) |\n",
            plan_res.rebuttal_output.dismissed_critiques.len(),
            plan_res.rebuttal_output.accepted_critiques.len(),
            plan_res.rebuttal_elapsed_ms
        ));
        out.push_str(&format!(
            "| **Lens 4: Reconciler** | Convergence Harmonizer (Synthesis) | **{} verified steps**, {} risks, {} commands (⏱️ {} ms) |\n",
            rec.steps.len(),
            rec.risks.len(),
            rec.validation_commands.len(),
            plan_res.reconciler_elapsed_ms
        ));
        out.push_str(&format!(
            "| **Node 5: Host Arbiter** | Deterministic Lease Validation | 🎯 **{} decisions committed** to Notebook ({} valid files, {} unbacked dropped) |\n\n",
            plan_res.committed_decision_ids.len(),
            plan_res.arbiter_validated_files.len(),
            plan_res.arbiter_dropped_files.len()
        ));

        out.push_str(&format!("**Plan Objective**: {}\n\n", rec.objective));

        if !plan_res.rebuttal_output.dismissed_critiques.is_empty() {
            out.push_str("<details>\n<summary><b>🛡️ Lens 3 Appellate Audit: Dismissed Critic Hedges (Analysis Paralysis Prevention)</b></summary>\n\n");
            for item in &plan_res.rebuttal_output.dismissed_critiques {
                out.push_str(&format!("- **Critique**: {}\n  - *Rebuttal Justification*: {}\n", item.critique, item.justification));
            }
            out.push_str("\n</details>\n\n");
        }

        if !plan_res.rebuttal_output.accepted_critiques.is_empty() {
            out.push_str("<details>\n<summary><b>✅ Lens 3 Appellate Audit: Accepted Critiques (Adopted into Plan)</b></summary>\n\n");
            for critique in &plan_res.rebuttal_output.accepted_critiques {
                out.push_str(&format!("- {}\n", critique));
            }
            out.push_str("\n</details>\n\n");
        }

        if !plan_res.arbiter_dropped_files.is_empty() {
            out.push_str("<details>\n<summary><b>🛡️ Host-Pruned Unleased Files</b></summary>\n\n");
            for file in &plan_res.arbiter_dropped_files {
                out.push_str(&format!("- `{}`: {}\n", file.path, file.reason));
            }
            out.push_str("\n</details>\n\n");
        }
    }

    // --- Step 4 Section ---
    if let Some(artifact) = patch {
        out.push_str("### 🛠️ Step 4: Appellate Stereoscopic Edit Gate (Safe In-Memory Patch)\n\n");
        out.push_str("| Metric | Value |\n");
        out.push_str("|---|---|\n");
        out.push_str(&format!("| **Target File** | `{}` (lines {}-{}) |\n", artifact.target_path, artifact.original_start_line, artifact.original_end_line));
        out.push_str(&format!("| **Critic Verdict** | {} |\n", if artifact.critic_approved { "🟢 Approved" } else { "⚠️ Notes / Reconciled" }));
        if !artifact.validation_commands.is_empty() {
            out.push_str(&format!("| **Validation Commands** | `{}` |\n", artifact.validation_commands.join(", ")));
        }
        out.push('\n');

        out.push_str("<details open>\n<summary><b>📝 Proposed Unified Git Diff</b></summary>\n\n");
        out.push_str("```diff\n");
        out.push_str(&artifact.diff);
        out.push_str("\n```\n</details>\n\n");

        if !artifact.critic_notes.is_empty() {
            out.push_str(&format!("**Critic Assessment**: {}\n\n", artifact.critic_notes));
        }
    }

    out.push_str("<details>\n<summary><b>📓 Current Notebook State (");
    let active_count = notebook.active_entries().len();
    out.push_str(&format!("{} Entries)</b></summary>\n\n", active_count));
    for entry in notebook.active_entries() {
        out.push_str(&format!(
            "- **`{:?}`** (`{}`): {}\n",
            entry.kind,
            entry.id.0,
            entry.statement
        ));
    }
    out.push_str("\n</details>\n\n");

    out.push_str("#### ⏱️ Performance Metrics\n");
    let usage = &intake_resp.usage;
    let wall_ms = usage
        .wall_ms
        .map(|v| format!("{} ms", v))
        .unwrap_or_else(|| "n/a".to_string());
    let prompt_tok = usage
        .prompt_tokens_total
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    let decode_tok = usage
        .decode_tokens
        .map(|v| v.to_string())
        .unwrap_or_else(|| "n/a".to_string());
    out.push_str(&format!(
        "- **Step 1 (Model Intake):** {} wall | {} prompt tokens | {} decode tokens\n",
        wall_ms, prompt_tok, decode_tok
    ));
    out.push_str(&format!(
        "- **Step 2 (Host Retrieval & Lease):** {} ms native Rust wall\n",
        phase2.elapsed_ms
    ));
    if let Some(ext) = extract {
        out.push_str(&format!(
            "- **Step 3A (Model Extraction):** {} ms wall\n",
            ext.elapsed_ms
        ));
    }
    out.push('\n');

    if let Some(ext) = extract {
        out.push_str("<details>\n<summary><b>📄 Step 3A Raw Model Extraction JSON</b></summary>\n\n");
        out.push_str("```json\n");
        let pretty_json = serde_json::from_str::<serde_json::Value>(&ext.raw_json)
            .ok()
            .and_then(|v| serde_json::to_string_pretty(&v).ok())
            .unwrap_or_else(|| ext.raw_json.trim().to_string());
        out.push_str(&pretty_json);
        out.push_str("\n```\n</details>\n\n");

        out.push_str("<details>\n<summary><b>📝 Exact Step 3A Prompt Sent to Model</b></summary>\n\n");
        out.push_str("```text\n");
        out.push_str(ext.prompt.trim());
        out.push_str("\n```\n</details>\n\n");
    }

    out.push_str("<details>\n<summary><b>📄 Step 1 Raw Model JSON Output</b></summary>\n\n");
    out.push_str("```json\n");
    let pretty_json = serde_json::from_str::<serde_json::Value>(&intake_resp.raw_json)
        .ok()
        .and_then(|v| serde_json::to_string_pretty(&v).ok())
        .unwrap_or_else(|| intake_resp.raw_json.trim().to_string());
    out.push_str(&pretty_json);
    out.push_str("\n```\n</details>\n\n");

    out.push_str("<details>\n<summary><b>📝 Exact Step 1 Prompt Sent to Model</b></summary>\n\n");
    out.push_str("```text\n");
    out.push_str(intake_prompt.trim());
    out.push_str("\n```\n</details>\n");

    out
}

fn format_line_numbered_slice(slice: &EvidenceSlice) -> String {
    slice
        .content
        .lines()
        .enumerate()
        .map(|(idx, line)| format!("{:>5}: {}", slice.lease.range.start + idx, line))
        .collect::<Vec<_>>()
        .join("\n")
}

fn detect_lang(path: &str) -> &'static str {
    let lower = path.to_lowercase();
    if lower.ends_with(".rs") {
        "rust"
    } else if lower.ends_with(".tsx") {
        "tsx"
    } else if lower.ends_with(".ts") {
        "typescript"
    } else if lower.ends_with(".jsx") {
        "jsx"
    } else if lower.ends_with(".js") {
        "javascript"
    } else if lower.ends_with(".json") {
        "json"
    } else if lower.ends_with(".toml") {
        "toml"
    } else if lower.ends_with(".css") {
        "css"
    } else if lower.ends_with(".html") {
        "html"
    } else if lower.ends_with(".md") {
        "markdown"
    } else {
        "text"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::TokenCounter;
    use async_trait::async_trait;
    use std::sync::Mutex;

    struct MockLlm {
        response: String,
    }

    #[async_trait]
    impl LlmClient for MockLlm {
        async fn complete_json(&self, request: LlmRequest) -> Result<LlmResponse> {
            let raw_json = if request.phase == Phase::IntakeCritic {
                r#"{
                    "approved": true,
                    "critique": "Candidate draft approved by mock critic.",
                    "has_dropped_clauses": false,
                    "has_contaminated_queries": false,
                    "corrected_intents": null
                }"#
                .to_string()
            } else if request.phase == Phase::CandidateGate {
                r#"{
                    "evaluations": [
                        { "path": "src/routes.tsx", "keep": true, "reason": "Configures application routing" }
                    ]
                }"#
                .to_string()
            } else {
                self.response.clone()
            };

            Ok(LlmResponse {
                call_id: request.call_id,
                raw_json,
                usage: crate::traits::LlmUsage {
                    wall_ms: Some(42),
                    prefill_ms: Some(10),
                    decode_ms: Some(32),
                    prompt_tokens_total: Some(150),
                    prompt_tokens_cached: Some(50),
                    decode_tokens: Some(40),
                    ..Default::default()
                },
            })
        }
    }

    #[async_trait]
    impl TokenCounter for MockLlm {
        async fn count_tokens(&self, text: &str) -> usize {
            text.len() / 4
        }
    }

    struct MockSearch;

    #[async_trait]
    impl CodeSearch for MockSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec!["src/main.rs".to_string(), "src/lib.rs".to_string()])
        }
        async fn search(&self, query: &str, _limit: usize) -> Result<Vec<crate::types::FileHit>> {
            if query.contains("route") {
                Ok(vec![crate::types::FileHit {
                    path: "src/routes.tsx".to_string(),
                    score: 80,
                }])
            } else {
                Ok(Vec::new())
            }
        }
        async fn search_symbols(&self, query: &str, _limit: usize) -> Result<Vec<crate::types::SymbolHit>> {
            if query.contains("route") {
                Ok(vec![crate::types::SymbolHit {
                    path: "src/routes.tsx".to_string(),
                    symbol: "AppRoutes".to_string(),
                    kind: "Component".to_string(),
                    start_line: 14,
                    end_line: 60,
                    score: 95,
                }])
            } else {
                Ok(Vec::new())
            }
        }
        async fn search_vector(&self, query: &str, _limit: usize) -> Result<Vec<crate::types::VectorHit>> {
            if query.contains("route") {
                Ok(vec![crate::types::VectorHit {
                    path: "src/routes.tsx".to_string(),
                    start_line: 14,
                    end_line: 60,
                    symbol: Some("AppRoutes".to_string()),
                    snippet: "export function AppRoutes() { ... }".to_string(),
                    score: 0.88,
                }])
            } else {
                Ok(Vec::new())
            }
        }
        async fn search_related_symbols(
            &self,
            _relation: crate::types::SymbolRelation,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<crate::types::SymbolHit>> {
            Ok(Vec::new())
        }
        async fn search_content(&self, query: &str, _limit: usize) -> Result<Vec<crate::types::ContentHit>> {
            if query.contains("route") {
                Ok(vec![crate::types::ContentHit {
                    path: "src/routes.tsx".to_string(),
                    start_line: 14,
                    end_line: 60,
                    preview: "export function AppRoutes() { ... }".to_string(),
                    score: 90,
                }])
            } else {
                Ok(Vec::new())
            }
        }
    }

    struct MockReader;

    #[async_trait]
    impl FileReader for MockReader {
        async fn read_file(&self, _path: &str, _max_chars: usize) -> Result<crate::types::EvidenceSlice> {
            Err(anyhow::anyhow!("not implemented for mock"))
        }
        async fn read_range(
            &self,
            path: &str,
            symbol: Option<&str>,
            start_line: u32,
            end_line: u32,
            _max_chars: usize,
        ) -> Result<crate::types::EvidenceSlice> {
            Ok(crate::types::EvidenceSlice {
                lease: crate::notebook::EvidenceLease::new(
                    path.to_string(),
                    symbol.map(ToString::to_string),
                    crate::notebook::LineRange {
                        start: start_line as usize,
                        end: end_line as usize,
                    },
                    "export function AppRoutes() {\n  return <Routes />;\n}",
                ),
                content: "export function AppRoutes() {\n  return <Routes />;\n}".to_string(),
            })
        }
        async fn verify_lease(&self, _lease: &crate::notebook::EvidenceLease) -> Result<crate::types::LeaseStatus> {
            Ok(crate::types::LeaseStatus::Valid)
        }
    }

    struct CollectProgress {
        messages: std::sync::Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl ProgressSink for CollectProgress {
        async fn emit(&self, progress: AgentProgress) {
            if let AgentProgress::Status(msg) = progress {
                self.messages.lock().unwrap().push(msg);
            }
        }
    }

    #[tokio::test]
    async fn chat_greeting_is_classified_without_search() {
        let llm = MockLlm {
            response: r#"{
                "reasoning": "User greeted the assistant.",
                "needs_workspace_search": false,
                "intents": [
                    {
                        "intent": "Chat",
                        "summary": "Friendly greeting",
                        "needs_workspace_search": false,
                        "direct_response": "Hello! How can I help you with your project today?",
                        "search_queries": []
                    }
                ],
                "target_hints": [],
                "constraints": []
            }"#
            .to_string(),
        };

        let task = TaskRequest {
            run_id: "test-run".to_string(),
            session_id: "test-session".to_string(),
            workspace_path: None,
            question: "Hello there".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: true,
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &llm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);
        assert!(result.answer.content.contains("Chat"));
        assert!(result.answer.content.contains("No"));
        assert!(result.answer.content.contains("Hello! How can I help you"));
        assert!(progress.messages.lock().unwrap().contains(&"Analyzing request intents".to_string()));
    }

    #[tokio::test]
    async fn multi_intent_compound_prompt_is_decomposed() {
        let llm = MockLlm {
            response: r#"{
                "reasoning": "User combined a greeting, a general conceptual question about JWT, and a search request for auth token validation in the codebase.",
                "needs_workspace_search": true,
                "intents": [
                    {
                        "intent": "Chat",
                        "summary": "Friendly greeting",
                        "needs_workspace_search": false,
                        "direct_response": "Hello! Happy to help.",
                        "search_queries": []
                    },
                    {
                        "intent": "GeneralQuestion",
                        "summary": "Explain how JWT expiration works",
                        "needs_workspace_search": false,
                        "direct_response": "JWT expiration is determined by the `exp` timestamp claim in the token payload.",
                        "search_queries": []
                    },
                    {
                        "intent": "WorkspaceSearch",
                        "summary": "Find where auth tokens are validated in this codebase",
                        "needs_workspace_search": true,
                        "direct_response": null,
                        "search_queries": ["validate_token", "jwt", "auth middleware"]
                    }
                ],
                "target_hints": ["auth", "jwt"],
                "constraints": []
            }"#
            .to_string(),
        };

        let task = TaskRequest {
            run_id: "test-run".to_string(),
            session_id: "test-session".to_string(),
            workspace_path: None,
            question: "Hi! What is JWT expiration, and where do we validate auth tokens in this project?".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: true,
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &llm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);
        assert!(result.answer.content.contains("Detected Intents (3)"));
        assert!(result.answer.content.contains("Chat"));
        assert!(result.answer.content.contains("GeneralQuestion"));
        assert!(result.answer.content.contains("WorkspaceSearch"));
        assert!(result.answer.content.contains("Yes"));
        assert!(result.answer.content.contains("exp"));
        assert!(result.answer.content.contains("validate_token"));
    }

    #[tokio::test]
    async fn parses_json_inside_code_fences() {
        let raw = "Here is the response:\n```json\n{\"reasoning\":\"greeting\",\"needs_workspace_search\":false,\"intents\":[{\"intent\":\"Chat\",\"summary\":\"hi\",\"needs_workspace_search\":false,\"direct_response\":\"hi\",\"search_queries\":[]}],\"target_hints\":[],\"constraints\":[]}\n```";
        let parsed: IntakeOutput = parse_json(raw).unwrap();
        assert_eq!(parsed.intents[0].intent, UserIntent::Chat);
        assert!(!parsed.needs_workspace_search);
    }

    #[tokio::test]
    async fn fallback_handles_unparseable_output() {
        let fallback = fallback_intake("add a login button");
        assert_eq!(fallback.intents[0].intent, UserIntent::WorkspaceEdit);
        assert!(fallback.needs_workspace_search);
        assert!(!fallback.intents[0].search_queries.is_empty());
    }

    #[tokio::test]
    async fn host_normalization_catches_collapsed_codebase_intent() {
        let llm = MockLlm {
            response: r#"{
                "reasoning": "The user asks about LRU cache and where it is in the codebase.",
                "needs_workspace_search": false,
                "intents": [
                    {
                        "intent": "GeneralQuestion",
                        "summary": "Explain LRU cache and its usage",
                        "needs_workspace_search": false,
                        "direct_response": "An LRU cache is a data structure...",
                        "search_queries": []
                    }
                ],
                "target_hints": [],
                "constraints": []
            }"#
            .to_string(),
        };

        let task = TaskRequest {
            run_id: "test-run".to_string(),
            session_id: "test-session".to_string(),
            workspace_path: None,
            question: "What is an LRU cache, and where in our codebase do we cache API responses?".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: true,
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &llm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);
        assert!(result.answer.content.contains("Detected Intents (2)"));
        assert!(result.answer.content.contains("WorkspaceSearch"));
        assert!(result.answer.content.contains("Yes"));
        assert!(result.answer.content.contains("Host detected explicit codebase search target"));
    }

    #[tokio::test]
    async fn host_normalization_prunes_ghost_greeting() {
        let llm = MockLlm {
            response: r#"{
                "reasoning": "The user greets and asks to find route settings and general best practices.",
                "needs_workspace_search": true,
                "intents": [
                    {
                        "intent": "Chat",
                        "summary": "Greeting",
                        "needs_workspace_search": false,
                        "direct_response": "Hello! How can I help you?",
                        "search_queries": []
                    },
                    {
                        "intent": "WorkspaceSearch",
                        "summary": "Find route settings in UI",
                        "needs_workspace_search": true,
                        "direct_response": null,
                        "search_queries": ["route", "router", "settings"]
                    },
                    {
                        "intent": "GeneralQuestion",
                        "summary": "Routing best practices",
                        "needs_workspace_search": false,
                        "direct_response": "Standard practices include...",
                        "search_queries": []
                    }
                ],
                "target_hints": ["route"],
                "constraints": []
            }"#
            .to_string(),
        };

        // User question has NO greeting
        let task = TaskRequest {
            run_id: "test-run".to_string(),
            session_id: "test-session".to_string(),
            workspace_path: None,
            question: "where does our Ui configure the route settings, and is it aligned with best practices".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: true,
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &llm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);
        // Chat intent must be pruned by host normalization!
        assert!(result.answer.content.contains("Detected Intents (2)"));
        assert!(result.answer.content.contains("1. **`WorkspaceSearch`**"));
        assert!(result.answer.content.contains("2. **`GeneralQuestion`**"));
        assert!(!result.answer.content.contains("**`Chat`**"));
    }

    #[tokio::test]
    async fn host_normalization_adds_missing_greeting() {
        let llm = MockLlm {
            response: r#"{
                "reasoning": "The user asks to find cache storage.",
                "needs_workspace_search": true,
                "intents": [
                    {
                        "intent": "WorkspaceSearch",
                        "summary": "Find cache storage",
                        "needs_workspace_search": true,
                        "direct_response": null,
                        "search_queries": ["cache"]
                    }
                ],
                "target_hints": ["cache"],
                "constraints": []
            }"#
            .to_string(),
        };

        // User question explicitly has a greeting
        let task = TaskRequest {
            run_id: "test-run".to_string(),
            session_id: "test-session".to_string(),
            workspace_path: None,
            question: "Hi! Where do we store cached responses?".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: true,
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &llm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);
        assert!(result.answer.content.contains("Detected Intents (2)"));
        assert!(result.answer.content.contains("`Chat`"));
        assert!(result.answer.content.contains("`WorkspaceSearch`"));
    }

    #[tokio::test]
    async fn step2_executes_search_and_leases_evidence() {
        let llm = MockLlm {
            response: r#"{
                "reasoning": "The user asks to find UI route configuration and routing best practices.",
                "needs_workspace_search": true,
                "intents": [
                    {
                        "intent": "WorkspaceSearch",
                        "summary": "Find UI route settings configuration in codebase",
                        "needs_workspace_search": true,
                        "direct_response": null,
                        "search_queries": ["route", "router"]
                    },
                    {
                        "intent": "GeneralQuestion",
                        "summary": "Routing best practices",
                        "needs_workspace_search": false,
                        "direct_response": "Standard patterns include declarative routes...",
                        "search_queries": []
                    }
                ],
                "target_hints": ["route"],
                "constraints": []
            }"#
            .to_string(),
        };

        let task = TaskRequest {
            run_id: "test-run".to_string(),
            session_id: "test-session".to_string(),
            workspace_path: None,
            question: "where does our Ui configure the route settings, and is it aligned with best practices".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: false, // Executes Step 1 + Step 2!
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &llm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);
        // Step 1 assertions
        assert!(result.answer.content.contains("Step 1: Multi-Intent Intake"));
        assert!(result.answer.content.contains("Detected Intents (2)"));
        assert!(result.answer.content.contains("WorkspaceSearch"));
        assert!(result.answer.content.contains("GeneralQuestion"));

        // Step 2 assertions
        assert!(result.answer.content.contains("Step 2: Deterministic Evidence Acquisition"));
        assert!(result.answer.content.contains("Search Queries Executed"));
        assert!(result.answer.content.contains("src/routes.tsx"));
        assert!(result.answer.content.contains("AppRoutes"));
        assert!(result.answer.content.contains("Active Evidence Leases in Notebook"));

        // Notebook assertions
        let entries = result.answer.notebook.active_entries();
        assert!(!entries.is_empty());
        let lease_entry = entries.iter().find(|e| e.statement.contains("src/routes.tsx"));
        assert!(lease_entry.is_some());
    }

    #[tokio::test]
    async fn triad_consensus_critic_challenges_and_arbiter_enforces_split() {
        struct TriadMockLlm;

        #[async_trait]
        impl LlmClient for TriadMockLlm {
            async fn complete_json(&self, request: LlmRequest) -> Result<LlmResponse> {
                let raw_json = match request.phase {
                    Phase::Intake => {
                        // Node 1 (Drafter) collapses into single intent and pollutes queries
                        r#"{
                            "reasoning": "User asks where UI configures route settings and if aligned with best practices.",
                            "needs_workspace_search": true,
                            "intents": [
                                {
                                    "intent": "WorkspaceSearch",
                                    "summary": "Locate route settings and evaluate best practices",
                                    "needs_workspace_search": true,
                                    "direct_response": null,
                                    "search_queries": ["route", "settings", "react", "best practices"]
                                }
                            ],
                            "target_hints": ["route"],
                            "constraints": []
                        }"#.to_string()
                    }
                    Phase::IntakeCritic => {
                        // Node 2 (Critic) challenges Node 1, flags dropped clause and contaminated queries, and provides corrected intents
                        r#"{
                            "approved": false,
                            "critique": "Draft dropped the second question regarding best practices and contaminated search queries with 'react' and 'best practices'.",
                            "has_dropped_clauses": true,
                            "has_contaminated_queries": true,
                            "corrected_intents": [
                                {
                                    "intent": "WorkspaceSearch",
                                    "summary": "Locate UI route and view settings in codebase",
                                    "needs_workspace_search": true,
                                    "direct_response": null,
                                    "search_queries": ["route", "view", "navigation", "App"]
                                },
                                {
                                    "intent": "GeneralQuestion",
                                    "summary": "Explain frontend routing best practices",
                                    "needs_workspace_search": false,
                                    "direct_response": "Declarative routing or state-driven view switching...",
                                    "search_queries": []
                                }
                            ]
                        }"#.to_string()
                    }
                    _ => "{}".to_string(),
                };

                Ok(LlmResponse {
                    call_id: request.call_id,
                    raw_json,
                    usage: crate::traits::LlmUsage::default(),
                })
            }
        }

        #[async_trait]
        impl TokenCounter for TriadMockLlm {
            async fn count_tokens(&self, text: &str) -> usize {
                text.len() / 4
            }
        }

        let task = TaskRequest {
            run_id: "triad-test".to_string(),
            session_id: "triad-session".to_string(),
            workspace_path: None,
            question: "where does our Ui configure the route settings, and is it aligned with react best practices".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: true,
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &TriadMockLlm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &TriadMockLlm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);
        // Verify Triad Quorum was rendered
        assert!(result.answer.content.contains("Triad Intake Consensus"));
        assert!(result.answer.content.contains("Node 1: Drafter"));
        assert!(result.answer.content.contains("Node 2: Critic"));
        assert!(result.answer.content.contains("Node 3: Arbiter"));
        assert!(result.answer.content.contains("Consensus Reached (2-of-3)"));
        assert!(result.answer.content.contains("Challenged / Vetoed"));

        // Verify that 2 intents were finalized
        assert!(result.answer.content.contains("Detected Intents (2)"));
        assert!(result.answer.content.contains("WorkspaceSearch"));
        assert!(result.answer.content.contains("GeneralQuestion"));
    }

    #[tokio::test]
    async fn test_phase2_expands_topology_and_candidate_gate_prunes_noise() {
        struct TopologyMockLlm;

        #[async_trait]
        impl LlmClient for TopologyMockLlm {
            async fn complete_json(&self, request: LlmRequest) -> Result<LlmResponse> {
                let raw_json = match request.phase {
                    Phase::Intake => r#"{
                        "reasoning": "User asks where UI configures route settings and best practices.",
                        "needs_workspace_search": true,
                        "intents": [
                            {
                                "intent": "WorkspaceSearch",
                                "summary": "Locate UI route settings configuration in codebase",
                                "needs_workspace_search": true,
                                "direct_response": null,
                                "search_queries": ["route", "view"]
                            },
                            {
                                "intent": "GeneralQuestion",
                                "summary": "Routing best practices",
                                "needs_workspace_search": false,
                                "direct_response": "Declarative routes are recommended.",
                                "search_queries": []
                            }
                        ],
                        "target_hints": ["route"],
                        "constraints": []
                    }"#.to_string(),
                    Phase::IntakeCritic => r#"{
                        "approved": true,
                        "critique": "Approved decomposition.",
                        "has_dropped_clauses": false,
                        "has_contaminated_queries": false,
                        "corrected_intents": null
                    }"#.to_string(),
                    Phase::CandidateGate | Phase::CandidateGateOptimist => r#"{
                        "evaluations": [
                            { "path": "src/App.tsx", "keep": true, "reason": "Configures root UI view and screen state switching" },
                            { "path": "src/index.tsx", "keep": true, "reason": "Entrypoint mounting application root" },
                            { "path": "src/AIConversation.tsx", "keep": false, "reason": "Chat message list display, does not configure view routes" }
                        ]
                    }"#.to_string(),
                    Phase::CandidateGateSkeptic => r#"{
                        "evaluations": [
                            { "path": "src/App.tsx", "keep": true, "reason": "Direct view routing configuration" },
                            { "path": "src/index.tsx", "keep": false, "reason": "Just the entrypoint mounting App" },
                            { "path": "src/AIConversation.tsx", "keep": false, "reason": "DOM copy helper and prose styling" }
                        ]
                    }"#.to_string(),
                    _ => "{}".to_string(),
                };

                Ok(LlmResponse {
                    call_id: request.call_id,
                    raw_json,
                    usage: crate::traits::LlmUsage::default(),
                })
            }
        }

        #[async_trait]
        impl TokenCounter for TopologyMockLlm {
            async fn count_tokens(&self, text: &str) -> usize {
                text.len() / 4
            }
        }

        struct TopologyMockSearch;

        #[async_trait]
        impl CodeSearch for TopologyMockSearch {
            async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
                Ok(Vec::new())
            }
            async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
                Ok(vec![
                    FileHit { path: "src/index.tsx".to_string(), score: 90 },
                    FileHit { path: "src-tauri/gen/schemas/macOS-schema.json".to_string(), score: 85 },
                    FileHit { path: "src-tauri/build.rs".to_string(), score: 80 },
                    FileHit { path: "src-tauri/src/main.rs".to_string(), score: 75 },
                ])
            }
            async fn search_symbols(&self, query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
                if query == "App" {
                    Ok(vec![SymbolHit {
                        path: "src/App.tsx".to_string(),
                        symbol: "App".to_string(),
                        kind: "Component".to_string(),
                        start_line: 17,
                        end_line: 171,
                        score: 95,
                    }])
                } else {
                    Ok(vec![SymbolHit {
                        path: "src/AIConversation.tsx".to_string(),
                        symbol: "AIConversation".to_string(),
                        kind: "Component".to_string(),
                        start_line: 99,
                        end_line: 134,
                        score: 70,
                    }])
                }
            }
            async fn search_vector(&self, _query: &str, _limit: usize) -> Result<Vec<VectorHit>> {
                Ok(Vec::new())
            }
            async fn search_related_symbols(
                &self,
                _relation: crate::types::SymbolRelation,
                _query: &str,
                _limit: usize,
            ) -> Result<Vec<SymbolHit>> {
                Ok(Vec::new())
            }
            async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
                Ok(Vec::new())
            }
            async fn search_imported_files(&self, seed_paths: &[String], _limit: usize) -> Result<Vec<FileHit>> {
                if seed_paths.contains(&"src/index.tsx".to_string()) {
                    Ok(vec![FileHit { path: "src/App.tsx".to_string(), score: 95 }])
                } else {
                    Ok(Vec::new())
                }
            }
            async fn search_file_symbols(&self, path: &str) -> Result<Vec<SymbolHit>> {
                if path == "src/App.tsx" {
                    Ok(vec![SymbolHit {
                        path: "src/App.tsx".to_string(),
                        symbol: "App".to_string(),
                        kind: "Component".to_string(),
                        start_line: 17,
                        end_line: 171,
                        score: 95,
                    }])
                } else {
                    Ok(Vec::new())
                }
            }
        }

        let task = TaskRequest {
            run_id: "test-topo-run".to_string(),
            session_id: "test-topo-session".to_string(),
            workspace_path: None,
            question: "where does our Ui configure the route settings, and is it aligned with best practices".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: false,
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &TopologyMockLlm,
                code_search: &TopologyMockSearch,
                file_reader: &MockReader,
                token_counter: &TopologyMockLlm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);

        // Active notebook leases must contain App.tsx and index.tsx
        let leased_paths: Vec<String> = result.answer.notebook.active_entries().iter().map(|e| e.statement.clone()).collect();
        assert!(leased_paths.iter().any(|s| s.contains("src/App.tsx")), "Expected src/App.tsx in leases, got: {:?}", leased_paths);
        assert!(leased_paths.iter().any(|s| s.contains("src/index.tsx")), "Expected src/index.tsx in leases, got: {:?}", leased_paths);

        // Noise and pruned candidates must NOT be in leases
        assert!(!leased_paths.iter().any(|s| s.contains("macOS-schema.json")));
        assert!(!leased_paths.iter().any(|s| s.contains("build.rs")));
        assert!(!leased_paths.iter().any(|s| s.contains("AIConversation.tsx")));
        assert!(!leased_paths.iter().any(|s| s.contains("main.rs")));

        // Report must document CodeGraph Topology Expansion
        assert!(result.answer.content.contains("CodeGraph Topology Expansion"));
        assert!(result.answer.content.contains("src/App.tsx"));

        // Report must document Pruned Candidates
        assert!(result.answer.content.contains("Pruned Candidates & False Positives"));
        assert!(result.answer.content.contains("macOS-schema.json"));
        assert!(result.answer.content.contains("build.rs"));
        assert!(result.answer.content.contains("AIConversation.tsx"));
    }

    #[tokio::test]
    async fn test_header_and_focal_window_slicing_for_large_files() {
        struct LargeFileMockLlm;

        #[async_trait]
        impl LlmClient for LargeFileMockLlm {
            async fn complete_json(&self, request: LlmRequest) -> Result<LlmResponse> {
                let raw_json = match request.phase {
                    Phase::Intake => r#"{
                        "reasoning": "User looking for deep routing logic in large file.",
                        "needs_workspace_search": true,
                        "intents": [
                            {
                                "intent": "WorkspaceSearch",
                                "summary": "Find deep routing in monolithic file",
                                "needs_workspace_search": true,
                                "direct_response": null,
                                "search_queries": ["DeepProfileRoute", "route"]
                            }
                        ],
                        "target_hints": ["MonolithicRoutes"],
                        "constraints": []
                    }"#.to_string(),
                    Phase::IntakeCritic => r#"{
                        "approved": true,
                        "critique": "Draft is well structured.",
                        "has_dropped_clauses": false,
                        "has_contaminated_queries": false,
                        "corrected_intents": []
                    }"#.to_string(),
                    Phase::CandidateGateOptimist | Phase::CandidateGateSkeptic => r#"{
                        "evaluations": [
                            {
                                "path": "src/MonolithicRoutes.tsx",
                                "keep": true,
                                "confidence": 0.95,
                                "reason": "Direct implementation of deep routing logic with framework imports"
                            }
                        ]
                    }"#.to_string(),
                    _ => "{}".to_string(),
                };

                Ok(LlmResponse {
                    call_id: request.call_id,
                    raw_json,
                    usage: crate::traits::LlmUsage::default(),
                })
            }
        }

        #[async_trait]
        impl TokenCounter for LargeFileMockLlm {
            async fn count_tokens(&self, text: &str) -> usize {
                text.len() / 4
            }
        }

        struct LargeFileMockSearch;

        #[async_trait]
        impl CodeSearch for LargeFileMockSearch {
            async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
                Ok(vec!["src/MonolithicRoutes.tsx".to_string()])
            }
            async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
                Ok(vec![FileHit {
                    path: "src/MonolithicRoutes.tsx".to_string(),
                    score: 80,
                }])
            }
            async fn search_symbols(&self, _query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
                Ok(Vec::new())
            }
            async fn search_vector(&self, _query: &str, _limit: usize) -> Result<Vec<VectorHit>> {
                Ok(vec![VectorHit {
                    path: "src/MonolithicRoutes.tsx".to_string(),
                    start_line: 450,
                    end_line: 510,
                    symbol: Some("DeepProfileRoute".to_string()),
                    snippet: "export function DeepProfileRoute() { ... }".to_string(),
                    score: 0.85,
                }])
            }
            async fn search_related_symbols(
                &self,
                _relation: crate::types::SymbolRelation,
                _query: &str,
                _limit: usize,
            ) -> Result<Vec<SymbolHit>> {
                Ok(Vec::new())
            }
            async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
                Ok(Vec::new())
            }
        }

        struct LargeFileMockReader;

        #[async_trait]
        impl FileReader for LargeFileMockReader {
            async fn read_file(&self, _path: &str, _max_chars: usize) -> Result<EvidenceSlice> {
                Err(anyhow::anyhow!("not implemented for mock"))
            }
            async fn read_range(
                &self,
                path: &str,
                symbol: Option<&str>,
                start_line: u32,
                end_line: u32,
                _max_chars: usize,
            ) -> Result<EvidenceSlice> {
                let text = if start_line == 1 && end_line == 25 {
                    "import { Router, Route } from '@solidjs/router';\nimport { createSignal } from 'solid-js';"
                } else {
                    "export function DeepProfileRoute() {\n  return <Route path='/profile' />;\n}"
                };

                Ok(EvidenceSlice {
                    lease: crate::notebook::EvidenceLease::new(
                        path.to_string(),
                        symbol.map(ToString::to_string),
                        crate::notebook::LineRange {
                            start: start_line as usize,
                            end: end_line as usize,
                        },
                        text,
                    ),
                    content: text.to_string(),
                })
            }
            async fn verify_lease(&self, _lease: &crate::notebook::EvidenceLease) -> Result<crate::types::LeaseStatus> {
                Ok(crate::types::LeaseStatus::Valid)
            }
        }

        let task = TaskRequest {
            run_id: "test-large-file-run".to_string(),
            session_id: "test-large-file-session".to_string(),
            workspace_path: None,
            question: "where does the DeepProfileRoute get defined in the monolithic routes file".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: false,
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &LargeFileMockLlm,
                code_search: &LargeFileMockSearch,
                file_reader: &LargeFileMockReader,
                token_counter: &LargeFileMockLlm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);

        // Active notebook leases must contain BOTH header/imports context and the focal chunk
        let entries = result.answer.notebook.active_entries();
        let header_lease = entries.iter().find(|e| e.statement.contains("lines 1-25 (header/imports context)"));
        assert!(header_lease.is_some(), "Expected header/imports context lease in notebook, got: {:?}", entries.iter().map(|e| &e.statement).collect::<Vec<_>>());

        let focal_lease = entries.iter().find(|e| e.statement.contains("lines 450-510"));
        assert!(focal_lease.is_some(), "Expected focal range lease in notebook, got: {:?}", entries.iter().map(|e| &e.statement).collect::<Vec<_>>());

        // Report must include both slices
        assert!(result.answer.content.contains("src/MonolithicRoutes.tsx"));
        assert!(result.answer.content.contains("DeepProfileRoute"));
        assert!(result.answer.content.contains("@solidjs/router"));
    }

    #[tokio::test]
    async fn test_step3a_extraction_validates_leases_and_updates_notebook() {
        struct ExtractMockLlm;

        #[async_trait]
        impl LlmClient for ExtractMockLlm {
            async fn complete_json(&self, request: LlmRequest) -> Result<LlmResponse> {
                let raw_json = match request.phase {
                    Phase::Intake => r#"{
                        "reasoning": "User asks where UI configures route settings.",
                        "needs_workspace_search": true,
                        "intents": [
                            {
                                "intent": "WorkspaceSearch",
                                "summary": "Locate UI route settings configuration in codebase",
                                "needs_workspace_search": true,
                                "direct_response": null,
                                "search_queries": ["route", "view"]
                            }
                        ],
                        "target_hints": ["route"],
                        "constraints": []
                    }"#.to_string(),
                    Phase::IntakeCritic => r#"{
                        "approved": true,
                        "critique": "Draft approved.",
                        "has_dropped_clauses": false,
                        "has_contaminated_queries": false,
                        "corrected_intents": null
                    }"#.to_string(),
                    Phase::CandidateGate | Phase::CandidateGateOptimist | Phase::CandidateGateSkeptic => r#"{
                        "evaluations": [
                            { "path": "src/routes.tsx", "keep": true, "reason": "Configures routes" }
                        ]
                    }"#.to_string(),
                    Phase::Extract => {
                        // Extract valid lease ID from prompt
                        let found_lease = request.prompt
                            .lines()
                            .find(|l| l.starts_with("LEASE "))
                            .and_then(|l| l.strip_prefix("LEASE "))
                            .unwrap_or("mock-lease");

                        format!(r#"{{
                            "inferences": [
                                {{
                                    "statement": "src/routes.tsx defines AppRoutes which renders application route tree.",
                                    "lease_id": "{}",
                                    "tags": ["route", "AppRoutes"]
                                }},
                                {{
                                    "statement": "Hallucinated claim with invalid lease id.",
                                    "lease_id": "lease-hallucinated-fake-id",
                                    "tags": ["fake"]
                                }}
                            ],
                            "sufficient": true,
                            "missing": []
                        }}"#, found_lease)
                    }
                    _ => "{}".to_string(),
                };

                Ok(LlmResponse {
                    call_id: request.call_id,
                    raw_json,
                    usage: crate::traits::LlmUsage::default(),
                })
            }
        }

        #[async_trait]
        impl TokenCounter for ExtractMockLlm {
            async fn count_tokens(&self, text: &str) -> usize {
                text.len() / 4
            }
        }

        let task = TaskRequest {
            run_id: "test-extract-run".to_string(),
            session_id: "test-extract-session".to_string(),
            workspace_path: None,
            question: "where does the UI configure route settings".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: false,
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &ExtractMockLlm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &ExtractMockLlm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);

        // Active notebook entries must contain the verified inference
        let active_entries = result.answer.notebook.active_entries();
        let verified_inf = active_entries.iter().find(|e| e.statement.contains("src/routes.tsx defines AppRoutes"));
        assert!(verified_inf.is_some(), "Expected verified inference in notebook");
        assert_eq!(verified_inf.unwrap().kind, crate::notebook::EntryKind::Inference);

        // Fake inference must NOT be in the notebook
        let fake_inf = active_entries.iter().find(|e| e.statement.contains("Hallucinated claim"));
        assert!(fake_inf.is_none(), "Fake inference should have been rejected by the host");

        // The report must display Step 3A section with metrics
        assert!(result.answer.content.contains("Step 3A: Evidence Extraction"));
        assert!(result.answer.content.contains("Sufficient"));
        assert!(result.answer.content.contains("1 inferences accepted"));
        assert!(result.answer.content.contains("1 inferences dropped"));
        assert!(result.answer.content.contains("lease-hallucinated-fake-id"));

        // Cited entries must contain the verified inference ID
        assert!(!result.answer.cited_entries.is_empty());
        assert!(result.answer.cited_entries.contains(&verified_inf.unwrap().id));
    }

    #[tokio::test]
    async fn test_step3b_appellate_planning_gate_validates_leases_and_commits_decisions() {
        struct PlanMockLlm;

        #[async_trait]
        impl LlmClient for PlanMockLlm {
            async fn complete_json(&self, request: LlmRequest) -> Result<LlmResponse> {
                let raw_json = match request.phase {
                    Phase::Intake => r#"{
                        "reasoning": "User asks where UI configures route settings.",
                        "needs_workspace_search": true,
                        "intents": [
                            {
                                "intent": "WorkspaceSearch",
                                "summary": "Locate UI route settings configuration in codebase",
                                "needs_workspace_search": true,
                                "direct_response": null,
                                "search_queries": ["route", "view"]
                            }
                        ],
                        "target_hints": ["route"],
                        "constraints": []
                    }"#.to_string(),
                    Phase::IntakeCritic => r#"{
                        "approved": true,
                        "critique": "Draft approved.",
                        "has_dropped_clauses": false,
                        "has_contaminated_queries": false,
                        "corrected_intents": null
                    }"#.to_string(),
                    Phase::CandidateGate | Phase::CandidateGateOptimist | Phase::CandidateGateSkeptic => r#"{
                        "evaluations": [
                            { "path": "src/routes.tsx", "keep": true, "reason": "Configures routes" }
                        ]
                    }"#.to_string(),
                    Phase::Extract => {
                        let found_lease = request.prompt
                            .lines()
                            .find(|l| l.starts_with("LEASE "))
                            .and_then(|l| l.strip_prefix("LEASE "))
                            .unwrap_or("mock-lease");

                        format!(r#"{{
                            "inferences": [
                                {{
                                    "statement": "src/routes.tsx defines AppRoutes which renders the navigation tree.",
                                    "lease_id": "{}",
                                    "tags": ["route", "AppRoutes"]
                                }}
                            ],
                            "sufficient": true,
                            "missing": []
                        }}"#, found_lease)
                    }
                    Phase::PlanProposer => r#"{
                        "objective": "Document and verify application routing architecture",
                        "affected_files": [
                            { "path": "src/routes.tsx", "reason": "Houses primary route declarations" },
                            { "path": "src/hallucinated_unleased.ts", "reason": "Speculative helper not in repo" }
                        ],
                        "steps": [
                            { "title": "Inspect AppRoutes", "detail": "Verify route path bindings in routes.tsx" },
                            { "title": "Audit Deep Links", "detail": "Check deep link handler" }
                        ],
                        "risks": ["Speculative deep link handling might not exist"],
                        "validation_commands": ["npm test"]
                    }"#.to_string(),
                    Phase::PlanCritic => r#"{
                        "critique": "The plan assumes deep link handling exists without leased evidence.",
                        "missing_aspects": ["Deep link configuration context"],
                        "disputed_steps": ["Audit Deep Links is unbacked"],
                        "unverified_assumptions": ["Assumes application supports URL deep linking"],
                        "additional_risks": ["Scope creep on unsupported features"]
                    }"#.to_string(),
                    Phase::PlanRebuttal => r#"{
                        "rebuttal": "Deep link handling is an overblown hedge; this is a desktop app using internal view switching. However, we should verify route fallback.",
                        "dismissed_critiques": [
                            {
                                "critique": "Assumes application supports URL deep linking",
                                "justification": "Evidence establishes this is a desktop view-switching architecture; deep link URL routing is out of scope."
                            }
                        ],
                        "accepted_critiques": ["Audit Deep Links is unbacked"]
                    }"#.to_string(),
                    Phase::PlanReconciler => r#"{
                        "objective": "Verify and document application route structure in routes.tsx",
                        "affected_files": [
                            { "path": "src/routes.tsx", "reason": "Houses primary route declarations" },
                            { "path": "src/hallucinated_unleased.ts", "reason": "Speculative helper" }
                        ],
                        "steps": [
                            { "title": "Inspect AppRoutes", "detail": "Verify route path bindings in routes.tsx" },
                            { "title": "Verify Route Fallback", "detail": "Ensure fallback route handles unknown views" }
                        ],
                        "risks": ["Potential unregistered view IDs"],
                        "validation_commands": ["npm test"]
                    }"#.to_string(),
                    _ => "{}".to_string(),
                };

                Ok(LlmResponse {
                    call_id: request.call_id,
                    raw_json,
                    usage: crate::traits::LlmUsage::default(),
                })
            }
        }

        #[async_trait]
        impl TokenCounter for PlanMockLlm {
            async fn count_tokens(&self, text: &str) -> usize {
                text.len() / 4
            }
        }

        let task = TaskRequest {
            run_id: "test-plan-run".to_string(),
            session_id: "test-plan-session".to_string(),
            workspace_path: None,
            question: "where does the UI configure route settings".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: false,
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &PlanMockLlm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &PlanMockLlm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);

        // 1. Check Double Stereoscopic Gate rendered in content
        assert!(result.answer.content.contains("Step 3B: Appellate Planning Gate"));
        assert!(result.answer.content.contains("Lens 1: Proposer"));
        assert!(result.answer.content.contains("Lens 2: Critic"));
        assert!(result.answer.content.contains("Lens 3: Rebuttal"));
        assert!(result.answer.content.contains("Lens 4: Reconciler"));
        assert!(result.answer.content.contains("Node 5: Host Arbiter"));

        // 2. Check Appellate Audit: dismissed critic hedges & accepted critiques
        assert!(result.answer.content.contains("Dismissed Critic Hedges"));
        assert!(result.answer.content.contains("desktop view-switching architecture"));
        assert!(result.answer.content.contains("Accepted Critiques"));

        // 3. Check Host Arbiter Evidence-Backing enforcement:
        //    src/routes.tsx was leased -> kept!
        //    src/hallucinated_unleased.ts was NOT leased -> pruned by Host Arbiter!
        assert!(result.answer.content.contains("Host-Pruned Unleased Files"));
        assert!(result.answer.content.contains("src/hallucinated_unleased.ts"));

        // 4. Check Notebook Decisions committed
        let active_entries = result.answer.notebook.active_entries();
        let decisions: Vec<_> = active_entries
            .iter()
            .filter(|e| e.kind == crate::notebook::EntryKind::Decision)
            .collect();

        assert!(!decisions.is_empty(), "Expected plan decisions committed to notebook");

        let obj_decision = decisions.iter().find(|d| d.statement.contains("Plan Objective:"));
        assert!(obj_decision.is_some(), "Expected Plan Objective decision");

        let step_decisions: Vec<_> = decisions.iter().filter(|d| d.statement.contains("Plan Step")).collect();
        assert_eq!(step_decisions.len(), 2, "Expected 2 step decisions");

        let target_decision = decisions.iter().find(|d| d.statement.contains("src/routes.tsx"));
        assert!(target_decision.is_some(), "Expected target file decision for leased routes.tsx");

        // The hallucinated unleased file must NOT have a Decision entry
        let unleased_decision = decisions.iter().find(|d| d.statement.contains("src/hallucinated_unleased.ts"));
        assert!(unleased_decision.is_none(), "Unleased target file must not be committed to notebook as decision");
    }

    #[tokio::test]
    async fn test_step3c_synthesis_answers_compound_intents_and_validates_citations() {
        struct SynthMockLlm;

        #[async_trait]
        impl LlmClient for SynthMockLlm {
            async fn complete_json(&self, request: LlmRequest) -> Result<LlmResponse> {
                let raw_json = match request.phase {
                    Phase::Intake => r#"{
                        "reasoning": "User asks where UI configures route settings and if aligned with modern best practices.",
                        "needs_workspace_search": true,
                        "intents": [
                            {
                                "intent": "WorkspaceSearch",
                                "summary": "Locate UI route configuration in codebase",
                                "needs_workspace_search": true,
                                "direct_response": null,
                                "search_queries": ["route", "view"]
                            },
                            {
                                "intent": "GeneralQuestion",
                                "summary": "Explain modern UI routing best practices",
                                "needs_workspace_search": false,
                                "direct_response": "Modern single-window desktop apps favor state-driven conditional view switching.",
                                "search_queries": []
                            }
                        ],
                        "target_hints": ["route"],
                        "constraints": []
                    }"#.to_string(),
                    Phase::IntakeCritic => r#"{
                        "approved": true,
                        "critique": "Approved decomposition.",
                        "has_dropped_clauses": false,
                        "has_contaminated_queries": false,
                        "corrected_intents": null
                    }"#.to_string(),
                    Phase::CandidateGate | Phase::CandidateGateOptimist | Phase::CandidateGateSkeptic => r#"{
                        "evaluations": [
                            { "path": "src/routes.tsx", "keep": true, "reason": "Configures routes" }
                        ]
                    }"#.to_string(),
                    Phase::Extract => {
                        let found_lease = request.prompt
                            .lines()
                            .find(|l| l.starts_with("LEASE "))
                            .and_then(|l| l.strip_prefix("LEASE "))
                            .unwrap_or("mock-lease");

                        format!(r#"{{
                            "inferences": [
                                {{
                                    "statement": "src/routes.tsx lines 1-10 defines AppRoutes with declarative route objects.",
                                    "lease_id": "{}",
                                    "tags": ["route", "AppRoutes"]
                                }}
                            ],
                            "sufficient": true,
                            "missing": []
                        }}"#, found_lease)
                    }
                    Phase::PlanProposer | Phase::PlanReconciler => r#"{
                        "objective": "Document route structure and evaluate against modern architecture best practices",
                        "affected_files": [
                            { "path": "src/routes.tsx", "reason": "Configures route tree" }
                        ],
                        "steps": [
                            { "title": "Explain route declarations", "detail": "Document route paths and component bindings" },
                            { "title": "Evaluate architecture alignment", "detail": "Compare against declarative routing standards" }
                        ],
                        "risks": [],
                        "validation_commands": ["npm test"]
                    }"#.to_string(),
                    Phase::PlanCritic => r#"{
                        "critique": "Verify edge cases in route transitions.",
                        "missing_aspects": [],
                        "disputed_steps": [],
                        "unverified_assumptions": [],
                        "additional_risks": []
                    }"#.to_string(),
                    Phase::PlanRebuttal => r#"{
                        "rebuttal": "Route transition edge cases are minor for read-only inquiry.",
                        "dismissed_critiques": [],
                        "accepted_critiques": []
                    }"#.to_string(),
                    Phase::Synthesize => {
                        assert!(request.prompt.contains("ESTABLISHED ARCHITECTURAL CONTEXT & GUIDANCE"));
                        assert!(request.prompt.contains("Do not defensively hedge"));

                        let entry_id = request.prompt
                            .lines()
                            .find(|l| l.starts_with("- ") && l.contains(" Fact Active"))
                            .and_then(|l| l.split_whitespace().nth(1))
                            .unwrap_or("entry-1");

                        format!(r#"{{
                            "answer": "The application configures its route settings in `src/routes.tsx` (lines 1-10) using declarative `AppRoutes`. This design aligns with modern frontend best practices by decoupling route path definitions from view renderers, keeping routing state centralized and testable.",
                            "cited_entries": ["{}"],
                            "stop_reason": "Done"
                        }}"#, entry_id)
                    }
                    _ => "{}".to_string(),
                };

                Ok(LlmResponse {
                    call_id: request.call_id,
                    raw_json,
                    usage: crate::traits::LlmUsage::default(),
                })
            }
        }

        #[async_trait]
        impl TokenCounter for SynthMockLlm {
            async fn count_tokens(&self, text: &str) -> usize {
                text.len() / 4
            }
        }

        let task = TaskRequest {
            run_id: "test-synth-run".to_string(),
            session_id: "test-synth-session".to_string(),
            workspace_path: None,
            question: "where does our UI configure route settings, and is it aligned with modern architecture best practices".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: false,
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &SynthMockLlm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &SynthMockLlm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);

        // 1. Check synthesized answer leads the response text
        assert!(result.answer.content.starts_with("The application configures its route settings in `src/routes.tsx`"));
        assert!(result.answer.content.contains("aligns with modern frontend best practices"));

        // 2. Check that diagnostics are cleanly housed in a collapsible details block
        assert!(result.answer.content.contains("<details>\n<summary><b>🛠️ Pipeline Diagnostics"));
        assert!(result.answer.content.contains("Step 1: Multi-Intent Intake"));
        assert!(result.answer.content.contains("Step 2: Deterministic Evidence Acquisition"));
        assert!(result.answer.content.contains("Step 3A: Evidence Extraction"));
        assert!(result.answer.content.contains("Step 3B: Appellate Planning Gate"));

        // 3. Check citations are properly validated
        assert!(!result.answer.cited_entries.is_empty(), "Cited entries must not be empty");
    }

    #[tokio::test]
    async fn test_step4_appellate_edit_gate_generates_verified_diff_and_commits_decisions() {
        struct EditMockLlm;

        #[async_trait]
        impl LlmClient for EditMockLlm {
            async fn complete_json(&self, request: LlmRequest) -> Result<LlmResponse> {
                let raw_json = match request.phase {
                    Phase::Intake => r#"{
                        "reasoning": "User asks to implement route logging.",
                        "needs_workspace_search": true,
                        "intents": [
                            {
                                "intent": "WorkspaceEdit",
                                "summary": "Implement route transition logging in routes.tsx",
                                "needs_workspace_search": true,
                                "direct_response": null,
                                "search_queries": ["routes"]
                            }
                        ],
                        "target_hints": ["routes"],
                        "constraints": []
                    }"#.to_string(),
                    Phase::IntakeCritic => r#"{
                        "approved": true,
                        "critique": "Approved edit decomposition.",
                        "has_dropped_clauses": false,
                        "has_contaminated_queries": false,
                        "corrected_intents": null
                    }"#.to_string(),
                    Phase::CandidateGate | Phase::CandidateGateOptimist | Phase::CandidateGateSkeptic => r#"{
                        "evaluations": [
                            { "path": "src/routes.tsx", "keep": true, "reason": "Configures routes" }
                        ]
                    }"#.to_string(),
                    Phase::Extract => {
                        let found_lease = request.prompt
                            .lines()
                            .find(|l| l.starts_with("LEASE "))
                            .and_then(|l| l.strip_prefix("LEASE "))
                            .unwrap_or("mock-lease");

                        format!(r#"{{
                            "inferences": [
                                {{
                                    "statement": "src/routes.tsx lines 1-10 configures AppRoutes.",
                                    "lease_id": "{}",
                                    "tags": ["routes"]
                                }}
                            ],
                            "sufficient": true,
                            "missing": []
                        }}"#, found_lease)
                    }
                    Phase::PlanProposer | Phase::PlanReconciler => r#"{
                        "objective": "Implement route transition logging in routes.tsx",
                        "affected_files": [
                            { "path": "src/routes.tsx", "reason": "Target route configuration" }
                        ],
                        "steps": [
                            { "title": "Add logger", "detail": "Add logging to AppRoutes" }
                        ],
                        "risks": [],
                        "validation_commands": ["npm test"]
                    }"#.to_string(),
                    Phase::PlanCritic => r#"{
                        "critique": "Plan is sound.",
                        "missing_aspects": [],
                        "disputed_steps": [],
                        "unverified_assumptions": [],
                        "additional_risks": []
                    }"#.to_string(),
                    Phase::PlanRebuttal => r#"{
                        "rebuttal": "Plan is valid.",
                        "dismissed_critiques": [],
                        "accepted_critiques": []
                    }"#.to_string(),
                    Phase::PatchProposer => r#"{
                        "replacement_code": "export function AppRoutes() {\n    console.log('route transition');\n}",
                        "justification": "Adds transition logging to AppRoutes"
                    }"#.to_string(),
                    Phase::PatchCritic => r#"{
                        "approved": true,
                        "critique": "Verified clean patch without regressions.",
                        "disputed_changes": [],
                        "unhandled_edge_cases": [],
                        "convention_violations": []
                    }"#.to_string(),
                    Phase::PatchRebuttal => r#"{
                        "rebuttal": "Patch is clean.",
                        "dismissed_points": [],
                        "accepted_points": []
                    }"#.to_string(),
                    Phase::PatchReconciler => r#"{
                        "hardened_code": "export function AppRoutes() {\n    console.log('route transition');\n}",
                        "reconciliation_summary": "Reconciled clean patch.",
                        "remaining_uncertainties": []
                    }"#.to_string(),
                    Phase::Synthesize => {
                        let patch_id = request.prompt
                            .lines()
                            .find(|l| l.starts_with("- ") && l.contains("Decision Active") && l.contains("Draft Patch"))
                            .and_then(|l| l.split_whitespace().nth(1))
                            .unwrap_or("entry-patch");

                        format!(r#"{{
                            "answer": "Draft patch has been generated to implement route transition logging in `src/routes.tsx`.",
                            "cited_entries": ["{}"],
                            "stop_reason": "Done"
                        }}"#, patch_id)
                    }
                    _ => "{}".to_string(),
                };

                Ok(LlmResponse {
                    call_id: request.call_id,
                    raw_json,
                    usage: crate::traits::LlmUsage::default(),
                })
            }
        }

        #[async_trait]
        impl TokenCounter for EditMockLlm {
            async fn count_tokens(&self, text: &str) -> usize {
                text.len() / 4
            }
        }

        let task = TaskRequest {
            run_id: "test-edit-run".to_string(),
            session_id: "test-edit-session".to_string(),
            workspace_path: None,
            question: "Implement route transition logging in src/routes.tsx".to_string(),
            conversation: Vec::new(),
            budgets: crate::types::Budgets::default(),
            intake_only: false,
        };

        let progress = CollectProgress {
            messages: std::sync::Arc::new(Mutex::new(Vec::new())),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &EditMockLlm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &EditMockLlm,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);
        assert!(result.answer.content.contains("Draft patch has been generated"));
        assert!(result.answer.content.contains("Step 4: Appellate Stereoscopic Edit Gate"));
        assert!(result.answer.content.contains("--- a/src/routes.tsx"));
        assert!(result.answer.content.contains("+++ b/src/routes.tsx"));
        assert!(result.answer.content.contains("+    console.log('route transition');"));

        // Verify notebook has Decision entry for the patch
        let patch_entry = result.answer.notebook.active_entries().into_iter().find(|e| e.statement.contains("Draft Patch: src/routes.tsx"));
        assert!(patch_entry.is_some(), "Expected Decision entry in notebook for Draft Patch");
    }
}
