use std::collections::HashSet;
use crate::notebook::{EntryId, Notebook, Provenance};
use crate::prompt::{
    evidence_block, extract_instruction, global_prefix, intake_critic_instruction,
    intake_instruction, notebook_block, phase_prefix, prompt_candidate_gate_optimist,
    prompt_candidate_gate_skeptic, schema_candidate_gate, schema_extract, schema_intake,
    schema_intake_critic, task_block, CandidateGateItem, PromptParts,
};
use crate::trace::{Trace, TraceEvent};
use crate::traits::{
    AgentProgress, CodeSearch, FileReader, LlmClient, LlmRequest, LlmResponse, ProgressSink,
    TokenCounter,
};
use crate::types::{
    AgentAnswer, CandidateGateOutput, ContentHit, CriticOutput, EvidenceSlice, ExtractOutput,
    FileHit, IntakeOutput, IntentItem, MissingEvidence, Phase, StopReason, SymbolHit, TaskRequest,
    TriadConsensus, UserIntent, VectorHit,
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
        phase: Phase::Extract,
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
        phase: Phase::Extract,
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

    let content = format_step1_and_step2_report(
        &task,
        &intake_prompt,
        &intake_resp,
        &intake,
        &critic_output,
        &triad,
        &phase2,
        extract_result.as_ref(),
        &notebook,
    );

    let answer = AgentAnswer {
        content,
        stop_reason: StopReason::Done,
        cited_entries,
        notebook,
    };

    record_trace!(TraceEvent::RunFinished {
        stop_reason: StopReason::Done,
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
                    "In frontend desktop and single-window applications, navigation is commonly structured either via state-driven view switching (e.g. conditional rendering based on active session or application state) or via dedicated declarative routers (e.g. React Router, Solid Router, or TanStack Router). When URLs or browser history are not required, state-driven view switching is clean, minimal, and avoids unnecessary routing dependencies.".to_string(),
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
}
