use crate::notebook::{EntryId, EvidenceLease, LineRange, Notebook, Provenance};
use crate::prompt::{
    evidence_block, extract_instruction, global_prefix, intake_instruction, notebook_block,
    phase_prefix, plan_instruction, schema_extract, schema_intake, schema_plan, schema_synthesize,
    synthesize_instruction, task_block, PromptParts,
};
use crate::trace::{Trace, TraceEvent};
use crate::traits::{
    AgentProgress, CodeSearch, FileReader, LlmClient, LlmRequest, ProgressSink, TokenCounter,
};
use crate::types::{
    AgentAnswer, EditPlan, EvidenceSlice, ExtractOutput, IntakeOutput, LeaseStatus,
    MissingEvidence, MissingEvidenceKind, Phase, StopReason, SymbolRelation, SynthesizeOutput,
    TaskRequest, TaskType,
};
use anyhow::{anyhow, Context, Result};
use serde::de::DeserializeOwned;
use std::collections::{BTreeMap, BTreeSet};
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

pub async fn run_agent(task: TaskRequest, runtime: AgentRuntime<'_>) -> Result<AgentRunResult> {
    let mut trace = Trace::default();
    trace.push(TraceEvent::RunStarted {
        run_id: task.run_id.clone(),
        question: task.question.clone(),
    });

    let repo_map = runtime.code_search.repo_map(80).await.unwrap_or_default();
    let global = global_prefix(&repo_map);
    let mut notebook = Notebook::default();
    let mut llm_calls = 0usize;

    runtime
        .progress
        .emit(AgentProgress::Status(
            "Classifying the user's request".to_string(),
        ))
        .await;
    trace.push(TraceEvent::PhaseStarted {
        phase: Phase::Intake,
    });

    let intake_parts = PromptParts {
        global_prefix: global.clone(),
        phase_prefix: phase_prefix(Phase::Intake, "IntakeOutput"),
        task: task_block(&task),
        notebook: notebook_block(&notebook, &[], 16),
        evidence: evidence_block(&[]),
        question: intake_instruction(),
    };
    let intake_prompt = intake_parts.render();

    let intake_req = LlmRequest {
        call_id: call_id("intake"),
        phase: Phase::Intake,
        schema_name: "IntakeOutput".to_string(),
        schema: schema_intake(),
        prompt_parts: Some(intake_parts),
        prompt: intake_prompt,
        temperature: 0.0,
    };
    trace.push(TraceEvent::NotebookSnapshot {
        phase: Phase::Intake,
        notebook: notebook.deterministic_checkpoint(),
    });
    runtime
        .progress
        .emit(AgentProgress::Status(
            "Planning workspace search".to_string(),
        ))
        .await;
    let intake_resp = runtime.llm.complete_json(intake_req.clone()).await?;
    llm_calls += 1;
    trace.push(TraceEvent::LlmCall {
        request: intake_req,
        response: intake_resp.clone(),
    });
    let mut intake: IntakeOutput =
        parse_json(&intake_resp.raw_json).unwrap_or_else(|_| fallback_intake(&task.question));

    normalize_intake(&mut intake, &task);
    notebook.append_decision(
        format!("Task classified as {:?}", intake.task_type),
        Provenance::Model {
            call_id: intake_resp.call_id,
        },
        Phase::Intake,
        Vec::new(),
        intake.target_hints.clone(),
    );

    if intake.task_type == TaskType::Unclear {
        let answer = AgentAnswer {
            content:
                "I need a more specific code question before I can inspect the workspace usefully."
                    .to_string(),
            stop_reason: StopReason::NeedsUser,
            cited_entries: Vec::new(),
            notebook,
        };
        trace.push(TraceEvent::RunFinished {
            stop_reason: StopReason::NeedsUser,
        });
        return Ok(AgentRunResult { answer, trace });
    }

    runtime
        .progress
        .emit(AgentProgress::Status(format!(
            "Searching for {}",
            format_list(&intake.search_queries, task.budgets.max_search_queries)
        )))
        .await;

    runtime
        .progress
        .emit(AgentProgress::Status(format!(
            "Checking {} focused workspace queries",
            intake
                .search_queries
                .len()
                .min(task.budgets.max_search_queries)
        )))
        .await;

    let mut file_paths = BTreeMap::<String, i64>::new();
    let mut symbol_hits = BTreeMap::<(String, String, u32, u32), crate::types::SymbolHit>::new();
    let mut content_hits = BTreeMap::<(String, u32, u32), crate::types::ContentHit>::new();
    let mut changed_paths = Vec::<String>::new();
    let relation_intents = relation_intents_for_question(&task.question);
    for query in intake
        .search_queries
        .iter()
        .take(task.budgets.max_search_queries)
    {
        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Finding files related to {}",
                plain_query(query)
            )))
            .await;
        let hits = runtime
            .code_search
            .search(query, task.budgets.max_files_to_read)
            .await
            .unwrap_or_default();
        trace.push(TraceEvent::ToolCall {
            name: "code_search".to_string(),
            args: serde_json::json!({ "query": query }),
            result_count: hits.len(),
        });
        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Found {} file path{} for {}",
                hits.len(),
                plural(hits.len()),
                plain_query(query)
            )))
            .await;
        for hit in hits {
            file_paths
                .entry(hit.path)
                .and_modify(|score| *score += hit.score)
                .or_insert(hit.score);
        }

        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Finding symbols related to {}",
                plain_query(query)
            )))
            .await;
        let symbols = runtime
            .code_search
            .search_symbols(query, task.budgets.max_files_to_read)
            .await
            .unwrap_or_default();
        trace.push(TraceEvent::ToolCall {
            name: "code_search_symbols".to_string(),
            args: serde_json::json!({ "query": query }),
            result_count: symbols.len(),
        });
        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Found {} symbol{} for {}",
                symbols.len(),
                plural(symbols.len()),
                plain_query(query)
            )))
            .await;
        for hit in symbols {
            symbol_hits
                .entry((
                    hit.path.clone(),
                    hit.symbol.clone(),
                    hit.start_line,
                    hit.end_line,
                ))
                .and_modify(|existing| existing.score += hit.score)
                .or_insert(hit);
        }

        for relation in &relation_intents {
            runtime
                .progress
                .emit(AgentProgress::Status(format!(
                    "Finding {:?} symbols for {}",
                    relation,
                    plain_query(query)
                )))
                .await;
            let related = runtime
                .code_search
                .search_related_symbols(*relation, query, task.budgets.max_files_to_read)
                .await
                .unwrap_or_default();
            trace.push(TraceEvent::ToolCall {
                name: "code_search_related_symbols".to_string(),
                args: serde_json::json!({
                    "query": query,
                    "relation": format!("{:?}", relation),
                    "initial": true
                }),
                result_count: related.len(),
            });
            for hit in related {
                symbol_hits
                    .entry((
                        hit.path.clone(),
                        hit.symbol.clone(),
                        hit.start_line,
                        hit.end_line,
                    ))
                    .and_modify(|existing| existing.score += hit.score)
                    .or_insert(hit);
            }
        }

        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Finding text matches for {}",
                plain_query(query)
            )))
            .await;
        let text_hits = runtime
            .code_search
            .search_content(query, task.budgets.max_files_to_read * 3)
            .await
            .unwrap_or_default();
        trace.push(TraceEvent::ToolCall {
            name: "code_search_content".to_string(),
            args: serde_json::json!({ "query": query }),
            result_count: text_hits.len(),
        });
        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Found {} text match{} for {}",
                text_hits.len(),
                plural(text_hits.len()),
                plain_query(query)
            )))
            .await;
        for hit in text_hits {
            content_hits
                .entry((hit.path.clone(), hit.start_line, hit.end_line))
                .and_modify(|existing| existing.score += hit.score)
                .or_insert(hit);
        }
    }

    if wants_changed_file_context(&intake, &task.question) {
        runtime
            .progress
            .emit(AgentProgress::Status(
                "Checking current workspace changes".to_string(),
            ))
            .await;
        let changed = runtime
            .code_search
            .changed_files(task.budgets.max_files_to_read)
            .await
            .unwrap_or_default();
        trace.push(TraceEvent::ToolCall {
            name: "changed_files".to_string(),
            args: serde_json::json!({ "limit": task.budgets.max_files_to_read }),
            result_count: changed.len(),
        });
        for hit in changed {
            changed_paths.push(hit.path.clone());
            file_paths
                .entry(hit.path)
                .and_modify(|score| *score += hit.score)
                .or_insert(hit.score);
        }
    }

    if wants_adjacent_file_context(&intake, &task.question) {
        let seed_paths = adjacent_seed_paths(
            &file_paths,
            &symbol_hits,
            &content_hits,
            &changed_paths,
            task.budgets.max_files_to_read,
        );
        if !seed_paths.is_empty() {
            runtime
                .progress
                .emit(AgentProgress::Status(format!(
                    "Finding files adjacent to {}",
                    format_list(&seed_paths, task.budgets.max_files_to_read)
                )))
                .await;
            let adjacent = runtime
                .code_search
                .search_adjacent_files(&seed_paths, task.budgets.max_files_to_read)
                .await
                .unwrap_or_default();
            trace.push(TraceEvent::ToolCall {
                name: "code_search_adjacent_files".to_string(),
                args: serde_json::json!({ "seed_paths": seed_paths }),
                result_count: adjacent.len(),
            });
            for hit in adjacent {
                file_paths
                    .entry(hit.path)
                    .and_modify(|score| *score += hit.score)
                    .or_insert(hit.score);
            }
        }
    }

    if wants_imported_file_context(&intake, &task.question) {
        let seed_paths = adjacent_seed_paths(
            &file_paths,
            &symbol_hits,
            &content_hits,
            &changed_paths,
            task.budgets.max_files_to_read,
        );
        if !seed_paths.is_empty() {
            runtime
                .progress
                .emit(AgentProgress::Status(format!(
                    "Finding files imported by {}",
                    format_list(&seed_paths, task.budgets.max_files_to_read)
                )))
                .await;
            let imported = runtime
                .code_search
                .search_imported_files(&seed_paths, task.budgets.max_files_to_read)
                .await
                .unwrap_or_default();
            trace.push(TraceEvent::ToolCall {
                name: "code_search_imported_files".to_string(),
                args: serde_json::json!({ "seed_paths": seed_paths }),
                result_count: imported.len(),
            });
            for hit in imported {
                file_paths
                    .entry(hit.path)
                    .and_modify(|score| *score += hit.score)
                    .or_insert(hit.score);
            }
        }
    }

    if wants_importing_file_context(&intake, &task.question) {
        let seed_paths = adjacent_seed_paths(
            &file_paths,
            &symbol_hits,
            &content_hits,
            &changed_paths,
            task.budgets.max_files_to_read,
        );
        if !seed_paths.is_empty() {
            runtime
                .progress
                .emit(AgentProgress::Status(format!(
                    "Finding files that import {}",
                    format_list(&seed_paths, task.budgets.max_files_to_read)
                )))
                .await;
            let importing = runtime
                .code_search
                .search_importing_files(&seed_paths, task.budgets.max_files_to_read)
                .await
                .unwrap_or_default();
            trace.push(TraceEvent::ToolCall {
                name: "code_search_importing_files".to_string(),
                args: serde_json::json!({ "seed_paths": seed_paths }),
                result_count: importing.len(),
            });
            for hit in importing {
                file_paths
                    .entry(hit.path)
                    .and_modify(|score| *score += hit.score)
                    .or_insert(hit.score);
            }
        }
    }

    if wants_scope_file_context(&intake, &task.question) {
        let seed_paths = adjacent_seed_paths(
            &file_paths,
            &symbol_hits,
            &content_hits,
            &changed_paths,
            task.budgets.max_files_to_read,
        );
        if !seed_paths.is_empty() {
            runtime
                .progress
                .emit(AgentProgress::Status(format!(
                    "Finding local project context for {}",
                    format_list(&seed_paths, task.budgets.max_files_to_read)
                )))
                .await;
            let scope_files = runtime
                .code_search
                .search_scope_files(&seed_paths, task.budgets.max_files_to_read)
                .await
                .unwrap_or_default();
            trace.push(TraceEvent::ToolCall {
                name: "code_search_scope_files".to_string(),
                args: serde_json::json!({ "seed_paths": seed_paths }),
                result_count: scope_files.len(),
            });
            for hit in scope_files {
                file_paths
                    .entry(hit.path)
                    .and_modify(|score| *score += hit.score)
                    .or_insert(hit.score);
            }
        }
    }

    if wants_context_doc_context(&intake, &task.question) {
        let seed_paths = adjacent_seed_paths(
            &file_paths,
            &symbol_hits,
            &content_hits,
            &changed_paths,
            task.budgets.max_files_to_read,
        );
        if !seed_paths.is_empty() {
            let queries = intake
                .search_queries
                .iter()
                .take(task.budgets.max_search_queries)
                .cloned()
                .collect::<Vec<_>>();
            if !queries.is_empty() {
                runtime
                    .progress
                    .emit(AgentProgress::Status(format!(
                        "Finding nearby docs for {}",
                        format_list(&queries, task.budgets.max_search_queries)
                    )))
                    .await;
                let context_docs = runtime
                    .code_search
                    .search_context_docs(&queries, &seed_paths, task.budgets.max_files_to_read)
                    .await
                    .unwrap_or_default();
                trace.push(TraceEvent::ToolCall {
                    name: "code_search_context_docs".to_string(),
                    args: serde_json::json!({ "queries": queries, "seed_paths": seed_paths }),
                    result_count: context_docs.len(),
                });
                for hit in context_docs {
                    file_paths
                        .entry(hit.path)
                        .and_modify(|score| *score += hit.score)
                        .or_insert(hit.score);
                }
            }
        }
    }

    let explicit_directories = extract_directory_queries(&task.question);
    if !explicit_directories.is_empty() {
        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Finding files under {}",
                format_list(&explicit_directories, task.budgets.max_files_to_read)
            )))
            .await;
        let directory_files = runtime
            .code_search
            .search_directory_files(&explicit_directories, task.budgets.max_files_to_read)
            .await
            .unwrap_or_default();
        trace.push(TraceEvent::ToolCall {
            name: "code_search_directory_files".to_string(),
            args: serde_json::json!({ "directories": explicit_directories }),
            result_count: directory_files.len(),
        });
        for hit in directory_files {
            file_paths
                .entry(hit.path)
                .and_modify(|score| *score += hit.score)
                .or_insert(hit.score);
        }
    }

    let mut sorted_paths: Vec<_> = file_paths.into_iter().collect();
    sorted_paths.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut evidence = Vec::new();
    let mut evidence_tokens = 0usize;
    let mut lease_fact_entries = BTreeMap::new();
    let mut read_paths = BTreeSet::<String>::new();
    let explicit_ranges = extract_file_range_queries(&task.question);
    let explicit_range_paths = explicit_ranges
        .iter()
        .map(|(path, _, _)| path.clone())
        .collect::<BTreeSet<_>>();
    for (path, start_line, end_line) in explicit_ranges
        .into_iter()
        .take(task.budgets.max_files_to_read)
    {
        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Inspecting requested range `{}` lines {}-{}",
                path, start_line, end_line
            )))
            .await;
        let slice_result = runtime
            .file_reader
            .read_range(&path, None, start_line, end_line, per_slice_budget(&task))
            .await;
        trace.push(TraceEvent::ToolCall {
            name: "read_explicit_file_range".to_string(),
            args: serde_json::json!({
                "path": path,
                "start_line": start_line,
                "end_line": end_line
            }),
            result_count: usize::from(slice_result.is_ok()),
        });
        match slice_result {
            Ok(slice) => {
                let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                    runtime
                        .progress
                        .emit(AgentProgress::Status(format!(
                            "Skipping `{}` lines {}-{} because the evidence budget is full",
                            slice.lease.path, slice.lease.range.start, slice.lease.range.end
                        )))
                        .await;
                    continue;
                }
                let fact_id = notebook.append_tool_fact(
                    format!(
                        "Read explicit requested range in {} lines {}-{} under lease {}",
                        slice.lease.path,
                        slice.lease.range.start,
                        slice.lease.range.end,
                        slice.lease.id.0
                    ),
                    Provenance::File {
                        path: slice.lease.path.clone(),
                        symbol: None,
                        range: slice.lease.range,
                        lease_id: slice.lease.id.clone(),
                    },
                    Phase::Extract,
                    vec![slice.lease.path.clone()],
                );
                lease_fact_entries.insert(slice.lease.id.clone(), fact_id);
                read_paths.insert(slice.lease.path.clone());
                evidence_tokens += token_count;
                evidence.push(slice);
            }
            Err(err) => {
                notebook.append_tool_fact(
                    format!(
                        "Could not read explicit requested range in {} lines {}-{}: {}",
                        path, start_line, end_line, err
                    ),
                    Provenance::Tool {
                        name: "read_range".to_string(),
                        call_id: call_id("read_range"),
                    },
                    Phase::Extract,
                    vec![path],
                );
            }
        }
    }
    let explicit_files = extract_file_path_queries(&task.question);
    for path in explicit_files
        .into_iter()
        .filter(|path| !explicit_range_paths.contains(path))
        .take(task.budgets.max_files_to_read)
    {
        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Inspecting requested file `{}`",
                path
            )))
            .await;
        let slice_result = runtime
            .file_reader
            .read_file(&path, per_slice_budget(&task))
            .await;
        trace.push(TraceEvent::ToolCall {
            name: "read_explicit_file".to_string(),
            args: serde_json::json!({ "path": path }),
            result_count: usize::from(slice_result.is_ok()),
        });
        match slice_result {
            Ok(slice) => {
                let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                    runtime
                        .progress
                        .emit(AgentProgress::Status(format!(
                            "Skipping `{}` lines {}-{} because the evidence budget is full",
                            slice.lease.path, slice.lease.range.start, slice.lease.range.end
                        )))
                        .await;
                    continue;
                }
                let fact_id = notebook.append_tool_fact(
                    format!(
                        "Read explicit requested file {} lines {}-{} under lease {}",
                        slice.lease.path,
                        slice.lease.range.start,
                        slice.lease.range.end,
                        slice.lease.id.0
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
                lease_fact_entries.insert(slice.lease.id.clone(), fact_id);
                read_paths.insert(slice.lease.path.clone());
                evidence_tokens += token_count;
                evidence.push(slice);
            }
            Err(err) => {
                notebook.append_tool_fact(
                    format!("Could not read explicit requested file {}: {}", path, err),
                    Provenance::Tool {
                        name: "read_file".to_string(),
                        call_id: call_id("read_file"),
                    },
                    Phase::Extract,
                    vec![path],
                );
            }
        }
    }
    dedup(&mut changed_paths);
    for path in changed_paths.iter().take(task.budgets.max_files_to_read) {
        match runtime
            .file_reader
            .read_diff(path, per_slice_budget(&task))
            .await
        {
            Ok(slice) => {
                let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                    runtime
                        .progress
                        .emit(AgentProgress::Status(format!(
                            "Skipping diff for `{}` because the evidence budget is full",
                            path
                        )))
                        .await;
                    continue;
                }
                trace.push(TraceEvent::ToolCall {
                    name: "read_changed_diff".to_string(),
                    args: serde_json::json!({ "path": path }),
                    result_count: 1,
                });
                let fact_id = notebook.append_tool_fact(
                    format!(
                        "Read diff for {} lines {}-{} under lease {}",
                        slice.lease.path,
                        slice.lease.range.start,
                        slice.lease.range.end,
                        slice.lease.id.0
                    ),
                    Provenance::File {
                        path: slice.lease.path.clone(),
                        symbol: slice.lease.symbol.clone(),
                        range: slice.lease.range,
                        lease_id: slice.lease.id.clone(),
                    },
                    Phase::Extract,
                    vec![slice.lease.path.clone(), "git diff".to_string()],
                );
                lease_fact_entries.insert(slice.lease.id.clone(), fact_id);
                evidence_tokens += token_count;
                evidence.push(slice);
            }
            Err(_) => {
                trace.push(TraceEvent::ToolCall {
                    name: "read_changed_diff".to_string(),
                    args: serde_json::json!({ "path": path }),
                    result_count: 0,
                });
            }
        }
    }
    let mut sorted_symbols = symbol_hits.into_values().collect::<Vec<_>>();
    sorted_symbols.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.start_line.cmp(&b.start_line))
            .then_with(|| a.symbol.cmp(&b.symbol))
    });
    let selected_symbol_hits = sorted_symbols
        .into_iter()
        .take(task.budgets.max_files_to_read)
        .collect::<Vec<_>>();
    for hit in selected_symbol_hits.iter().cloned() {
        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Inspecting `{}` in `{}` lines {}-{}",
                hit.symbol, hit.path, hit.start_line, hit.end_line
            )))
            .await;
        match runtime
            .file_reader
            .read_range(
                &hit.path,
                Some(&hit.symbol),
                hit.start_line,
                hit.end_line,
                per_slice_budget(&task),
            )
            .await
        {
            Ok(slice) => {
                let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                    runtime
                        .progress
                        .emit(AgentProgress::Status(format!(
                            "Skipping `{}` lines {}-{} because the evidence budget is full",
                            slice.lease.path, slice.lease.range.start, slice.lease.range.end
                        )))
                        .await;
                    notebook.append_tool_fact(
                        format!(
                            "Skipped symbol {} in {} lines {}-{} because it would exceed the evidence token budget",
                            hit.symbol,
                            slice.lease.path,
                            slice.lease.range.start,
                            slice.lease.range.end
                        ),
                        Provenance::Tool {
                            name: "token_budget".to_string(),
                            call_id: call_id("token_budget"),
                        },
                        Phase::Extract,
                        vec![slice.lease.path.clone(), hit.symbol],
                    );
                    continue;
                }
                let fact_id = notebook.append_tool_fact(
                    format!(
                        "Read symbol {} in {} lines {}-{} under lease {}",
                        hit.symbol,
                        slice.lease.path,
                        slice.lease.range.start,
                        slice.lease.range.end,
                        slice.lease.id.0
                    ),
                    Provenance::File {
                        path: slice.lease.path.clone(),
                        symbol: slice.lease.symbol.clone(),
                        range: slice.lease.range,
                        lease_id: slice.lease.id.clone(),
                    },
                    Phase::Extract,
                    vec![slice.lease.path.clone(), hit.symbol],
                );
                lease_fact_entries.insert(slice.lease.id.clone(), fact_id);
                read_paths.insert(slice.lease.path.clone());
                evidence_tokens += token_count;
                evidence.push(slice);
                runtime
                    .progress
                    .emit(AgentProgress::Status(format!(
                        "Using `{}` lines {}-{} as evidence",
                        evidence
                            .last()
                            .map(|slice| slice.lease.path.as_str())
                            .unwrap_or("unknown"),
                        evidence
                            .last()
                            .map(|slice| slice.lease.range.start)
                            .unwrap_or_default(),
                        evidence
                            .last()
                            .map(|slice| slice.lease.range.end)
                            .unwrap_or_default()
                    )))
                    .await;
            }
            Err(err) => {
                notebook.append_tool_fact(
                    format!(
                        "Could not read symbol {} in {} lines {}-{}: {}",
                        hit.symbol, hit.path, hit.start_line, hit.end_line, err
                    ),
                    Provenance::Tool {
                        name: "read_range".to_string(),
                        call_id: call_id("read_range"),
                    },
                    Phase::Extract,
                    vec![hit.path, hit.symbol],
                );
            }
        }
    }

    for expansion in related_context_expansions(&intake, &task.question) {
        for seed in selected_symbol_hits.iter().take(2) {
            runtime
                .progress
                .emit(AgentProgress::Status(format!(
                    "{} `{}`",
                    expansion.progress, seed.symbol
                )))
                .await;
            let related = runtime
                .code_search
                .search_related_symbols(expansion.relation, &seed.symbol, expansion.limit)
                .await
                .unwrap_or_default();
            trace.push(TraceEvent::ToolCall {
                name: "code_search_related_symbols".to_string(),
                args: serde_json::json!({
                    "query": seed.symbol,
                    "relation": format!("{:?}", expansion.relation),
                    "context_expansion": true,
                    "context_kind": expansion.kind
                }),
                result_count: related.len(),
            });
            for hit in related {
                if evidence_overlaps_range(&evidence, &hit.path, hit.start_line, hit.end_line) {
                    continue;
                }
                let Ok(slice) = runtime
                    .file_reader
                    .read_range(
                        &hit.path,
                        Some(&hit.symbol),
                        hit.start_line,
                        hit.end_line,
                        per_slice_budget(&task),
                    )
                    .await
                else {
                    continue;
                };
                let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                    runtime
                        .progress
                        .emit(AgentProgress::Status(format!(
                            "Skipping `{}` lines {}-{} because the evidence budget is full",
                            slice.lease.path, slice.lease.range.start, slice.lease.range.end
                        )))
                        .await;
                    continue;
                }
                let fact_id = notebook.append_tool_fact(
                    format!(
                        "Read {} {} for {} in {} lines {}-{} under lease {}",
                        expansion.fact_label,
                        hit.symbol,
                        seed.symbol,
                        slice.lease.path,
                        slice.lease.range.start,
                        slice.lease.range.end,
                        slice.lease.id.0
                    ),
                    Provenance::File {
                        path: slice.lease.path.clone(),
                        symbol: slice.lease.symbol.clone(),
                        range: slice.lease.range,
                        lease_id: slice.lease.id.clone(),
                    },
                    Phase::Extract,
                    vec![slice.lease.path.clone(), hit.symbol, seed.symbol.clone()],
                );
                lease_fact_entries.insert(slice.lease.id.clone(), fact_id);
                read_paths.insert(slice.lease.path.clone());
                evidence_tokens += token_count;
                evidence.push(slice);
            }
        }
    }

    let mut sorted_content_hits = content_hits.into_values().collect::<Vec<_>>();
    sorted_content_hits.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.start_line.cmp(&b.start_line))
    });
    for hit in sorted_content_hits
        .into_iter()
        .take(task.budgets.max_files_to_read)
    {
        if evidence_overlaps_range(&evidence, &hit.path, hit.start_line, hit.end_line) {
            continue;
        }
        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Inspecting text match in `{}` lines {}-{}",
                hit.path, hit.start_line, hit.end_line
            )))
            .await;
        match runtime
            .file_reader
            .read_range(
                &hit.path,
                None,
                hit.start_line,
                hit.end_line,
                per_slice_budget(&task),
            )
            .await
        {
            Ok(slice) => {
                let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                    runtime
                        .progress
                        .emit(AgentProgress::Status(format!(
                            "Skipping `{}` lines {}-{} because the evidence budget is full",
                            slice.lease.path, slice.lease.range.start, slice.lease.range.end
                        )))
                        .await;
                    notebook.append_tool_fact(
                        format!(
                            "Skipped text match in {} lines {}-{} because it would exceed the evidence token budget",
                            slice.lease.path, slice.lease.range.start, slice.lease.range.end
                        ),
                        Provenance::Tool {
                            name: "token_budget".to_string(),
                            call_id: call_id("token_budget"),
                        },
                        Phase::Extract,
                        vec![slice.lease.path.clone()],
                    );
                    continue;
                }
                let fact_id = notebook.append_tool_fact(
                    format!(
                        "Read text match in {} lines {}-{} under lease {}",
                        slice.lease.path,
                        slice.lease.range.start,
                        slice.lease.range.end,
                        slice.lease.id.0
                    ),
                    Provenance::File {
                        path: slice.lease.path.clone(),
                        symbol: None,
                        range: slice.lease.range,
                        lease_id: slice.lease.id.clone(),
                    },
                    Phase::Extract,
                    vec![slice.lease.path.clone()],
                );
                lease_fact_entries.insert(slice.lease.id.clone(), fact_id);
                read_paths.insert(slice.lease.path.clone());
                evidence_tokens += token_count;
                evidence.push(slice);
                runtime
                    .progress
                    .emit(AgentProgress::Status(format!(
                        "Using `{}` lines {}-{} as evidence",
                        evidence
                            .last()
                            .map(|slice| slice.lease.path.as_str())
                            .unwrap_or("unknown"),
                        evidence
                            .last()
                            .map(|slice| slice.lease.range.start)
                            .unwrap_or_default(),
                        evidence
                            .last()
                            .map(|slice| slice.lease.range.end)
                            .unwrap_or_default()
                    )))
                    .await;
            }
            Err(err) => {
                notebook.append_tool_fact(
                    format!(
                        "Could not read text match in {} lines {}-{}: {}",
                        hit.path, hit.start_line, hit.end_line, err
                    ),
                    Provenance::Tool {
                        name: "read_range".to_string(),
                        call_id: call_id("read_range"),
                    },
                    Phase::Extract,
                    vec![hit.path],
                );
            }
        }
    }

    for (path, _) in sorted_paths
        .into_iter()
        .take(task.budgets.max_files_to_read)
    {
        if read_paths.contains(&path) && !evidence.is_empty() {
            continue;
        }
        runtime
            .progress
            .emit(AgentProgress::Status(format!("Inspecting `{}`", path)))
            .await;
        match runtime
            .file_reader
            .read_file(&path, per_slice_budget(&task))
            .await
        {
            Ok(slice) => {
                runtime
                    .progress
                    .emit(AgentProgress::Status(format!(
                        "Finding the relevant section in `{}`",
                        slice.lease.path
                    )))
                    .await;
                let slice = focus_location_slice(slice, &task.question, &intake.search_queries);
                let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                    runtime
                        .progress
                        .emit(AgentProgress::Status(format!(
                            "Skipping `{}` lines {}-{} because the evidence budget is full",
                            slice.lease.path, slice.lease.range.start, slice.lease.range.end
                        )))
                        .await;
                    notebook.append_tool_fact(
                        format!(
                            "Skipped {} lines {}-{} because it would exceed the evidence token budget",
                            slice.lease.path, slice.lease.range.start, slice.lease.range.end
                        ),
                        Provenance::Tool {
                            name: "token_budget".to_string(),
                            call_id: call_id("token_budget"),
                        },
                        Phase::Extract,
                        vec![slice.lease.path.clone()],
                    );
                    continue;
                }
                let fact_id = notebook.append_tool_fact(
                    format!(
                        "Read {} lines {}-{} under lease {}",
                        slice.lease.path,
                        slice.lease.range.start,
                        slice.lease.range.end,
                        slice.lease.id.0
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
                lease_fact_entries.insert(slice.lease.id.clone(), fact_id);
                evidence_tokens += token_count;
                evidence.push(slice);
                runtime
                    .progress
                    .emit(AgentProgress::Status(format!(
                        "Using `{}` lines {}-{} as evidence",
                        evidence
                            .last()
                            .map(|slice| slice.lease.path.as_str())
                            .unwrap_or("unknown"),
                        evidence
                            .last()
                            .map(|slice| slice.lease.range.start)
                            .unwrap_or_default(),
                        evidence
                            .last()
                            .map(|slice| slice.lease.range.end)
                            .unwrap_or_default()
                    )))
                    .await;
            }
            Err(err) => {
                notebook.append_tool_fact(
                    format!("Could not read {}: {}", path, err),
                    Provenance::Tool {
                        name: "read_file".to_string(),
                        call_id: call_id("read_file"),
                    },
                    Phase::Extract,
                    vec![path],
                );
            }
        }
    }

    if evidence.is_empty() {
        let answer = AgentAnswer {
            content: "I could not find matching indexed files for this request.".to_string(),
            stop_reason: StopReason::NotFound,
            cited_entries: Vec::new(),
            notebook,
        };
        trace.push(TraceEvent::RunFinished {
            stop_reason: StopReason::NotFound,
        });
        return Ok(AgentRunResult { answer, trace });
    }

    let mut target_tags = intake.target_hints.clone();
    target_tags.extend(evidence.iter().map(|slice| slice.lease.path.clone()));
    dedup(&mut target_tags);

    let mut inference_entry_ids = Vec::<EntryId>::new();
    let mut previous_missing = BTreeSet::<String>::new();

    for round in 0..task.budgets.max_extract_rounds {
        if llm_calls >= task.budgets.max_llm_calls {
            break;
        }

        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Extracting supported findings, pass {}",
                round + 1
            )))
            .await;
        trace.push(TraceEvent::PhaseStarted {
            phase: Phase::Extract,
        });

        let extract_parts = PromptParts {
            global_prefix: global.clone(),
            phase_prefix: phase_prefix(Phase::Extract, "ExtractOutput"),
            task: task_block(&task),
            notebook: notebook_block(&notebook, &target_tags, 32),
            evidence: evidence_block(&evidence),
            question: extract_instruction(),
        };
        let extract_prompt = extract_parts.render();

        let extract_req = LlmRequest {
            call_id: call_id("extract"),
            phase: Phase::Extract,
            schema_name: "ExtractOutput".to_string(),
            schema: schema_extract(),
            prompt_parts: Some(extract_parts),
            prompt: extract_prompt,
            temperature: 0.0,
        };
        trace.push(TraceEvent::NotebookSnapshot {
            phase: Phase::Extract,
            notebook: notebook.deterministic_checkpoint(),
        });
        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Summarizing evidence into findings, pass {}",
                round + 1
            )))
            .await;
        let extract_resp = runtime.llm.complete_json(extract_req.clone()).await?;
        llm_calls += 1;
        trace.push(TraceEvent::LlmCall {
            request: extract_req,
            response: extract_resp.clone(),
        });

        let extract: ExtractOutput =
            parse_json(&extract_resp.raw_json).unwrap_or_else(|_| fallback_extract(&evidence));
        let lease_index = evidence
            .iter()
            .map(|slice| (slice.lease.id.clone(), slice.lease.clone()))
            .collect::<BTreeMap<_, _>>();

        let before_count = inference_entry_ids.len();
        for inference in extract.inferences {
            if let Some(lease) = lease_index.get(&inference.lease_id) {
                runtime
                    .progress
                    .emit(AgentProgress::Status(format!(
                        "Checking freshness for `{}` lines {}-{}",
                        lease.path, lease.range.start, lease.range.end
                    )))
                    .await;
                if runtime.file_reader.verify_lease(lease).await? == LeaseStatus::Expired {
                    notebook.mark_stale_by_lease(&lease.id);
                    trace.push(TraceEvent::LeaseInvalidated {
                        lease_id: lease.id.clone(),
                    });
                    continue;
                }
                let tags = if inference.tags.is_empty() {
                    vec![lease.path.clone()]
                } else {
                    inference.tags
                };
                let depends_on = lease_fact_entries
                    .get(&lease.id)
                    .cloned()
                    .into_iter()
                    .collect::<Vec<_>>();
                let entry_id = notebook.append_model_inference(
                    inference.statement,
                    Provenance::File {
                        path: lease.path.clone(),
                        symbol: lease.symbol.clone(),
                        range: lease.range,
                        lease_id: lease.id.clone(),
                    },
                    Phase::Extract,
                    depends_on,
                    tags,
                );
                inference_entry_ids.push(entry_id);
            }
        }
        let added_count = inference_entry_ids.len().saturating_sub(before_count);
        runtime
            .progress
            .emit(AgentProgress::Status(format!(
                "Accepted {} supported finding{} from pass {}",
                added_count,
                plural(added_count),
                round + 1
            )))
            .await;

        if extract.sufficient {
            break;
        }

        let mut new_missing = Vec::new();
        for missing in extract.missing {
            let target = missing.target.trim();
            let key = format!("{:?}:{}", missing.kind, target);
            if previous_missing.insert(key) && !target.is_empty() {
                new_missing.push(MissingEvidence {
                    kind: missing.kind,
                    target: target.to_string(),
                    reason: missing.reason,
                });
            }
        }

        if new_missing.is_empty() {
            break;
        }

        if inference_entry_ids.len() == before_count && round + 1 >= task.budgets.max_extract_rounds
        {
            break;
        }

        for missing in new_missing
            .into_iter()
            .take(task.budgets.max_search_queries.saturating_sub(1))
        {
            let query = missing.target;
            runtime
                .progress
                .emit(AgentProgress::Status(format!(
                    "Finding additional evidence for {}",
                    plain_query(&query)
                )))
                .await;

            if matches!(missing.kind, MissingEvidenceKind::FileRange) {
                if let Some((path, start_line, end_line)) = parse_file_range_target(&query) {
                    if evidence.iter().any(|slice| {
                        slice.lease.path == path
                            && slice.lease.range.start == start_line as usize
                            && slice.lease.range.end == end_line as usize
                    }) {
                        continue;
                    }
                    let slice_result = runtime
                        .file_reader
                        .read_range(&path, None, start_line, end_line, per_slice_budget(&task))
                        .await;
                    trace.push(TraceEvent::ToolCall {
                        name: "read_file_range".to_string(),
                        args: serde_json::json!({
                            "target": query,
                            "path": path,
                            "start_line": start_line,
                            "end_line": end_line,
                            "follow_up": true
                        }),
                        result_count: usize::from(slice_result.is_ok()),
                    });
                    if let Ok(slice) = slice_result {
                        let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                        if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                            continue;
                        }
                        let fact_id = notebook.append_tool_fact(
                            format!(
                                "Read requested file range in {} lines {}-{} under lease {}",
                                slice.lease.path,
                                slice.lease.range.start,
                                slice.lease.range.end,
                                slice.lease.id.0
                            ),
                            Provenance::File {
                                path: slice.lease.path.clone(),
                                symbol: None,
                                range: slice.lease.range,
                                lease_id: slice.lease.id.clone(),
                            },
                            Phase::Extract,
                            vec![slice.lease.path.clone(), query.clone()],
                        );
                        lease_fact_entries.insert(slice.lease.id.clone(), fact_id);
                        read_paths.insert(slice.lease.path.clone());
                        target_tags.push(slice.lease.path.clone());
                        target_tags.push(query.clone());
                        evidence_tokens += token_count;
                        evidence.push(slice);
                        dedup(&mut target_tags);
                        continue;
                    }
                }
            }

            if let Some(relation) = symbol_relation_for_missing(&missing.kind) {
                let related = runtime
                    .code_search
                    .search_related_symbols(relation, &query, task.budgets.max_files_to_read)
                    .await
                    .unwrap_or_default();
                trace.push(TraceEvent::ToolCall {
                    name: "code_search_related_symbols".to_string(),
                    args: serde_json::json!({
                        "query": query,
                        "relation": format!("{:?}", relation),
                        "follow_up": true
                    }),
                    result_count: related.len(),
                });
                for hit in related {
                    if evidence_overlaps_range(&evidence, &hit.path, hit.start_line, hit.end_line) {
                        continue;
                    }
                    let Ok(slice) = runtime
                        .file_reader
                        .read_range(
                            &hit.path,
                            Some(&hit.symbol),
                            hit.start_line,
                            hit.end_line,
                            per_slice_budget(&task),
                        )
                        .await
                    else {
                        continue;
                    };
                    let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                    if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                        continue;
                    }
                    let fact_id = notebook.append_tool_fact(
                        format!(
                            "Read follow-up {:?} symbol {} in {} lines {}-{} under lease {}",
                            relation,
                            hit.symbol,
                            slice.lease.path,
                            slice.lease.range.start,
                            slice.lease.range.end,
                            slice.lease.id.0
                        ),
                        Provenance::File {
                            path: slice.lease.path.clone(),
                            symbol: slice.lease.symbol.clone(),
                            range: slice.lease.range,
                            lease_id: slice.lease.id.clone(),
                        },
                        Phase::Extract,
                        vec![slice.lease.path.clone(), hit.symbol, query.clone()],
                    );
                    lease_fact_entries.insert(slice.lease.id.clone(), fact_id);
                    read_paths.insert(slice.lease.path.clone());
                    target_tags.push(slice.lease.path.clone());
                    target_tags.push(query.clone());
                    evidence_tokens += token_count;
                    evidence.push(slice);
                }
            }

            let symbols = runtime
                .code_search
                .search_symbols(&query, task.budgets.max_files_to_read)
                .await
                .unwrap_or_default();
            trace.push(TraceEvent::ToolCall {
                name: "code_search_symbols".to_string(),
                args: serde_json::json!({ "query": query, "follow_up": true }),
                result_count: symbols.len(),
            });
            for hit in symbols {
                if evidence_overlaps_range(&evidence, &hit.path, hit.start_line, hit.end_line) {
                    continue;
                }
                let Ok(slice) = runtime
                    .file_reader
                    .read_range(
                        &hit.path,
                        Some(&hit.symbol),
                        hit.start_line,
                        hit.end_line,
                        per_slice_budget(&task),
                    )
                    .await
                else {
                    continue;
                };
                let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                    continue;
                }
                let fact_id = notebook.append_tool_fact(
                    format!(
                        "Read follow-up symbol {} in {} lines {}-{} under lease {}",
                        hit.symbol,
                        slice.lease.path,
                        slice.lease.range.start,
                        slice.lease.range.end,
                        slice.lease.id.0
                    ),
                    Provenance::File {
                        path: slice.lease.path.clone(),
                        symbol: slice.lease.symbol.clone(),
                        range: slice.lease.range,
                        lease_id: slice.lease.id.clone(),
                    },
                    Phase::Extract,
                    vec![slice.lease.path.clone(), hit.symbol, query.clone()],
                );
                lease_fact_entries.insert(slice.lease.id.clone(), fact_id);
                read_paths.insert(slice.lease.path.clone());
                target_tags.push(slice.lease.path.clone());
                target_tags.push(query.clone());
                evidence_tokens += token_count;
                evidence.push(slice);
            }

            let text_hits = runtime
                .code_search
                .search_content(&query, task.budgets.max_files_to_read * 3)
                .await
                .unwrap_or_default();
            trace.push(TraceEvent::ToolCall {
                name: "code_search_content".to_string(),
                args: serde_json::json!({ "query": query, "follow_up": true }),
                result_count: text_hits.len(),
            });
            for hit in text_hits {
                if evidence_overlaps_range(&evidence, &hit.path, hit.start_line, hit.end_line) {
                    continue;
                }
                let Ok(slice) = runtime
                    .file_reader
                    .read_range(
                        &hit.path,
                        None,
                        hit.start_line,
                        hit.end_line,
                        per_slice_budget(&task),
                    )
                    .await
                else {
                    continue;
                };
                let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                    continue;
                }
                let fact_id = notebook.append_tool_fact(
                    format!(
                        "Read follow-up text match in {} lines {}-{} under lease {}",
                        slice.lease.path,
                        slice.lease.range.start,
                        slice.lease.range.end,
                        slice.lease.id.0
                    ),
                    Provenance::File {
                        path: slice.lease.path.clone(),
                        symbol: None,
                        range: slice.lease.range,
                        lease_id: slice.lease.id.clone(),
                    },
                    Phase::Extract,
                    vec![slice.lease.path.clone(), query.clone()],
                );
                lease_fact_entries.insert(slice.lease.id.clone(), fact_id);
                read_paths.insert(slice.lease.path.clone());
                target_tags.push(slice.lease.path.clone());
                target_tags.push(query.clone());
                evidence_tokens += token_count;
                evidence.push(slice);
            }

            let hits = runtime
                .code_search
                .search(&query, task.budgets.max_files_to_read)
                .await
                .unwrap_or_default();
            trace.push(TraceEvent::ToolCall {
                name: "code_search".to_string(),
                args: serde_json::json!({ "query": query, "follow_up": true }),
                result_count: hits.len(),
            });
            for hit in hits {
                if evidence.iter().any(|slice| slice.lease.path == hit.path) {
                    continue;
                }
                if let Ok(slice) = runtime
                    .file_reader
                    .read_file(&hit.path, per_slice_budget(&task))
                    .await
                {
                    let slice =
                        focus_location_slice(slice, &task.question, std::slice::from_ref(&query));
                    let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                    if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                        continue;
                    }
                    let fact_id = notebook.append_tool_fact(
                        format!(
                            "Read follow-up file {} lines {}-{} under lease {}",
                            slice.lease.path,
                            slice.lease.range.start,
                            slice.lease.range.end,
                            slice.lease.id.0
                        ),
                        Provenance::File {
                            path: slice.lease.path.clone(),
                            symbol: slice.lease.symbol.clone(),
                            range: slice.lease.range,
                            lease_id: slice.lease.id.clone(),
                        },
                        Phase::Extract,
                        vec![slice.lease.path.clone(), query.clone()],
                    );
                    lease_fact_entries.insert(slice.lease.id.clone(), fact_id);
                    read_paths.insert(slice.lease.path.clone());
                    target_tags.push(slice.lease.path.clone());
                    target_tags.push(query.clone());
                    evidence_tokens += token_count;
                    evidence.push(slice);
                }
            }
            dedup(&mut target_tags);
        }
    }

    runtime
        .progress
        .emit(AgentProgress::Status("Preparing the answer".to_string()))
        .await;

    if intake.task_type == TaskType::Edit {
        if llm_calls >= task.budgets.max_llm_calls {
            let content = fallback_plan_answer(&task, &notebook, &inference_entry_ids);
            let answer = AgentAnswer {
                content,
                stop_reason: StopReason::BudgetExhausted,
                cited_entries: active_entries(&notebook, inference_entry_ids),
                notebook,
            };
            trace.push(TraceEvent::NotebookSnapshot {
                phase: Phase::Plan,
                notebook: answer.notebook.deterministic_checkpoint(),
            });
            trace.push(TraceEvent::RunFinished {
                stop_reason: StopReason::BudgetExhausted,
            });
            return Ok(AgentRunResult { answer, trace });
        }

        trace.push(TraceEvent::PhaseStarted { phase: Phase::Plan });
        runtime
            .progress
            .emit(AgentProgress::Status(
                "Drafting an implementation plan".to_string(),
            ))
            .await;
        let plan_parts = PromptParts {
            global_prefix: global,
            phase_prefix: phase_prefix(Phase::Plan, "EditPlan"),
            task: task_block(&task),
            notebook: notebook_block(&notebook, &target_tags, 64),
            evidence: "EPHEMERAL EVIDENCE\nUse notebook entries as the source of truth for this plan. Do not draft replacement code or claim edits were applied.".to_string(),
            question: plan_instruction(),
        };
        let plan_prompt = plan_parts.render();
        let plan_req = LlmRequest {
            call_id: call_id("plan"),
            phase: Phase::Plan,
            schema_name: "EditPlan".to_string(),
            schema: schema_plan(),
            prompt_parts: Some(plan_parts),
            prompt: plan_prompt,
            temperature: 0.0,
        };
        trace.push(TraceEvent::NotebookSnapshot {
            phase: Phase::Plan,
            notebook: notebook.deterministic_checkpoint(),
        });
        let plan_resp = runtime.llm.complete_json(plan_req.clone()).await?;
        trace.push(TraceEvent::LlmCall {
            request: plan_req,
            response: plan_resp.clone(),
        });
        let mut plan: EditPlan = parse_json(&plan_resp.raw_json)
            .unwrap_or_else(|_| fallback_edit_plan(&task, &notebook, &inference_entry_ids));
        constrain_edit_plan_to_evidence(&mut plan, &notebook, &inference_entry_ids);
        let cited_entries = active_entries(&notebook, inference_entry_ids);
        let answer = AgentAnswer {
            content: render_edit_plan(&plan),
            stop_reason: StopReason::Done,
            cited_entries,
            notebook,
        };
        trace.push(TraceEvent::RunFinished {
            stop_reason: StopReason::Done,
        });
        return Ok(AgentRunResult { answer, trace });
    }

    if llm_calls >= task.budgets.max_llm_calls {
        let content = fallback_answer(
            &task,
            &notebook,
            &inference_entry_ids,
            StopReason::BudgetExhausted,
        );
        let answer = AgentAnswer {
            content,
            stop_reason: StopReason::BudgetExhausted,
            cited_entries: active_entries(&notebook, inference_entry_ids),
            notebook,
        };
        trace.push(TraceEvent::NotebookSnapshot {
            phase: Phase::Synthesize,
            notebook: answer.notebook.deterministic_checkpoint(),
        });
        trace.push(TraceEvent::RunFinished {
            stop_reason: StopReason::BudgetExhausted,
        });
        return Ok(AgentRunResult { answer, trace });
    }

    trace.push(TraceEvent::PhaseStarted {
        phase: Phase::Synthesize,
    });
    runtime
        .progress
        .emit(AgentProgress::Status(
            "Selecting supporting sources".to_string(),
        ))
        .await;
    let synth_parts = PromptParts {
        global_prefix: global,
        phase_prefix: phase_prefix(Phase::Synthesize, "SynthesizeOutput"),
        task: task_block(&task),
        notebook: notebook_block(&notebook, &target_tags, 64),
        evidence: "EPHEMERAL EVIDENCE\nUse notebook entries as the citation surface. Do not cite leases directly in the final answer.".to_string(),
        question: synthesize_instruction(),
    };
    let synth_prompt = synth_parts.render();

    let synth_req = LlmRequest {
        call_id: call_id("synthesize"),
        phase: Phase::Synthesize,
        schema_name: "SynthesizeOutput".to_string(),
        schema: schema_synthesize(),
        prompt_parts: Some(synth_parts),
        prompt: synth_prompt,
        temperature: 0.0,
    };
    trace.push(TraceEvent::NotebookSnapshot {
        phase: Phase::Synthesize,
        notebook: notebook.deterministic_checkpoint(),
    });
    runtime
        .progress
        .emit(AgentProgress::Status(
            "Constructing the final response".to_string(),
        ))
        .await;
    let synth_resp = runtime.llm.complete_json(synth_req.clone()).await?;
    trace.push(TraceEvent::LlmCall {
        request: synth_req,
        response: synth_resp.clone(),
    });

    let synth: SynthesizeOutput = parse_json(&synth_resp.raw_json)
        .unwrap_or_else(|_| fallback_synthesize(&task, &notebook, &inference_entry_ids));
    let mut cited_entries = sort_citations_by_specificity(
        &notebook,
        synth
            .cited_entries
            .into_iter()
            .filter(|id| {
                notebook
                    .get(id)
                    .map(|entry| entry.status == crate::notebook::EntryStatus::Active)
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>(),
    );
    if cited_entries.is_empty() {
        cited_entries = active_entries(&notebook, inference_entry_ids.clone());
    }
    let stop_reason = synth.stop_reason;
    let answer = AgentAnswer {
        content: synth.answer,
        stop_reason,
        cited_entries,
        notebook,
    };
    trace.push(TraceEvent::RunFinished { stop_reason });

    Ok(AgentRunResult { answer, trace })
}

fn plural(count: usize) -> &'static str {
    if count == 1 {
        ""
    } else {
        "s"
    }
}

fn plain_query(value: &str) -> String {
    value
        .trim()
        .trim_matches('`')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn parse_file_range_target(target: &str) -> Option<(String, u32, u32)> {
    let cleaned = target.trim().trim_matches('`').trim();
    if cleaned.is_empty() {
        return None;
    }

    if let Some((path, range)) = cleaned.rsplit_once(':') {
        if let Some((start, end)) = parse_line_range(range) {
            let path = path.trim().trim_matches('`').to_string();
            if !path.is_empty() {
                return Some((path, start, end));
            }
        }
    }

    let lower = cleaned.to_lowercase();
    for marker in [" lines ", " line "] {
        if let Some(idx) = lower.rfind(marker) {
            let path = cleaned[..idx].trim().trim_matches('`').to_string();
            let range = cleaned[idx + marker.len()..].trim();
            if !path.is_empty() {
                if let Some((start, end)) = parse_line_range(range) {
                    return Some((path, start, end));
                }
            }
        }
    }

    None
}

fn extract_file_range_queries(question: &str) -> Vec<(String, u32, u32)> {
    let tokens = question
        .split_whitespace()
        .map(|token| {
            token
                .trim_matches(|ch: char| {
                    matches!(
                        ch,
                        '`' | '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';'
                    )
                })
                .trim_end_matches(|ch: char| matches!(ch, '.' | '?' | '!' | ','))
                .to_string()
        })
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>();
    let mut ranges = Vec::new();
    let mut seen = BTreeSet::new();

    for start in 0..tokens.len() {
        for width in 1..=4 {
            if start + width > tokens.len() {
                break;
            }
            let target = tokens[start..start + width].join(" ");
            let Some((path, start_line, end_line)) = parse_file_range_target(&target) else {
                continue;
            };
            if !is_path_like_query(&path) {
                continue;
            }
            let key = (path.clone(), start_line, end_line);
            if seen.insert(key) {
                ranges.push((path, start_line, end_line));
            }
        }
    }

    ranges
}

fn parse_line_range(value: &str) -> Option<(u32, u32)> {
    let trimmed = value
        .trim()
        .trim_start_matches("lines")
        .trim_start_matches("line")
        .trim();
    let (start, end) = match trimmed.split_once('-') {
        Some((start, end)) => (parse_line_number(start)?, parse_line_number(end)?),
        None => {
            let line = parse_line_number(trimmed)?;
            (line, line)
        }
    };
    (start > 0 && end >= start).then_some((start, end))
}

fn parse_line_number(value: &str) -> Option<u32> {
    value
        .trim()
        .trim_start_matches('L')
        .trim_start_matches('l')
        .parse::<u32>()
        .ok()
}

fn symbol_relation_for_missing(kind: &MissingEvidenceKind) -> Option<SymbolRelation> {
    match kind {
        MissingEvidenceKind::Definition => Some(SymbolRelation::Definition),
        MissingEvidenceKind::References => Some(SymbolRelation::References),
        MissingEvidenceKind::Callers => Some(SymbolRelation::Callers),
        MissingEvidenceKind::Callees => Some(SymbolRelation::Callees),
        MissingEvidenceKind::FileRange | MissingEvidenceKind::Search => None,
    }
}

fn relation_intents_for_question(question: &str) -> Vec<SymbolRelation> {
    let lower = question.to_lowercase();
    let mut relations = Vec::new();

    if lower.contains("definition")
        || lower.contains("defined")
        || lower.contains("where is")
        || lower.contains("where are")
    {
        relations.push(SymbolRelation::Definition);
    }
    if lower.contains("reference")
        || lower.contains("references")
        || lower.contains("usage")
        || lower.contains("usages")
        || lower.contains("where used")
        || lower.contains("where is used")
        || lower.contains("where are used")
    {
        relations.push(SymbolRelation::References);
    }
    if lower.contains("caller")
        || lower.contains("callers")
        || lower.contains("who calls")
        || lower.contains("what calls")
        || lower.contains("called by")
    {
        relations.push(SymbolRelation::Callers);
    }
    if lower.contains("callee")
        || lower.contains("callees")
        || lower.contains("what does")
            && (lower.contains(" call") || lower.contains(" invoke") || lower.contains(" use"))
        || lower.contains("what do")
            && (lower.contains(" call") || lower.contains(" invoke") || lower.contains(" use"))
        || lower.contains("calls into")
        || lower.contains("invokes")
    {
        relations.push(SymbolRelation::Callees);
    }

    relations.dedup();
    relations
}

#[derive(Clone, Copy)]
struct RelatedContextExpansion {
    relation: SymbolRelation,
    kind: &'static str,
    fact_label: &'static str,
    progress: &'static str,
    limit: usize,
}

fn related_context_expansions(
    intake: &IntakeOutput,
    question: &str,
) -> Vec<RelatedContextExpansion> {
    let explicit_relations = relation_intents_for_question(question);
    match intake.task_type {
        TaskType::Explain if !explicit_relations.contains(&SymbolRelation::Callees) => {
            vec![RelatedContextExpansion {
                relation: SymbolRelation::Callees,
                kind: "direct_callees",
                fact_label: "direct callee",
                progress: "Finding calls made by",
                limit: 2,
            }]
        }
        TaskType::Edit => {
            let mut expansions = Vec::new();
            if !explicit_relations.contains(&SymbolRelation::Callers) {
                expansions.push(RelatedContextExpansion {
                    relation: SymbolRelation::Callers,
                    kind: "edit_callers",
                    fact_label: "edit caller",
                    progress: "Finding callers of",
                    limit: 2,
                });
            }
            if !explicit_relations.contains(&SymbolRelation::References) {
                expansions.push(RelatedContextExpansion {
                    relation: SymbolRelation::References,
                    kind: "edit_references",
                    fact_label: "edit reference",
                    progress: "Finding references to",
                    limit: 2,
                });
            }
            expansions
        }
        _ => Vec::new(),
    }
}

fn wants_changed_file_context(intake: &IntakeOutput, question: &str) -> bool {
    if intake.task_type == TaskType::Edit {
        return true;
    }
    let lower = question.to_lowercase();
    [
        "changed",
        "changes",
        "diff",
        "dirty",
        "working tree",
        "worktree",
        "current work",
        "review",
        "audit",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn wants_adjacent_file_context(intake: &IntakeOutput, question: &str) -> bool {
    if intake.task_type == TaskType::Edit {
        return true;
    }
    let lower = question.to_lowercase();
    [
        "review",
        "audit",
        "architecture",
        "architectural",
        "research",
        "understand",
        "overview",
        "project",
        "codebase",
        "repo",
        "repository",
        "related",
        "nearby",
        "sibling",
        "context",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn wants_imported_file_context(intake: &IntakeOutput, question: &str) -> bool {
    if intake.task_type == TaskType::Edit {
        return true;
    }
    let lower = question.to_lowercase();
    [
        "review",
        "audit",
        "architecture",
        "architectural",
        "research",
        "understand",
        "overview",
        "project",
        "codebase",
        "repo",
        "repository",
        "import",
        "imports",
        "dependency",
        "dependencies",
        "module",
        "modules",
        "flow",
        "context",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn wants_importing_file_context(intake: &IntakeOutput, question: &str) -> bool {
    if intake.task_type == TaskType::Edit {
        return true;
    }
    let lower = question.to_lowercase();
    [
        "review",
        "audit",
        "architecture",
        "architectural",
        "research",
        "understand",
        "overview",
        "project",
        "codebase",
        "repo",
        "repository",
        "import",
        "imports",
        "imported by",
        "dependent",
        "dependents",
        "dependency",
        "dependencies",
        "impact",
        "usage",
        "usages",
        "module",
        "modules",
        "flow",
        "context",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn wants_scope_file_context(intake: &IntakeOutput, question: &str) -> bool {
    if intake.task_type == TaskType::Edit {
        return true;
    }
    let lower = question.to_lowercase();
    [
        "review",
        "audit",
        "architecture",
        "architectural",
        "research",
        "understand",
        "overview",
        "project",
        "codebase",
        "repo",
        "repository",
        "package",
        "manifest",
        "config",
        "configuration",
        "setup",
        "build",
        "test",
        "validation",
        "context",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn wants_context_doc_context(intake: &IntakeOutput, question: &str) -> bool {
    if intake.task_type == TaskType::Edit {
        return true;
    }
    let lower = question.to_lowercase();
    [
        "review",
        "audit",
        "architecture",
        "architectural",
        "research",
        "understand",
        "overview",
        "project",
        "codebase",
        "repo",
        "repository",
        "docs",
        "documentation",
        "design",
        "guide",
        "flow",
        "workflow",
        "lifecycle",
        "context",
        "how does",
        "how do",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
}

fn adjacent_seed_paths(
    file_paths: &BTreeMap<String, i64>,
    symbol_hits: &BTreeMap<(String, String, u32, u32), crate::types::SymbolHit>,
    content_hits: &BTreeMap<(String, u32, u32), crate::types::ContentHit>,
    changed_paths: &[String],
    limit: usize,
) -> Vec<String> {
    let mut seeds = Vec::new();
    seeds.extend(changed_paths.iter().cloned());
    seeds.extend(symbol_hits.values().map(|hit| hit.path.clone()));
    seeds.extend(content_hits.values().map(|hit| hit.path.clone()));
    seeds.extend(file_paths.keys().cloned());
    dedup(&mut seeds);
    seeds.truncate(limit);
    seeds
}

fn format_list(values: &[String], limit: usize) -> String {
    let visible = values
        .iter()
        .take(limit)
        .map(|value| plain_query(value))
        .collect::<Vec<_>>();
    match visible.as_slice() {
        [] => "no queries".to_string(),
        [only] => only.clone(),
        _ => visible.join(", "),
    }
}

fn fallback_intake(question: &str) -> IntakeOutput {
    let mut terms = question
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != '.')
        .filter(|term| term.len() > 2)
        .map(ToString::to_string)
        .take(5)
        .collect::<Vec<_>>();
    if terms.is_empty() {
        terms.push(question.to_string());
    }
    IntakeOutput {
        task_type: if looks_like_edit_request(question) {
            TaskType::Edit
        } else {
            TaskType::Explain
        },
        target_hints: terms.clone(),
        constraints: Vec::new(),
        search_queries: terms,
    }
}

fn normalize_intake(intake: &mut IntakeOutput, task: &TaskRequest) {
    if intake.search_queries.is_empty() {
        *intake = fallback_intake(&task.question);
    }
    if intake.task_type == TaskType::Unclear && looks_actionable_code_request(&task.question) {
        let fallback = fallback_intake(&task.question);
        intake.task_type = fallback.task_type;
        if intake.target_hints.is_empty() {
            intake.target_hints = fallback.target_hints;
        }
        if intake.search_queries.is_empty() {
            intake.search_queries = fallback.search_queries;
        }
    }
    preserve_question_query(intake, &task.question);
    preserve_path_queries(intake, &task.question);
    augment_search_queries(intake, &task.question);
    dedup(&mut intake.search_queries);
    dedup(&mut intake.target_hints);
}

fn preserve_path_queries(intake: &mut IntakeOutput, question: &str) {
    let paths = extract_path_queries(question);
    if paths.is_empty() {
        return;
    }

    let mut insert_at = usize::from(!intake.search_queries.is_empty());
    for path in paths {
        if intake
            .search_queries
            .iter()
            .any(|existing| normalize_query(existing) == normalize_query(&path))
        {
            intake.target_hints.push(path);
            continue;
        }
        intake.search_queries.insert(insert_at, path.clone());
        intake.target_hints.push(path);
        insert_at += 1;
    }
}

fn preserve_question_query(intake: &mut IntakeOutput, question: &str) {
    let Some(query) = compact_question_query(question) else {
        return;
    };
    if intake
        .search_queries
        .iter()
        .any(|existing| normalize_query(existing) == normalize_query(&query))
    {
        return;
    }

    let insert_at = usize::from(!intake.search_queries.is_empty());
    intake.search_queries.insert(insert_at, query.clone());
    intake.target_hints.push(query);
}

fn compact_question_query(question: &str) -> Option<String> {
    let mut terms = question
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '-' && ch != '.')
        .map(str::trim)
        .filter(|term| term.len() > 2)
        .filter(|term| !is_query_filler(term))
        .take(8)
        .map(ToString::to_string)
        .collect::<Vec<_>>();
    if terms.is_empty() {
        let trimmed = question.trim();
        if trimmed.len() > 2 && trimmed.len() <= 120 {
            terms.push(trimmed.to_string());
        }
    }
    (!terms.is_empty()).then(|| terms.join(" "))
}

fn extract_path_queries(question: &str) -> Vec<String> {
    let mut paths = Vec::new();
    for raw in question.split_whitespace() {
        let candidate = raw
            .trim_matches(|ch: char| {
                matches!(
                    ch,
                    '`' | '\'' | '"' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';'
                )
            })
            .trim_end_matches(|ch: char| matches!(ch, '.' | '?' | '!' | ','));
        if candidate.is_empty() || candidate.contains("://") {
            continue;
        }
        let path = strip_path_line_suffix(candidate)
            .trim_start_matches("./")
            .trim_start_matches('/')
            .replace('\\', "/");
        if is_path_like_query(&path) {
            paths.push(path);
        }
    }
    dedup(&mut paths);
    paths
}

fn extract_file_path_queries(question: &str) -> Vec<String> {
    let mut paths = extract_path_queries(question);
    paths.retain(|path| has_known_file_extension(path));
    paths
}

fn extract_directory_queries(question: &str) -> Vec<String> {
    let mut dirs = extract_path_queries(question);
    dirs.retain(|path| path.contains('/') && !has_known_file_extension(path));
    dirs.iter_mut()
        .for_each(|path| *path = path.trim_end_matches('/').to_string());
    dedup(&mut dirs);
    dirs
}

fn strip_path_line_suffix(value: &str) -> &str {
    let Some((path, suffix)) = value.rsplit_once(':') else {
        return value;
    };
    if parse_line_range(suffix).is_some() {
        path
    } else {
        value
    }
}

fn is_path_like_query(value: &str) -> bool {
    if value.is_empty()
        || value.contains(' ')
        || value.contains("://")
        || value.contains("..")
        || value.starts_with('-')
        || value.starts_with('#')
    {
        return false;
    }
    value.contains('/') || has_known_file_extension(value)
}

fn has_known_file_extension(value: &str) -> bool {
    let extension = value.rsplit('.').next().unwrap_or_default();
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "rs" | "ts"
            | "tsx"
            | "js"
            | "jsx"
            | "css"
            | "scss"
            | "html"
            | "md"
            | "json"
            | "toml"
            | "yaml"
            | "yml"
            | "sql"
            | "py"
            | "go"
            | "java"
            | "kt"
            | "swift"
            | "c"
            | "cc"
            | "cpp"
            | "h"
            | "hpp"
            | "xml"
    )
}

fn is_query_filler(term: &str) -> bool {
    matches!(
        term.to_ascii_lowercase().as_str(),
        "the"
            | "and"
            | "for"
            | "with"
            | "our"
            | "this"
            | "that"
            | "what"
            | "where"
            | "when"
            | "how"
            | "does"
            | "into"
            | "explain"
            | "locate"
            | "find"
            | "change"
            | "update"
            | "review"
            | "audit"
    )
}

fn normalize_query(query: &str) -> String {
    query
        .split_whitespace()
        .map(|term| term.to_ascii_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

fn augment_search_queries(intake: &mut IntakeOutput, question: &str) {
    let lower = question.to_lowercase();
    let mentions_app_startup = lower.contains("splash")
        || lower.contains("fullscreen")
        || lower.contains("full screen")
        || lower.contains("loaded")
        || lower.contains("startup")
        || lower.contains("boot");

    if mentions_app_startup {
        intake.search_queries.extend(
            [
                "splash",
                "startup",
                "boot",
                "loading",
                "app shell",
                "window",
                "fullscreen",
            ]
            .into_iter()
            .map(ToString::to_string),
        );
        intake.target_hints.extend(
            ["app startup", "window behavior", "user-facing loading UI"]
                .into_iter()
                .map(ToString::to_string),
        );
    }

    let wants_validation_context = intake.task_type == TaskType::Edit
        || [
            "test",
            "build",
            "validate",
            "validation",
            "run",
            "script",
            "check",
            "lint",
            "project setup",
        ]
        .iter()
        .any(|needle| lower.contains(needle));

    if wants_validation_context {
        let mut expanded = intake
            .search_queries
            .iter()
            .take(1)
            .cloned()
            .collect::<Vec<_>>();
        expanded.extend(
            intake
                .search_queries
                .iter()
                .skip(1)
                .filter(|query| is_path_like_query(query))
                .take(2)
                .cloned(),
        );
        expanded.extend(
            [
                "project manifest",
                "test script",
                "build script",
                "related tests",
                "validation command",
                "workspace config",
            ]
            .into_iter()
            .map(ToString::to_string),
        );
        expanded.extend(
            intake
                .search_queries
                .iter()
                .skip(1)
                .filter(|query| !is_path_like_query(query))
                .cloned(),
        );
        intake.search_queries = expanded;
        intake.target_hints.extend(
            [
                "project manifest",
                "build scripts",
                "related tests",
                "validation commands",
            ]
            .into_iter()
            .map(ToString::to_string),
        );
    }

    let wants_project_research_context = [
        "review",
        "audit",
        "architecture",
        "architectural",
        "research",
        "understand",
        "overview",
        "how is",
        "how are",
        "how does",
        "design",
        "structure",
        "project",
        "codebase",
        "repo",
        "repository",
    ]
    .iter()
    .any(|needle| lower.contains(needle));

    if wants_project_research_context {
        let mut expanded = intake
            .search_queries
            .iter()
            .take(1)
            .cloned()
            .collect::<Vec<_>>();
        expanded.extend(
            intake
                .search_queries
                .iter()
                .skip(1)
                .filter(|query| is_path_like_query(query))
                .take(2)
                .cloned(),
        );
        expanded.extend(
            [
                "project overview",
                "readme",
                "repository instructions",
                "project entrypoints",
                "architecture docs",
                "design docs",
                "developer guide",
            ]
            .into_iter()
            .map(ToString::to_string),
        );
        expanded.extend(
            intake
                .search_queries
                .iter()
                .skip(1)
                .filter(|query| !is_path_like_query(query))
                .cloned(),
        );
        intake.search_queries = expanded;
        intake.target_hints.extend(
            [
                "project overview",
                "repository instructions",
                "project entrypoints",
                "architecture documentation",
                "developer documentation",
            ]
            .into_iter()
            .map(ToString::to_string),
        );
    }
}

fn looks_like_edit_request(question: &str) -> bool {
    let lower = question.to_lowercase();
    [
        "update",
        "change",
        "fix",
        "add",
        "remove",
        "make",
        "wrap",
        "display",
        "show",
        "refactor",
        "rename",
        "style",
        "implement",
        "create",
    ]
    .iter()
    .any(|word| lower.contains(word))
}

fn looks_actionable_code_request(question: &str) -> bool {
    let lower = question.to_lowercase();
    let action_words = [
        "update", "change", "fix", "add", "remove", "make", "wrap", "display", "show", "inspect",
        "explain", "locate",
    ];
    let code_words = [
        "ui",
        "component",
        "panel",
        "workspace",
        "app",
        "tsx",
        "rust",
        "tauri",
        "code",
        "file",
    ];
    action_words.iter().any(|word| lower.contains(word))
        && code_words.iter().any(|word| lower.contains(word))
}

fn fallback_extract(evidence: &[EvidenceSlice]) -> ExtractOutput {
    ExtractOutput {
        inferences: evidence
            .iter()
            .take(3)
            .map(|slice| crate::types::ModelInference {
                statement: format!(
                    "Relevant evidence was found in {} lines {}-{}",
                    slice.lease.path, slice.lease.range.start, slice.lease.range.end
                ),
                lease_id: slice.lease.id.clone(),
                tags: vec![slice.lease.path.clone()],
            })
            .collect(),
        sufficient: true,
        missing: Vec::new(),
    }
}

fn fallback_synthesize(
    task: &TaskRequest,
    notebook: &Notebook,
    inference_entry_ids: &[EntryId],
) -> SynthesizeOutput {
    SynthesizeOutput {
        answer: fallback_answer(task, notebook, inference_entry_ids, StopReason::Done),
        cited_entries: active_entries(notebook, inference_entry_ids.to_vec()),
        stop_reason: StopReason::Done,
    }
}

fn fallback_answer(
    task: &TaskRequest,
    notebook: &Notebook,
    inference_entry_ids: &[EntryId],
    stop_reason: StopReason,
) -> String {
    let mut lines = vec![format!("I inspected the workspace for: {}", task.question)];
    if inference_entry_ids.is_empty() {
        lines.push("I found files, but did not produce strong cited inferences.".to_string());
    } else {
        lines.push("Relevant findings:".to_string());
        for id in inference_entry_ids {
            if let Some(entry) = notebook.get(id) {
                lines.push(format!("- {}", entry.statement));
            }
        }
    }
    if stop_reason == StopReason::BudgetExhausted {
        lines.push("The run stopped because the LLM call budget was exhausted.".to_string());
    }
    lines.join("\n")
}

fn fallback_edit_plan(
    task: &TaskRequest,
    notebook: &Notebook,
    inference_entry_ids: &[EntryId],
) -> EditPlan {
    let affected_files = active_entries(notebook, inference_entry_ids.to_vec())
        .into_iter()
        .filter_map(|id| {
            let entry = notebook.get(&id)?;
            match &entry.provenance {
                Provenance::File { path, .. } => Some(crate::types::PlannedFile {
                    path: path.clone(),
                    reason: entry.statement.clone(),
                }),
                _ => None,
            }
        })
        .collect::<Vec<_>>();

    EditPlan {
        objective: task.question.clone(),
        affected_files,
        steps: vec![
            crate::types::PlanStep {
                title: "Confirm the target code".to_string(),
                detail:
                    "Use the cited evidence to confirm the exact component, function, or file range to change."
                        .to_string(),
            },
            crate::types::PlanStep {
                title: "Draft a host-validated edit artifact".to_string(),
                detail:
                    "Generate a whole-file or block-level replacement in the next edit phase, then preview the resulting diff before writing."
                        .to_string(),
            },
        ],
        risks: vec![
            "This is a planning-only response; no file changes have been written.".to_string(),
        ],
        validation_commands: Vec::new(),
    }
}

fn constrain_edit_plan_to_evidence(
    plan: &mut EditPlan,
    notebook: &Notebook,
    inference_entry_ids: &[EntryId],
) {
    let evidence_entry_ids = active_entries(notebook, inference_entry_ids.to_vec())
        .into_iter()
        .flat_map(|id| {
            let mut ids = vec![id.clone()];
            if let Some(entry) = notebook.get(&id) {
                ids.extend(entry.depends_on.clone());
            }
            ids.into_iter()
        })
        .collect::<BTreeSet<_>>();

    let evidence_paths = evidence_entry_ids
        .iter()
        .filter_map(|id| {
            notebook.get(&id).and_then(|entry| match &entry.provenance {
                Provenance::File { path, .. } => Some(path.clone()),
                _ => None,
            })
        })
        .collect::<BTreeSet<_>>();

    if evidence_paths.is_empty() {
        plan.affected_files.clear();
        if !plan
            .risks
            .iter()
            .any(|risk| risk.contains("No evidence-backed files"))
        {
            plan.risks.push(
                "No evidence-backed files were confirmed, so file targets were withheld."
                    .to_string(),
            );
        }
        return;
    }

    plan.affected_files
        .retain(|file| evidence_paths.contains(&file.path));
    let evidence_text = evidence_entry_ids
        .iter()
        .filter_map(|id| notebook.get(id))
        .map(|entry| {
            let tags = entry.tags.iter().cloned().collect::<Vec<_>>().join(" ");
            format!("{} {}", entry.statement, tags)
        })
        .chain(evidence_paths.iter().cloned())
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase();
    plan.validation_commands
        .retain(|command| validation_command_is_evidence_backed(command, &evidence_text));
    if plan.affected_files.is_empty() {
        plan.risks.push(
            "The model proposed files that were not supported by active evidence, so they were omitted."
                .to_string(),
        );
    }
}

fn validation_command_is_evidence_backed(command: &str, evidence_text: &str) -> bool {
    let meaningful_tokens = command
        .split(|ch: char| !ch.is_ascii_alphanumeric() && ch != '_' && ch != '-')
        .map(|token| token.trim().to_lowercase())
        .filter(|token| token.len() > 2)
        .filter(|token| {
            !matches!(
                token.as_str(),
                "run" | "test" | "tests" | "check" | "build" | "lint" | "ci" | "all"
            )
        })
        .collect::<Vec<_>>();

    !meaningful_tokens.is_empty()
        && meaningful_tokens
            .iter()
            .any(|token| evidence_text.contains(token))
}

fn fallback_plan_answer(
    task: &TaskRequest,
    notebook: &Notebook,
    inference_entry_ids: &[EntryId],
) -> String {
    let mut plan = fallback_edit_plan(task, notebook, inference_entry_ids);
    plan.risks
        .push("The run stopped before a model-authored plan could be generated.".to_string());
    render_edit_plan(&plan)
}

fn render_edit_plan(plan: &EditPlan) -> String {
    let mut lines = vec![
        "## Implementation Plan".to_string(),
        String::new(),
        format!("**Objective:** {}", plan.objective),
        String::new(),
        "**Likely files:**".to_string(),
    ];

    if plan.affected_files.is_empty() {
        lines.push("- No specific files were confirmed yet.".to_string());
    } else {
        for file in &plan.affected_files {
            lines.push(format!("- `{}`: {}", file.path, file.reason));
        }
    }

    lines.push(String::new());
    lines.push("**Steps:**".to_string());
    if plan.steps.is_empty() {
        lines.push("- No implementation steps were produced.".to_string());
    } else {
        for (idx, step) in plan.steps.iter().enumerate() {
            lines.push(format!("{}. **{}**: {}", idx + 1, step.title, step.detail));
        }
    }

    lines.push(String::new());
    lines.push("**Risks:**".to_string());
    if plan.risks.is_empty() {
        lines.push("- No specific risks identified.".to_string());
    } else {
        for risk in &plan.risks {
            lines.push(format!("- {}", risk));
        }
    }

    lines.push(String::new());
    lines.push("**Validation:**".to_string());
    if plan.validation_commands.is_empty() {
        lines.push("- No validation commands identified yet.".to_string());
    } else {
        for command in &plan.validation_commands {
            lines.push(format!("- `{}`", command));
        }
    }

    lines.push(String::new());
    lines.push(
        "No files were changed. This is the plan-only phase before edit artifacts and diff preview."
            .to_string(),
    );

    lines.join("\n")
}

fn active_entries(notebook: &Notebook, ids: Vec<EntryId>) -> Vec<EntryId> {
    sort_citations_by_specificity(
        notebook,
        ids.into_iter()
            .filter(|id| {
                notebook
                    .get(id)
                    .map(|entry| entry.status == crate::notebook::EntryStatus::Active)
                    .unwrap_or(false)
            })
            .collect(),
    )
}

fn sort_citations_by_specificity(notebook: &Notebook, mut ids: Vec<EntryId>) -> Vec<EntryId> {
    ids.sort_by_key(|id| {
        notebook
            .get(id)
            .map(|entry| match &entry.provenance {
                Provenance::File { range, symbol, .. } => {
                    let width = range.end.saturating_sub(range.start);
                    (0usize, symbol.is_none(), width, range.start)
                }
                _ => (1usize, true, usize::MAX, usize::MAX),
            })
            .unwrap_or((2usize, true, usize::MAX, usize::MAX))
    });
    ids
}

fn per_slice_budget(task: &TaskRequest) -> usize {
    let fair_share = task.budgets.max_evidence_chars / task.budgets.max_files_to_read.max(1);
    fair_share.max(8_000).min(task.budgets.max_evidence_chars)
}

fn should_accept_evidence(
    evidence: &[EvidenceSlice],
    current_tokens: usize,
    next_tokens: usize,
    task: &TaskRequest,
) -> bool {
    evidence.is_empty()
        || current_tokens.saturating_add(next_tokens) <= task.budgets.max_evidence_tokens
}

fn focus_location_slice(
    slice: EvidenceSlice,
    question: &str,
    search_queries: &[String],
) -> EvidenceSlice {
    if !looks_like_location_question(question) || slice.lease.symbol.is_some() {
        return slice;
    }

    let lines = slice.content.lines().collect::<Vec<_>>();
    if lines.len() <= 80 {
        return slice;
    }

    let terms = location_terms(question, search_queries);
    if terms.is_empty() {
        return slice;
    }

    let Some(best_idx) = best_matching_line(&lines, &terms) else {
        return slice;
    };
    let (start_idx, end_idx) = focused_line_window(&lines, best_idx);
    if start_idx == 0 && end_idx + 1 == lines.len() {
        return slice;
    }

    let content = lines[start_idx..=end_idx].join("\n");
    let range = LineRange {
        start: slice.lease.range.start + start_idx,
        end: slice.lease.range.start + end_idx,
    };
    EvidenceSlice {
        lease: EvidenceLease::new(slice.lease.path, None, range, &content),
        content,
    }
}

fn looks_like_location_question(question: &str) -> bool {
    let lower = question.to_lowercase();
    ["where", "display", "render", "show", "locate", "which file"]
        .iter()
        .any(|needle| lower.contains(needle))
}

fn location_terms(question: &str, search_queries: &[String]) -> Vec<String> {
    let mut terms =
        question
            .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            .chain(search_queries.iter().flat_map(|query| {
                query.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
            }))
            .map(|term| term.to_lowercase())
            .filter(|term| term.len() > 2)
            .filter(|term| {
                !matches!(
                    term.as_str(),
                    "where"
                        | "which"
                        | "what"
                        | "the"
                        | "our"
                        | "does"
                        | "display"
                        | "render"
                        | "show"
                        | "locate"
                        | "file"
                        | "code"
                )
            })
            .flat_map(|term| {
                let stem = term
                    .trim_end_matches("ing")
                    .trim_end_matches("ed")
                    .trim_end_matches('s')
                    .to_string();
                if stem != term && stem.len() > 2 {
                    vec![term, stem]
                } else {
                    vec![term]
                }
            })
            .collect::<Vec<_>>();
    dedup(&mut terms);
    terms
}

fn best_matching_line(lines: &[&str], terms: &[String]) -> Option<usize> {
    lines
        .iter()
        .enumerate()
        .filter_map(|(idx, line)| {
            let lower = line.to_lowercase();
            let mut score = terms
                .iter()
                .filter(|term| lower.contains(term.as_str()))
                .count() as i64;
            if lower.contains("list") {
                score += 1;
            }
            if lower.contains("files") || lower.contains("file") {
                score += 1;
            }
            if lower.contains("{/*") {
                score += 2;
            }
            if score > 0 {
                Some((idx, score))
            } else {
                None
            }
        })
        .max_by(|(left_idx, left_score), (right_idx, right_score)| {
            left_score
                .cmp(right_score)
                .then_with(|| right_idx.cmp(left_idx))
        })
        .map(|(idx, _)| idx)
}

fn focused_line_window(lines: &[&str], best_idx: usize) -> (usize, usize) {
    let mut start = best_idx.saturating_sub(34);
    let mut end = (best_idx + 34).min(lines.len().saturating_sub(1));

    if lines[best_idx].contains("{/*") {
        start = best_idx;
        if let Some(next_comment) = lines
            .iter()
            .enumerate()
            .skip(best_idx + 1)
            .find_map(|(idx, line)| line.contains("{/*").then_some(idx))
        {
            end = next_comment.saturating_sub(2).max(start);
        }
    }

    (start, end)
}

fn parse_json<T: DeserializeOwned>(raw: &str) -> Result<T> {
    let trimmed = raw.trim();
    if let Ok(parsed) = serde_json::from_str(trimmed) {
        return Ok(parsed);
    }

    let cleaned = trimmed
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    if let Ok(parsed) = serde_json::from_str(cleaned) {
        return Ok(parsed);
    }

    let start = cleaned
        .find('{')
        .ok_or_else(|| anyhow!("No JSON object in model output"))?;
    let end = cleaned
        .rfind('}')
        .ok_or_else(|| anyhow!("No JSON object terminator in model output"))?;
    serde_json::from_str(&cleaned[start..=end]).context("Failed to parse extracted JSON")
}

fn dedup(values: &mut Vec<String>) {
    let mut seen = BTreeSet::new();
    values.retain(|value| seen.insert(value.clone()));
}

fn call_id(prefix: &str) -> String {
    format!("{}_{}", prefix, Uuid::new_v4().simple())
}

fn evidence_overlaps_range(
    evidence: &[EvidenceSlice],
    path: &str,
    start_line: u32,
    end_line: u32,
) -> bool {
    let start = start_line.max(1) as usize;
    let end = end_line.max(start_line.max(1)) as usize;
    evidence.iter().any(|slice| {
        slice.lease.path == path
            && ranges_overlap(slice.lease.range.start, slice.lease.range.end, start, end)
    })
}

fn ranges_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start <= b_end && b_start <= a_end
}

#[allow(dead_code)]
fn lease_for_text(path: String, text: &str) -> EvidenceSlice {
    let line_count = text.lines().count().max(1);
    EvidenceSlice {
        lease: EvidenceLease::new(
            path,
            None,
            LineRange {
                start: 1,
                end: line_count,
            },
            text,
        ),
        content: text.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notebook::EvidenceLease;
    use crate::traits::{ApproxTokenCounter, LlmResponse, LlmUsage, NoopProgressSink};
    use crate::types::{Budgets, ContentHit, FileHit, SymbolHit};
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex};

    struct MockLlm {
        responses: Mutex<Vec<String>>,
    }

    #[async_trait]
    impl LlmClient for MockLlm {
        async fn complete_json(&self, request: LlmRequest) -> Result<LlmResponse> {
            let mut lock = self.responses.lock().unwrap();
            let raw_json = lock.remove(0);
            Ok(LlmResponse {
                call_id: request.call_id,
                raw_json,
                usage: LlmUsage::default(),
            })
        }
    }

    struct MockSearch;

    #[async_trait]
    impl CodeSearch for MockSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec!["src/main.rs".to_string()])
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(vec![FileHit {
                path: "src/main.rs".to_string(),
                score: 10,
            }])
        }

        async fn search_symbols(&self, _query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_related_symbols(
            &self,
            _relation: SymbolRelation,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            Ok(Vec::new())
        }
    }

    struct ContentOnlySearch;

    #[async_trait]
    impl CodeSearch for ContentOnlySearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec!["src/styles.css".to_string()])
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(Vec::new())
        }

        async fn search_symbols(&self, _query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_related_symbols(
            &self,
            _relation: SymbolRelation,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            Ok(vec![ContentHit {
                path: "src/styles.css".to_string(),
                start_line: 10,
                end_line: 14,
                preview: ".status-pill { display: flex; }".to_string(),
                score: 90,
            }])
        }
    }

    struct ChangedFilesSearch;

    #[async_trait]
    impl CodeSearch for ChangedFilesSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec!["src/changed.rs".to_string()])
        }

        async fn changed_files(&self, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(vec![FileHit {
                path: "src/changed.rs".to_string(),
                score: 120,
            }])
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(Vec::new())
        }

        async fn search_symbols(&self, _query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_related_symbols(
            &self,
            _relation: SymbolRelation,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            Ok(Vec::new())
        }
    }

    struct SymbolAndContentSameFileSearch;

    #[async_trait]
    impl CodeSearch for SymbolAndContentSameFileSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec!["src/status.rs".to_string()])
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(Vec::new())
        }

        async fn search_symbols(&self, _query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            Ok(vec![SymbolHit {
                path: "src/status.rs".to_string(),
                symbol: "StatusPanel".to_string(),
                kind: "fn".to_string(),
                start_line: 4,
                end_line: 8,
                score: 120,
            }])
        }

        async fn search_related_symbols(
            &self,
            _relation: SymbolRelation,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            Ok(vec![ContentHit {
                path: "src/status.rs".to_string(),
                start_line: 30,
                end_line: 34,
                preview: "status panel styles".to_string(),
                score: 90,
            }])
        }
    }

    struct AdjacentFileSearch;

    #[async_trait]
    impl CodeSearch for AdjacentFileSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec![
                "src/status-panel.tsx".to_string(),
                "src/status-panel.test.tsx".to_string(),
            ])
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(Vec::new())
        }

        async fn search_symbols(&self, query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            if query == "status panel" {
                Ok(vec![SymbolHit {
                    path: "src/status-panel.tsx".to_string(),
                    symbol: "StatusPanel".to_string(),
                    kind: "component".to_string(),
                    start_line: 4,
                    end_line: 12,
                    score: 120,
                }])
            } else {
                Ok(Vec::new())
            }
        }

        async fn search_related_symbols(
            &self,
            _relation: SymbolRelation,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            Ok(Vec::new())
        }

        async fn search_adjacent_files(
            &self,
            seed_paths: &[String],
            _limit: usize,
        ) -> Result<Vec<FileHit>> {
            if seed_paths.iter().any(|path| path == "src/status-panel.tsx") {
                Ok(vec![FileHit {
                    path: "src/status-panel.test.tsx".to_string(),
                    score: 100,
                }])
            } else {
                Ok(Vec::new())
            }
        }
    }

    struct ImportedFileSearch;

    #[async_trait]
    impl CodeSearch for ImportedFileSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec![
                "src/status-panel.tsx".to_string(),
                "src/status-utils.ts".to_string(),
            ])
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(Vec::new())
        }

        async fn search_symbols(&self, query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            if query == "status panel" {
                Ok(vec![SymbolHit {
                    path: "src/status-panel.tsx".to_string(),
                    symbol: "StatusPanel".to_string(),
                    kind: "component".to_string(),
                    start_line: 4,
                    end_line: 12,
                    score: 120,
                }])
            } else {
                Ok(Vec::new())
            }
        }

        async fn search_related_symbols(
            &self,
            _relation: SymbolRelation,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            Ok(Vec::new())
        }

        async fn search_imported_files(
            &self,
            seed_paths: &[String],
            _limit: usize,
        ) -> Result<Vec<FileHit>> {
            if seed_paths.iter().any(|path| path == "src/status-panel.tsx") {
                Ok(vec![FileHit {
                    path: "src/status-utils.ts".to_string(),
                    score: 95,
                }])
            } else {
                Ok(Vec::new())
            }
        }
    }

    struct ImportingFileSearch;

    #[async_trait]
    impl CodeSearch for ImportingFileSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec![
                "src/status-utils.ts".to_string(),
                "src/status-panel.tsx".to_string(),
            ])
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(Vec::new())
        }

        async fn search_symbols(&self, query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            if query == "status utils" {
                Ok(vec![SymbolHit {
                    path: "src/status-utils.ts".to_string(),
                    symbol: "formatStatus".to_string(),
                    kind: "fn".to_string(),
                    start_line: 1,
                    end_line: 5,
                    score: 120,
                }])
            } else {
                Ok(Vec::new())
            }
        }

        async fn search_related_symbols(
            &self,
            _relation: SymbolRelation,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            Ok(Vec::new())
        }

        async fn search_importing_files(
            &self,
            seed_paths: &[String],
            _limit: usize,
        ) -> Result<Vec<FileHit>> {
            if seed_paths.iter().any(|path| path == "src/status-utils.ts") {
                Ok(vec![FileHit {
                    path: "src/status-panel.tsx".to_string(),
                    score: 100,
                }])
            } else {
                Ok(Vec::new())
            }
        }
    }

    struct ScopeFileSearch;

    #[async_trait]
    impl CodeSearch for ScopeFileSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec![
                "packages/web/src/status-panel.tsx".to_string(),
                "packages/web/package.json".to_string(),
            ])
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(Vec::new())
        }

        async fn search_symbols(&self, query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            if query == "status panel" {
                Ok(vec![SymbolHit {
                    path: "packages/web/src/status-panel.tsx".to_string(),
                    symbol: "StatusPanel".to_string(),
                    kind: "component".to_string(),
                    start_line: 4,
                    end_line: 12,
                    score: 120,
                }])
            } else {
                Ok(Vec::new())
            }
        }

        async fn search_related_symbols(
            &self,
            _relation: SymbolRelation,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            Ok(Vec::new())
        }

        async fn search_scope_files(
            &self,
            seed_paths: &[String],
            _limit: usize,
        ) -> Result<Vec<FileHit>> {
            if seed_paths
                .iter()
                .any(|path| path == "packages/web/src/status-panel.tsx")
            {
                Ok(vec![FileHit {
                    path: "packages/web/package.json".to_string(),
                    score: 125,
                }])
            } else {
                Ok(Vec::new())
            }
        }
    }

    struct ContextDocSearch;

    #[async_trait]
    impl CodeSearch for ContextDocSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec![
                "packages/web/src/status-panel.tsx".to_string(),
                "packages/web/docs/status-panel.md".to_string(),
            ])
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(Vec::new())
        }

        async fn search_symbols(&self, query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            if query == "status panel" {
                Ok(vec![SymbolHit {
                    path: "packages/web/src/status-panel.tsx".to_string(),
                    symbol: "StatusPanel".to_string(),
                    kind: "component".to_string(),
                    start_line: 4,
                    end_line: 12,
                    score: 120,
                }])
            } else {
                Ok(Vec::new())
            }
        }

        async fn search_related_symbols(
            &self,
            _relation: SymbolRelation,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            Ok(Vec::new())
        }

        async fn search_context_docs(
            &self,
            queries: &[String],
            seed_paths: &[String],
            _limit: usize,
        ) -> Result<Vec<FileHit>> {
            if queries.iter().any(|query| query == "status panel")
                && seed_paths
                    .iter()
                    .any(|path| path == "packages/web/src/status-panel.tsx")
            {
                Ok(vec![FileHit {
                    path: "packages/web/docs/status-panel.md".to_string(),
                    score: 130,
                }])
            } else {
                Ok(Vec::new())
            }
        }
    }

    struct DirectoryFileSearch;

    #[async_trait]
    impl CodeSearch for DirectoryFileSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec![
                "packages/web/src/App.tsx".to_string(),
                "packages/web/src/main.tsx".to_string(),
            ])
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(Vec::new())
        }

        async fn search_symbols(&self, _query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_related_symbols(
            &self,
            _relation: SymbolRelation,
            _query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            Ok(Vec::new())
        }

        async fn search_directory_files(
            &self,
            directories: &[String],
            _limit: usize,
        ) -> Result<Vec<FileHit>> {
            if directories.iter().any(|path| path == "packages/web/src") {
                Ok(vec![FileHit {
                    path: "packages/web/src/App.tsx".to_string(),
                    score: 100,
                }])
            } else {
                Ok(Vec::new())
            }
        }
    }

    struct DirectCalleeContextSearch;

    #[async_trait]
    impl CodeSearch for DirectCalleeContextSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec!["src/main.rs".to_string(), "src/helper.rs".to_string()])
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(Vec::new())
        }

        async fn search_symbols(&self, query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            if query == "main" {
                Ok(vec![SymbolHit {
                    path: "src/main.rs".to_string(),
                    symbol: "main".to_string(),
                    kind: "fn".to_string(),
                    start_line: 1,
                    end_line: 5,
                    score: 120,
                }])
            } else {
                Ok(Vec::new())
            }
        }

        async fn search_related_symbols(
            &self,
            relation: SymbolRelation,
            query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            if relation == SymbolRelation::Callees && query == "main" {
                Ok(vec![SymbolHit {
                    path: "src/helper.rs".to_string(),
                    symbol: "helper".to_string(),
                    kind: "fn".to_string(),
                    start_line: 10,
                    end_line: 14,
                    score: 100,
                }])
            } else {
                Ok(Vec::new())
            }
        }

        async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            Ok(Vec::new())
        }
    }

    struct EditImpactContextSearch;

    #[async_trait]
    impl CodeSearch for EditImpactContextSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec![
                "src/helper.rs".to_string(),
                "src/caller.rs".to_string(),
            ])
        }

        async fn search(&self, _query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            Ok(Vec::new())
        }

        async fn search_symbols(&self, query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            if query == "helper" {
                Ok(vec![SymbolHit {
                    path: "src/helper.rs".to_string(),
                    symbol: "helper".to_string(),
                    kind: "fn".to_string(),
                    start_line: 3,
                    end_line: 7,
                    score: 120,
                }])
            } else {
                Ok(Vec::new())
            }
        }

        async fn search_related_symbols(
            &self,
            relation: SymbolRelation,
            query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            match (relation, query) {
                (SymbolRelation::Callers, "helper") => Ok(vec![SymbolHit {
                    path: "src/caller.rs".to_string(),
                    symbol: "call_helper".to_string(),
                    kind: "fn".to_string(),
                    start_line: 20,
                    end_line: 24,
                    score: 100,
                }]),
                (SymbolRelation::References, "helper") => Ok(vec![SymbolHit {
                    path: "src/caller.rs".to_string(),
                    symbol: "call_helper".to_string(),
                    kind: "fn".to_string(),
                    start_line: 20,
                    end_line: 24,
                    score: 90,
                }]),
                _ => Ok(Vec::new()),
            }
        }

        async fn search_content(&self, _query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            Ok(Vec::new())
        }
    }

    struct FollowUpSearch;

    #[async_trait]
    impl CodeSearch for FollowUpSearch {
        async fn repo_map(&self, _max_entries: usize) -> Result<Vec<String>> {
            Ok(vec!["src/main.rs".to_string(), "src/status.rs".to_string()])
        }

        async fn search(&self, query: &str, _limit: usize) -> Result<Vec<FileHit>> {
            if query == "main" {
                Ok(vec![FileHit {
                    path: "src/main.rs".to_string(),
                    score: 10,
                }])
            } else {
                Ok(Vec::new())
            }
        }

        async fn search_symbols(&self, _query: &str, _limit: usize) -> Result<Vec<SymbolHit>> {
            Ok(Vec::new())
        }

        async fn search_related_symbols(
            &self,
            relation: SymbolRelation,
            query: &str,
            _limit: usize,
        ) -> Result<Vec<SymbolHit>> {
            if relation == SymbolRelation::Callees && query == "main" {
                Ok(vec![SymbolHit {
                    path: "src/helper.rs".to_string(),
                    symbol: "helper".to_string(),
                    kind: "fn".to_string(),
                    start_line: 20,
                    end_line: 24,
                    score: 120,
                }])
            } else {
                Ok(Vec::new())
            }
        }

        async fn search_content(&self, query: &str, _limit: usize) -> Result<Vec<ContentHit>> {
            if query == "status pill" {
                Ok(vec![ContentHit {
                    path: "src/status.rs".to_string(),
                    start_line: 4,
                    end_line: 8,
                    preview: "render_status_pill()".to_string(),
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
        async fn read_file(&self, path: &str, _max_chars: usize) -> Result<EvidenceSlice> {
            let content = "fn main() { println!(\"hi\"); }";
            Ok(EvidenceSlice {
                lease: EvidenceLease::new(
                    path.to_string(),
                    None,
                    LineRange { start: 1, end: 1 },
                    content,
                ),
                content: content.to_string(),
            })
        }

        async fn read_diff(&self, path: &str, _max_chars: usize) -> Result<EvidenceSlice> {
            let content = "diff --git a/src/changed.rs b/src/changed.rs\n+fn changed() {}";
            Ok(EvidenceSlice {
                lease: EvidenceLease::new(
                    path.to_string(),
                    Some("__git_diff__".to_string()),
                    LineRange { start: 1, end: 2 },
                    content,
                ),
                content: content.to_string(),
            })
        }

        async fn read_range(
            &self,
            path: &str,
            symbol: Option<&str>,
            start_line: u32,
            end_line: u32,
            _max_chars: usize,
        ) -> Result<EvidenceSlice> {
            let content = "fn main() { println!(\"hi\"); }";
            Ok(EvidenceSlice {
                lease: EvidenceLease::new(
                    path.to_string(),
                    symbol.map(ToString::to_string),
                    LineRange {
                        start: start_line as usize,
                        end: end_line as usize,
                    },
                    content,
                ),
                content: content.to_string(),
            })
        }

        async fn verify_lease(&self, _lease: &EvidenceLease) -> Result<LeaseStatus> {
            Ok(LeaseStatus::Valid)
        }
    }

    struct ExpiredReader;

    #[async_trait]
    impl FileReader for ExpiredReader {
        async fn read_file(&self, path: &str, _max_chars: usize) -> Result<EvidenceSlice> {
            let content = "fn main() { println!(\"hi\"); }";
            Ok(EvidenceSlice {
                lease: EvidenceLease::new(
                    path.to_string(),
                    None,
                    LineRange { start: 1, end: 1 },
                    content,
                ),
                content: content.to_string(),
            })
        }

        async fn read_range(
            &self,
            path: &str,
            symbol: Option<&str>,
            start_line: u32,
            end_line: u32,
            _max_chars: usize,
        ) -> Result<EvidenceSlice> {
            let content = "fn main() { println!(\"hi\"); }";
            Ok(EvidenceSlice {
                lease: EvidenceLease::new(
                    path.to_string(),
                    symbol.map(ToString::to_string),
                    LineRange {
                        start: start_line as usize,
                        end: end_line as usize,
                    },
                    content,
                ),
                content: content.to_string(),
            })
        }

        async fn verify_lease(&self, _lease: &EvidenceLease) -> Result<LeaseStatus> {
            Ok(LeaseStatus::Expired)
        }
    }

    struct CollectProgress {
        messages: Arc<Mutex<Vec<String>>>,
    }

    #[async_trait]
    impl ProgressSink for CollectProgress {
        async fn emit(&self, progress: AgentProgress) {
            if let AgentProgress::Status(message) = progress {
                self.messages.lock().unwrap().push(message);
            }
        }
    }

    #[tokio::test]
    async fn deterministic_loop_returns_answer() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["main"],"constraints":[],"search_queries":["main"]}"#.to_string(),
                r#"{"inferences":[{"statement":"main prints hi","lease_id":"placeholder","tags":["src/main.rs"]}],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The app entry point prints hi.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "What does main do?".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };
        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);
        assert!(result.answer.content.contains("entry point"));
    }

    #[tokio::test]
    async fn progress_reports_concrete_agent_steps() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Locate","target_hints":["main"],"constraints":[],"search_queries":["main"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The app entry point is in src/main.rs.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let messages = Arc::new(Mutex::new(Vec::new()));
        let progress = CollectProgress {
            messages: messages.clone(),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "where is main?".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &progress,
            },
        )
        .await
        .unwrap();

        let joined = messages.lock().unwrap().join("\n");
        assert!(joined.contains("Classifying the user's request"));
        assert!(joined.contains("Searching for main"));
        assert!(joined.contains("Found 1 file path for main"));
        assert!(joined.contains("Using `src/main.rs` lines 1-1 as evidence"));
        assert!(joined.contains("Accepted 0 supported findings from pass 1"));
    }

    #[tokio::test]
    async fn edit_request_returns_plan_without_synthesis() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Edit","target_hints":["main"],"constraints":[],"search_queries":["main"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"objective":"Update the main entry point.","affected_files":[{"path":"src/main.rs","reason":"It contains the entry point."}],"steps":[{"title":"Adjust main","detail":"Update the relevant entry point behavior after confirming the intended change."}],"risks":["The exact desired runtime behavior needs confirmation."],"validation_commands":["cargo test --workspace"]}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "update main".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert_eq!(result.answer.stop_reason, StopReason::Done);
        assert!(result.answer.content.contains("## Implementation Plan"));
        assert!(result.answer.content.contains("No files were changed"));
        assert!(result
            .trace
            .events
            .iter()
            .any(|event| matches!(event, TraceEvent::PhaseStarted { phase: Phase::Plan })));
        assert!(!result.trace.events.iter().any(|event| matches!(
            event,
            TraceEvent::PhaseStarted {
                phase: Phase::Synthesize
            }
        )));
    }

    #[tokio::test]
    async fn edit_tasks_expand_caller_and_reference_context_for_impact() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Edit","target_hints":["helper"],"constraints":[],"search_queries":["helper"]}"#.to_string(),
                "not valid extraction json".to_string(),
                r#"{"objective":"Change helper safely.","affected_files":[{"path":"src/helper.rs","reason":"It contains the edited helper."},{"path":"src/caller.rs","reason":"It calls helper and may need updates."}],"steps":[{"title":"Update helper","detail":"Adjust helper and check callers."}],"risks":["Caller behavior may change."],"validation_commands":[]}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "change helper behavior".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &EditImpactContextSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, args, result_count }
                    if name == "code_search_related_symbols"
                        && *result_count == 1
                        && args.get("context_expansion").and_then(|value| value.as_bool()) == Some(true)
                        && args.get("context_kind").and_then(|value| value.as_str()) == Some("edit_callers")
                        && args.get("relation").and_then(|value| value.as_str()) == Some("Callers")
            )
        }));
        assert!(result
            .answer
            .content
            .contains("`src/helper.rs`: It contains the edited helper."));
        assert!(result
            .answer
            .content
            .contains("`src/caller.rs`: It calls helper and may need updates."));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read edit caller call_helper for helper in src/caller.rs lines 20-24")));
    }

    #[test]
    fn startup_edit_queries_include_stack_neutral_boot_and_window_targets() {
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "add a splash screen and open the app in full screen when loaded".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };
        let mut intake = IntakeOutput {
            task_type: TaskType::Edit,
            target_hints: vec!["splash".to_string()],
            constraints: Vec::new(),
            search_queries: vec!["splash screen".to_string()],
        };

        normalize_intake(&mut intake, &task);

        assert!(intake.search_queries.contains(&"startup".to_string()));
        assert!(intake.search_queries.contains(&"boot".to_string()));
        assert!(intake.search_queries.contains(&"loading".to_string()));
        assert!(intake.search_queries.contains(&"window".to_string()));
        assert!(intake.search_queries.contains(&"fullscreen".to_string()));
    }

    #[test]
    fn edit_requests_include_validation_context_queries_within_budget() {
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "update the status panel".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };
        let mut intake = IntakeOutput {
            task_type: TaskType::Edit,
            target_hints: vec!["status panel".to_string()],
            constraints: Vec::new(),
            search_queries: vec![
                "status panel".to_string(),
                "StatusPanel".to_string(),
                "status view".to_string(),
                "render status".to_string(),
            ],
        };

        normalize_intake(&mut intake, &task);
        let budgeted = intake
            .search_queries
            .iter()
            .take(task.budgets.max_search_queries)
            .cloned()
            .collect::<Vec<_>>();

        assert!(budgeted.contains(&"project manifest".to_string()));
        assert!(budgeted.contains(&"test script".to_string()));
        assert!(budgeted.contains(&"build script".to_string()));
        assert!(budgeted.contains(&"related tests".to_string()));
    }

    #[test]
    fn normalization_preserves_user_question_terms_within_budget() {
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "explain the status panel rendering flow".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };
        let mut intake = IntakeOutput {
            task_type: TaskType::Explain,
            target_hints: vec!["status".to_string()],
            constraints: Vec::new(),
            search_queries: vec!["status".to_string(), "renderStatus".to_string()],
        };

        normalize_intake(&mut intake, &task);
        let budgeted = intake
            .search_queries
            .iter()
            .take(task.budgets.max_search_queries)
            .cloned()
            .collect::<Vec<_>>();

        assert!(budgeted.contains(&"status panel rendering flow".to_string()));
    }

    #[test]
    fn normalization_preserves_path_mentions_within_budget() {
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question:
                "review `src-tauri/src/orchestrator.rs:442-480` and crates/agent-core/src/pipeline.rs"
                    .to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };
        let mut intake = IntakeOutput {
            task_type: TaskType::Explain,
            target_hints: vec!["orchestrator".to_string()],
            constraints: Vec::new(),
            search_queries: vec!["orchestrator".to_string(), "agent loop".to_string()],
        };

        normalize_intake(&mut intake, &task);
        let budgeted = intake
            .search_queries
            .iter()
            .take(task.budgets.max_search_queries)
            .cloned()
            .collect::<Vec<_>>();

        assert!(budgeted.contains(&"src-tauri/src/orchestrator.rs".to_string()));
        assert!(budgeted.contains(&"crates/agent-core/src/pipeline.rs".to_string()));
        assert!(intake
            .target_hints
            .contains(&"src-tauri/src/orchestrator.rs".to_string()));
    }

    #[test]
    fn path_query_extraction_skips_urls_and_parent_paths() {
        let paths = extract_path_queries(
            "compare https://example.com/src/main.rs with ../outside.rs and ./src/App.tsx.",
        );

        assert_eq!(paths, vec!["src/App.tsx".to_string()]);
    }

    #[test]
    fn research_requests_include_project_overview_queries_within_budget() {
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "review the architecture of this codebase".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };
        let mut intake = IntakeOutput {
            task_type: TaskType::Explain,
            target_hints: vec!["architecture".to_string()],
            constraints: Vec::new(),
            search_queries: vec![
                "agent loop".to_string(),
                "code discovery".to_string(),
                "context filling".to_string(),
                "retrieval".to_string(),
            ],
        };

        normalize_intake(&mut intake, &task);
        let budgeted = intake
            .search_queries
            .iter()
            .take(task.budgets.max_search_queries)
            .cloned()
            .collect::<Vec<_>>();

        assert!(budgeted.contains(&"project overview".to_string()));
        assert!(budgeted.contains(&"readme".to_string()));
        assert!(budgeted.contains(&"repository instructions".to_string()));
        assert!(budgeted.contains(&"project entrypoints".to_string()));
    }

    #[test]
    fn edit_plan_drops_files_without_active_file_evidence() {
        let mut notebook = Notebook::default();
        let fact = notebook.append_tool_fact(
            "Read user-facing app shell in app/root.ui lines 22-120",
            Provenance::File {
                path: "app/root.ui".to_string(),
                symbol: Some("App".to_string()),
                range: LineRange {
                    start: 22,
                    end: 120,
                },
                lease_id: crate::notebook::LeaseId("lease".to_string()),
            },
            Phase::Extract,
            vec!["app/root.ui".to_string(), "AppShell".to_string()],
        );
        let inference = notebook.append_model_inference(
            "app/root.ui owns the booting and ready UI states.",
            Provenance::Model {
                call_id: "extract".to_string(),
            },
            Phase::Extract,
            vec![fact],
            vec!["app/root.ui".to_string()],
        );
        let mut plan = EditPlan {
            objective: "Add a splash screen".to_string(),
            affected_files: vec![
                crate::types::PlannedFile {
                    path: "app/root.ui".to_string(),
                    reason: "Evidence shows it owns the boot UI.".to_string(),
                },
                crate::types::PlannedFile {
                    path: "services/agent_orchestrator.lang".to_string(),
                    reason: "The model guessed this unrelated orchestration file.".to_string(),
                },
            ],
            steps: Vec::new(),
            risks: Vec::new(),
            validation_commands: vec!["framework-cli test".to_string()],
        };

        constrain_edit_plan_to_evidence(&mut plan, &notebook, &[inference]);

        assert_eq!(plan.affected_files.len(), 1);
        assert_eq!(plan.affected_files[0].path, "app/root.ui");
        assert!(plan.validation_commands.is_empty());
    }

    #[test]
    fn evidence_range_overlap_detects_contained_ranges_only() {
        let evidence = vec![EvidenceSlice {
            lease: EvidenceLease::new(
                "src/status.rs".to_string(),
                Some("StatusPanel".to_string()),
                LineRange { start: 10, end: 20 },
                "fn status_panel() {}",
            ),
            content: "fn status_panel() {}".to_string(),
        }];

        assert!(evidence_overlaps_range(&evidence, "src/status.rs", 12, 14));
        assert!(evidence_overlaps_range(&evidence, "src/status.rs", 1, 10));
        assert!(!evidence_overlaps_range(&evidence, "src/status.rs", 21, 24));
        assert!(!evidence_overlaps_range(&evidence, "src/other.rs", 12, 14));
    }

    #[tokio::test]
    async fn content_search_hits_are_read_as_evidence() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Locate","target_hints":["status pill"],"constraints":[],"search_queries":["status pill"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The status pill styling is in src/styles.css.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "where are status pill styles?".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &ContentOnlySearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall {
                    name,
                    result_count: 1,
                    ..
                } if name == "code_search_content"
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read text match in src/styles.css lines 10-14")));
    }

    #[tokio::test]
    async fn review_requests_read_changed_files_as_evidence() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["current changes"],"constraints":[],"search_queries":["current changes"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The changed file was inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "review current changes".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &ChangedFilesSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, result_count, .. }
                    if name == "changed_files" && *result_count == 1
            )
        }));
        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, result_count, .. }
                    if name == "read_changed_diff" && *result_count == 1
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read diff for src/changed.rs lines 1-2")));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry.statement.contains("Read src/changed.rs lines 1-1")));
    }

    #[tokio::test]
    async fn disjoint_content_hit_in_symbol_file_is_kept_as_evidence() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["status panel"],"constraints":[],"search_queries":["status panel"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The status panel implementation and related text were inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "explain the status panel implementation and related styles".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &SymbolAndContentSameFileSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        let statements = result
            .answer
            .notebook
            .iter_in_append_order()
            .map(|entry| entry.statement.as_str())
            .collect::<Vec<_>>();
        assert!(statements
            .iter()
            .any(|statement| statement
                .contains("Read symbol StatusPanel in src/status.rs lines 4-8")));
        assert!(statements
            .iter()
            .any(|statement| statement.contains("Read text match in src/status.rs lines 30-34")));
    }

    #[tokio::test]
    async fn review_tasks_read_adjacent_files_from_seed_evidence() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["status panel"],"constraints":[],"search_queries":["status panel"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The status panel and adjacent test were inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "review status panel behavior".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &AdjacentFileSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, result_count, .. }
                    if name == "code_search_adjacent_files" && *result_count == 1
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read src/status-panel.test.tsx lines 1-1")));
    }

    #[tokio::test]
    async fn review_tasks_read_imported_files_from_seed_evidence() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["status panel"],"constraints":[],"search_queries":["status panel"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The status panel and imported helper were inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "review status panel dependency flow".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &ImportedFileSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, result_count, .. }
                    if name == "code_search_imported_files" && *result_count == 1
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read src/status-utils.ts lines 1-1")));
    }

    #[tokio::test]
    async fn review_tasks_read_importing_files_from_seed_evidence() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["status utils"],"constraints":[],"search_queries":["status utils"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The helper and dependent status panel were inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "review status utils impact".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &ImportingFileSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, result_count, .. }
                    if name == "code_search_importing_files" && *result_count == 1
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read src/status-panel.tsx lines 1-1")));
    }

    #[tokio::test]
    async fn review_tasks_read_scope_files_from_seed_evidence() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["status panel"],"constraints":[],"search_queries":["status panel"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The status panel and local package context were inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "review status panel package context".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &ScopeFileSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, result_count, .. }
                    if name == "code_search_scope_files" && *result_count == 1
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read packages/web/package.json lines 1-1")));
    }

    #[tokio::test]
    async fn research_tasks_read_nearby_context_docs_from_seed_evidence() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["status panel"],"constraints":[],"search_queries":["status panel"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The status panel and nearby docs were inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "research how the status panel works".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &ContextDocSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, result_count, .. }
                    if name == "code_search_context_docs" && *result_count == 1
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read packages/web/docs/status-panel.md lines 1-1")));
    }

    #[tokio::test]
    async fn directory_mentions_expand_to_ranked_files_without_file_read_failures() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["web src"],"constraints":[],"search_queries":["web src"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The requested directory was inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "review packages/web/src".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &DirectoryFileSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, args, result_count }
                    if name == "code_search_directory_files"
                        && *result_count == 1
                        && args.get("directories")
                            .and_then(|value| value.as_array())
                            .map(|items| items.iter().any(|item| item.as_str() == Some("packages/web/src")))
                            == Some(true)
            )
        }));
        assert!(!result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Could not read explicit requested file packages/web/src")));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read packages/web/src/App.tsx lines 1-1")));
    }

    #[tokio::test]
    async fn explicit_file_ranges_are_read_before_model_follow_up() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["detail"],"constraints":[],"search_queries":["detail"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The requested range was inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "review `src/detail.rs:12-18` before explaining detail".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, args, result_count }
                    if name == "read_explicit_file_range"
                        && *result_count == 1
                        && args.get("path").and_then(|value| value.as_str()) == Some("src/detail.rs")
                        && args.get("start_line").and_then(|value| value.as_u64()) == Some(12)
                        && args.get("end_line").and_then(|value| value.as_u64()) == Some(18)
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read explicit requested range in src/detail.rs lines 12-18")));
    }

    #[tokio::test]
    async fn explicit_file_mentions_are_read_before_ranked_path_fallback() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["detail"],"constraints":[],"search_queries":["detail"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The requested file was inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "review src/detail.rs before explaining detail".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, args, result_count }
                    if name == "read_explicit_file"
                        && *result_count == 1
                        && args.get("path").and_then(|value| value.as_str()) == Some("src/detail.rs")
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read explicit requested file src/detail.rs lines 1-1")));
    }

    #[tokio::test]
    async fn explicit_range_mentions_do_not_duplicate_whole_file_reads() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["detail"],"constraints":[],"search_queries":["detail"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The requested range was inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "review src/detail.rs:12-18 before explaining detail".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, .. } if name == "read_explicit_file_range"
            )
        }));
        assert!(!result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, args, .. }
                    if name == "read_explicit_file"
                        && args.get("path").and_then(|value| value.as_str()) == Some("src/detail.rs")
            )
        }));
    }

    #[tokio::test]
    async fn missing_evidence_follow_up_uses_content_search_and_records_lease() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["main"],"constraints":[],"search_queries":["main"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":false,"missing":[{"kind":"Search","target":"status pill","reason":"Need the render text."}]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The follow-up evidence was inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let mut budgets = Budgets::default();
        budgets.max_extract_rounds = 2;
        budgets.max_llm_calls = 4;
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "explain main".to_string(),
            conversation: Vec::new(),
            budgets,
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &FollowUpSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, args, .. }
                    if name == "code_search_content"
                        && args.get("follow_up").and_then(|value| value.as_bool()) == Some(true)
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read follow-up text match in src/status.rs")));
    }

    #[tokio::test]
    async fn missing_file_range_follow_up_reads_exact_leased_range() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["main"],"constraints":[],"search_queries":["main"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":false,"missing":[{"kind":"FileRange","target":"src/detail.rs:12-18","reason":"Need the nearby implementation."}]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The requested range was inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let mut budgets = Budgets::default();
        budgets.max_extract_rounds = 2;
        budgets.max_llm_calls = 4;
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "explain main and nearby detail range".to_string(),
            conversation: Vec::new(),
            budgets,
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, args, result_count }
                    if name == "read_file_range"
                        && *result_count == 1
                        && args.get("path").and_then(|value| value.as_str()) == Some("src/detail.rs")
                        && args.get("start_line").and_then(|value| value.as_u64()) == Some(12)
                        && args.get("end_line").and_then(|value| value.as_u64()) == Some(18)
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read requested file range in src/detail.rs lines 12-18")));
    }

    #[tokio::test]
    async fn missing_symbol_relation_uses_related_symbol_search() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["main"],"constraints":[],"search_queries":["main"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":false,"missing":[{"kind":"Callees","target":"main","reason":"Need symbols called by main."}]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The callee evidence was inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let mut budgets = Budgets::default();
        budgets.max_extract_rounds = 2;
        budgets.max_llm_calls = 4;
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "explain what main calls".to_string(),
            conversation: Vec::new(),
            budgets,
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &FollowUpSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, args, result_count }
                    if name == "code_search_related_symbols"
                        && *result_count == 1
                        && args.get("relation").and_then(|value| value.as_str()) == Some("Callees")
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read follow-up Callees symbol helper in src/helper.rs lines 20-24")));
    }

    #[tokio::test]
    async fn initial_relation_intent_reads_related_symbols_before_extraction_follow_up() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["main"],"constraints":[],"search_queries":["main"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The callee evidence was already available.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "what does main call?".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &FollowUpSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, args, result_count }
                    if name == "code_search_related_symbols"
                        && *result_count == 1
                        && args.get("initial").and_then(|value| value.as_bool()) == Some(true)
                        && args.get("relation").and_then(|value| value.as_str()) == Some("Callees")
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read symbol helper in src/helper.rs lines 20-24")));
    }

    #[tokio::test]
    async fn explain_tasks_expand_direct_callee_context_from_seed_symbols() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["main"],"constraints":[],"search_queries":["main"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":true,"missing":[]}"#.to_string(),
                r#"{"answer":"The main implementation and direct helper were inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "explain main".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &DirectCalleeContextSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result.trace.events.iter().any(|event| {
            matches!(
                event,
                TraceEvent::ToolCall { name, args, result_count }
                    if name == "code_search_related_symbols"
                        && *result_count == 1
                        && args.get("context_expansion").and_then(|value| value.as_bool()) == Some(true)
                        && args.get("relation").and_then(|value| value.as_str()) == Some("Callees")
            )
        }));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .any(|entry| entry
                .statement
                .contains("Read direct callee helper for main in src/helper.rs lines 10-14")));
    }

    #[test]
    fn relation_intents_detect_common_code_navigation_questions() {
        assert_eq!(
            relation_intents_for_question("what does main call?"),
            vec![SymbolRelation::Callees]
        );
        assert_eq!(
            relation_intents_for_question("who calls renderStatus?"),
            vec![SymbolRelation::Callers]
        );
        assert!(
            relation_intents_for_question("where is StatusPanel defined?")
                .contains(&SymbolRelation::Definition)
        );
        assert!(
            relation_intents_for_question("find references to workspace_id")
                .contains(&SymbolRelation::References)
        );
    }

    #[test]
    fn file_range_missing_targets_parse_common_location_formats() {
        assert_eq!(
            parse_file_range_target("src/main.rs:10-20"),
            Some(("src/main.rs".to_string(), 10, 20))
        );
        assert_eq!(
            parse_file_range_target("`src/main.rs` lines 10-20"),
            Some(("src/main.rs".to_string(), 10, 20))
        );
        assert_eq!(
            parse_file_range_target("src/main.rs:L10-L20"),
            Some(("src/main.rs".to_string(), 10, 20))
        );
        assert_eq!(parse_file_range_target("src/main.rs:20-10"), None);
    }

    #[test]
    fn explicit_file_range_extraction_finds_common_question_mentions() {
        let ranges = extract_file_range_queries(
            "inspect `src/main.rs:10-20`, src/lib.rs lines 3-4, and https://site.test/app.rs:1-2",
        );

        assert_eq!(
            ranges,
            vec![
                ("src/main.rs".to_string(), 10, 20),
                ("src/lib.rs".to_string(), 3, 4)
            ]
        );
    }

    #[tokio::test]
    async fn synthesis_uses_active_findings_when_model_omits_citations() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["main"],"constraints":[],"search_queries":["main"]}"#.to_string(),
                "not json".to_string(),
                r#"{"answer":"The entry point was inspected.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "What does main do?".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(!result.answer.cited_entries.is_empty());
        assert!(result.answer.cited_entries.iter().all(|id| result
            .answer
            .notebook
            .get(id)
            .map(|entry| entry.status == crate::notebook::EntryStatus::Active)
            .unwrap_or(false)));
    }

    #[tokio::test]
    async fn extraction_stops_when_no_new_evidence_is_requested() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["main"],"constraints":[],"search_queries":["main"]}"#.to_string(),
                r#"{"inferences":[],"sufficient":false,"missing":[]}"#.to_string(),
                r#"{"answer":"No supported finding yet.","cited_entries":[],"stop_reason":"NoProgress"}"#.to_string(),
            ]),
        };
        let mut budgets = Budgets::default();
        budgets.max_extract_rounds = 3;
        budgets.max_llm_calls = 5;
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "explain main".to_string(),
            conversation: Vec::new(),
            budgets,
        };

        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &MockReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        let extract_phases = result
            .trace
            .events
            .iter()
            .filter(|event| {
                matches!(
                    event,
                    TraceEvent::PhaseStarted {
                        phase: Phase::Extract
                    }
                )
            })
            .count();
        assert_eq!(extract_phases, 1);
    }

    #[test]
    fn fallback_answer_hides_notebook_entry_ids() {
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "What does main do?".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };
        let mut notebook = Notebook::default();
        let inference = notebook.append_model_inference(
            "main prints hi.",
            Provenance::File {
                path: "src/main.rs".to_string(),
                symbol: Some("main".to_string()),
                range: LineRange { start: 1, end: 1 },
                lease_id: crate::notebook::LeaseId("lease".to_string()),
            },
            Phase::Extract,
            Vec::new(),
            vec!["src/main.rs".to_string()],
        );

        let answer = fallback_answer(&task, &notebook, &[inference.clone()], StopReason::Done);

        assert!(answer.contains("- main prints hi."));
        assert!(!answer.contains(&inference.0));
        assert!(!answer.contains("- ["));
    }

    #[tokio::test]
    async fn expired_lease_invalidates_notebook_and_rejects_inference() {
        let llm = MockLlm {
            responses: Mutex::new(vec![
                r#"{"task_type":"Explain","target_hints":["main"],"constraints":[],"search_queries":["main"]}"#.to_string(),
                "not json".to_string(),
                r#"{"answer":"No current inference accepted.","cited_entries":[],"stop_reason":"Done"}"#.to_string(),
            ]),
        };
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "What does main do?".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };
        let result = run_agent(
            task,
            AgentRuntime {
                llm: &llm,
                code_search: &MockSearch,
                file_reader: &ExpiredReader,
                token_counter: &ApproxTokenCounter,
                progress: &NoopProgressSink,
            },
        )
        .await
        .unwrap();

        assert!(result
            .trace
            .events
            .iter()
            .any(|event| matches!(event, TraceEvent::LeaseInvalidated { .. })));
        assert!(result
            .answer
            .notebook
            .iter_in_append_order()
            .all(|entry| entry.kind != crate::notebook::EntryKind::Inference));
    }

    #[test]
    fn actionable_ui_prompt_overrides_unclear_intake_as_edit_when_change_requested() {
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "can you update the workspace displayed in the left panel of our UI so it’s not truncated and wraps appropriately".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };
        let mut intake = IntakeOutput {
            task_type: TaskType::Unclear,
            target_hints: Vec::new(),
            constraints: Vec::new(),
            search_queries: Vec::new(),
        };

        normalize_intake(&mut intake, &task);

        assert_eq!(intake.task_type, TaskType::Edit);
        assert!(!intake.search_queries.is_empty());
    }

    #[test]
    fn citations_prefer_narrower_file_ranges() {
        let lease = EvidenceLease::new(
            "src/GitPanel.tsx".to_string(),
            None,
            LineRange { start: 1, end: 160 },
            "whole file",
        );
        let narrow_lease = EvidenceLease::new(
            "src/GitPanel.tsx".to_string(),
            Some("ChangedFilesList".to_string()),
            LineRange {
                start: 106,
                end: 140,
            },
            "changed files list",
        );
        let mut notebook = Notebook::default();
        let broad = notebook.append_tool_fact(
            "Read src/GitPanel.tsx lines 1-160",
            Provenance::File {
                path: lease.path,
                symbol: lease.symbol,
                range: lease.range,
                lease_id: lease.id,
            },
            Phase::Extract,
            vec!["src/GitPanel.tsx".to_string()],
        );
        let narrow = notebook.append_tool_fact(
            "Read src/GitPanel.tsx lines 106-140",
            Provenance::File {
                path: narrow_lease.path,
                symbol: narrow_lease.symbol,
                range: narrow_lease.range,
                lease_id: narrow_lease.id,
            },
            Phase::Extract,
            vec!["src/GitPanel.tsx".to_string()],
        );

        let sorted = sort_citations_by_specificity(&notebook, vec![broad, narrow.clone()]);

        assert_eq!(sorted.first(), Some(&narrow));
    }

    #[test]
    fn per_slice_budget_keeps_ui_render_blocks_in_view() {
        let task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "where in the UI do we display git change file list?".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };

        assert!(per_slice_budget(&task) >= 8_000);
    }

    #[test]
    fn evidence_token_budget_allows_first_slice_then_caps_more() {
        let mut task = TaskRequest {
            run_id: "run".to_string(),
            session_id: "session".to_string(),
            workspace_path: None,
            question: "explain main".to_string(),
            conversation: Vec::new(),
            budgets: Budgets::default(),
        };
        task.budgets.max_evidence_tokens = 2;
        let first = lease_for_text("src/main.rs".to_string(), "one two three");

        assert!(should_accept_evidence(&[], 0, 3, &task));
        assert!(!should_accept_evidence(&[first], 3, 1, &task));
    }

    #[test]
    fn location_question_focuses_whole_file_slice_to_matching_jsx_section() {
        let mut lines = Vec::new();
        for idx in 1..=228 {
            let line = match idx {
                106 => "      {/* Changed Files List (Top Half) */}".to_string(),
                107 => "      <div class=\"shrink-0 max-h-60 overflow-y-auto\">".to_string(),
                113 => "        <For each={files()}>".to_string(),
                127 => {
                    "              <span class=\"truncate block\" title={file.path}>".to_string()
                }
                133 => "                class={`font-mono text-xs ${getStatusColor(file.status)}`}"
                    .to_string(),
                140 => "      </div>".to_string(),
                142 => "      {/* Diff Viewer (Bottom Half) */}".to_string(),
                _ => format!("      <div>line {idx}</div>"),
            };
            lines.push(line);
        }
        let content = lines.join("\n");
        let slice = EvidenceSlice {
            lease: EvidenceLease::new(
                "src/GitPanel.tsx".to_string(),
                None,
                LineRange { start: 1, end: 228 },
                &content,
            ),
            content,
        };

        let focused = focus_location_slice(
            slice,
            "where in the UI do we display git change file list?",
            &["git change file list".to_string()],
        );

        assert_eq!(focused.lease.range.start, 106);
        assert_eq!(focused.lease.range.end, 140);
        assert!(focused.content.contains("<For each={files()}>"));
        assert!(!focused.content.contains("Diff Viewer"));
    }

    #[test]
    fn non_location_question_keeps_whole_file_slice() {
        let slice = lease_for_text(
            "src/GitPanel.tsx".to_string(),
            &(1..=100)
                .map(|idx| format!("line {idx}"))
                .collect::<Vec<_>>()
                .join("\n"),
        );

        let focused = focus_location_slice(slice, "explain this file", &[]);

        assert_eq!(focused.lease.range.start, 1);
        assert_eq!(focused.lease.range.end, 100);
    }
}
