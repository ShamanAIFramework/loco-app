use crate::notebook::{EntryId, EvidenceLease, LineRange, Notebook, Provenance};
use crate::prompt::{
    evidence_block, extract_instruction, global_prefix, intake_instruction, notebook_block,
    phase_prefix, schema_extract, schema_intake, schema_synthesize, synthesize_instruction,
    task_block, PromptParts,
};
use crate::trace::{Trace, TraceEvent};
use crate::traits::{
    AgentProgress, CodeSearch, FileReader, LlmClient, LlmRequest, ProgressSink, TokenCounter,
};
use crate::types::{
    AgentAnswer, EvidenceSlice, ExtractOutput, IntakeOutput, LeaseStatus, MissingEvidenceKind,
    Phase, StopReason, SymbolHit, SynthesizeOutput, TaskRequest, TaskType,
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
    let mut symbol_hits = BTreeMap::<(String, String, u32, u32), SymbolHit>::new();
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
    }

    let mut sorted_paths: Vec<_> = file_paths.into_iter().collect();
    sorted_paths.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));

    let mut evidence = Vec::new();
    let mut evidence_tokens = 0usize;
    let mut lease_fact_entries = BTreeMap::new();
    let mut read_paths = BTreeSet::<String>::new();
    let mut sorted_symbols = symbol_hits.into_values().collect::<Vec<_>>();
    sorted_symbols.sort_by(|a, b| {
        b.score
            .cmp(&a.score)
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.start_line.cmp(&b.start_line))
            .then_with(|| a.symbol.cmp(&b.symbol))
    });
    for hit in sorted_symbols
        .into_iter()
        .take(task.budgets.max_files_to_read)
    {
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

        let mut new_queries = Vec::new();
        for missing in extract.missing {
            let key = format!("{:?}:{}", missing.kind, missing.target);
            if previous_missing.insert(key) && matches!(missing.kind, MissingEvidenceKind::Search) {
                new_queries.push(missing.target);
            }
        }

        if new_queries.is_empty() || inference_entry_ids.len() == before_count {
            if round + 1 >= task.budgets.max_extract_rounds {
                break;
            }
        }

        for query in new_queries
            .into_iter()
            .take(task.budgets.max_search_queries.saturating_sub(1))
        {
            runtime
                .progress
                .emit(AgentProgress::Status(format!(
                    "Finding additional evidence for {}",
                    plain_query(&query)
                )))
                .await;
            let hits = runtime
                .code_search
                .search(&query, task.budgets.max_files_to_read)
                .await
                .unwrap_or_default();
            for hit in hits {
                if evidence.iter().any(|slice| slice.lease.path == hit.path) {
                    continue;
                }
                if let Ok(slice) = runtime.file_reader.read_file(&hit.path, 4_000).await {
                    let token_count = runtime.token_counter.count_tokens(&slice.content).await;
                    if !should_accept_evidence(&evidence, evidence_tokens, token_count, &task) {
                        continue;
                    }
                    evidence_tokens += token_count;
                    evidence.push(slice);
                }
            }
        }
    }

    runtime
        .progress
        .emit(AgentProgress::Status("Preparing the answer".to_string()))
        .await;

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
    let cited_entries = sort_citations_by_specificity(
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
        task_type: TaskType::Explain,
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
    dedup(&mut intake.search_queries);
    dedup(&mut intake.target_hints);
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
        cited_entries: inference_entry_ids.to_vec(),
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
                lines.push(format!("- [{}] {}", entry.id.0, entry.statement));
            }
        }
    }
    if stop_reason == StopReason::BudgetExhausted {
        lines.push("The run stopped because the LLM call budget was exhausted.".to_string());
    }
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
    use crate::types::{Budgets, FileHit};
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
    fn actionable_ui_prompt_overrides_unclear_intake() {
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

        assert_eq!(intake.task_type, TaskType::Explain);
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
