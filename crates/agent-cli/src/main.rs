use agent_core::{Trace, TraceEvent};
use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "agent-cli")]
#[command(about = "Headless tools for Loco's deterministic local agent")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Validate and summarize a saved trace JSONL file.
    Replay {
        trace_path: PathBuf,
    },
    /// Run a deterministic no-LLM path baseline over an eval task JSONL file.
    Baseline {
        task_path: PathBuf,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Score the no-LLM baseline against expected files/symbols in a task file.
    Eval {
        task_path: PathBuf,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct EvalTask {
    id: String,
    repo_path: PathBuf,
    prompt: String,
    #[serde(default)]
    task_type: Option<String>,
    #[serde(default)]
    expected_files: Vec<String>,
    #[serde(default)]
    expected_symbols: Vec<String>,
    #[serde(default)]
    expected_outcome: Option<String>,
    #[serde(default)]
    notes: Option<String>,
    #[serde(default)]
    trace_path: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize)]
struct BaselineResult {
    id: String,
    predicted_files: Vec<String>,
    predicted_symbols: Vec<String>,
    expected_files: Vec<String>,
    expected_symbols: Vec<String>,
    file_recall: f64,
    file_precision: f64,
    symbol_recall: f64,
    symbol_precision: f64,
    trace: Option<TraceMetrics>,
    expected_outcome: Option<String>,
    outcome_correct: Option<bool>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct TraceMetrics {
    citation_validity: f64,
    invalid_citations: usize,
    stop_reason: Option<String>,
    llm_calls: usize,
    tool_calls: usize,
    prompt_tokens_total: u64,
    decode_tokens: u64,
    wall_ms: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Replay { trace_path } => replay(&trace_path),
        Command::Baseline { task_path, limit } => baseline(&task_path, limit),
        Command::Eval { task_path, limit } => eval(&task_path, limit),
    }
}

fn replay(trace_path: &Path) -> Result<()> {
    let trace = read_trace(trace_path)?;
    let mut run_started = 0usize;
    let mut llm_calls = 0usize;
    let mut regenerated_prompts = 0usize;
    let mut tool_calls = 0usize;
    let mut run_finished = 0usize;
    let mut phases = BTreeSet::new();

    for event in &trace.events {
        match event {
            TraceEvent::RunStarted { .. } => run_started += 1,
            TraceEvent::PhaseStarted { phase } => {
                phases.insert(format!("{:?}", phase));
            }
            TraceEvent::LlmCall { request, response } => {
                llm_calls += 1;
                if request.call_id != response.call_id {
                    return Err(anyhow!(
                        "LLM call id mismatch: request={} response={}",
                        request.call_id,
                        response.call_id
                    ));
                }
                if request.prompt.trim().is_empty() {
                    return Err(anyhow!("LLM request {} has an empty prompt", request.call_id));
                }
                if let Some(parts) = &request.prompt_parts {
                    regenerated_prompts += 1;
                    let regenerated = parts.render();
                    if regenerated != request.prompt {
                        return Err(anyhow!(
                            "Prompt replay mismatch for {}: regenerated {} bytes, recorded {} bytes",
                            request.call_id,
                            regenerated.len(),
                            request.prompt.len()
                        ));
                    }
                } else {
                    return Err(anyhow!(
                        "LLM request {} is missing prompt_parts for deterministic replay",
                        request.call_id
                    ));
                }
            }
            TraceEvent::ToolCall { .. } => tool_calls += 1,
            TraceEvent::NotebookSnapshot { .. } => {}
            TraceEvent::LeaseInvalidated { .. } => {}
            TraceEvent::RunFinished { .. } => run_finished += 1,
        }
    }

    if run_started != 1 {
        return Err(anyhow!("Expected exactly one RunStarted event, saw {}", run_started));
    }
    if run_finished != 1 {
        return Err(anyhow!("Expected exactly one RunFinished event, saw {}", run_finished));
    }

    println!(
        "trace ok: llm_calls={} regenerated_prompts={} tool_calls={} phases={}",
        llm_calls,
        regenerated_prompts,
        tool_calls,
        phases.into_iter().collect::<Vec<_>>().join(",")
    );
    Ok(())
}

fn baseline(task_path: &Path, limit: usize) -> Result<()> {
    for task in read_tasks(task_path)? {
        let predicted_files = baseline_files(&task.repo_path, &task.prompt, limit)?;
        println!(
            "{}",
            serde_json::to_string(&serde_json::json!({
                "id": task.id,
                "predicted_files": predicted_files,
            }))?
        );
    }
    Ok(())
}

fn eval(task_path: &Path, limit: usize) -> Result<()> {
    let mut results = Vec::new();
    for task in read_tasks(task_path)? {
        let predicted_files = baseline_files(&task.repo_path, &task.prompt, limit)?;
        let predicted_symbols = baseline_symbols(&task.repo_path, &task.prompt, limit)?;
        let (file_recall, file_precision) = score_items(&predicted_files, &task.expected_files);
        let (symbol_recall, symbol_precision) =
            score_items(&predicted_symbols, &task.expected_symbols);
        let trace = task
            .trace_path
            .as_ref()
            .map(|path| read_trace(path).and_then(|trace| trace_metrics(&trace)))
            .transpose()?;
        let outcome_correct = task.expected_outcome.as_ref().and_then(|expected| {
            trace
                .as_ref()
                .and_then(|metrics| metrics.stop_reason.as_ref())
                .map(|actual| outcome_matches(expected, actual))
        });
        results.push(BaselineResult {
            id: task.id,
            predicted_files,
            predicted_symbols,
            expected_files: task.expected_files,
            expected_symbols: task.expected_symbols,
            file_recall,
            file_precision,
            symbol_recall,
            symbol_precision,
            trace,
            expected_outcome: task.expected_outcome,
            outcome_correct,
        });
    }

    let avg_file_recall = average(results.iter().map(|r| r.file_recall));
    let avg_file_precision = average(results.iter().map(|r| r.file_precision));
    let avg_symbol_recall = average(results.iter().map(|r| r.symbol_recall));
    let avg_symbol_precision = average(results.iter().map(|r| r.symbol_precision));
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "task_count": results.len(),
            "average_file_recall": avg_file_recall,
            "average_file_precision": avg_file_precision,
            "average_symbol_recall": avg_symbol_recall,
            "average_symbol_precision": avg_symbol_precision,
            "stop_reason_distribution": stop_reason_distribution(&results),
            "outcome_accuracy": outcome_accuracy(&results),
            "results": results,
        }))?
    );
    Ok(())
}

fn read_trace(trace_path: &Path) -> Result<Trace> {
    let raw = fs::read_to_string(trace_path)
        .with_context(|| format!("Failed to read {}", trace_path.display()))?;
    let mut trace = Trace::default();
    for (idx, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let event: TraceEvent = serde_json::from_str(line)
            .with_context(|| format!("Invalid trace JSON on line {}", idx + 1))?;
        trace.events.push(event);
    }
    Ok(trace)
}

fn read_tasks(task_path: &Path) -> Result<Vec<EvalTask>> {
    let raw = fs::read_to_string(task_path)
        .with_context(|| format!("Failed to read {}", task_path.display()))?;
    let mut tasks = Vec::new();
    for (idx, line) in raw.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        tasks.push(
            serde_json::from_str::<EvalTask>(line)
                .with_context(|| format!("Invalid eval task JSON on line {}", idx + 1))?,
        );
    }
    Ok(tasks)
}

fn baseline_files(repo_path: &Path, prompt: &str, limit: usize) -> Result<Vec<String>> {
    let terms = search_terms(prompt);
    let mut scored = Vec::<(String, i64)>::new();
    collect_files(repo_path, repo_path, &terms, &mut scored)?;
    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(limit);
    Ok(scored.into_iter().map(|(path, _)| path).collect())
}

fn baseline_symbols(repo_path: &Path, prompt: &str, limit: usize) -> Result<Vec<String>> {
    let terms = search_terms(prompt);
    let mut scored = Vec::<(String, i64)>::new();
    collect_symbols(repo_path, repo_path, &terms, &mut scored)?;
    scored.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    scored.truncate(limit);
    Ok(scored.into_iter().map(|(symbol, _)| symbol).collect())
}

fn collect_files(
    root: &Path,
    current: &Path,
    terms: &[String],
    scored: &mut Vec<(String, i64)>,
) -> Result<()> {
    let entries = match fs::read_dir(current) {
        Ok(entries) => entries,
        Err(_) => return Ok(()),
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if should_skip(&name) {
            continue;
        }

        if path.is_dir() {
            collect_files(root, &path, terms, scored)?;
        } else if path.is_file() {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let score = score_path(&rel, terms);
            if score > 0 {
                scored.push((rel, score));
            }
        }
    }
    Ok(())
}

fn collect_symbols(
    root: &Path,
    current: &Path,
    terms: &[String],
    scored: &mut Vec<(String, i64)>,
) -> Result<()> {
    let entries = match fs::read_dir(current) {
        Ok(entries) => entries,
        Err(_) => return Ok(()),
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = entry.file_name();
        let name = file_name.to_string_lossy();
        if should_skip(&name) {
            continue;
        }

        if path.is_dir() {
            collect_symbols(root, &path, terms, scored)?;
        } else if path.is_file() && looks_like_source(&path) {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let content = match fs::read_to_string(&path) {
                Ok(content) => content,
                Err(_) => continue,
            };
            for symbol in extract_symbol_names(&content) {
                let score = score_symbol(&rel, &symbol, terms);
                if score > 0 {
                    scored.push((symbol, score));
                }
            }
        }
    }
    Ok(())
}

fn should_skip(name: &str) -> bool {
    name.starts_with('.')
        || matches!(
            name,
            "node_modules" | "target" | "dist" | "build" | ".git" | "vendor"
        )
}

fn search_terms(prompt: &str) -> Vec<String> {
    let mut seen = BTreeSet::new();
    prompt
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-' && c != '.')
        .map(str::trim)
        .filter(|term| term.len() > 2)
        .map(|term| term.to_lowercase())
        .filter(|term| seen.insert(term.clone()))
        .collect()
}

fn score_path(path: &str, terms: &[String]) -> i64 {
    let path_lower = path.to_lowercase();
    terms
        .iter()
        .map(|term| {
            if path_lower == *term {
                100
            } else if path_lower.ends_with(term) {
                75
            } else if path_lower.contains(term) {
                50
            } else {
                0
            }
        })
        .sum()
}

fn score_symbol(path: &str, symbol: &str, terms: &[String]) -> i64 {
    let symbol_lower = symbol.to_lowercase();
    score_path(path, terms)
        + terms
            .iter()
            .map(|term| {
                if symbol_lower == *term {
                    120
                } else if symbol_lower.contains(term) {
                    90
                } else {
                    0
                }
            })
            .sum::<i64>()
}

fn score_items(predicted: &[String], expected: &[String]) -> (f64, f64) {
    if expected.is_empty() {
        return (1.0, if predicted.is_empty() { 1.0 } else { 0.0 });
    }

    let predicted_set = predicted.iter().collect::<BTreeSet<_>>();
    let expected_set = expected.iter().collect::<BTreeSet<_>>();
    let hits = expected_set
        .iter()
        .filter(|expected| predicted_set.contains(*expected))
        .count();
    let recall = hits as f64 / expected_set.len() as f64;
    let precision = if predicted_set.is_empty() {
        0.0
    } else {
        hits as f64 / predicted_set.len() as f64
    };
    (recall, precision)
}

fn trace_metrics(trace: &Trace) -> Result<TraceMetrics> {
    let mut metrics = TraceMetrics::default();
    let mut latest_notebook = None;
    let mut cited_ids = Vec::<String>::new();

    for event in &trace.events {
        match event {
            TraceEvent::LlmCall { response, .. } => {
                metrics.llm_calls += 1;
                if let Some(value) = response.usage.prompt_tokens_total {
                    metrics.prompt_tokens_total += value;
                }
                if let Some(value) = response.usage.decode_tokens {
                    metrics.decode_tokens += value;
                }
                if let Some(value) = response.usage.wall_ms {
                    metrics.wall_ms += value;
                }
                if let Ok(value) = serde_json::from_str::<serde_json::Value>(&response.raw_json) {
                    if let Some(ids) = value.get("cited_entries").and_then(|v| v.as_array()) {
                        cited_ids = ids
                            .iter()
                            .filter_map(|id| id.as_str().map(ToString::to_string))
                            .collect();
                    }
                }
            }
            TraceEvent::ToolCall { .. } => metrics.tool_calls += 1,
            TraceEvent::NotebookSnapshot { notebook, .. } => {
                latest_notebook = Some(notebook);
            }
            TraceEvent::RunFinished { stop_reason } => {
                metrics.stop_reason = Some(format!("{:?}", stop_reason));
            }
            TraceEvent::RunStarted { .. }
            | TraceEvent::PhaseStarted { .. }
            | TraceEvent::LeaseInvalidated { .. } => {}
        }
    }

    if cited_ids.is_empty() {
        metrics.citation_validity = 1.0;
        return Ok(metrics);
    }

    let Some(notebook) = latest_notebook else {
        metrics.invalid_citations = cited_ids.len();
        metrics.citation_validity = 0.0;
        return Ok(metrics);
    };

    let valid = cited_ids
        .iter()
        .filter(|id| {
            notebook
                .get(&agent_core::EntryId((*id).clone()))
                .map(|entry| entry.status == agent_core::EntryStatus::Active)
                .unwrap_or(false)
        })
        .count();
    metrics.invalid_citations = cited_ids.len().saturating_sub(valid);
    metrics.citation_validity = valid as f64 / cited_ids.len() as f64;
    Ok(metrics)
}

fn outcome_matches(expected: &str, actual_stop_reason: &str) -> bool {
    match expected {
        "answer" => actual_stop_reason == "Done",
        "not_found" => actual_stop_reason == "NotFound",
        "needs_user" => actual_stop_reason == "NeedsUser",
        other => other == actual_stop_reason,
    }
}

fn outcome_accuracy(results: &[BaselineResult]) -> Option<f64> {
    let mut correct = 0usize;
    let mut count = 0usize;
    for result in results {
        if let Some(value) = result.outcome_correct {
            count += 1;
            if value {
                correct += 1;
            }
        }
    }
    (count > 0).then_some(correct as f64 / count as f64)
}

fn stop_reason_distribution(results: &[BaselineResult]) -> serde_json::Value {
    let mut counts = std::collections::BTreeMap::<String, usize>::new();
    for result in results {
        if let Some(trace) = &result.trace {
            if let Some(stop_reason) = &trace.stop_reason {
                *counts.entry(stop_reason.clone()).or_default() += 1;
            }
        }
    }
    serde_json::to_value(counts).unwrap_or_else(|_| serde_json::json!({}))
}

fn looks_like_source(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|ext| ext.to_str()),
        Some("rs" | "ts" | "tsx" | "js" | "jsx" | "mjs" | "cjs" | "py" | "go")
    )
}

fn extract_symbol_names(content: &str) -> Vec<String> {
    content
        .lines()
        .filter_map(parse_symbol_name)
        .collect::<Vec<_>>()
}

fn parse_symbol_name(line: &str) -> Option<String> {
    let trimmed = line.trim_start();
    parse_after_keyword(trimmed, "pub async fn")
        .or_else(|| parse_after_keyword(trimmed, "pub fn"))
        .or_else(|| parse_after_keyword(trimmed, "async fn"))
        .or_else(|| parse_after_keyword(trimmed, "fn"))
        .or_else(|| parse_after_keyword(trimmed, "pub struct"))
        .or_else(|| parse_after_keyword(trimmed, "struct"))
        .or_else(|| parse_after_keyword(trimmed, "pub enum"))
        .or_else(|| parse_after_keyword(trimmed, "enum"))
        .or_else(|| parse_after_keyword(trimmed, "export function"))
        .or_else(|| parse_after_keyword(trimmed, "function"))
        .or_else(|| parse_after_keyword(trimmed, "export class"))
        .or_else(|| parse_after_keyword(trimmed, "class"))
        .or_else(|| parse_const_arrow(trimmed))
}

fn parse_after_keyword(line: &str, keyword: &str) -> Option<String> {
    let rest = line.strip_prefix(keyword)?.trim_start();
    let name = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
        .collect::<String>();
    (!name.is_empty()).then_some(name)
}

fn parse_const_arrow(line: &str) -> Option<String> {
    let rest = line
        .strip_prefix("export const")
        .or_else(|| line.strip_prefix("const"))?
        .trim_start();
    if !(rest.contains("=>") || rest.contains("function")) {
        return None;
    }
    let name = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
        .collect::<String>();
    (!name.is_empty()).then_some(name)
}

fn average(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in values {
        sum += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_core::{LlmRequest, LlmResponse, LlmUsage, Phase, PromptParts, StopReason};

    #[test]
    fn search_terms_are_deduplicated_and_filtered() {
        assert_eq!(
            search_terms("Explain src/main.rs and main main"),
            vec!["explain", "src", "main.rs", "and", "main"]
        );
    }

    #[test]
    fn path_scoring_prefers_exact_and_suffix_matches() {
        let terms = vec!["main.rs".to_string(), "lib".to_string()];
        assert!(score_path("src/main.rs", &terms) > score_path("src/other.rs", &terms));
        assert!(score_path("lib", &terms) > score_path("src/lib.rs", &terms));
    }

    #[test]
    fn file_score_handles_empty_expected_set() {
        let (recall, precision) = score_items(&[], &[]);
        assert_eq!(recall, 1.0);
        assert_eq!(precision, 1.0);
    }

    #[test]
    fn file_score_counts_exact_file_hits() {
        let predicted = vec!["src/main.rs".to_string(), "src/lib.rs".to_string()];
        let expected = vec!["src/main.rs".to_string()];
        let (recall, precision) = score_items(&predicted, &expected);
        assert_eq!(recall, 1.0);
        assert_eq!(precision, 0.5);
    }

    #[test]
    fn replay_validates_prompt_parts_against_prompt() {
        let path = std::env::temp_dir().join(format!(
            "loco-trace-{}.jsonl",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let parts = PromptParts {
            global_prefix: "global".to_string(),
            phase_prefix: "phase".to_string(),
            task: "task".to_string(),
            notebook: "notebook".to_string(),
            evidence: "evidence".to_string(),
            question: "question".to_string(),
        };
        let request = LlmRequest {
            call_id: "call-1".to_string(),
            phase: Phase::Intake,
            schema_name: "Test".to_string(),
            schema: serde_json::json!({ "type": "object" }),
            prompt_parts: Some(parts.clone()),
            prompt: parts.render(),
            temperature: 0.0,
        };
        let response = LlmResponse {
            call_id: "call-1".to_string(),
            raw_json: "{}".to_string(),
            usage: LlmUsage::default(),
        };
        let lines = vec![
            serde_json::to_string(&TraceEvent::RunStarted {
                run_id: "run".to_string(),
                question: "question".to_string(),
            })
            .unwrap(),
            serde_json::to_string(&TraceEvent::LlmCall { request, response }).unwrap(),
            serde_json::to_string(&TraceEvent::RunFinished {
                stop_reason: StopReason::Done,
            })
            .unwrap(),
        ]
        .join("\n");
        fs::write(&path, lines).unwrap();
        let result = replay(&path);
        let _ = fs::remove_file(&path);
        assert!(result.is_ok());
    }
}
