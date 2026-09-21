# Agent Rebuild Progress

This file tracks the reconstruction of Loco's agent loop toward the local-first deterministic architecture in `/Users/ruckus/Downloads/local-coding-agent-build-prompt.md`.

## Current Objective

Rebuild the orchestrator and `agentv2` behavior so deterministic Rust code controls workflow, evidence acquisition, budgets, notebook state, leases, tracing, and stop conditions. The model should act only as a bounded typed-output reasoning coprocessor.

## Completed

- Created `crates/agent-core` as a headless Rust library with no Tauri dependency.
- Added core architecture types:
  - task request and budgets
  - phases and stop reasons
  - typed LLM outputs for intake, extraction, and synthesis
  - file hits and evidence slices
- Added notebook implementation:
  - typed entries
  - provenance
  - evidence leases
  - append-order iteration
  - stale cascade by lease
  - deterministic checkpoint compaction
- Added deterministic prompt assembly:
  - global prefix
  - phase prefix
  - task block
  - notebook block
  - ephemeral evidence block
  - JSON schemas for each model call type
- Added trait surface for host adapters:
  - `LlmClient`
  - `CodeSearch`
  - `FileReader`
  - `TokenCounter`
  - `ProgressSink`
- Added first deterministic pipeline:
  - `INTAKE`
  - host-driven code search
  - host-driven file reads
  - `EXTRACT`
  - notebook inference insertion only when lease ids are valid
  - `SYNTHESIZE`
  - trace event collection
- Replaced Tauri `prompt_agent_v2` internals with adapters into `agent-core`.
- Kept the existing frontend IPC shape: Solid still invokes `prompt_agent_v2` and receives `Status` and `Chunk` events.
- Added trace writing to `~/.loco/traces/<run_id>.jsonl`.
- Added TOML budget loading from `~/.loco/agent.toml`, falling back to `Budgets::default()`.
- Added file checksums to initial indexing and watcher-driven updates.
- Added prompt-stability unit coverage for stable prefix preservation.
- Added root Cargo workspace for headless crates, explicitly excluding `src-tauri` so the existing Tauri server keeps its current Cargo setup.
- Added `crates/agent-cli` with:
  - `replay <trace_path>` for trace JSONL validation and summary
  - `baseline <task_path>` for deterministic no-LLM path search
  - `eval <task_path>` for baseline scoring against expected files
- Added unit coverage for CLI term extraction and file scoring.
- Extended CodeGraph schema with migration support for:
  - file `index_version`
  - symbol `start_line`
  - symbol `end_line`
  - symbol `content_hash`
- Added conservative symbol extraction in the indexer for common Rust and TypeScript or JavaScript declarations.
- Added CodeGraph symbol search.
- Extended `agent-core` with `SymbolHit`, `CodeSearch::search_symbols`, and `FileReader::read_range`.
- Updated the deterministic pipeline to prefer symbol-range evidence before falling back to whole-file reads.
- Updated the Tauri adapter to provide symbol search and range-based evidence leases.
- Added structured prompt parts to every traced `LlmRequest`.
- Upgraded `agent-cli replay` so it re-renders prompt parts and requires byte-for-byte equality with the recorded prompt.
- Added CLI replay coverage with a synthetic trace.
- Expanded `agent-cli eval` beyond file-only scoring:
  - symbol baseline predictions
  - symbol recall and precision
  - optional trace-derived citation validity
  - optional expected outcome accuracy for `answer`, `not_found`, and `needs_user`
  - optional trace-derived stop reason distribution
  - optional trace-derived LLM/tool call, token, and wall-time totals
- Added notebook snapshots to traces before LLM calls.
- Added `LeaseStatus` and `FileReader::verify_lease`.
- Added lease invalidation trace events.
- Updated the deterministic pipeline to verify cited leases before accepting model inferences.
- Linked model inferences to the fact entry that supplied their lease, enabling stale cascade.
- Added a core test proving expired leases invalidate notebook state and reject model inferences.
- Added lightweight CodeGraph symbol edges:
  - `symbol_edges` table
  - edge upsert/list APIs
  - symbol search can surface symbols connected to a queried target
  - indexer creates simple intra-file reference or call edges among parsed symbols
- Added a shared single-slot LLM queue in Tauri app state so concurrent UI runs cannot overlap llama-server JSON calls.
- Added `queue_wait_ms` to traced LLM usage.
- Added runtime metadata trace events with the Tauri sidecar endpoint, best-effort `/v1/models` model id, and queue policy.
- Added deterministic intake fallback for non-empty actionable code/UI prompts when the model incorrectly labels them `Unclear`.
- Updated the left-panel workspace path text to wrap instead of truncating.
- Improved location-answer quality:
  - evidence snippets now include line numbers in prompts
  - notebook context now includes file provenance and line ranges
  - broad file reads use a larger per-slice budget so nearby render blocks are less likely to be truncated
  - whole-file evidence for location-style questions is deterministically narrowed to the best matching section before extraction
  - synthesis/extraction prompts ask for exact file and line ranges on UI/code location questions
  - final citation ordering prefers the narrowest active file range
  - rendered sources now include file and line provenance, not only notebook ids
- Added regression coverage for line-numbered evidence, wider UI-location slice budgets, deterministic GitPanel-style JSX section focusing, non-location whole-file preservation, and citation specificity ordering.
- Improved chat progress visibility:
  - agent-core emits more fine-grained status updates for model calls, searches, reads, focusing, lease checks, and synthesis
  - progress now reports task classification, search intent, path/symbol hit counts, accepted/skipped evidence ranges, extraction pass counts, and missing-evidence follow-up queries
  - status labels avoid implementation details such as local-model calls and omit decorative ellipses
  - chat UI shows an elapsed timer for the current progress step
  - chat UI shows only the active step, not a completed-subtask trail
- Improved TypeScript/TSX symbol ranges:
  - `export default function` declarations are now indexed as function symbols
  - symbol end ranges are brace-aware when possible, so component symbols can include nested helpers and JSX render blocks
  - added TSX component regression coverage for a GitPanel-style render section
- Added token-aware evidence budgeting:
  - `AgentRuntime` now receives a `TokenCounter`
  - Tauri asks llama-server `/tokenize` for evidence token counts when available
  - token counting falls back to the existing approximate counter if the tokenizer endpoint is unavailable
  - evidence accumulation now respects `max_evidence_tokens`, while always allowing the first evidence slice
  - older budget configs without `max_evidence_tokens` still deserialize with a default
- Improved source display:
  - Tauri now sends structured `Sources` events instead of appending raw source markdown with notebook and lease ids
  - source filtering prefers file ranges mentioned by the final answer and deduplicates file/range pairs
  - chat messages can render compact source chips with path, line range, label, and click-to-copy behavior
- Audited and tightened prompts for local coding:
  - global prompt now emphasizes local code truth, no file edits, no tool claims, and repo-map-as-navigation-only
  - task context now includes a compact recent-conversation block, skips the duplicated current request, and caps prior turns for local-model context windows
  - intake prompt biases toward proceeding on actionable local-code questions and generating path/symbol/UI-label queries
  - extraction prompt requires exact lease ids, file/range/block purpose, and UI render block identification
  - synthesis prompt forbids markdown source sections and internal ids because sources are rendered structurally by the UI
  - schemas now include local-code-oriented field descriptions
  - added prompt contract tests for these rules
- Applied local-model harness guidance:
  - added `AGENTS.md` with project rules for separated phases, stable context, source UI, and future whole-file edit mode
  - added local Codex skill `local-agent-prompting` with a reference guide for local/open-weight coding-agent prompt design
  - prompt text now explicitly says each call is a separated harness phase, not a ReAct loop
- Started plan-only edit loop:
  - added `TaskType::Edit` and `Phase::Plan` in `agent-core`
  - added structured `EditPlan`, `PlannedFile`, and `PlanStep` types
  - intake can classify change requests as `Edit`, then the existing evidence loop feeds a plan-only JSON phase
  - edit requests now return a markdown implementation plan and explicitly state that no files were changed
  - added regression coverage that edit requests enter `Plan` and do not enter synthesis
- Added local content-search retrieval:
  - `CodeSearch` now includes `search_content` alongside path and symbol search
  - Tauri content search scans indexed workspace files locally and returns ranked line-window matches
  - content search supports case-insensitive regex/phrase matching and merges overlapping snippet windows
  - the evidence loop reads content hits as leased ranges before falling back to whole-file reads
  - progress now reports text-match search and matched line-window inspection
  - added regression coverage that content-search hits become notebook evidence
- Improved stable repo-map context:
  - repo map entries now include compact symbol outlines and line ranges instead of only file paths
  - symbol-bearing files are prioritized before plain file path entries inside the repo-map budget
- Added Tree-sitter-backed structural indexing:
  - Rust, TypeScript, JavaScript, JSX, and TSX files now use Tree-sitter parsing before the conservative line scanner fallback
  - indexed symbols include functions, classes, methods, structs, enums, traits, impls, and function-valued callbacks
  - repo maps, symbol search, and leased range evidence benefit from parser-derived definition ranges when the file parses cleanly
  - added regression coverage for Rust impl/function extraction and TSX component/callback extraction
- Improved lightweight reference discovery:
  - the indexer now records deterministic symbol mention and call edges against all indexed project symbols, including cross-file references discovered from parsed symbol bodies
  - symbol search can surface definitions and local callers/referencers for the queried symbol name
  - added regression coverage for a cross-file TypeScript helper call
- Improved local retrieval ranking and acceleration:
  - symbol search now boosts symbols that call or reference the queried symbol through indexed graph edges
  - repo-map context now prefers files with stronger graph connectivity before isolated symbol outlines
  - content search can use `rg` to preselect candidate files when available, then falls back to the deterministic in-process scanner
  - added regression coverage for graph-edge scoring and `rg` fallback behavior
- Added bounded multi-hop graph-distance retrieval:
  - symbol search now includes two-hop candidates, so callers of callers can surface for a queried target symbol
  - symbol scoring applies a smaller boost to two-hop graph matches than direct graph matches
  - added regression coverage for a `main -> helper -> println` symbol chain
- Reduced graph-reference noise:
  - symbol edge extraction now requires identifier-boundary matches instead of raw substring matches
  - call edges are detected from boundary-matched symbol names followed by an opening parenthesis
  - added regression coverage so `helperText` and `my_helper` do not create false `helper` references

## Verified

- `cargo test --workspace`
  - 24 tests passed.
- `cargo test --manifest-path src-tauri/Cargo.toml`
  - 14 Tauri-side tests passed.
- `cargo test --manifest-path src-tauri/Cargo.toml indexer::tests::tree_sitter -- --nocapture`
  - 2 Tree-sitter indexer tests passed.
- `cargo check --manifest-path src-tauri/Cargo.toml`
  - completed successfully.
- `npm run build`
  - completed successfully with Vite's existing large-chunk warning.

## Intentional Temporary Limitations

- `CodeGraphStore` now provides symbol ranges, lightweight mention/call edges, and bounded graph-distance lookup, but it does not yet provide robust LSP/type-aware cross-file call graphs or references.
- Evidence leases can now cover symbol and content-search ranges when hits are available. Symbol parsing is conservative, so complex language constructs may still fall back to whole-file slices.
- The Tauri adapter uses deterministic path, Tree-sitter-backed symbol, regex/phrase content search, `rg` candidate acceleration, lightweight symbol mention/call edges, graph-edge symbol boosts, graph-aware repo-map ordering, and bounded two-hop graph-distance retrieval. It does not yet provide semantic/vector search, LSP-backed references, or robust type-aware cross-file references.
- The final answer is streamed to the UI as one chunk after synthesis rather than token-streamed.
- Plan mode is non-writing only. It does not yet produce edit artifacts, preview diffs, accept approvals, apply changes, run validation commands, or repair failed validation.
- Token counting is tokenizer-backed when llama-server exposes `/tokenize`, with an approximate fallback for compatibility.
- `agent-cli replay` can now regenerate prompt bytes from recorded prompt parts. It still does not replay from notebook snapshots alone.
- Eval now scores files and symbols and can include trace-derived citation validity, outcome correctness, stop reasons, calls, tokens, queue wait, and wall time when a task includes `trace_path`. It is still a baseline/evaluation harness, not a full task runner.
- Lease verification is implemented against the current filesystem range hash. Whole-file fallback leases may be weaker for very large truncated files, so UI verification should focus first on symbol-range cases.
- Runtime metadata currently records the OpenAI-compatible local endpoint and best-effort model id. The sidecar server version remains `null` until we add a stable llama-server version source.

## Next Milestones

1. Add draft edit artifacts without writes:
   - whole-file or block-level replacements
   - prior hash/range validation
   - host-computed diff preview
2. Add plan approval/refinement UI controls before artifact generation.
3. Capture a stable llama.cpp/server version if the sidecar exposes one.
4. Improve graph extraction beyond local symbol definitions:
   - LSP references/definitions where available
   - Tree-sitter reference extraction beyond local definitions
   - richer graph-distance ranking using typed references when available
5. Add semantic/vector retrieval as a supplement to lexical and structural search.
6. Add UI smoke verification against a small workspace:
   - ask for a known symbol
   - edit that symbol while the app is open
   - ask again
   - confirm stale evidence is rejected
   - run `agent-cli replay` on the trace
7. Consider removing or renaming the now-empty `src-tauri/src/agentv2.rs` compatibility module once all references have settled.

## Resume Notes

Start with:

```bash
cargo test --manifest-path crates/agent-core/Cargo.toml
cargo test --workspace
cargo test --manifest-path src-tauri/Cargo.toml
cargo check --manifest-path src-tauri/Cargo.toml
```

Then inspect any compile failures in:

- `crates/agent-core/src/pipeline.rs`
- `crates/agent-cli/src/main.rs`
- `src-tauri/src/orchestrator.rs`
- `src-tauri/Cargo.toml`
