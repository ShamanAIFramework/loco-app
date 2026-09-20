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

## Verified

- `cargo test --workspace`
  - 10 tests passed.
- `cargo test --manifest-path src-tauri/Cargo.toml`
  - 5 Tauri-side tests passed.
- `cargo check --manifest-path src-tauri/Cargo.toml`
  - completed successfully.

## Intentional Temporary Limitations

- `CodeGraphStore` now provides symbol ranges and lightweight intra-file edges, but it does not yet provide robust cross-file call graphs, references, or graph-distance ranking.
- Evidence leases can now cover symbol ranges when symbol hits are available. Symbol parsing is conservative and regex-free, so complex language constructs may still fall back to whole-file slices.
- The Tauri adapter uses current file-path and symbol-name search, scored deterministically by simple matching.
- The final answer is streamed to the UI as one chunk after synthesis rather than token-streamed.
- Token counting is still approximate in `agent-core`; the host does not yet query llama-server tokenizer.
- `agent-cli replay` can now regenerate prompt bytes from recorded prompt parts. It still does not replay from notebook snapshots alone.
- Eval now scores files and symbols and can include trace-derived citation validity, outcome correctness, stop reasons, calls, tokens, queue wait, and wall time when a task includes `trace_path`. It is still a baseline/evaluation harness, not a full task runner.
- Lease verification is implemented against the current filesystem range hash. Whole-file fallback leases may be weaker for very large truncated files, so UI verification should focus first on symbol-range cases.
- Runtime metadata currently records the OpenAI-compatible local endpoint and best-effort model id. The sidecar server version remains `null` until we add a stable llama-server version source.

## Next Milestones

1. Add tokenizer-backed token counting instead of approximate counting.
2. Capture a stable llama.cpp/server version if the sidecar exposes one.
3. Improve parsers and graph extraction beyond the conservative line scanner.
4. Add UI smoke verification against a small workspace:
   - ask for a known symbol
   - edit that symbol while the app is open
   - ask again
   - confirm stale evidence is rejected
   - run `agent-cli replay` on the trace
5. Consider removing or renaming the now-empty `src-tauri/src/agentv2.rs` compatibility module once all references have settled.

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
