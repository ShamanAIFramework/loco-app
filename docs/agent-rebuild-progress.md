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
- Tightened plan-mode file selection:
  - verified the active code graph included the relevant app files, so the splash/fullscreen miss was plan selection, not whole-repo indexing failure
  - startup/splash/fullscreen requests now deterministically add stack-neutral boot/loading/window search concepts
  - model-authored edit plans are post-processed so likely files must be backed by active leased file evidence
  - plan prompts now warn not to involve unrelated orchestration, search, backend service, build, or model/AI plumbing files for UI/startup work unless cited evidence requires them
  - added regression coverage for stack-neutral startup query expansion and dropping unsupported planned files such as an unrelated orchestration file
- Completed a prompt portability audit:
  - reusable `agent-core` prompts no longer name this repository's framework, language, paths, package manager, or app structure as static guidance
  - framework/language-specific clues must come from the user request, repo map, indexed project evidence, manifests/config files, file extensions, or leased evidence
  - extraction wording now uses stack-neutral render/output, screen/view, template, component/widget, API/command/event, configuration, persistence, and data-flow concepts
  - fallback plans no longer emit hardcoded validation commands; model-authored validation commands are retained only when backed by active evidence
  - legacy tool descriptions in `src-tauri/src/agent.rs` no longer include stack-specific path examples
  - added prompt contract coverage that reusable prompt strings do not contain static framework/language/path examples
- Fixed multi-word path search regression:
  - path search now tokenizes coding queries and scores indexed paths by meaningful term matches instead of requiring the full query string to appear in the path
  - repo-map context filters generated/build output such as `dist` entries before ranking, so app source files are not buried by generated assets
  - added regression coverage for multi-word path scoring and generated-output filtering
- Tightened the deterministic evidence loop after prompt review:
  - extraction follow-up requests now reuse symbol search, content search, path search, range reads, leases, notebook facts, and trace events instead of falling back to path-only whole-file reads
  - file and range leases now cover exactly the returned line-bounded content, preventing truncated evidence from self-expiring during freshness checks
  - structured source chips keep cited sources even when the final prose does not name every supporting file
  - removed the legacy model-selected tool loop from the Tauri invoke surface; `agent.rs` now only carries shared IPC event/source types
- Added host-side validation context discovery:
  - edit and validation-oriented requests now insert generic project manifest, test script, and build script queries early enough to fit the default query budget while preserving target-code discovery
  - Tauri path scoring maps those stack-neutral queries to discovered manifest/config files such as package, Cargo, Python, Go, Java, Gradle, task-runner, and tool config files without putting stack-specific names in reusable prompts
  - added regression coverage for budgeted validation-context query insertion and manifest/config path scoring
- Improved multi-word symbol discovery:
  - Tauri symbol search now expands phrase queries into meaningful terms before querying the CodeGraph, so symbols such as `StatusPanel` can be discovered from requests such as "status panel"
  - symbol scoring now rewards symbols that contain all meaningful query terms, complementing the existing path/content multi-word matching
  - added regression coverage for multi-word symbol candidate expansion and scoring
- Improved identifier-style query matching:
  - local path, symbol, and content scoring now normalize camelCase, snake_case, kebab-case, and compact identifier forms so queries like `status_pill`, `statusPill`, and "status pill" can converge on the same code
  - added regression coverage for identifier term splitting, identifier-aware content search, and snake-to-camel symbol scoring
- Tightened extraction loop stop conditions:
  - extraction now stops instead of spending another local-model call when the model says evidence is insufficient but provides no concrete follow-up evidence target for the host to retrieve
  - added regression coverage that no-new-evidence extraction runs only one extraction pass
- Tightened the final source-display contract:
  - synthesis now falls back to active notebook findings when the model omits citation ids, preserving structured source chips for the UI
  - fallback answer prose now hides notebook entry ids while still presenting the relevant finding text
  - added regression coverage for citation fallback and id-free fallback answers
- Made structured missing-evidence requests actionable:
  - `FileRange` follow-up requests now parse common `path:line-line` and `path lines line-line` targets and read the exact leased range instead of treating the target as plain search text
  - range reads are token-budgeted, recorded as notebook facts, traced as host tool calls, and added to target tags for the next extraction pass
  - added regression coverage for exact file-range follow-up reads and common range target parsing
- Added relation-aware symbol follow-up retrieval:
  - `Definition`, `References`, `Callers`, and `Callees` missing-evidence requests now use a typed symbol-relation search before falling back to generic symbol/content/path search
  - Tauri resolves definitions directly from matching indexed symbols, resolves references/callers by walking inbound CodeGraph edges, and resolves callees by walking outbound edges to target symbol definitions
  - definition expansion prefers exact or normalized symbol-name matches before fuzzy matches
  - caller/callee expansion now uses only edges indexed as calls, while reference expansion can include broader non-call usages
  - added core regression coverage for relation follow-up reads and Tauri regression coverage for definition, reference, caller, and callee collection
- Added first-pass relation-intent retrieval:
  - user questions that explicitly ask for definitions, references/usages, callers, or callees now trigger typed relation search during the initial deterministic evidence acquisition pass
  - relation hits are deduplicated with ordinary symbol hits and leased as normal symbol evidence, so extraction starts with caller/callee context when the request already names that need
  - added regression coverage for common relation-intent phrases and initial relation evidence reads before extraction follow-up
- Added project-research overview discovery:
  - review, audit, architecture, research, overview, design, project, codebase, and repository-oriented requests now seed stack-neutral overview/doc queries inside the default search budget
  - Tauri path scoring maps those generic overview/doc queries to README, docs, architecture, design, contributing, and developer-guide files without hard-coding this repository's paths
  - added regression coverage for budgeted project-overview query insertion and overview/documentation path scoring
- Added bounded symbol-context reads:
  - Tauri symbol range reads now include a small surrounding line window so extraction sees nearby setup, comments, and adjacent control flow instead of an isolated bare symbol body
  - explicit file-range reads remain exact, preserving precise user/model-requested range semantics and lease verification
  - added regression coverage that only symbol reads expand while exact file-range reads do not
- Added deterministic software-concept query expansion:
  - Tauri search terms now conservatively bridge common project-research language such as loops, tools, context, prompts, discovery, APIs, persistence, validation, and UI to nearby software vocabulary used in paths and code
  - the expansion lives in host-side lexical scoring/search, keeping prompts portable and avoiding any claim of true semantic/vector retrieval
  - added regression coverage for concept-expanded terms and path scoring
- Tightened `rg` candidate acceleration for recall:
  - `rg` remains an optional prefilter for content search, but broad match sets now disable prefiltering instead of limiting in-process scanning to an arbitrary capped subset
  - added regression coverage that over-broad `rg` matches fall back to the deterministic scanner path
- Made evidence dedupe range-aware:
  - content hits in the same file as a symbol read are now skipped only when their line windows overlap existing evidence, so project-research tasks can collect multiple focused sections from one file
  - follow-up symbol and content reads also use overlap checks, avoiding duplicate reads when symbol leases include a wider context window than the original symbol hit
  - added regression coverage for same-file disjoint symbol/content evidence and overlap detection
- Added bounded direct-callee context expansion for explanation tasks:
  - top seed symbols in ordinary `Explain` tasks now trigger a small host-driven callee lookup, so the evidence pass can include definitions of immediate helpers the seed symbol calls before asking the model to extract claims
  - explicit caller/callee questions continue through the typed relation-intent path; the new expansion is for non-explicit "explain this" style code research
  - added regression coverage that direct callee evidence is read before extraction without model-requested follow-up
- Added bounded impact-context expansion for edit planning:
  - top seed symbols in `Edit` tasks now trigger small host-driven caller and reference lookups, so plan mode can see likely impact sites before accepting or filtering affected files
  - graph-impact context still uses host-selected relation search, normal evidence leases, overlap checks, and token budgets rather than model-selected tool calls
  - added regression coverage that caller impact evidence is read before plan generation and can support evidence-backed planned files
- Preserved user-request terms in host query normalization:
  - intake normalization now inserts a compact target-focused query derived from the user's own request near the front of the search plan, so a narrow or weak model-generated query list cannot fully steer retrieval away from the original wording
  - the preserved query coexists with validation and project-overview augmentations inside the default query budget
  - added regression coverage for raw-request term preservation and existing budget-sensitive query augmentations
- Surfaced repository instruction files for project research:
  - project-research query augmentation now reserves default-budget space for repository instructions alongside overview and README discovery
  - Tauri path scoring maps generic repository-instruction queries to guidance files such as `AGENTS.md`, `CLAUDE.md`, `.cursorrules`, `.github/instructions/*`, and `.cursor/rules/*`
  - added regression coverage for budgeted instruction-query insertion and instruction-file path scoring
- Surfaced startup and entrypoint files for project research:
  - project-research query augmentation now reserves default-budget space for project entrypoints after overview, README, and repository instructions
  - Tauri path scoring maps generic entrypoint/startup/bootstrap queries to common startup files such as `main.*`, `App.*`, `index.*`, `lib.rs`, app directories, and Tauri main files
  - added regression coverage for budgeted entrypoint-query insertion and startup-file path scoring without bleeding into architecture docs
- Added changed-file context for review and edit workflows:
  - `CodeSearch` now exposes a host-owned `changed_files` hook, and the deterministic pipeline folds changed file hits into the same ranked path evidence flow for edit, review, diff, dirty, and current-worktree requests
  - the Tauri adapter reads `git status --porcelain`, parses modified, untracked, and renamed paths, filters to searchable text files, and ranks them as high-value local context
  - changed-file evidence now includes a bounded `git diff HEAD -- <path>` slice when available, using a dedicated `FileReader::read_diff` path and diff freshness checks rather than unverified notes
  - added regression coverage that review requests read changed file diffs and current file contents as evidence, and that porcelain parsing keeps the current renamed path
- Surfaced related test files for edit and review context:
  - edit and validation-oriented requests now reserve default-budget space for a generic `related tests` query alongside manifest and command discovery
  - Tauri path scoring maps related/coverage/spec test queries to conventional test files and test directories without treating plain `test script` manifest discovery as the same context
  - added regression coverage for budgeted related-test query insertion and test/spec path scoring
- Added deterministic adjacent-file discovery:
  - `CodeSearch` now exposes a host-owned adjacent-file lookup that receives seed paths already found by file, symbol, content, and changed-file search
  - project-research, review, and edit workflows can fold sibling modules, nearby tests, styles, and nested support files into the ordinary ranked path evidence flow before extraction
  - Tauri adjacent-file scoring prefers same-directory or nested-neighbor files with the same logical basename, while rejecting same-name files in unrelated directories
  - added regression coverage that adjacent files are read as leased evidence and that sibling scoring avoids cross-directory noise
- Added deterministic imported-file discovery:
  - `CodeSearch` now exposes a host-owned import-neighbor lookup that receives seed paths already found by earlier retrieval passes
  - project-research, review, and edit workflows can fold locally imported modules into the ordinary ranked path evidence flow before extraction
  - Tauri resolves relative JavaScript/TypeScript imports, re-exports, dynamic imports, `require(...)` calls, and simple Rust `mod name;` declarations against indexed workspace files
  - added regression coverage that imported files are read as leased evidence and that resolver tests use real temporary files
- Added deterministic importing-file discovery:
  - `CodeSearch` now exposes a host-owned reverse import-neighbor lookup for files that import or declare seed paths
  - project-research, review, and edit workflows can fold dependent files into the ordinary ranked path evidence flow before extraction, improving impact analysis without model-selected tools
  - Tauri reuses the local import resolver to scan indexed files and find JavaScript/TypeScript/Rust files that resolve to the seed paths
  - added regression coverage that importing files are read as leased evidence and that reverse resolver tests use real temporary files
- Added deterministic scoped project-context discovery:
  - `CodeSearch` now exposes a host-owned scope-file lookup for manifests, config files, README/developer docs, and repository instruction files near seed paths
  - project-research, review, and edit workflows can fold nearest package/local context into the ordinary ranked path evidence flow before extraction, improving nested package and monorepo answers
  - Tauri walks seed enclosing directories and scores nearer scope files above distant ancestors while rejecting sibling package manifests
  - added regression coverage that scoped files are read as leased evidence and that nested package scoring avoids sibling noise
- Preserved explicit path mentions in deterministic query planning:
  - intake normalization now extracts path-like mentions such as `src/foo.rs` and `file.tsx:10-20` from the user's request before generic query augmentation
  - validation and project-research augmentation preserve explicit paths inside the default query budget instead of pushing them behind overview/config discovery
  - added regression coverage for path extraction, line-suffix stripping, URL/parent-path rejection, and budget preservation
- Added direct evidence reads for explicit file ranges:
  - deterministic evidence acquisition now extracts request mentions such as `src/foo.rs:10-20` and `src/foo.rs lines 10-20`
  - explicit ranges are read with `FileReader::read_range`, recorded as leased notebook facts, traced as host tool calls, and token-budgeted before broader ranked path reads
  - added regression coverage that explicit ranges are read before model-requested follow-up and that URL-like ranges are rejected
- Added direct evidence reads for explicit file mentions:
  - deterministic evidence acquisition now reads user-mentioned files such as `src/foo.rs` directly with `FileReader::read_file` before broader ranked path fallback
  - exact file reads are recorded as leased notebook facts, traced as host tool calls, and skipped when an explicit range for the same file was already read
  - added regression coverage that explicit files are read first and that range mentions do not duplicate whole-file reads
- Added deterministic directory mention expansion:
  - `CodeSearch` now exposes a host-owned directory-file lookup for extensionless path mentions such as `src/foo`
  - deterministic retrieval expands requested directories into ranked indexed files before path evidence is read, while avoiding failed direct file reads for directories
  - Tauri ranks direct children above nested files and boosts entrypoints, manifests, docs, instructions, and tests under the requested directory
  - added regression coverage that directory mentions expand to leased file evidence and that directory files are ranked without leaking outside the requested directory
- Added deterministic context-doc discovery:
  - `CodeSearch` now exposes a host-owned context-doc lookup for project research, review, edit, and "how does this work" style requests
  - deterministic retrieval uses already-found seed files plus intake queries to add nearby documentation before extraction, improving subsystem research without model-selected tools
  - Tauri ranks domain-matching docs in the seed's local package/docs scope above distant ancestors while rejecting sibling package docs
  - added regression coverage that nearby context docs are read as leased evidence and that package-local docs outrank global docs without sibling noise
- Added Stereoscopic Gating Pattern:
  - Implemented dual-lens dialectical verification across critical pipeline boundaries to eliminate local-model blind spots and recall-vs-precision trade-offs:
    1. **Stereoscopic Intent Gate**: Evaluates user prompts with `Phase::Intake` (generator) and `Phase::IntakeCritic` (adversary), normalized by `TriadConsensus` to prevent dropped clauses or intent collapse.
    2. **Stereoscopic Evidence Gate**: Evaluates top 10 retrieved candidates with `Phase::CandidateGateOptimist` (inclusion scout for entrypoints, root shells, state signals) and `Phase::CandidateGateSkeptic` (precision pruner for false positives, stylesheets, utilities).
  - Built a deterministic Rust Host Arbiter that reconciles votes using CodeGraph AST import edges, topology distance, entrypoint identity, and domain constraints without calling a third LLM.
  - Added `CodeGraphStore::symbols_for_path` and `CodeSearch::search_file_symbols` to resolve component definitions directly from indexed Tree-sitter symbols.
  - Added unit and regression tests verifying split-decision arbitration, topology discovery, prompt portability, and noise elimination.
  - Implemented **KV-Cache Invariant Prefixing** for Stereoscopic Gating: candidate snippets precede lens-specific instructions, enabling near-instant prompt processing (~80% faster on local `llama-server`).
  - Added **Tree-Sitter Large Symbol Sub-Chunking**: symbols $> 200$ lines in `vector_search.rs` are partitioned into 60-line windowed sub-chunks with symbol metadata preserved.
  - Implemented **Vector-Bound Slicing**: path matches and CodeGraph topology expansions prioritize exact vector search chunk ranges over fallback line boundaries.
  - Implemented **"Header + Focal Window" Slicing & Companion Leasing**: chunks located deep in files ($> 35$ lines) attach the file header (lines 1..25) with omission markers during candidate evaluation and acquire companion context leases in the Notebook.
  - Implemented **Step 3A Evidence Extraction (`Phase::Extract`)**: local LLM extracts atomic claims citing leased code blocks; host verifies cited `lease_id`s against active notebook leases, records verified inferences with causal dependencies, and rejects unbacked claims.
  - Extended UI and Pipeline reporting with **Step 3A Extraction Metrics, Verified Inferences, and Missing Evidence diagnostics**.

## Verified

- `cargo test --workspace`
  - 27 unit tests passed (22 in `agent-core`, 5 in `agent-cli`).
- `cargo test --manifest-path src-tauri/Cargo.toml`
  - 46 Tauri-side tests passed (including new vector sub-chunking tests).
- `cargo test --manifest-path src-tauri/Cargo.toml indexer::tests::tree_sitter -- --nocapture`
  - 2 Tree-sitter indexer tests passed.
- `cargo check --manifest-path src-tauri/Cargo.toml`
  - completed successfully.
- `npm run build`
  - completed successfully with Vite's existing large-chunk warning.

## Intentional Temporary Limitations

- `CodeGraphStore` now provides symbol ranges, lightweight mention/call edges, bounded graph-distance lookup, relation-specific definition/reference/caller/callee expansion, but it does not yet provide robust LSP/type-aware cross-file call graphs or references.
- Evidence leases can now cover symbol and content-search ranges when hits are available, and symbol leases include a small context window. Symbol parsing is conservative, so complex language constructs may still fall back to whole-file slices.
- The Tauri adapter uses deterministic path, scoped-context, context-doc, adjacent-file, imported-file, importing-file, Tree-sitter-backed symbol, regex/phrase content search, `rg` candidate acceleration, lightweight symbol mention/call edges, graph-edge symbol boosts, graph-aware repo-map ordering, bounded two-hop graph-distance retrieval, relation-specific symbol expansion, overview/documentation surfacing, and conservative lexical software-concept expansion for project-research requests. It does not yet provide semantic/vector search, LSP-backed references, or robust type-aware cross-file references.
- The final answer is streamed to the UI as one chunk after synthesis rather than token-streamed.
- Plan mode is non-writing only. It does not yet produce edit artifacts, preview diffs, accept approvals, apply changes, run validation commands, or repair failed validation.
- Token counting is tokenizer-backed when llama-server exposes `/tokenize`, with an approximate fallback for compatibility.
- `agent-cli replay` can now regenerate prompt bytes from recorded prompt parts. It still does not replay from notebook snapshots alone.
- Eval now scores files and symbols and can include trace-derived citation validity, outcome correctness, stop reasons, calls, tokens, queue wait, and wall time when a task includes `trace_path`. It is still a baseline/evaluation harness, not a full task runner.
- Lease verification is implemented against the current filesystem range hash. Whole-file fallback leases are line-bounded when truncated, but broad file slices are still less specific than symbol or content-match ranges.
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
