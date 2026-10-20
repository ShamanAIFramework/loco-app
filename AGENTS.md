# Loco Agent Guidance

This project is a local-first coding agent built around a deterministic Rust harness and a Solid/Tauri UI. Favor harness engineering over clever prompt-only behavior.

## Architecture Rules

- Keep model calls separated by phase. The current loop is intake, deterministic search/read, extraction, optional plan-only edit planning, and synthesis. Do not collapse this into a ReAct loop where the model decides tools and actions in one prompt.
- The Rust host owns tools, budgets, evidence acquisition, leases, tracing, source display, and stop conditions. The model is a bounded reasoning coprocessor.
- Preserve stable prompt prefixes where possible. Put durable system/phase/context structure before volatile request, evidence, and notebook content.
- Keep context small and specific. Prefer indexed file/symbol search, narrow evidence ranges, and structured notebook entries over dumping broad tool lists or whole repo text.
- Use source chips/events for UI source display. Do not put notebook ids, lease ids, hashes, or raw source bullet lists into the assistant answer text.

## Stereoscopic Gating Pattern

To overcome the recall-vs-precision trade-offs and context limits of local models (7B–14B), critical judgment boundaries use **Stereoscopic Gating** (an asymmetric dialectic reconciled by the deterministic Rust host). Do not collapse critical filtering or classification boundaries into a single monocular prompt.

### Pattern Structure
1. **Lens 1 (Inclusion / Recall - Thesis)**: A prompt explicitly incentivized to maximize recall, detect systemic/architectural value, entrypoints, root shells, state signals, and indirect callers.
2. **Lens 2 (Precision / Pruning - Antithesis)**: An independent adversarial prompt incentivized to maximize precision, challenge false positives, eliminate coincidental keyword matches, generic utilities, and noise.
3. **Node 3 (Deterministic Host Arbiter - Synthesis)**: The Rust engine reconciles votes using ground truth (CodeGraph AST import edges, topology distance, file domain alignment, and token budgets). The host breaks split decisions without calling a third LLM.

### Active Stereoscopic Gates in the Pipeline
- **Gate 1: Stereoscopic Intent Gate (Input Boundary)**:
  - *Lens 1*: `Phase::Intake` drafts intent decomposition and search queries.
  - *Lens 2*: `Phase::IntakeCritic` actively challenges dropped prompt clauses, intent collapse, and query contamination.
  - *Host Arbiter*: `TriadConsensus` normalizes the decomposition, restores general questions, and sanitizes queries before execution.
- **Gate 2: Stereoscopic Evidence Gate (Retrieval Boundary)**:
  - *Lens 1*: `Phase::CandidateGateOptimist` acts as an Architectural Scout, advocating for framework entrypoints, root components, and view-switching shells.
  - *Lens 2*: `Phase::CandidateGateSkeptic` acts as a Precision Pruner, rejecting stylesheets, isolated utilities, and superficial keyword hits.
  - *Host Arbiter*: Evaluates the consensus matrix. On split decisions, breaks ties in favor of candidates backed by CodeGraph AST import topology, entrypoint identity, or domain alignment.

### Rules for Future Agents and Loops
- **Never collapse a Stereoscopic Gate into a single prompt**: Single-perspective prompts on local models inevitably skew into either hyper-permissive noise or hyper-restrictive rejections of context.
- **Keep the Host as the Arbiter**: Do not invoke a third LLM to arbitrate unless qualitative trade-offs cannot be resolved by ground truth. Prefer host-mediated rules based on CodeGraph AST topology, checksums, and budgets.
- **Apply Stereoscopic Gating to new decision boundaries**: If adding new capabilities (e.g. implementation plan validation, edit candidate selection, or patch verification), structure them as stereoscopic gates (Proposer/Optimist + Adversary/Critic + Deterministic Host Arbiter).

## Prompting Rules For Local Models

- Prompts should be short, direct, and phase-specific. Avoid ornate roleplay, large tool catalogs, and fragile formatting instructions.
- For the current read-only and plan-only loop, use host-enforced JSON outputs for internal phases. The model should return only the required JSON object for intake, extraction, planning, or synthesis.
- Do not ask the model to call tools. The harness decides which searches and reads happen.
- Do not introduce XML/tagged tool calls for the current loop. XML is only a future compatibility option if Loco adds model-selected actions and native/tool-schema calling is unavailable or unreliable.
- Plan mode may draft an implementation plan, affected files, risks, and validation commands. It must not draft replacement code, claim edits were applied, or mutate files.
- If edit artifacts are added later, prefer a host-mediated whole-file or block-level replacement workflow before attempting diff/search-replace formats. The host must validate and apply changes; the model should not directly mutate files.

## Prompt Portability Rules

- Prompt text in `agent-core` must be generalized for software engineering across languages, frameworks, runtimes, and app shapes. Do not hard-code this repository's stack, paths, filenames, frameworks, languages, UI library, package manager, or server architecture into reusable agent prompts.
- Framework, language, and architecture-specific guidance may appear only when it is derived from the current user request, indexed repository evidence, manifests/config files, file extensions, or leased evidence. Prefer host-side detection and context injection over static prompt wording.
- Use generic software surfaces in reusable prompts: entrypoints, screens/views, components/widgets, templates, styles, routing/navigation, APIs, services, persistence, configuration, build scripts, tests, command/event boundaries, and generated artifacts.
- Deterministic query expansion must use stack-neutral concepts unless a specific stack clue appears in the user request or repo evidence. Never add fixed paths such as this app's frontend files, Tauri config, Rust modules, or model/agent files as generic hints.
- Plan-mode affected files must stay evidence-backed. If the model proposes files not supported by active leased evidence, host logic should omit them or mark the uncertainty rather than rendering them as likely targets.
- Prompt tests should include portability checks: no static framework/language/path examples in reusable prompt strings unless the test is explicitly about detected evidence or a fixture-local fake path.

## Source And Evidence Rules

- Leased evidence and active notebook entries are the code truth. Repo maps are navigation hints only.
- Prefer layered retrieval before broad reads: path search, Tree-sitter-backed symbol search, regex/phrase content search, then narrow leased ranges before whole-file evidence.
- For UI/code location answers, preserve exact file paths, line ranges, symbol/block names, and render-block purpose.
- Prefer the narrowest relevant citation. Avoid background citations unless the answer explicitly depends on them.

## Verification

For changes to the agent loop or prompts, run:

```bash
cargo test --workspace
cargo test --manifest-path src-tauri/Cargo.toml
cargo check --manifest-path src-tauri/Cargo.toml
npm run build
```

After indexer changes, restart or reindex the app workspace before judging retrieval behavior in the UI.
