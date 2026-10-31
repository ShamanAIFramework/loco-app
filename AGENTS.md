# Loco Agent Guidance

This project is a local-first coding agent built around a deterministic Rust harness and a Solid/Tauri UI. Favor harness engineering over clever prompt-only behavior.

## Architecture Rules

- Keep model calls separated by phase. The current loop is intake, deterministic search/read, extraction, optional plan-only edit planning, and synthesis. Do not collapse this into a ReAct loop where the model decides tools and actions in one prompt.
- The Rust host owns tools, budgets, evidence acquisition, leases, tracing, source display, and stop conditions. The model is a bounded reasoning coprocessor.
- Preserve stable prompt prefixes where possible. Put durable system/phase/context structure before volatile request, evidence, and notebook content.
- Keep context small and specific. Prefer indexed file/symbol search, narrow evidence ranges, and structured notebook entries over dumping broad tool lists or whole repo text.
- Use source chips/events for UI source display. Do not put notebook ids, lease ids, hashes, or raw source bullet lists into the assistant answer text.

## Stereoscopic Gating Pattern

To overcome the recall-vs-precision trade-offs and context limits of local models (7B–14B), critical judgment boundaries use **Stereoscopic Gating**—an asymmetric dialectic governed by an **Intelligent Consensus Partition**. Critical filtering, classification, and validation boundaries are never collapsed into a single monocular prompt.

### Pattern Structure
1. **Lens 1 (Inclusion / Proposer / Recall - Thesis)**: A prompt incentivized to maximize recall, detect systemic architectural value, entrypoints, root components, state signals, and solution paths.
2. **Lens 2 (Precision / Critic / Pruning - Antithesis)**: An independent adversarial prompt incentivized to maximize precision, challenge false positives, eliminate coincidental keyword matches, isolate regressions, and prune noise.
3. **Consensus Partition (Synthesis & Arbitration)**:
   - **Unanimous Consensus (Fast Path)**: When both lenses agree (unanimous keep, unanimous drop, or unreserved critique approval), the decision is accepted immediately. Downstream arbitration and rebuttal phases are bypassed entirely to preserve token budget and reduce latency.
   - **Dispute Resolution (Deep Path)**: When lenses disagree (split keep/drop votes, missing aspects, or active critique disputes), the decision routes to arbitration or appellate reconciliation primed with host ground truth:
     - *Candidate Evidence*: `Phase::CandidateGateArbiter` resolves split candidate decisions using a tightly constrained prompt primed with ground-truth CodeGraph AST import topology, entrypoint identity, and domain alignment (backed by a deterministic host fallback).
     - *Implementation Plans & Edit Patches*: Appellate rebuttal and reconciliation phases resolve substantive technical disputes, followed by host-enforced lease freshness and Tree-sitter syntax validation before disk writes.

### Active Stereoscopic Gates in the Pipeline
- **Gate 1: Stereoscopic Intent Gate (Input Boundary)**:
  - *Lens 1*: `Phase::Intake` drafts intent decomposition and search queries.
  - *Lens 2*: `Phase::IntakeCritic` actively challenges dropped prompt clauses, intent collapse, and query contamination.
  - *Synthesis*: Unanimous decompositions proceed directly; disputes are normalized by host `TriadConsensus` to restore dropped questions and sanitize queries before execution.
- **Gate 2: Stereoscopic Evidence Gate (Retrieval Boundary)**:
  - *Lens 1*: `Phase::CandidateGateOptimist` acts as an Architectural Scout, advocating for framework entrypoints, root components, and view-switching shells.
  - *Lens 2*: `Phase::CandidateGateSkeptic` acts as a Precision Pruner, rejecting stylesheets, isolated utilities, and superficial keyword hits.
  - *Synthesis*: Unanimous keep/drop decisions are committed immediately. Split decisions invoke `Phase::CandidateGateArbiter` primed with host ground-truth AST topology, entrypoint identity, and domain alignment (with deterministic host fallback).
- **Gate 3: Appellate Planning Gate (Implementation Boundary)**:
  - *Lens 1*: `Phase::PlanProposer` drafts implementation steps, risks, and affected files from leased evidence.
  - *Lens 2*: `Phase::PlanCritic` challenges unverified assumptions, missing architectural aspects, and disputed steps.
  - *Synthesis*: Unanimous approvals (no missing aspects, disputed steps, or unverified assumptions) advance immediately to synthesis. Disputed plans invoke `Phase::PlanRebuttal` and `Phase::PlanReconciler` to distinguish genuine flaws from speculative nitpicks before committing to the notebook ledger.
- **Gate 4: Appellate Stereoscopic Edit Gate (Execution & Patch Boundary)**:
  - *Lens 1*: `Phase::PatchProposer` generates in-memory replacement unified diffs against leased focal windows.
  - *Lens 2*: `Phase::PatchCritic` audits syntax regressions, disputed changes, and unhandled boundary conditions.
  - *Synthesis*: Approved diffs without disputes bypass appellate rounds directly to host validation. Disputed patches invoke `Phase::PatchRebuttal` and `Phase::PatchReconciler`. Final diffs are verified by the Rust host for lease freshness and Tree-sitter AST syntax validity before staging in sandboxed Git worktrees.

### Rules for Future Agents and Loops
- **Never collapse a Stereoscopic Gate into a single prompt**: Single-perspective prompts on local models inevitably skew into either hyper-permissive noise or hyper-restrictive rejections of context.
- **Apply Consensus Partitioning to all dual-lens boundaries**: Always evaluate unanimous agreement first. Bypass multi-turn rebuttal/arbitration passes when both lenses converge to eliminate redundant local inference overhead.
- **Ground Arbiters in Host Truth**: When an LLM arbiter or reconciler is required to resolve qualitative disputes, strictly prime the prompt with host-verified ground truth (CodeGraph AST import edges, entrypoint identity, lease checksums, and token budgets). Maintain deterministic host fallback paths when LLM arbitration fails or times out.
- **Keep File and Worktree Mutation Host-Enforced**: The model generates diffs or plans, but the Rust host exclusively validates leases, parses AST syntax, and commits changes to disk.

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
