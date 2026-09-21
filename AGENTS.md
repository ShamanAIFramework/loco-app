# Loco Agent Guidance

This project is a local-first coding agent built around a deterministic Rust harness and a Solid/Tauri UI. Favor harness engineering over clever prompt-only behavior.

## Architecture Rules

- Keep model calls separated by phase. The current loop is intake, deterministic search/read, extraction, optional plan-only edit planning, and synthesis. Do not collapse this into a ReAct loop where the model decides tools and actions in one prompt.
- The Rust host owns tools, budgets, evidence acquisition, leases, tracing, source display, and stop conditions. The model is a bounded reasoning coprocessor.
- Preserve stable prompt prefixes where possible. Put durable system/phase/context structure before volatile request, evidence, and notebook content.
- Keep context small and specific. Prefer indexed file/symbol search, narrow evidence ranges, and structured notebook entries over dumping broad tool lists or whole repo text.
- Use source chips/events for UI source display. Do not put notebook ids, lease ids, hashes, or raw source bullet lists into the assistant answer text.

## Prompting Rules For Local Models

- Prompts should be short, direct, and phase-specific. Avoid ornate roleplay, large tool catalogs, and fragile formatting instructions.
- For the current read-only and plan-only loop, use host-enforced JSON outputs for internal phases. The model should return only the required JSON object for intake, extraction, planning, or synthesis.
- Do not ask the model to call tools. The harness decides which searches and reads happen.
- Do not introduce XML/tagged tool calls for the current loop. XML is only a future compatibility option if Loco adds model-selected actions and native/tool-schema calling is unavailable or unreliable.
- Plan mode may draft an implementation plan, affected files, risks, and validation commands. It must not draft replacement code, claim edits were applied, or mutate files.
- If edit artifacts are added later, prefer a host-mediated whole-file or block-level replacement workflow before attempting diff/search-replace formats. The host must validate and apply changes; the model should not directly mutate files.

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
