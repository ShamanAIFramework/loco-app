use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KnowledgeDomain {
    Architecture,
    CodeReview,
    StackIdiom,
    Security,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct KnowledgeCard {
    pub id: String,
    pub domain: KnowledgeDomain,
    pub match_tags: Vec<String>,
    pub title: String,
    pub summary: String,
    pub rules: Vec<String>,
}

#[derive(Clone, Debug, Default)]
pub struct KnowledgeEngine {
    cards: Vec<KnowledgeCard>,
}

impl KnowledgeEngine {
    pub fn new() -> Self {
        Self { cards: Vec::new() }
    }

    pub fn new_with_builtin() -> Self {
        let mut engine = Self::new();
        engine.load_builtin_cards();
        engine
    }

    pub fn add_card(&mut self, card: KnowledgeCard) {
        self.cards.push(card);
    }

    pub fn cards(&self) -> &[KnowledgeCard] {
        &self.cards
    }

    pub fn load_builtin_cards(&mut self) {
        self.cards = vec![
            KnowledgeCard {
                id: "desktop-client-architecture".to_string(),
                domain: KnowledgeDomain::Architecture,
                match_tags: vec![
                    "desktop".to_string(),
                    "window".to_string(),
                    "navigation".to_string(),
                    "routing".to_string(),
                    "tauri".to_string(),
                    "electron".to_string(),
                    "view".to_string(),
                    "route".to_string(),
                    "routes".to_string(),
                ],
                title: "Desktop Client Navigation & Windowing Architecture".to_string(),
                summary: "In desktop single-window client applications, navigation is commonly structured via reactive state-driven view switching rather than browser-style URL routing.".to_string(),
                rules: vec![
                    "Use reactive state (session signals, view enums, or tab state) for window view transitions when browser history or URL address bars are unnecessary.".to_string(),
                    "Avoid introducing multi-page URL routing frameworks unless deep-linking or nested URL path history is explicitly required by the project.".to_string(),
                    "Decouple the outer window chrome and layout shell from active view components.".to_string(),
                    "Ensure backgrounding, session hydration, and window lifecycle transitions do not block main thread rendering.".to_string(),
                ],
            },
            KnowledgeCard {
                id: "reactive-ui-signals".to_string(),
                domain: KnowledgeDomain::StackIdiom,
                match_tags: vec![
                    "reactive".to_string(),
                    "signal".to_string(),
                    "solid".to_string(),
                    "react".to_string(),
                    "props".to_string(),
                    "frontend".to_string(),
                    "component".to_string(),
                    "state".to_string(),
                    "render".to_string(),
                ],
                title: "Reactive UI & Fine-Grained Signal Patterns".to_string(),
                summary: "Fine-grained reactive UI frameworks track dependencies synchronously; avoid breaking reactivity through premature destructuring or unmemoized allocations.".to_string(),
                rules: vec![
                    "Preserve reactive getters and signal accessors; do not destructure reactive props in component setup scopes.".to_string(),
                    "Isolate high-frequency state updates to leaf components to prevent redundant root component re-renders.".to_string(),
                    "Derive state with pure computed/memo values instead of synchronizing multiple mutable state variables with imperative side effects.".to_string(),
                    "Clean up event listeners, timers, and background tasks in component teardown hooks.".to_string(),
                ],
            },
            KnowledgeCard {
                id: "systems-rust-engineering".to_string(),
                domain: KnowledgeDomain::StackIdiom,
                match_tags: vec![
                    "rust".to_string(),
                    "systems".to_string(),
                    "error".to_string(),
                    "concurrency".to_string(),
                    "memory".to_string(),
                    "resource".to_string(),
                    "backend".to_string(),
                    "harness".to_string(),
                ],
                title: "Robust Systems & Deterministic Resource Management".to_string(),
                summary: "Host and systems harnesses must prioritize explicit error propagation, bounded budgets, and deterministic resource lifecycles.".to_string(),
                rules: vec![
                    "Prefer explicit Result propagation with contextual error messages (e.g. anyhow::Context or typed error enums) over unwrap() or expect() in production code.".to_string(),
                    "Enforce deterministic bounds: bound buffer sizes, candidate pools, search windows, and execution timeouts.".to_string(),
                    "Model state transitions with explicit enums and pattern matching rather than loose boolean flags.".to_string(),
                    "Keep host orchestration deterministic: the host owns tools, leases, and budgets while isolating model coprocessor calls.".to_string(),
                ],
            },
            KnowledgeCard {
                id: "engineering-code-review-rubric".to_string(),
                domain: KnowledgeDomain::CodeReview,
                match_tags: vec![
                    "review".to_string(),
                    "plan".to_string(),
                    "critique".to_string(),
                    "quality".to_string(),
                    "design".to_string(),
                    "testing".to_string(),
                    "complexity".to_string(),
                    "rubric".to_string(),
                ],
                title: "Software Engineering Code Review & Planning Rubric".to_string(),
                summary: "High quality plans and code changes must minimize blast radius, guarantee testability, and account for failure modes.".to_string(),
                rules: vec![
                    "Design & Scope: Keep changes minimal and focused strictly on the stated objective; avoid bundling unrelated refactoring.".to_string(),
                    "Failure Handling: Explicitly identify edge cases, missing data, and error states; do not assume happy path execution.".to_string(),
                    "Complexity & Dependencies: Avoid over-engineering, premature abstractions, or introducing unnecessary external dependencies.".to_string(),
                    "Verifiability: Every planned change must be accompanied by explicit, automated validation commands and tests.".to_string(),
                ],
            },
            KnowledgeCard {
                id: "ipc-and-host-security".to_string(),
                domain: KnowledgeDomain::Security,
                match_tags: vec![
                    "security".to_string(),
                    "ipc".to_string(),
                    "tauri".to_string(),
                    "electron".to_string(),
                    "boundary".to_string(),
                    "shell".to_string(),
                    "invoke".to_string(),
                    "api".to_string(),
                ],
                title: "IPC Boundary Security & Input Validation".to_string(),
                summary: "Client-to-host IPC boundaries must treat all incoming client payloads as untrusted and enforce typed schema validation.".to_string(),
                rules: vec![
                    "Always deserialize and validate IPC arguments using strict typed schemas before delegating to host capabilities.".to_string(),
                    "Never construct shell commands via raw string concatenation; use parameterized command runners with explicit whitelists.".to_string(),
                    "Constrain filesystem and network capabilities to explicitly scoped workspace boundaries.".to_string(),
                    "Ensure sensitive secrets, tokens, or internal trace identifiers are not leaked to client renderer logs or unmasked output.".to_string(),
                ],
            },
        ];
    }

    /// Match relevant knowledge cards against detected repository tags and user query terms.
    pub fn match_cards(
        &self,
        query_text: &str,
        detected_tags: &[String],
        domains: Option<&[KnowledgeDomain]>,
        limit: usize,
    ) -> Vec<&KnowledgeCard> {
        let q_lower = query_text.to_lowercase();
        let query_words: Vec<&str> = q_lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| w.len() > 2)
            .collect();

        let mut scored: Vec<(&KnowledgeCard, usize)> = self
            .cards
            .iter()
            .filter(|card| {
                if let Some(allowed) = domains {
                    allowed.contains(&card.domain)
                } else {
                    true
                }
            })
            .filter_map(|card| {
                let mut score = 0;

                // Match against detected repository tags
                for detected in detected_tags {
                    let d_lower = detected.to_lowercase();
                    if card.match_tags.iter().any(|t| t == &d_lower) {
                        score += 3;
                    }
                }

                // Match against query words
                for word in &query_words {
                    if card.match_tags.iter().any(|t| t == *word) {
                        score += 2;
                    }
                    if card.title.to_lowercase().contains(word) {
                        score += 2;
                    }
                    if card.summary.to_lowercase().contains(word) {
                        score += 1;
                    }
                }

                if score > 0 {
                    Some((card, score))
                } else {
                    None
                }
            })
            .collect();

        scored.sort_by(|a, b| b.1.cmp(&a.1));
        scored.into_iter().take(limit).map(|(card, _)| card).collect()
    }

    /// Format matched cards into an authoritative guidance block for prompt injection.
    pub fn format_cards_for_prompt(&self, cards: &[&KnowledgeCard]) -> String {
        if cards.is_empty() {
            return String::new();
        }

        let mut out = String::from("AUTHORITATIVE ENGINEERING KNOWLEDGE & BEST PRACTICES\n");
        for card in cards {
            out.push_str(&format!("### {} ({:?})\n", card.title, card.domain));
            out.push_str(&format!("{}\n", card.summary));
            for rule in &card.rules {
                out.push_str(&format!("- {}\n", rule));
            }
            out.push('\n');
        }
        out.push_str("Incorporate this architectural context authoritatively. Do not defensively hedge when applying these established conventions.");
        out
    }
}

/// Detect project stack tags from repo map, question, and leased file paths.
pub fn detect_stack_tags(
    repo_map: &[String],
    question: &str,
    leased_paths: &[String],
) -> Vec<String> {
    let mut tags = Vec::new();
    let q_lower = question.to_lowercase();

    // Check query terms
    if q_lower.contains("route") || q_lower.contains("routing") || q_lower.contains("navigat") {
        tags.push("routing".to_string());
        tags.push("navigation".to_string());
    }
    if q_lower.contains("state") || q_lower.contains("signal") || q_lower.contains("reactive") {
        tags.push("state".to_string());
        tags.push("reactive".to_string());
    }
    if q_lower.contains("security") || q_lower.contains("ipc") || q_lower.contains("invoke") {
        tags.push("security".to_string());
        tags.push("ipc".to_string());
    }
    if q_lower.contains("review") || q_lower.contains("plan") || q_lower.contains("practice") {
        tags.push("review".to_string());
        tags.push("quality".to_string());
    }

    // Check repo map and file paths
    let all_paths = repo_map.iter().chain(leased_paths.iter());
    for path in all_paths {
        let p_lower = path.to_lowercase();
        if p_lower.contains("tauri.conf") || p_lower.contains("src-tauri") {
            tags.push("tauri".to_string());
            tags.push("desktop".to_string());
        }
        if p_lower.ends_with(".rs") || p_lower.contains("cargo.toml") {
            tags.push("rust".to_string());
            tags.push("systems".to_string());
        }
        if p_lower.ends_with(".tsx") || p_lower.ends_with(".jsx") || p_lower.contains("package.json") {
            tags.push("frontend".to_string());
            tags.push("component".to_string());
        }
        if p_lower.contains("electron") {
            tags.push("electron".to_string());
            tags.push("desktop".to_string());
        }
    }

    tags.sort();
    tags.dedup();
    tags
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_engine_matches_desktop_and_routing() {
        let engine = KnowledgeEngine::new_with_builtin();
        let detected = vec!["desktop".to_string(), "tauri".to_string()];
        let matched = engine.match_cards(
            "where does our UI configure route settings and is it aligned with best practices",
            &detected,
            None,
            3,
        );

        assert!(!matched.is_empty());
        assert!(matched.iter().any(|c| c.id == "desktop-client-architecture"));
    }

    #[test]
    fn test_format_cards_for_prompt() {
        let engine = KnowledgeEngine::new_with_builtin();
        let detected = vec!["desktop".to_string()];
        let matched = engine.match_cards("route settings", &detected, None, 1);
        let prompt_block = engine.format_cards_for_prompt(&matched);

        assert!(prompt_block.contains("AUTHORITATIVE ENGINEERING KNOWLEDGE & BEST PRACTICES"));
        assert!(prompt_block.contains("Desktop Client Navigation & Windowing Architecture"));
        assert!(prompt_block.contains("Do not defensively hedge"));
    }

    #[test]
    fn test_detect_stack_tags() {
        let repo_map = vec![
            "src-tauri/src/main.rs".to_string(),
            "src/App.tsx".to_string(),
            "package.json".to_string(),
        ];
        let tags = detect_stack_tags(&repo_map, "how does navigation work", &[]);
        assert!(tags.contains(&"desktop".to_string()));
        assert!(tags.contains(&"tauri".to_string()));
        assert!(tags.contains(&"rust".to_string()));
        assert!(tags.contains(&"frontend".to_string()));
        assert!(tags.contains(&"navigation".to_string()));
    }
}
