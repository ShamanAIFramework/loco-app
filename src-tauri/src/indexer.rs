// src-tauri/src/indexer.rs
//! Scans the local filesystem to index source files into the CodeGraph store.
//! Runs synchronously and ignores standard build and hidden directories.

use crate::codegraph::{CodeGraphStore, Symbol};
use crate::AppState;
use anyhow::Result;
use notify_debouncer_mini::notify::RecommendedWatcher;
use notify_debouncer_mini::{new_debouncer, notify::RecursiveMode, Debouncer};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::Path;
use std::time::Duration;
use tauri::Manager;
use tree_sitter::{Node, Parser};

pub fn index_workspace(store: &CodeGraphStore, project_id: &str, root_path: &Path) -> Result<()> {
    log::info!("Indexing workspace: {:?}", root_path);
    walk_and_index(store, project_id, root_path, root_path)?;
    refresh_project_symbol_edges(store, project_id, root_path)
}

fn walk_and_index(
    store: &CodeGraphStore,
    project_id: &str,
    current_dir: &Path,
    root_path: &Path,
) -> Result<()> {
    let entries = match fs::read_dir(current_dir) {
        Ok(e) => e,
        Err(_) => return Ok(()), // Ignore inaccessible directories safely
    };

    for entry in entries.flatten() {
        let path = entry.path();
        let file_name = entry.file_name();
        let file_name_str = file_name.to_string_lossy();

        if should_skip_entry(&file_name_str) {
            continue;
        }

        if path.is_dir() {
            let _ = walk_and_index(store, project_id, &path, root_path);
        } else if path.is_file() {
            if let Ok(rel_path) = path.strip_prefix(root_path) {
                let rel_str = rel_path.to_string_lossy().to_string();
                let _ = index_workspace_file(store, project_id, &path, &rel_str);
            }
        }
    }
    Ok(())
}

/// Spawns a background thread that listens to OS filesystem events and incrementally updates SQLite.
pub fn start_watcher(
    app: tauri::AppHandle,
    project_id: String,
    root_path: std::path::PathBuf,
) -> Result<Debouncer<RecommendedWatcher>> {
    // Create an MPSC channel to receive debounced events
    let (tx, rx) = std::sync::mpsc::channel();

    // Debounce events every 500ms to ignore editor temp-file spam
    let mut debouncer = new_debouncer(Duration::from_millis(500), tx)
        .map_err(|e| anyhow::anyhow!("Failed to create watcher: {}", e))?;

    debouncer
        .watcher()
        .watch(&root_path, RecursiveMode::Recursive)
        .map_err(|e| anyhow::anyhow!("Failed to watch path: {}", e))?;

    let root_clone = root_path.clone();

    // Spawn the background worker thread
    std::thread::spawn(move || {
        for res in rx {
            match res {
                Ok(events) => {
                    let state = app.state::<AppState>();

                    for event in events {
                        let path = event.path;
                        let path_str = path.to_string_lossy();

                        // 1. Ignore build directories and hidden files
                        if path_str.contains("/.")
                            || path_str.contains("\\.")
                            || path_str.contains("/node_modules/")
                            || path_str.contains("\\node_modules\\")
                            || path_str.contains("/target/")
                            || path_str.contains("\\target\\")
                            || path_str.contains("/models/")
                            || path_str.contains("\\models\\")
                            || path_str.contains("/binaries/")
                            || path_str.contains("\\binaries\\")
                        {
                            continue;
                        }

                        // 2. Calculate the relative path
                        if let Ok(rel_path) = path.strip_prefix(&root_clone) {
                            let rel_str = rel_path.to_string_lossy().to_string();

                            // 3. Incrementally update SQLite based on file existence
                            if path.exists() && path.is_file() {
                                // File was created or modified
                                let _ = index_workspace_file(
                                    &state.codegraph,
                                    &project_id,
                                    &path,
                                    &rel_str,
                                );
                            } else if !path.exists() {
                                // File was deleted
                                let _ = state.codegraph.remove_file(&project_id, &rel_str);
                            }
                        }
                    }
                }
                Err(e) => log::error!("Watch error: {:?}", e),
            }
        }
    });

    // Return the debouncer so it doesn't get dropped (which would stop the watching)
    Ok(debouncer)
}

fn file_checksum(path: &Path) -> Result<String> {
    let bytes = fs::read(path)?;
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    Ok(format!("{:x}", hasher.finalize()))
}

fn index_workspace_file(
    store: &CodeGraphStore,
    project_id: &str,
    path: &Path,
    rel_path: &str,
) -> Result<()> {
    let is_source = looks_like_source(rel_path);
    let checksum = if is_source {
        file_checksum(path).ok()
    } else {
        None
    };
    let file = store.upsert_file(project_id, rel_path, checksum.as_deref())?;
    store.clear_file_symbols(&file.id)?;

    if !is_source {
        return Ok(());
    }

    let content = match fs::read_to_string(path) {
        Ok(content) => content,
        Err(_) => return Ok(()),
    };

    let parsed_symbols = extract_symbols_for_path(rel_path, &content);
    let mut inserted_symbols = Vec::<(Symbol, ParsedSymbol)>::new();
    for symbol in parsed_symbols {
        let hash = symbol_hash(&content, symbol.start_line, symbol.end_line);
        if let Ok(inserted) = store.upsert_symbol(
            &file.id,
            &symbol.name,
            &symbol.kind,
            symbol.start_line,
            symbol.end_line,
            hash.as_deref(),
        ) {
            inserted_symbols.push((inserted, symbol));
        }
    }

    let project_symbols = store.project_symbols(project_id).unwrap_or_default();
    upsert_reference_edges_for_symbols(store, &content, &inserted_symbols, &project_symbols);

    Ok(())
}

fn refresh_project_symbol_edges(
    store: &CodeGraphStore,
    project_id: &str,
    root_path: &Path,
) -> Result<()> {
    let project_symbols = store.project_symbols(project_id).unwrap_or_default();
    let project_files = store.project_files(project_id).unwrap_or_default();

    for file in project_files {
        let path = root_path.join(&file.path);
        let Ok(content) = fs::read_to_string(path) else {
            continue;
        };
        let file_symbols = project_symbols
            .iter()
            .filter(|symbol| symbol.file_id == file.id)
            .map(|symbol| {
                (
                    symbol.clone(),
                    ParsedSymbol {
                        name: symbol.name.clone(),
                        kind: symbol.kind.clone(),
                        start_line: symbol.start_line,
                        end_line: symbol.end_line,
                    },
                )
            })
            .collect::<Vec<_>>();
        upsert_reference_edges_for_symbols(store, &content, &file_symbols, &project_symbols);
    }

    Ok(())
}

fn upsert_reference_edges_for_symbols(
    store: &CodeGraphStore,
    content: &str,
    source_symbols: &[(Symbol, ParsedSymbol)],
    target_symbols: &[Symbol],
) {
    for (from_symbol, from_parsed) in source_symbols {
        let range_text = symbol_text(&content, from_parsed.start_line, from_parsed.end_line);
        for target in target_symbols {
            if target.id == from_symbol.id || target.name == from_parsed.name {
                continue;
            }
            if let Some(kind) = symbol_occurrence_kind(&range_text, &target.name) {
                let _ = store.upsert_symbol_edge(&from_symbol.id, &target.name, kind);
            }
        }
    }
}

fn symbol_occurrence_kind(text: &str, symbol: &str) -> Option<&'static str> {
    if symbol.is_empty() {
        return None;
    }
    let mut search_from = 0usize;
    while let Some(offset) = text[search_from..].find(symbol) {
        let start = search_from + offset;
        let end = start + symbol.len();
        let before = text[..start].chars().next_back();
        let after = text[end..].chars().next();
        if before.map(is_identifier_char).unwrap_or(false)
            || after.map(is_identifier_char).unwrap_or(false)
        {
            search_from = end;
            continue;
        }
        let after_symbol = text[end..].trim_start();
        return Some(if after_symbol.starts_with('(') {
            "call"
        } else {
            "reference"
        });
    }
    None
}

fn is_identifier_char(ch: char) -> bool {
    ch == '_' || ch.is_ascii_alphanumeric()
}

#[derive(Clone, Debug)]
struct ParsedSymbol {
    name: String,
    kind: String,
    start_line: u32,
    end_line: u32,
}

#[cfg(test)]
fn extract_symbols(content: &str) -> Vec<ParsedSymbol> {
    extract_symbols_with_line_scanner(content)
}

fn extract_symbols_for_path(path: &str, content: &str) -> Vec<ParsedSymbol> {
    let parsed = extract_symbols_with_tree_sitter(path, content);
    if parsed.is_empty() {
        extract_symbols_with_line_scanner(content)
    } else {
        parsed
    }
}

fn extract_symbols_with_line_scanner(content: &str) -> Vec<ParsedSymbol> {
    let lines = content.lines().collect::<Vec<_>>();
    let mut starts = Vec::<ParsedSymbol>::new();

    for (idx, line) in lines.iter().enumerate() {
        let Some((kind, name)) = parse_symbol_start(line) else {
            continue;
        };
        starts.push(ParsedSymbol {
            name,
            kind,
            start_line: (idx + 1) as u32,
            end_line: (idx + 1) as u32,
        });
    }

    for idx in 0..starts.len() {
        let next_start = starts
            .get(idx + 1)
            .map(|symbol| symbol.start_line.saturating_sub(1))
            .unwrap_or(lines.len() as u32);
        let brace_end = block_end_line(&lines, starts[idx].start_line as usize);
        starts[idx].end_line = brace_end.unwrap_or(next_start).max(starts[idx].start_line);
    }

    starts
}

fn extract_symbols_with_tree_sitter(path: &str, content: &str) -> Vec<ParsedSymbol> {
    let mut parser = Parser::new();
    let extension = Path::new(path)
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default();
    let language = match extension {
        "rs" => tree_sitter_rust::LANGUAGE.into(),
        "ts" | "js" | "jsx" => tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
        "tsx" => tree_sitter_typescript::LANGUAGE_TSX.into(),
        _ => return Vec::new(),
    };
    if parser.set_language(&language).is_err() {
        return Vec::new();
    }
    let Some(tree) = parser.parse(content, None) else {
        return Vec::new();
    };
    if tree.root_node().has_error() {
        return Vec::new();
    }

    let mut symbols = Vec::new();
    collect_tree_sitter_symbols(tree.root_node(), content, extension, &mut symbols);
    symbols.sort_by_key(|symbol| (symbol.start_line, symbol.end_line, symbol.name.clone()));
    symbols.dedup_by(|left, right| {
        left.name == right.name
            && left.kind == right.kind
            && left.start_line == right.start_line
            && left.end_line == right.end_line
    });
    symbols
}

fn collect_tree_sitter_symbols(
    node: Node<'_>,
    content: &str,
    extension: &str,
    symbols: &mut Vec<ParsedSymbol>,
) {
    if let Some(symbol) = tree_sitter_symbol(node, content, extension) {
        symbols.push(symbol);
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        collect_tree_sitter_symbols(child, content, extension, symbols);
    }
}

fn tree_sitter_symbol(node: Node<'_>, content: &str, extension: &str) -> Option<ParsedSymbol> {
    let kind = node.kind();
    let (symbol_kind, name) = match (extension, kind) {
        ("rs", "function_item") => ("fn", named_child_text(node, content, "name")?),
        ("rs", "struct_item") => ("struct", named_child_text(node, content, "name")?),
        ("rs", "enum_item") => ("enum", named_child_text(node, content, "name")?),
        ("rs", "trait_item") => ("trait", named_child_text(node, content, "name")?),
        ("rs", "impl_item") => {
            let name = named_child_text(node, content, "type")
                .or_else(|| impl_header_name(node, content))?;
            ("impl", name)
        }
        (_, "function_declaration") => ("function", named_child_text(node, content, "name")?),
        (_, "class_declaration") => ("class", named_child_text(node, content, "name")?),
        (_, "method_definition") => ("method", named_child_text(node, content, "name")?),
        (_, "variable_declarator") => {
            if !node_has_function_value(node, content) {
                return None;
            }
            ("function", named_child_text(node, content, "name")?)
        }
        _ => return None,
    };

    Some(ParsedSymbol {
        name,
        kind: symbol_kind.to_string(),
        start_line: (node.start_position().row + 1) as u32,
        end_line: (node.end_position().row + 1) as u32,
    })
}

fn named_child_text(node: Node<'_>, content: &str, field: &str) -> Option<String> {
    node.child_by_field_name(field)
        .and_then(|child| child.utf8_text(content.as_bytes()).ok())
        .map(|name| name.trim().to_string())
        .filter(|name| !name.is_empty())
}

fn node_has_function_value(node: Node<'_>, content: &str) -> bool {
    node.child_by_field_name("value")
        .map(|value| matches!(value.kind(), "arrow_function" | "function"))
        .unwrap_or_else(|| {
            node.utf8_text(content.as_bytes())
                .map(|text| text.contains("=>") || text.contains("function"))
                .unwrap_or(false)
        })
}

fn impl_header_name(node: Node<'_>, content: &str) -> Option<String> {
    let text = node.utf8_text(content.as_bytes()).ok()?;
    let first_line = text.lines().next()?.trim();
    let rest = first_line.strip_prefix("impl")?.trim();
    let name = rest
        .trim_start_matches('<')
        .split(|ch: char| ch.is_whitespace() || ch == '{' || ch == '<')
        .find(|part| !part.is_empty() && *part != "for")?;
    Some(name.to_string())
}

fn block_end_line(lines: &[&str], start_line: usize) -> Option<u32> {
    let start_idx = start_line.saturating_sub(1);
    let mut depth = 0i32;
    let mut saw_open = false;

    for (idx, line) in lines.iter().enumerate().skip(start_idx) {
        let scan = if idx == start_idx {
            match line.rfind('{') {
                Some(pos) => &line[pos..],
                None => line,
            }
        } else {
            line
        };
        for ch in scan.chars() {
            match ch {
                '{' => {
                    depth += 1;
                    saw_open = true;
                }
                '}' if saw_open => {
                    depth -= 1;
                    if depth <= 0 {
                        return Some((idx + 1) as u32);
                    }
                }
                _ => {}
            }
        }
    }

    None
}

fn parse_symbol_start(line: &str) -> Option<(String, String)> {
    let trimmed = line.trim_start();
    if trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with('*') {
        return None;
    }

    parse_after_keyword(trimmed, "pub fn", "fn")
        .or_else(|| parse_after_keyword(trimmed, "pub async fn", "fn"))
        .or_else(|| parse_after_keyword(trimmed, "async fn", "fn"))
        .or_else(|| parse_after_keyword(trimmed, "fn", "fn"))
        .or_else(|| parse_after_keyword(trimmed, "pub struct", "struct"))
        .or_else(|| parse_after_keyword(trimmed, "struct", "struct"))
        .or_else(|| parse_after_keyword(trimmed, "pub enum", "enum"))
        .or_else(|| parse_after_keyword(trimmed, "enum", "enum"))
        .or_else(|| parse_after_keyword(trimmed, "pub trait", "trait"))
        .or_else(|| parse_after_keyword(trimmed, "trait", "trait"))
        .or_else(|| parse_after_keyword(trimmed, "pub impl", "impl"))
        .or_else(|| parse_after_keyword(trimmed, "impl", "impl"))
        .or_else(|| parse_after_keyword(trimmed, "export default function", "function"))
        .or_else(|| parse_after_keyword(trimmed, "export function", "function"))
        .or_else(|| parse_after_keyword(trimmed, "function", "function"))
        .or_else(|| parse_after_keyword(trimmed, "export class", "class"))
        .or_else(|| parse_after_keyword(trimmed, "class", "class"))
        .or_else(|| parse_const_arrow(trimmed))
}

fn parse_after_keyword(line: &str, keyword: &str, kind: &str) -> Option<(String, String)> {
    let rest = line.strip_prefix(keyword)?.trim_start();
    let name = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
        .collect::<String>();
    if name.is_empty() {
        None
    } else {
        Some((kind.to_string(), name))
    }
}

fn parse_const_arrow(line: &str) -> Option<(String, String)> {
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
    if name.is_empty() {
        None
    } else {
        Some(("function".to_string(), name))
    }
}

fn symbol_hash(content: &str, start_line: u32, end_line: u32) -> Option<String> {
    let slice = symbol_text(content, start_line, end_line);
    if slice.is_empty() {
        None
    } else {
        let mut hasher = Sha256::new();
        hasher.update(slice.as_bytes());
        Some(format!("{:x}", hasher.finalize()))
    }
}

fn symbol_text(content: &str, start_line: u32, end_line: u32) -> String {
    content
        .lines()
        .enumerate()
        .filter(|(idx, _)| {
            let line_no = (*idx + 1) as u32;
            line_no >= start_line && line_no <= end_line
        })
        .map(|(_, line)| line)
        .collect::<Vec<_>>()
        .join("\n")
}

fn looks_like_source(path: &str) -> bool {
    matches!(
        Path::new(path).extension().and_then(|ext| ext.to_str()),
        Some(
            "rs" | "ts"
                | "tsx"
                | "js"
                | "jsx"
                | "mjs"
                | "cjs"
                | "py"
                | "go"
                | "java"
                | "kt"
                | "swift"
                | "c"
                | "h"
                | "cpp"
                | "hpp"
        )
    )
}

fn should_skip_entry(name: &str) -> bool {
    name.starts_with('.')
        || matches!(
            name,
            "node_modules" | "target" | "dist" | "build" | "models" | "binaries" | "resources"
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use uuid::Uuid;

    #[test]
    fn extracts_common_rust_symbols_with_ranges() {
        let source = r#"
pub struct AppState {
}

pub async fn run_app() {
}

impl AppState {
}
"#;
        let symbols = extract_symbols(source);
        assert_eq!(symbols.len(), 3);
        assert_eq!(symbols[0].name, "AppState");
        assert_eq!(symbols[0].kind, "struct");
        assert_eq!(symbols[0].start_line, 2);
        assert_eq!(symbols[0].end_line, 3);
        assert_eq!(symbols[1].name, "run_app");
    }

    #[test]
    fn tree_sitter_extracts_rust_symbols_with_impl_ranges() {
        let source = r#"
pub struct AppState {
}

impl AppState {
  pub fn new() -> Self {
    Self {}
  }
}
"#;
        let symbols = extract_symbols_for_path("src/lib.rs", source);

        assert!(symbols
            .iter()
            .any(|symbol| symbol.name == "AppState" && symbol.kind == "struct"));
        assert!(symbols
            .iter()
            .any(|symbol| symbol.name == "AppState" && symbol.kind == "impl"));
        assert!(symbols
            .iter()
            .any(|symbol| symbol.name == "new" && symbol.kind == "fn"));
    }

    #[test]
    fn extracts_common_typescript_symbols() {
        let source = r#"
export function useThing() {
}
const helper = () => {}
export class Widget {}
"#;
        let symbols = extract_symbols(source);
        assert_eq!(
            symbols.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(),
            vec!["useThing", "helper", "Widget"]
        );
        let use_thing = symbols.iter().find(|s| s.name == "useThing").unwrap();
        assert_eq!(use_thing.start_line, 2);
        assert_eq!(use_thing.end_line, 3);
    }

    #[test]
    fn tsx_component_range_can_cover_nested_helpers_and_render_block() {
        let source = r#"
export default function GitPanel(props: { workspacePath: string }) {
  const [files, setFiles] = createSignal([]);

  const fetchStatus = async () => {
    setFiles([]);
  };

  const loadDiff = async (path: string) => {
    console.log(path);
  };

  return (
    <div>
      {/* Changed Files List (Top Half) */}
      <div>
        <For each={files()}>
          {(file) => (
            <button>
              <span>{file.path}</span>
              <span>{file.status}</span>
            </button>
          )}
        </For>
      </div>
    </div>
  );
}
"#;
        let symbols = extract_symbols(source);
        let component = symbols.iter().find(|s| s.name == "GitPanel").unwrap();

        assert_eq!(component.start_line, 2);
        assert_eq!(component.end_line, 28);
        assert!(symbols.iter().any(|s| s.name == "fetchStatus"));
        assert!(symbols.iter().any(|s| s.name == "loadDiff"));
    }

    #[test]
    fn tree_sitter_extracts_tsx_component_and_callbacks() {
        let source = r#"
export default function GitPanel(props: { workspacePath: string }) {
  const fetchStatus = async () => {
    return props.workspacePath;
  };

  return <button onClick={fetchStatus}>Refresh</button>;
}
"#;
        let symbols = extract_symbols_for_path("src/GitPanel.tsx", source);
        let component = symbols.iter().find(|s| s.name == "GitPanel").unwrap();

        assert_eq!(component.kind, "function");
        assert_eq!(component.start_line, 2);
        assert_eq!(component.end_line, 8);
        assert!(symbols
            .iter()
            .any(|symbol| symbol.name == "fetchStatus" && symbol.kind == "function"));
    }

    #[test]
    fn symbol_text_extracts_requested_range() {
        let text = "one\ntwo\nthree\nfour";
        assert_eq!(symbol_text(text, 2, 3), "two\nthree");
    }

    #[test]
    fn symbol_occurrence_kind_respects_identifier_boundaries() {
        assert_eq!(
            symbol_occurrence_kind("return helper();", "helper"),
            Some("call")
        );
        assert_eq!(
            symbol_occurrence_kind("const selected = helper;", "helper"),
            Some("reference")
        );
        assert_eq!(
            symbol_occurrence_kind("const helperText = '';", "helper"),
            None
        );
        assert_eq!(
            symbol_occurrence_kind("const my_helper = 1;", "helper"),
            None
        );
    }

    #[test]
    fn indexes_cross_file_symbol_reference_edges() {
        let root = env::temp_dir().join(format!("loco_indexer_{}", Uuid::new_v4()));
        let db_path = env::temp_dir().join(format!("loco_indexer_{}.db", Uuid::new_v4()));
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(
            root.join("src/helpers.ts"),
            "export function helper() {\n  return 1;\n}\n",
        )
        .unwrap();
        fs::write(
            root.join("src/feature.ts"),
            "export function useHelper() {\n  return helper();\n}\n",
        )
        .unwrap();

        let store = CodeGraphStore::new(&db_path).unwrap();
        let project = store.register_project("Test", &root).unwrap();
        index_workspace(&store, &project.id, &root).unwrap();

        let edge_hits = store.search_symbols(&project.id, "helper").unwrap();
        assert!(edge_hits
            .iter()
            .any(|(_, symbol)| symbol.name == "useHelper"));
        let feature_file = edge_hits
            .iter()
            .find(|(_, symbol)| symbol.name == "useHelper")
            .map(|(file, _)| file.path.as_str());
        assert_eq!(feature_file, Some("src/feature.ts"));

        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_file(db_path);
    }

    #[test]
    fn skips_large_generated_and_resource_directories() {
        assert!(should_skip_entry("node_modules"));
        assert!(should_skip_entry("target"));
        assert!(should_skip_entry("models"));
        assert!(should_skip_entry("resources"));
        assert!(!should_skip_entry("src"));
    }
}
