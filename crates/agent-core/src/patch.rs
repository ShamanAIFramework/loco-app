use crate::notebook::{hash_content, EvidenceLease};
use anyhow::{anyhow, Result};
use std::path::Path;

/// Extract lines from content based on 1-based start and end line numbers (inclusive).
pub fn extract_line_range(content: &str, start_line: usize, end_line: usize) -> Result<String> {
    if start_line == 0 || end_line < start_line {
        return Err(anyhow!("Invalid line range: {}..={}", start_line, end_line));
    }
    let lines: Vec<&str> = content.lines().collect();
    if start_line > lines.len() {
        return Err(anyhow!(
            "start_line {} exceeds file line count {}",
            start_line,
            lines.len()
        ));
    }
    let actual_end = end_line.min(lines.len());
    let slice = &lines[(start_line - 1)..actual_end];
    Ok(slice.join("\n"))
}

/// Replace 1-based line range `start_line..=end_line` (inclusive) with `replacement`.
///
/// Preserves the original file's newline convention (LF or CRLF) and trailing newline.
pub fn apply_in_memory_patch(
    content: &str,
    start_line: usize,
    end_line: usize,
    replacement: &str,
) -> Result<String> {
    if start_line == 0 || end_line < start_line {
        return Err(anyhow!("Invalid line range: {}..={}", start_line, end_line));
    }

    let is_crlf = content.contains("\r\n");
    let newline = if is_crlf { "\r\n" } else { "\n" };

    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Ok(replacement.to_string());
    }

    if start_line > lines.len() + 1 {
        return Err(anyhow!(
            "start_line {} exceeds file line count + 1 (length: {})",
            start_line,
            lines.len()
        ));
    }

    let prefix_lines = &lines[..start_line.saturating_sub(1).min(lines.len())];
    let suffix_start = end_line.min(lines.len());
    let suffix_lines = &lines[suffix_start..];

    let mut result_parts: Vec<&str> = Vec::new();
    if !prefix_lines.is_empty() {
        result_parts.extend_from_slice(prefix_lines);
    }
    if !replacement.is_empty() {
        result_parts.push(replacement);
    }
    if !suffix_lines.is_empty() {
        result_parts.extend_from_slice(suffix_lines);
    }

    let mut patched = result_parts.join(newline);

    // If original content had a trailing newline, ensure the patched content preserves it
    if (content.ends_with('\n') || content.ends_with("\r\n")) && !patched.ends_with('\n') {
        patched.push_str(newline);
    }

    Ok(patched)
}

/// Verifies that the file content matching the lease range has not drifted from the lease's content_hash.
pub fn verify_lease_freshness(file_content: &str, lease: &EvidenceLease) -> bool {
    let actual_slice = match extract_line_range(file_content, lease.range.start, lease.range.end) {
        Ok(s) => s,
        Err(_) => return false,
    };
    let actual_hash = hash_content(&actual_slice);
    actual_hash == lease.content_hash
}

/// Discovers adjacent sibling files in the repository that share the same directory and file extension.
///
/// Useful for providing the model with sibling exemplars to mimic established codebase idioms.
pub fn find_sibling_exemplars(repo_map: &[String], target_path: &str, limit: usize) -> Vec<String> {
    let target = Path::new(target_path);
    let target_dir = target.parent().unwrap_or_else(|| Path::new(""));
    let target_ext = target.extension().and_then(|e| e.to_str()).unwrap_or("");

    let mut candidates: Vec<(String, usize)> = repo_map
        .iter()
        .filter(|p| p.as_str() != target_path)
        .filter_map(|p| {
            let path = Path::new(p);
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            
            // Skip non-matching extensions
            if ext != target_ext {
                return None;
            }

            let file_name = path.file_name().and_then(|f| f.to_str()).unwrap_or("");
            // Skip test or spec files for general exemplars
            if file_name.contains(".test.") || file_name.contains(".spec.") || file_name.ends_with("_test.rs") {
                return None;
            }

            let dir = path.parent().unwrap_or_else(|| Path::new(""));
            let score = if dir == target_dir {
                10 // Exact same directory
            } else if dir.starts_with(target_dir) || target_dir.starts_with(dir) {
                5 // Parent or child directory
            } else {
                1 // Different directory
            };

            Some((p.clone(), score))
        })
        .collect();

    candidates.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    candidates.into_iter().take(limit).map(|(p, _)| p).collect()
}

/// Extracts a representative header slice (e.g. imports and top component declaration) from an exemplar file.
pub fn extract_exemplar_header(content: &str, max_lines: usize) -> String {
    let lines: Vec<&str> = content.lines().take(max_lines).collect();
    lines.join("\n")
}

/// Generates a unified git-style diff header and hunk for a replaced line range.
pub fn generate_unified_diff(
    path: &str,
    start_line: usize,
    original_lines: &str,
    replacement_code: &str,
) -> String {
    let orig: Vec<&str> = if original_lines.is_empty() {
        Vec::new()
    } else {
        original_lines.lines().collect()
    };
    let repl: Vec<&str> = if replacement_code.is_empty() {
        Vec::new()
    } else {
        replacement_code.lines().collect()
    };

    let orig_count = orig.len();
    let repl_count = repl.len();

    let mut out = format!(
        "--- a/{}\n+++ b/{}\n@@ -{},{} +{},{} @@\n",
        path, path, start_line, orig_count, start_line, repl_count
    );

    for line in orig {
        out.push_str(&format!("-{}\n", line));
    }
    for line in repl {
        out.push_str(&format!("+{}\n", line));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::notebook::{EvidenceLease, LineRange};

    #[test]
    fn test_extract_line_range() {
        let content = "line 1\nline 2\nline 3\nline 4\nline 5";
        let extracted = extract_line_range(content, 2, 4).unwrap();
        assert_eq!(extracted, "line 2\nline 3\nline 4");
    }

    #[test]
    fn test_apply_in_memory_patch_middle() {
        let content = "line 1\nline 2\nline 3\nline 4\nline 5\n";
        let patched = apply_in_memory_patch(content, 2, 4, "replaced 2-4").unwrap();
        assert_eq!(patched, "line 1\nreplaced 2-4\nline 5\n");
    }

    #[test]
    fn test_apply_in_memory_patch_start() {
        let content = "line 1\nline 2\nline 3\n";
        let patched = apply_in_memory_patch(content, 1, 2, "new header").unwrap();
        assert_eq!(patched, "new header\nline 3\n");
    }

    #[test]
    fn test_apply_in_memory_patch_end() {
        let content = "line 1\nline 2\nline 3\n";
        let patched = apply_in_memory_patch(content, 3, 3, "new footer").unwrap();
        assert_eq!(patched, "line 1\nline 2\nnew footer\n");
    }

    #[test]
    fn test_apply_in_memory_patch_deletion() {
        let content = "line 1\nline 2\nline 3\nline 4\n";
        let patched = apply_in_memory_patch(content, 2, 3, "").unwrap();
        assert_eq!(patched, "line 1\nline 4\n");
    }

    #[test]
    fn test_verify_lease_freshness() {
        let content = "fn foo() {\n    let x = 1;\n    println!(\"{}\", x);\n}\n";
        let lease = EvidenceLease::new(
            "src/main.rs".to_string(),
            Some("foo".to_string()),
            LineRange { start: 2, end: 3 },
            "    let x = 1;\n    println!(\"{}\", x);",
        );

        assert!(verify_lease_freshness(content, &lease));

        // Tamper with the content
        let drifted = "fn foo() {\n    let x = 999;\n    println!(\"{}\", x);\n}\n";
        assert!(!verify_lease_freshness(drifted, &lease));
    }

    #[test]
    fn test_find_sibling_exemplars() {
        let repo_map = vec![
            "src/App.tsx".to_string(),
            "src/AiChat.tsx".to_string(),
            "src/NotebookPanel.tsx".to_string(),
            "src/AiChat.test.tsx".to_string(),
            "src/components/Button.tsx".to_string(),
            "src-tauri/src/main.rs".to_string(),
        ];

        let siblings = find_sibling_exemplars(&repo_map, "src/App.tsx", 2);
        assert_eq!(siblings.len(), 2);
        assert!(siblings.contains(&"src/AiChat.tsx".to_string()));
        assert!(siblings.contains(&"src/NotebookPanel.tsx".to_string()));
        assert!(!siblings.contains(&"src/AiChat.test.tsx".to_string()));
    }

    #[test]
    fn test_extract_exemplar_header() {
        let content = "import React from 'react';\n\nexport function Component() {\n  return <div>Hello</div>;\n}\n";
        let header = extract_exemplar_header(content, 3);
        assert_eq!(header, "import React from 'react';\n\nexport function Component() {");
    }

    #[test]
    fn test_generate_unified_diff() {
        let diff = generate_unified_diff(
            "src/App.tsx",
            10,
            "const x = 1;\nconst y = 2;",
            "const x = 100;\nconst y = 200;\nconst z = 300;",
        );

        assert!(diff.contains("--- a/src/App.tsx"));
        assert!(diff.contains("+++ b/src/App.tsx"));
        assert!(diff.contains("@@ -10,2 +10,3 @@"));
        assert!(diff.contains("-const x = 1;"));
        assert!(diff.contains("-const y = 2;"));
        assert!(diff.contains("+const x = 100;"));
        assert!(diff.contains("+const y = 200;"));
        assert!(diff.contains("+const z = 300;"));
    }
}
