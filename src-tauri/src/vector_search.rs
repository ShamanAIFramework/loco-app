// src-tauri/src/vector_search.rs
//! Local Vector Search Engine for semantic code retrieval.
//! Operates on code chunks and Tree-sitter symbols using vector space embeddings
//! and cosine similarity to enable dynamic, semantic discovery without hardcoded keyword mappings.

use agent_core::VectorHit;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use crate::codegraph::{SourceFile, Symbol};

/// A semantic code chunk with text content and location metadata.
#[derive(Clone, Debug)]
pub struct CodeChunk {
    pub path: String,
    pub symbol: Option<String>,
    pub start_line: u32,
    pub end_line: u32,
    pub text: String,
}

/// Tokenizes text into normalized words, camelCase/snake_case sub-tokens, and character n-grams.
pub fn tokenize_for_vector(text: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let lower = text.to_lowercase();

    // 1. Extract words and sub-tokens
    let mut current_word = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            current_word.push(ch);
        } else if !current_word.is_empty() {
            split_identifier_into_tokens(&current_word, &mut tokens);
            current_word.clear();
        }
    }
    if !current_word.is_empty() {
        split_identifier_into_tokens(&current_word, &mut tokens);
    }

    // 2. Character 3-grams for semantic and typo tolerance on key words
    for word in lower.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() >= 4 && w.len() <= 15) {
        let chars: Vec<char> = word.chars().collect();
        for window in chars.windows(3) {
            tokens.push(format!("3g:{}", window.iter().collect::<String>()));
        }
    }

    tokens
}

fn split_identifier_into_tokens(ident: &str, out: &mut Vec<String>) {
    let lower = ident.to_lowercase();
    if lower.len() >= 2 {
        out.push(lower.clone());
    }

    // Split CamelCase or snake_case
    let mut piece = String::new();
    let chars: Vec<char> = ident.chars().collect();
    for i in 0..chars.len() {
        let c = chars[i];
        if c == '_' || c == '-' {
            if piece.len() >= 2 {
                out.push(piece.to_lowercase());
                piece.clear();
            }
        } else if c.is_uppercase() && !piece.is_empty() && (i + 1 < chars.len() && chars[i + 1].is_lowercase()) {
            out.push(piece.to_lowercase());
            piece.clear();
            piece.push(c);
        } else {
            piece.push(c);
        }
    }
    if piece.len() >= 2 {
        out.push(piece.to_lowercase());
    }
}

/// Computes term vector with TF-IDF weights.
fn compute_term_vector(tokens: &[String], idf_map: &HashMap<String, f32>) -> (HashMap<String, f32>, f32) {
    let mut tf = HashMap::<String, f32>::new();
    for t in tokens {
        *tf.entry(t.clone()).or_insert(0.0) += 1.0;
    }

    let total = tokens.len() as f32;
    let mut vec = HashMap::new();
    let mut norm_sq = 0.0;

    for (term, count) in tf {
        let idf = idf_map.get(&term).copied().unwrap_or(1.0);
        let weight = (count / total) * idf;
        norm_sq += weight * weight;
        vec.insert(term, weight);
    }

    let norm = norm_sq.sqrt();
    (vec, norm)
}

/// Cosine similarity between query vector and document vector.
fn cosine_similarity(
    q_vec: &HashMap<String, f32>,
    q_norm: f32,
    d_vec: &HashMap<String, f32>,
    d_norm: f32,
) -> f32 {
    if q_norm == 0.0 || d_norm == 0.0 {
        return 0.0;
    }
    let mut dot = 0.0;
    for (term, q_w) in q_vec {
        if let Some(d_w) = d_vec.get(term) {
            dot += q_w * d_w;
        }
    }
    dot / (q_norm * d_norm)
}

/// Chunks indexed files into semantic search blocks.
pub fn extract_chunks_from_project(
    root_path: &Path,
    files: &[SourceFile],
    symbols: &[Symbol],
) -> Vec<CodeChunk> {
    let mut chunks = Vec::new();

    // Group symbols by file_id
    let mut symbols_by_file: HashMap<String, Vec<&Symbol>> = HashMap::new();
    for s in symbols {
        symbols_by_file.entry(s.file_id.clone()).or_default().push(s);
    }

    for file in files {
        let full_path = root_path.join(&file.path);
        let Ok(content) = fs::read_to_string(&full_path) else {
            continue;
        };
        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len() as u32;
        if total_lines == 0 {
            continue;
        }

        let mut covered_ranges: HashSet<(u32, u32)> = HashSet::new();

        // 1. Symbol-based chunks (Tree-sitter components, functions, classes)
        if let Some(file_syms) = symbols_by_file.get(&file.id) {
            for sym in file_syms {
                let start = sym.start_line.max(1);
                let end = sym.end_line.min(total_lines).max(start);
                if end - start >= 3 && end - start <= 200 {
                    let slice_text = lines[(start as usize - 1)..(end as usize)].join("\n");
                    chunks.push(CodeChunk {
                        path: file.path.clone(),
                        symbol: Some(sym.name.clone()),
                        start_line: start,
                        end_line: end,
                        text: slice_text,
                    });
                    covered_ranges.insert((start, end));
                }
            }
        }

        // 2. Sliding window chunks for uncovered sections or top-level render blocks
        let window_size = 40;
        let step = 25;
        let mut curr = 1;
        while curr <= total_lines {
            let end = (curr + window_size - 1).min(total_lines);
            let already_covered = covered_ranges.iter().any(|(s, e)| curr >= *s && end <= *e);
            if !already_covered && end >= curr {
                let slice_text = lines[(curr as usize - 1)..(end as usize)].join("\n");
                if !slice_text.trim().is_empty() {
                    chunks.push(CodeChunk {
                        path: file.path.clone(),
                        symbol: None,
                        start_line: curr,
                        end_line: end,
                        text: slice_text,
                    });
                }
            }
            if end == total_lines {
                break;
            }
            curr += step;
        }
    }

    chunks
}

/// Searches the project using vector space embeddings and cosine similarity.
pub fn search_project_vectors(
    root_path: &Path,
    files: &[SourceFile],
    symbols: &[Symbol],
    query: &str,
    limit: usize,
) -> Result<Vec<VectorHit>> {
    let query_clean = query.trim();
    if query_clean.is_empty() {
        return Ok(Vec::new());
    }

    let chunks = extract_chunks_from_project(root_path, files, symbols);
    if chunks.is_empty() {
        return Ok(Vec::new());
    }

    // 1. Build document frequency across chunks for IDF
    let mut doc_freq = HashMap::<String, usize>::new();
    let mut chunk_tokens = Vec::with_capacity(chunks.len());

    for chunk in &chunks {
        let tokens = tokenize_for_vector(&format!("{} {} {}", chunk.path, chunk.symbol.as_deref().unwrap_or(""), chunk.text));
        let unique_terms: HashSet<String> = tokens.iter().cloned().collect();
        for t in unique_terms {
            *doc_freq.entry(t).or_insert(0) += 1;
        }
        chunk_tokens.push(tokens);
    }

    let num_docs = chunks.len() as f32;
    let mut idf_map = HashMap::new();
    for (term, freq) in doc_freq {
        let idf = ((1.0 + num_docs) / (1.0 + freq as f32)).ln() + 1.0;
        idf_map.insert(term, idf);
    }

    // 2. Compute query vector
    let q_tokens = tokenize_for_vector(query_clean);
    let (q_vec, q_norm) = compute_term_vector(&q_tokens, &idf_map);
    if q_norm == 0.0 {
        return Ok(Vec::new());
    }

    // 3. Score chunks using cosine similarity
    let mut scored_hits = Vec::new();
    for (idx, chunk) in chunks.into_iter().enumerate() {
        let (d_vec, d_norm) = compute_term_vector(&chunk_tokens[idx], &idf_map);
        let sim = cosine_similarity(&q_vec, q_norm, &d_vec, d_norm);
        if sim > 0.05 {
            let snippet: String = chunk.text.lines().take(4).collect::<Vec<_>>().join("\n");
            scored_hits.push(VectorHit {
                path: chunk.path,
                start_line: chunk.start_line,
                end_line: chunk.end_line,
                symbol: chunk.symbol,
                snippet,
                score: sim,
            });
        }
    }

    // 4. Sort descending by score
    scored_hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored_hits.truncate(limit);

    Ok(scored_hits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenization_and_subwords() {
        let tokens = tokenize_for_vector("routeSettings activeSession");
        assert!(tokens.contains(&"route".to_string()));
        assert!(tokens.contains(&"settings".to_string()));
        assert!(tokens.contains(&"active".to_string()));
        assert!(tokens.contains(&"session".to_string()));
    }

    #[test]
    fn test_cosine_similarity() {
        let mut idf = HashMap::new();
        idf.insert("sample_token".to_string(), 2.0);
        idf.insert("another_token".to_string(), 1.5);

        let (q_vec, q_norm) = compute_term_vector(&["sample_token".to_string(), "another_token".to_string()], &idf);
        let (d_vec, d_norm) = compute_term_vector(&["sample_token".to_string(), "another_token".to_string(), "other".to_string()], &idf);

        let sim = cosine_similarity(&q_vec, q_norm, &d_vec, d_norm);
        assert!(sim > 0.7);
    }
}
