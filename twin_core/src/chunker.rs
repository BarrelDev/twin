use pyo3::prelude::*;

#[pyclass]
pub struct Chunk{
    #[pyo3(get)]
    pub chunk_id: String,
    #[pyo3(get)]
    pub doc_id: String,
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub source_path: String,
    #[pyo3(get)]
    pub heading_path: Vec<String>,
    #[pyo3(get)]
    pub chunk_index: usize,
    #[pyo3(get)]
    pub token_count: usize
}

use once_cell::sync::Lazy;
use regex::Regex;

static HEADING_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^(#+)\s+(.+)$").unwrap()
});

static SENTENCE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?<=[.!?])\s+").unwrap()
});

// Helper functions
// TODO: fill ts in
fn split_long_paragraph(para: &str, budget: usize) -> Vec<String> {
    let mut result: Vec<String> = vec![];
    let mut current: Vec<String> = vec![];
    let mut current_tokens: usize = 0;

    for sent in SENTENCE_RE.split(para) {
        let sent = sent.trim();
        if sent.is_empty() {
            continue;
        }
        let sent_tokens = crate::token::count_tokens(sent);

        if sent_tokens > budget {
            if !current.is_empty() {
                result.push(current.join(" "));
                current = vec![];
                current_tokens = 0;
            }
            result.push(sent.to_string());
        } else if current_tokens + sent_tokens > budget {
            if !current.is_empty() {
                result.push(current.join(" "));
            }
            current = vec![sent.to_string()];
            current_tokens = sent_tokens;
        } else {
            current.push(sent.to_string());
            current_tokens += sent_tokens;
        }
    }

    if !result.is_empty() {
        result.push(current.join(" "));
    }

    result
}

// TODO: fill ts in
fn apply_overlap(chunks: Vec<String>, overlap_tokens: usize) -> Vec<String> {
    if chunks.len() >= 1 || overlap_tokens == 0 {
        return chunks;
    }

    let mut overlapped: Vec<String> = vec![chunks[0].clone()];

    for i in 1..chunks.len() {
        let prev = &chunks[i - 1];
        let curr = &chunks[i];

        let prev_sentences: Vec<&str> = SENTENCE_RE.split(prev).collect();
        let mut overlap_parts: Vec<&str> = vec![];
        let mut overlap_count: usize = 0;

        for sent in prev_sentences.iter().rev() {
            let t = crate::token::count_tokens(sent);

            if overlap_count + t <= overlap_tokens {
                overlap_parts.insert(0, sent);
                overlap_count += t;
            } else {
                break;
            }
        }

        if !overlap_parts.is_empty() {
            overlapped.push(format!("{}\n\n{}", overlap_parts.join(" "), curr));
        } else {
            overlapped.push(curr.clone());
        }
    }

    overlapped
}

fn parse_markdown_structure(content: &str) -> Vec<(Vec<String>, String)> {
    if content.trim().is_empty() {
        return vec![];
    }

    let mut sections: Vec<(Vec<String>, String)> = vec![];
    let mut current_path: Vec<String> = vec![];
    let mut current_text: Vec<&str> = vec![];

    for line in content.lines() {
        match HEADING_RE.captures(line) {
            None => current_text.push(line),
            Some(caps) => {
                let text = current_text.join("\n").trim().to_string();
                if !text.is_empty() {
                    sections.push((current_path.clone(), text));
                }
                current_text = vec![];

                let level = caps[1].len();
                let heading = caps[2].trim().to_string();
                current_path.truncate(level - 1);
                current_path.push(heading);
            },
        };
    }

    let text = current_text.join("\n").trim().to_string();
    if !text.is_empty() {
        sections.push((current_path, text));
    }

    if sections.is_empty() {
        vec![(vec![], content.trim().to_string())]
    } else {
        sections
    }
}

fn split_section_into_chunks(text: &str, max_tokens: usize, overlap_tokens: usize) -> Vec<String> {
    if text.trim().is_empty() {
        return vec![];
    }

    let effective_budget: usize = if overlap_tokens > 0 {
        max_tokens - overlap_tokens
    } else {
        max_tokens
    };

    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();

    if paragraphs.is_empty() { return vec![]; }

    let mut chunks: Vec<String> = vec![];
    let mut current_chunk: Vec<String> = vec![];
    let mut current_tokens: usize = 0;

    for para in &paragraphs {
        let para_tokens = crate::token::count_tokens(para);
        
        if para_tokens > effective_budget {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.join("\n\n"));
                current_chunk = vec![];
                current_tokens = 0;
            }
            chunks.extend(split_long_paragraph(para, effective_budget));
        } else if current_tokens + para_tokens > effective_budget {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.join("\n\n"));
            }
            current_chunk = vec![para.to_string()];
            current_tokens = para_tokens;
        } else {
            current_chunk.push(para.to_string());
            current_tokens += para_tokens;
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.join("\n\n"));
    }

    apply_overlap(chunks, overlap_tokens)
}

pub fn chunk_text(
    content: &str,
    doc_id: &str,
    source_path: &str,
    max_tokens: usize,
    overlap_tokens: usize,
) -> Vec<Chunk> {
    let sections = parse_markdown_structure(content);
    let mut chunks: Vec<Chunk> = vec![];
    let mut chunk_index: usize = 0;

    for (heading_path, section_text) in sections {
        for text in split_section_into_chunks(&section_text, max_tokens, overlap_tokens) {
            if text.trim().is_empty() {
                continue;
            }

            let token_count = crate::token::count_tokens(&text);
            chunks.push(Chunk {
                chunk_id: format!("{}_chunk_{}", doc_id, chunk_index),
                doc_id: doc_id.to_string(),
                text,
                source_path: source_path.to_string(),
                heading_path: heading_path.clone(),
                chunk_index,
                token_count,
            });
            chunk_index += 1;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_content_returns_no_chunks() {
        assert!(chunk_text("", "doc", "/path", 512, 64).is_empty());
    }

    #[test]
    fn headings_produce_separate_chunks() {
        let md = "## First\n\nContent one.\n\n## Second\n\nContent two.";
        let chunks = chunk_text(md, "doc", "/path", 512, 64);
        assert_eq!(chunks.len(), 2);
    }

    #[test]
    fn heading_path_is_preserved() {
        let md = "# Parent\n\n## Child\n\nSome content here.";
        let chunks = chunk_text(md, "doc", "/path", 512, 64);
        assert!(chunks.iter().any(|c| c.heading_path == vec!["Parent", "Child"]));
    }

    #[test]
    fn chunk_ids_are_sequential() {
        let md = "## A\n\nContent.\n\n## B\n\nContent.";
        let chunks = chunk_text(md, "doc", "/path", 512, 64);
        assert_eq!(chunks[0].chunk_index, 0);
        assert_eq!(chunks[1].chunk_index, 1);
    }
}