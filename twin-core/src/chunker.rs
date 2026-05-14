use std::env::consts::EXE_SUFFIX;

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
    vec![] 
}

// TODO: fill ts in
fn apply_overlap(chunks: Vec<String>, overlap_tokens: usize) -> Vec<String> {
    vec![]
}

fn parse_markdown_structure(content: String) -> Vec<(Vec<String>, String)> {
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
        } else if current_token + para_tokens > effective_budget {
            if !current_chunk.is_empty() {
                chunks.push(current_chunk.join("\n\n"));
            }
            current_chunk = vec![para.to_string()];
            current_tokens = para_tokens;
        } else {
            current_chunk.push(para.to_string());
            current_chunk += para_tokens;
        }
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk.join("\n\n"));
    }

    apply_overlap(chunks, overlap_tokens)
}