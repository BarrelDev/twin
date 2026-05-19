mod chunker;
mod token;

use pyo3::prelude::*;

#[pyfunction]
fn chunk_text(
    content: &str,
    doc_id: &str,
    source_path: &str,
    max_tokens: usize,
    overlap_tokens: usize,
) -> Vec<chunker::Chunk> {
    chunker::chunk_text(content, doc_id, source_path, max_tokens, overlap_tokens)
}

#[pyfunction]
fn count_tokens(text: &str) -> usize {
    token::count_tokens(text)
}

#[pyfunction]
fn parse_markdown_structure(content: &str) -> Vec<(Vec<String>, String)> {
    chunker::parse_markdown_structure(content)
}

#[pyfunction]
fn split_section_into_chunks(
    text: &str,
    max_tokens: usize,
    overlap_tokens: usize,
) -> Vec<String> {
    chunker::split_section_into_chunks(text, max_tokens, overlap_tokens)
}

#[pymodule]
fn twin_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<chunker::Chunk>()?;
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(parse_markdown_structure, m)?)?;
    m.add_function(wrap_pyfunction!(split_section_into_chunks, m)?)?;
    Ok(())
}