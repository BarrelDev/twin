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

#[pymodule]
fn twin_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<chunker::Chunk>()?;
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    Ok(())
}