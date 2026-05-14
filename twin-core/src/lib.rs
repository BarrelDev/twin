mod chunker;
mod token;

use pyo3::prelude::*;

#[pymodule]
fn twin_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chunker::chunk_text, m)?)?;
    Ok(())
}