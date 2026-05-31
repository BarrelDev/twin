@echo off
where cargo >nul 2>&1 || (
    echo Rust is required. Install it at https://rustup.rs
    exit /b 1
)
uv sync || exit /b 1
pushd twin_core
uv run maturin develop || exit /b 1
popd
echo Twin is ready. Run: uv run twin --help