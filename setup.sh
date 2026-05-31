#!/bin/bash
if ! command -v cargo &> /dev/null; then
    echo "Rust is required. Install it at https://rustup.rs"
    exit 1
fi
set -e

uv sync
cd twin_core && uv run maturin develop && cd ..
echo "Twin is ready. Run: uv run twin --help"