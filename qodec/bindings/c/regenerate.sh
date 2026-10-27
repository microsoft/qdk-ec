#!/usr/bin/env bash
# Regenerate the checked-in C header from the Rust source.
#
# Run after changing any `extern "C"` signature or status constant in
# src/lib.rs. `header_is_in_sync` in tests/header_test.rs fails if you forget.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v cbindgen >/dev/null 2>&1; then
    echo "cbindgen not found; install it with: cargo install cbindgen --locked" >&2
    exit 1
fi

mkdir -p include
cbindgen --config cbindgen.toml --crate qodec-c --output include/qodec.h
echo "regenerated include/qodec.h"
