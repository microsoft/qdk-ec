#!/usr/bin/env bash
# Coverage for the PyO3 wrapper crate, driven by the Python test suite.
#
# `cargo llvm-cov` alone will not do this: it reports the workspace's own
# targets, and the wrappers only ever execute inside the extension module that
# Python loads. So the extension is built instrumented, pytest drives it, and
# the profile is read back against that exact `.so`.
#
# The trap worth naming: `llvm-cov` reports 0% for every file when the profile
# and the binary do not match, with no error. If you merge profraw from a
# directory that also holds `cargo test` profiles, that is what you get. This
# script writes to a dedicated directory and merges only from there.
#
# Usage:  tools/binding-coverage.sh
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT=$(pwd)
PROFILES="$ROOT/target/binding-coverage"
SO="$ROOT/bindings/python/python/qodec/_native.abi3.so"

LLVM_BIN="$(dirname "$(rustc --print target-libdir)")/bin"
if [[ ! -x "$LLVM_BIN/llvm-profdata" || ! -x "$LLVM_BIN/llvm-cov" ]]; then
    echo "error: llvm-profdata not found. Install it with: rustup component add llvm-tools-preview" >&2
    exit 1
fi

rm -rf "$PROFILES"
mkdir -p "$PROFILES"

# show-env exports the instrumentation RUSTFLAGS that maturin's cargo run picks
# up, plus LLVM_PROFILE_FILE. Override the latter so nothing else lands here.
eval "$(cargo llvm-cov show-env --sh)"
export LLVM_PROFILE_FILE="$PROFILES/qodec-%p-%16m.profraw"

echo "==> building the extension with coverage instrumentation"
(cd bindings/python && maturin develop >/dev/null)

echo "==> running the Python suite against it"
(cd bindings/python && python -m pytest -q)

count=$(find "$PROFILES" -name '*.profraw' | wc -l)
if [[ "$count" -eq 0 ]]; then
    echo "error: no profile written -- the .so pytest loaded was not the instrumented one" >&2
    exit 1
fi
echo "==> $count profile(s) collected"

"$LLVM_BIN/llvm-profdata" merge -sparse "$PROFILES"/*.profraw -o "$PROFILES/merged.profdata"
"$LLVM_BIN/llvm-cov" report \
    --instr-profile="$PROFILES/merged.profdata" \
    "$SO" \
    "$ROOT"/bindings/python/src/*.rs \
    "$@"

cat <<'EOF'

Note: the extension is left as an instrumented debug build. Restore the normal
one before benchmarking or shipping:

    cd bindings/python && maturin develop --release
EOF
