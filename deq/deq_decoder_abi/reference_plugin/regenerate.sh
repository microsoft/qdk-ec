#!/usr/bin/env bash
# Regenerate the deq decoder plugin C header from the reference plugin.
#
# cbindgen must expand the declare_decoder! macro to see the exported symbols,
# which requires nightly's -Zunpretty=expanded (RUSTC_BOOTSTRAP=1 enables it on
# the pinned toolchain). The header-sync test runs this same command.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="$here/../include/deq_decoder.h"

RUSTC_BOOTSTRAP=1 cbindgen \
    --config "$here/cbindgen.toml" \
    --crate deq-decoder-reference-plugin \
    --output "$out"

echo "wrote $out"
