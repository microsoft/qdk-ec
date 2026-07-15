//! Verifies the checked-in C header is in sync with the Rust ABI.
//!
//! Regenerates the header with cbindgen (the same `regenerate.sh` invocation) into a
//! temporary file and asserts it byte-matches `../include/deq_decoder.h`. If this
//! fails, the ABI changed without regenerating the header: run
//! `deq/deq_decoder_abi/reference_plugin/regenerate.sh` and commit the result.
//!
//! Requires the `cbindgen` binary on PATH (CI installs it). The header generation
//! needs nightly macro expansion, enabled via `RUSTC_BOOTSTRAP=1`.

use std::path::PathBuf;
use std::process::Command;

#[test]
fn header_matches_rust_abi() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let config = manifest_dir.join("cbindgen.toml");
    let committed = manifest_dir.join("../include/deq_decoder.h");

    // shell out to the cbindgen binary instead of using it as a library dep,
    // because header generation needs `--parse.expand` (a nested cargo
    // build) anyway, so the binary is the simplest faithful reproduction of
    // regenerate.sh.
    let output = Command::new("cbindgen")
        .env("RUSTC_BOOTSTRAP", "1")
        .arg("--config")
        .arg(&config)
        .arg("--crate")
        .arg("deq-decoder-reference-plugin")
        .output();

    let output =
        output.unwrap_or_else(|e| panic!("failed to run `cbindgen` (install with `cargo install cbindgen`): {e}"));
    assert!(
        output.status.success(),
        "cbindgen failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let generated = String::from_utf8(output.stdout).expect("cbindgen output is UTF-8");
    let committed_text = std::fs::read_to_string(&committed).expect("read committed header");

    assert_eq!(
        generated.replace("\r\n", "\n").trim_end(),
        committed_text.replace("\r\n", "\n").trim_end(),
        "{} is out of sync with the Rust ABI; regenerate with reference_plugin/regenerate.sh",
        committed.display()
    );
}
