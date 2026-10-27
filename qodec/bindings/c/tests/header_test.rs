//! Assert the checked-in C header matches what cbindgen produces from the
//! current source, so a changed signature cannot silently diverge from the
//! header shipped to C consumers.

use std::path::Path;
use std::process::Command;

#[test]
fn header_is_in_sync_with_the_source() {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let checked_in = manifest.join("include/qodec.h");
    let generated = generate_header(manifest);
    let existing = std::fs::read_to_string(&checked_in).expect("read the checked-in header");
    assert_eq!(
        normalize(&existing),
        normalize(&generated),
        "include/qodec.h is stale; regenerate it with bindings/c/regenerate.sh"
    );
}

fn generate_header(manifest: &Path) -> String {
    let cbindgen = Command::new("cbindgen")
        .arg("--config")
        .arg(manifest.join("cbindgen.toml"))
        .arg("--crate")
        .arg("qodec-c")
        .arg(manifest)
        .output();

    // cbindgen is a documented prerequisite. Skipping here reported green while
    // the header drifted, and `cargo test` swallows the explanation.
    let output =
        cbindgen.expect("cbindgen is required by this test; install it with `cargo install cbindgen --locked`");
    assert!(
        output.status.success(),
        "cbindgen failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    String::from_utf8(output.stdout).expect("cbindgen emits UTF-8")
}

/// Compare ignoring trailing whitespace and blank-line differences.
fn normalize(header: &str) -> String {
    header
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}
