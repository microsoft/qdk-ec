use std::path::{Path, PathBuf};

/// Every shipped example manifest, found by walking `examples/`.
///
/// Derived rather than listed: a hardcoded list silently skips a new example
/// from round-trip and schema validation. The Python audit suite walks the same
/// directory and asserts its own list matches.
pub(crate) fn example_manifests() -> Vec<PathBuf> {
    let mut manifests = Vec::new();
    collect_manifests(&Path::new(env!("CARGO_MANIFEST_DIR")).join("examples"), &mut manifests);
    manifests.sort();
    manifests
}

fn collect_manifests(directory: &Path, found: &mut Vec<PathBuf>) {
    let entries = std::fs::read_dir(directory).unwrap_or_else(|error| panic!("read {}: {error}", directory.display()));
    for entry in entries {
        let path = entry.expect("read example entry").path();
        if path.is_dir() {
            collect_manifests(&path, found);
        } else if path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().ends_with("qodec.yaml"))
        {
            found.push(path);
        }
    }
}
