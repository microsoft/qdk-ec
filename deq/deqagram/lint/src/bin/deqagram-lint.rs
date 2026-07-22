//! `deqagram-lint`: parse a `.deq` file and report semantic problems.
//!
//! Usage: `deqagram-lint <file.deq>...`. Prints one diagnostic per line as
//! `file:line:col: severity[rule]: message`. Exits non-zero if any file fails to
//! parse or produces an error-severity diagnostic.

use std::path::Path;
use std::process::ExitCode;

use deqagram::ast::DeqFile;
use deqagram_lint::{Severity, lint};

fn main() -> ExitCode {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("usage: deqagram-lint <file.deq>...");
        return ExitCode::from(2);
    }

    let mut had_error = false;
    for path in &paths {
        had_error |= lint_file(path);
    }

    if had_error {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

/// Lints one file, printing diagnostics. Returns `true` if the file failed to
/// parse or had any error-severity diagnostic.
fn lint_file(path: &str) -> bool {
    let source = match std::fs::read_to_string(path) {
        Ok(source) => source,
        Err(e) => {
            eprintln!("{path}: cannot read file: {e}");
            return true;
        }
    };

    let file: DeqFile = match source.parse() {
        Ok(file) => file,
        Err(e) => {
            // deqagram's parse error already carries a line/column and message.
            eprintln!("{path}: {e}");
            return true;
        }
    };

    let name = Path::new(path).display();
    let mut had_error = false;
    for diagnostic in lint(&file) {
        let (line, col) = diagnostic.span.line_col(&source).unwrap_or((0, 0));
        println!(
            "{name}:{line}:{col}: {}[{}]: {}",
            diagnostic.severity, diagnostic.rule, diagnostic.message
        );
        had_error |= diagnostic.severity == Severity::Error;
    }
    had_error
}
