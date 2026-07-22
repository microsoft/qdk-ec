//! Parse and roundtrip each `.deq` file passed on the command line.
//!
//! Usage: `cargo run --example parse-deq -- [--debug] path/to/a.deq ...`
//!
//! For each file it parses to a [`DeqFile`], displays it, re-parses, and checks
//! the two ASTs are equal. Reports any parse or roundtrip failure.
//!
//! With `--debug` (or `-d`), the full parsed AST of each file is printed to
//! stdout via `{:#?}` (diagnostics stay on stderr).

use std::process::ExitCode;

use deqagram::ast::DeqFile;

fn main() -> ExitCode {
    let mut debug = false;
    let mut paths = Vec::new();
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-d" | "--debug" => debug = true,
            other if other.starts_with('-') => {
                eprintln!("unknown flag: {other}");
                eprintln!("usage: parse-deq [--debug] <file.deq>...");
                return ExitCode::FAILURE;
            }
            _ => paths.push(arg),
        }
    }

    let mut failures = 0;
    let total = paths.len();
    for path in paths {
        let src = std::fs::read_to_string(&path).expect("read file");
        let ast: DeqFile = match src.parse() {
            Ok(ast) => ast,
            Err(e) => {
                failures += 1;
                eprintln!("PARSE FAIL {path}\n{e}\n");
                continue;
            }
        };
        if debug {
            println!("// {path}");
            println!("{ast:#?}");
        }
        let serialized = ast.to_string();
        match serialized.parse::<DeqFile>() {
            Ok(reparsed) if reparsed == ast => {}
            Ok(_) => {
                failures += 1;
                eprintln!("ROUNDTRIP MISMATCH {path}");
            }
            Err(e) => {
                failures += 1;
                eprintln!("REPARSE FAIL {path}\n{e}\n");
            }
        }
    }
    eprintln!("{}/{} ok", total - failures, total);
    if failures == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}
