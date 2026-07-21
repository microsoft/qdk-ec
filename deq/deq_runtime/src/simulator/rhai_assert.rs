//! Rhai scripting support for custom logical readout assertions.
//!
//! When `logical_assert_filepath` is set in the simulator config, or when a
//! Stim circuit file contains embedded `#!rhai` blocks, a Rhai script is
//! compiled once and its `fn is_logical_error(shot, readouts,
//! measurements)` is called for each shot.  The function
//! must return a `bool` — `true` means a logical error was detected.
//!
//! ## Embedding scripts in Stim files
//!
//! ```text
//! #!rhai
//! # fn is_logical_error(shot, readouts, measurements) {
//! #     readouts[0] != 0
//! # }
//! H 0
//! CNOT 0 1
//! M 0 1
//! ```
//!
//! Lines after `#!rhai` that start with `#` are collected as script code (the
//! leading `#` and one optional space are stripped).  The block ends at the
//! first non-`#` line.  Multiple `#!rhai` blocks are concatenated.
//!
//! Rhai's built-in `print` function writes to stdout, so users can emit debug
//! information from within their scripts.

use crate::misc::bit_vector;
use crate::util::BitVector;

/// A pre-compiled Rhai assertion engine.
///
/// The engine compiles the user script once at construction time and calls the
/// script's `is_logical_error` function for each shot.
pub struct RhaiAssertEngine {
    engine: rhai::Engine,
    ast: rhai::AST,
}

impl RhaiAssertEngine {
    /// Create a new engine by reading and compiling the script at `script_path`.
    pub fn new(script_path: &str) -> Self {
        let script = std::fs::read_to_string(script_path)
            .unwrap_or_else(|e| panic!("Failed to read Rhai script '{script_path}': {e}"));
        Self::from_source(script_path, &script)
    }

    /// Create a new engine by compiling a script from a string.
    ///
    /// `name` is used in error messages (e.g. `"embedded in circuit.stim"`).
    pub fn from_source(name: &str, source: &str) -> Self {
        let engine = rhai::Engine::new();
        let ast = engine
            .compile(source)
            .unwrap_or_else(|e| panic!("Failed to compile Rhai script '{name}': {e}"));
        Self { engine, ast }
    }

    /// Build a ``RhaiAssertEngine`` from the available script sources.
    ///
    /// Priority: ``logical_assert_filepath`` takes precedence over an
    /// embedded ``#!rhai`` block in the Stim file.  Panics with a helpful
    /// example if neither is provided.
    pub fn build(stim_filepath: &str, embedded_rhai_script: Option<&str>, logical_assert_filepath: Option<&str>) -> Self {
        if let Some(path) = logical_assert_filepath {
            return Self::new(path);
        }
        if let Some(script) = embedded_rhai_script {
            return Self::from_source(&format!("embedded in {stim_filepath}"), script);
        }
        panic!(
            "No logical assertion script found.  Either embed a #!rhai block \n\
             in your Stim circuit file or set `logical_assert_filepath` in the \n\
             simulator config.\n\
             \n\
             Example #!rhai block in a Stim file:\n\
             \n\
             #!rhai\n\
             # fn is_logical_error(shot, readouts, measurements) {{\n\
             #     readouts[0] != 0\n\
             # }}\n\
             \n\
             Or set in config JSON:\n\
             \n\
             {{\"logical_assert_filepath\": \"path/to/assert.rhai\"}}"
        )
    }

    /// Call the user's `is_logical_error` function with the shot context.
    ///
    /// Arguments passed to the script function:
    /// - `shot` (`i64`): current shot index
    /// - `readouts` (`Array` of `bool`): decoded readout bits
    /// - `measurements` (`Array` of `bool`): physical measurement outcomes
    ///
    /// Returns `true` if the script reports a logical error.
    pub fn is_logical_error(&self, shot: usize, readouts: Option<&BitVector>, measurements: &BitVector) -> bool {
        let ast = &self.ast;
        let readouts_arr: rhai::Array = match readouts {
            Some(r) => bit_vector::unpack_bits(&r.data, r.size)
                .into_iter()
                .map(rhai::Dynamic::from)
                .collect(),
            None => rhai::Array::new(),
        };
        let measurements_arr: rhai::Array = bit_vector::unpack_bits(&measurements.data, measurements.size)
            .into_iter()
            .map(rhai::Dynamic::from)
            .collect();

        let mut scope = rhai::Scope::new();
        let result: bool = self
            .engine
            .call_fn(
                &mut scope,
                ast,
                "is_logical_error",
                (shot as rhai::INT, readouts_arr, measurements_arr),
            )
            .unwrap_or_else(|e| {
                panic!(
                    "Rhai is_logical_error() failed at shot {shot}: {e}\n\
                     Hint: the function signature should be:\n  \
                     fn is_logical_error(shot, readouts, measurements) {{ ... }}"
                )
            });
        result
    }
}

/// Extract embedded Rhai script from a Stim circuit file's text.
///
/// Scans for `#!rhai` markers on their own line.  Subsequent lines starting
/// with `#` are collected as script code — the leading `#` and one optional
/// space are stripped.  The block ends at the first non-`#` line (or EOF).
/// Multiple `#!rhai` blocks are concatenated with newlines.
///
/// Returns `None` if no `#!rhai` block is found.
pub fn extract_rhai_script(stim_text: &str) -> Option<String> {
    let mut script = String::new();
    let mut in_rhai_block = false;

    for line in stim_text.lines() {
        let trimmed = line.trim();
        if trimmed == "#!rhai" {
            in_rhai_block = true;
            continue;
        }
        if in_rhai_block {
            if let Some(rest) = trimmed.strip_prefix('#') {
                // Strip one optional space after the '#'
                let code = rest.strip_prefix(' ').unwrap_or(rest);
                script.push_str(code);
                script.push('\n');
            } else {
                in_rhai_block = false;
            }
        }
    }

    if script.is_empty() { None } else { Some(script) }
}

/// Return the Stim circuit text with every `#!rhai` script block removed.
///
/// The `#!rhai` marker line and every subsequent `#`-prefixed line up to the
/// first non-`#` line (or EOF) are dropped.  The result is a Stim circuit safe
/// to hand to a third-party Stim parser (e.g. QDK's) that doesn't recognize
/// deq's `#!rhai` extension — the parser would otherwise treat `#!rhai` as an
/// unknown instruction name.
///
/// This is the mirror of [`extract_rhai_script`]: what `extract_rhai_script`
/// consumes is what `strip_rhai_scripts` drops.
pub fn strip_rhai_scripts(stim_text: &str) -> String {
    let mut out = String::with_capacity(stim_text.len());
    let mut in_rhai_block = false;
    for line in stim_text.lines() {
        let trimmed = line.trim();
        if trimmed == "#!rhai" {
            in_rhai_block = true;
            continue;
        }
        if in_rhai_block {
            if trimmed.starts_with('#') {
                continue;
            }
            in_rhai_block = false;
        }
        out.push_str(line);
        out.push('\n');
    }
    out
}
