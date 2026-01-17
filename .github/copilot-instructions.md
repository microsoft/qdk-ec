# Copilot Instructions

## Code Style

- This is a quantum computing library focused on Pauli operators, Clifford gates, and stabilizer simulation.
- The binar crate provides bit manipulation primitives; paulimer builds on it for Pauli algebra. The pauliverse crate provides stabilizer simulation; synthesis handles circuit synthesis.
- Use idiomatic Rust with proper error handling using Result types
- Rust code should be compliant with `clippy --all-targets --all-features -- -D clippy::pedantic`
- Follow the existing code style and patterns in this workspace
- For Python bindings, use PyO3 macros consistently with existing code
- Prefer const generics and compile-time checks where applicable
- Prioritize runtime performance and memory efficiency in Rust implementations. Speed is a critical feature. But remember that the best performance usually comes from simplicity and clarity.
- Use proptest or hypothesis for property-based testing where appropriate
- Python bindings should expose a Pythonic API, not just a direct Rust translation
- Use #[pyo3(name = "...")] to provide Python-friendly naming where Rust conventions differ
- When adding or otherwise modifying Python bindings, ensure that the .pyi files are updated accordingly.
- To build and install the Python bindings, use `maturin develop --release` from the `bindings/python/` directory for the corresponding crate.
- Before implementing a new feature, check how similar types in the codebase already handle it — consistency with existing patterns is valuable
- Use Criterion for rust benchmarks (see existing benchmarks in benches/ folders), and use asv for Python benchmarks.
- Benchmarks used to compare packages should typically use deterministic inputs for reproducibility
- When adding new functionality, include relevant unit tests and documentation.
- Avoid inline commments. They are usually redundant and often wrong. Prefer clear code and good variable names instead.
- Avoid abbreviations and single-letter variable names, unless they very clearly improve clarity.