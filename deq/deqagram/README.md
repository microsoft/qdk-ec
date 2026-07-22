# deqagram

`deqagram` provides a [pest](https://pest.rs/)-based parser and a typed AST for
the [deq](https://github.com/microsoft/qdk-ec) quantum error correction file
format:

- **`.deq`** — declarative QEC code, gadget, compose, and program definitions.

Files are parsed into a typed AST via `FromStr` and serialized back via
`Display`, with **roundtrip fidelity** as a core invariant: parsing a file and
displaying it produces output that re-parses to the same AST.

## Architecture

The crate (`src/`) has three layers:

1. **PEG grammar** (`src/deq.pest`) — the concrete syntax, ported from deq's own
   Lark grammar (`deq/circuit/deq.lark` in the deq repo), the single source of
   truth.
2. **Parser struct** (`src/lib.rs`) — a thin `#[derive(Parser)]` wrapper around
   the grammar (`DeqParser`). Parser-level tests live in `src/parser_tests.rs`.
3. **AST** (`src/ast.rs`) — a typed representation built from pest pairs via
   `FromStr`, serialized back via `Display`, and mirroring deq's `model.py`.
   Source spans live in `src/span.rs` and the error type in `src/error.rs`.

The grammar is whitespace- and newline-insensitive, so roundtrip fidelity is at
the AST level (parse → display → re-parse), not byte-for-byte; comments are not
retained. Following deq's transformer, the `CHECK`/`DETECTOR` and
`READOUT`/`OBSERVABLE_INCLUDE` aliases are conflated into single node kinds.
Mako-templated `.deq` files are out of scope: deq renders Mako before its own
Lark parse.

## Installation

This crate is published to the private **`errata`** Azure Artifacts registry
(not crates.io). Configure the registry in `.cargo/config.toml`:

```toml
[registries]
errata = { index = "sparse+https://pkgs.dev.azure.com/ms-quantum/_packaging/errata/Cargo/index/" }
```

Then depend on it from that registry:

```toml
[dependencies]
deqagram = { version = "0.1", registry = "errata" }
```

## Usage

Parse a `.deq` file, inspect the AST, and serialize it back:

```rust
use deqagram::ast::DeqFile;

let input = "\
CODE RepetitionCode [[3,1,3]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}
";
let file: DeqFile = input.parse().unwrap();

// Round-trips: displaying and re-parsing yields the same AST.
let serialized = file.to_string();
assert_eq!(file, serialized.parse().unwrap());
```

### Runnable example

Parse and roundtrip-check one or more `.deq` files:

```sh
cargo run --example parse-deq -- path/to/a.deq path/to/b.deq
```

## Development

```sh
cargo test                                      # all tests
cargo clippy --all-targets -- -W clippy::pedantic
cargo +nightly fmt                              # format (nightly: see rustfmt.toml)
```

See [HACKING.md](HACKING.md) for full contributor setup, the CI pipelines, and
the release process. Architecture notes and code conventions live in
[AGENTS.md](AGENTS.md).

This project uses Rust **edition 2024**.
