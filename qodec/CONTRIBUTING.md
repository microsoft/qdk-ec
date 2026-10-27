# Contributing to qodec

qodec is in early development (preview). Schemas and APIs are still evolving;
each pre-1.0 minor release may introduce breaking changes.

Follow the parent [contribution guide](../CONTRIBUTING.md) for issues, pull
requests, the contributor license agreement, and the code of conduct.
The parent [support policy](../SUPPORT.md) applies. For qodec issues, include
the relevant YAML snippets or error messages in addition to reproduction steps.

## Building and Testing

qodec has a Rust model, Python bindings, and a read-only C ABI. Run the following
from the **qodec directory**, the one containing `examples/` and `tools/`. Use
the Python environment selected for your work.

```bash
python tools/check.py all --dry-run
python tools/check.py all
```

The [verification runner](tools/README.md) covers Rust/C, Python types and tests,
documentation, coverage, and example audits. The example scope requires a
compatible development QDK with `ec` dependencies. It is stricter than the
current source CI, which does not yet install that audit engine. Do not count
CI's binding tests as evidence of example audits.

For a focused change, choose a runner scope (`rust`, `python`, `docs`,
`coverage`, `examples`, or `packaging`). See [prerequisites](../.github/instructions/qodec-checks.instructions.md#environment)
for compiler, header-generation, and coverage tools. A focused core test can use
`cargo test -p qodec --test round_trip_test`.

## Documentation

Model concepts and the YAML format live in [docs/](docs/README.md). Language
guidance lives with the corresponding API: crate-level rustdoc in
[src/lib.rs](src/lib.rs), Python usage and stub-generated reference in
[bindings/python/docs/](bindings/python/docs/index.rst), and the
[C binding guide](bindings/c/README.md) with its generated header.

Build locally from the qodec directory, using Python 3.11 or newer with the
Python binding installed for its examples:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc -p qodec -p qodec-python -p qodec-c --no-deps
cargo test -p qodec --doc
python -m pip install -r bindings/python/docs/requirements.txt
python -m sphinx -W --keep-going -b html bindings/python/docs target/python-docs/html
python -m sphinx -W --keep-going -b doctest bindings/python/docs target/python-docs/doctest
python bindings/python/docs/test_docs.py
```

Open `../target/doc/qodec/index.html` for Rust or
`target/python-docs/html/index.html` for Python. The C reference is
[bindings/c/include/qodec.h](bindings/c/include/qodec.h); regenerate it from the
Rust binding source as described in the C guide.

Python API pages read the `.pyi` signatures and docstrings without importing the
native extension. PyO3 constructors are documented as `__new__` methods. Keep
runtime docstrings useful for `help()`, and run mypy and stubtest after stub edits.
CI builds both documentation sets with warnings as errors and runs the Python
examples and generated-reference checks on Python 3.12. Generated files are not
checked in.

## Code Standards

Rust:

- Code must pass `python tools/check.py rust`, which formats, lints, and tests
  `qodec`, `qodec-python`, and `qodec-c` without including sibling packages.
  Clippy uses the parent's `--all-targets --all-features -- -D clippy::pedantic`
  policy. The C library is built before its tests.
- New features should include tests
- Public APIs should be documented

Python bindings:

- Expose a Pythonic API
- Keep the `.pyi` stub files in sync with the extension (run `mypy`)
- Add tests under `bindings/python/tests/`

On-disk format changes:

- Any breaking change to an artifact format must bump `schema_version` and be
  recorded in [`CHANGELOG.md`](CHANGELOG.md). See that file for the
  compatibility contract.

## Releasing

See [RELEASING.md](RELEASING.md) for versioning and release verification.
