---
description: 'Use before qodec builds, tests, lint, type checking, stubtest, coverage, documentation checks, CI debugging, or pre-merge verification. Includes environment prerequisites and working directories.'
---

# qodec Checks

[The parent build workflow](../workflows/build.yaml) runs qodec's
verification job on Python 3.11 and 3.12, with coverage, language docs, and
packaging checked on the 3.12 leg. These gates invoke
[tools/check.py](../../qodec/tools/check.py). The parent's cross-platform workspace
tests also cover qodec's Rust crates and build the C library first.
[The wheels workflow](../workflows/qodec-wheels.yaml) builds and
imports native Linux, Windows, and universal2 macOS wheels without publishing.
The parent [Azure build stage](../../.ado/stages/build.yaml) builds and
probes qodec wheels on all six native platforms, plus an sdist on Linux x86_64.
Publication uses the parent's [manual ESRP pipeline](../../.ado/publish.yaml),
as described in [RELEASING.md](../../qodec/RELEASING.md); do not run it to check code.
Azure retains a `<platform>-rust-timings` artifact from the workspace test build,
including after test failures. Use that report to investigate compile time before
changing release optimization settings.

The `packaging` scope discovers every `tools/test_*.py` test, including runner
and formatter tests. Keep qodec-specific verification commands in the runner;
workflow files own environment setup and platform selection. Match the parent
Clippy policy (`-D clippy::pedantic`); do not add a workspace lint table.
Packaging tests require `packaging` for Rust-to-PEP-440 version comparison.
Native-wheel validation invokes `tools/check_wheel.py` to install and probe the
wheel outside the checkout. Azure runs that probe on native ARM64 hosts as well.

## Repeatable Commands

Use the existing selected interpreter to run [tools/check.py](../../qodec/tools/check.py)
with scope `rust`, `python`, `docs`, `coverage`, `examples`, `packaging`, or `all`.
For example, from the qdk-ec repository root:

```bash
python qodec/tools/check.py python
python qodec/tools/check.py all --dry-run
```

Replace `python` with the selected executable, not an assumed terminal alias.
The runner sets child-process working directories and builds against the invoking
interpreter, leaving the rest of the environment alone. It prints executable paths,
verifies the rebuilt package imports from this checkout, and stops at the first
failed gate. `--dry-run` prints commands without running checks or installing tools.

The root [VS Code tasks](../../.vscode/tasks.json) expose the same scopes using
`${command:python.interpreterPath}`. Prefer the runner or these tasks to recreating
command chains in agents or chat. Neither requires checked-in interpreter settings.
The gate descriptions below explain what runs; individual commands remain available
for focused work.

## Environment

- Check the selected Python interpreter and available compiler/linker before
  running builds. Use the selected Python environment for `maturin`, `mypy`,
  `stubtest`, `pytest`, and Sphinx; do not silently replace it or assume a
  machine-specific conda environment. Ensure imports resolve to this checkout.
- Activate the environment you intend to use, or invoke its interpreter directly.
  `maturin develop` installs into the interpreter running it unless `VIRTUAL_ENV`
  or `CONDA_PREFIX` names another one, so avoid conflicting conda and virtualenv
  selections.
- Install native wheel tools from [requirements-build.txt](../../requirements-build.txt)
  using the selected interpreter. Check that `python -m maturin --version` agrees
  with its installed package metadata. Linux Zig builds use the shared
  [adapter](../../tools/zig.py) through `CARGO_ZIGBUILD_PYTHON_PATH`; see
  [native wheel build tools](../../CONTRIBUTING.md#native-wheel-build-tools).
- Display tests use `PyYAML`, `types-PyYAML`, and `ipython`, installed by CI
  alongside `pytest` and `mypy`. These are test tools, not runtime dependencies.
- Binding tests, documentation examples, and example audits need `stim>=1.16,<2`,
  supplied by the optional `qodec[parsers]` extra. The base package has no mandatory
  Stim dependency; missing-dependency tests must still exercise that boundary.
- Rust builds need a working linker, including for build scripts and procedural
  macros during `cargo check` or `clippy`. Python binding builds also need the
  selected Python interpreter available to PyO3.
- The C header consistency test needs `cbindgen`. Coverage needs `cargo-llvm-cov`
  and the Rust `llvm-tools-preview` component. Python docs dependencies are in
  [requirements.txt](../../qodec/bindings/python/docs/requirements.txt).
- The Python binding crate is an extension module with no Rust test target: an
  interpreter loads it, so its behavior is covered by `pytest`. No library-path
  setup is needed for `cargo test`.

## Focused Development Checks

- After a change, run the smallest relevant test or content check first. For
  Rust changes, use the owning crate and test filter; for Python changes, use the
  affected test file. Narrow checks do not replace the full pre-merge gates.
- Compare normalized filesystem paths with Rust `Path` or Python `pathlib.Path`,
  not separator-specific strings. Keep literal comparisons for authored YAML keys
  and preserved source text. Write raw-text fixtures without platform newline
  translation when their exact contents matter.
- Rebuild the Python extension using the build step below before `pytest` or
  `stubtest` when Rust core or binding changes affect it. Follow
  [qodec-python.instructions.md](qodec-python.instructions.md) for stub and
  runtime-value coverage.
- Report exactly which checks ran and any blocked or unverified gates.

### Example Audit

`python qodec/tools/check.py examples` runs `qdk.ec.audit` on every retained example.
It also runs in `all`. Use a compatible QDK build with its `ec` dependencies;
the suite never skips for a missing audit engine. Every error and every new or
increased warning fails. Only the per-example unsupported-verification warnings
documented in [examples/README.md](../../qodec/examples/README.md#correctness-tests)
are allowed. No pipeline installs this development QDK build yet, so a green CI
run is not evidence that the examples passed audit.

## Full Pre-merge Gates

All qodec gates must pass before merge. Run the Rust commands from the qdk-ec
repository root; package selection must not include sibling crates:

```bash
cargo fmt -p qodec -p qodec-python -p qodec-c -- --check
cargo clippy -p qodec -p qodec-python -p qodec-c --all-targets --all-features -- -D clippy::pedantic
cargo build -p qodec-c
cargo test -p qodec -p qodec-python -p qodec-c --all-features
```

Keep all three packages selected. Build the C staticlib before its Linux smoke
tests, using the same profile and target as the tests. The Rust ABI and generated
[header](../../qodec/bindings/c/include/qodec.h) checks run on all CI platforms;
compiled C callers are qualified only on Linux. Python runtime behavior is
tested through `pytest`, not the Rust suite.

From [qodec/bindings/python/](../../qodec/bindings/python), with the Python environment set:

```bash
maturin develop --release
mypy python/qodec tests
python -m mypy.stubtest qodec --allowlist stubtest-allowlist.txt
python -m pytest -q
```

Stubtest must check both missing runtime members and missing stub declarations,
including module exports. Keep its allowlist limited to the documented type-only
aliases; do not suppress all runtime members missing from stubs.

### Language Documentation

From the qodec directory, with the Python docs dependencies installed:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc -p qodec -p qodec-python -p qodec-c --no-deps
python -m sphinx -W --keep-going -b html bindings/python/docs target/python-docs/html
python -m sphinx -W --keep-going -b doctest bindings/python/docs target/python-docs/doctest
python bindings/python/docs/test_docs.py
```

### Coverage

From the qdk-ec repository root, enforce the core-only CI line-coverage floor:

```bash
cargo llvm-cov -p qodec --summary-only --fail-under-lines 88
```

Keep `-p qodec`: sibling crates and the bindings must not affect the core's 88%
floor. Use a fresh report, not a recorded percentage. The Rust report does not
measure Python wrapper execution inside the extension. Use
[tools/binding-coverage.sh](../../qodec/tools/binding-coverage.sh) to measure that work.
Run it in the selected Python environment with `llvm-tools-preview` installed
for the active Rust toolchain. It uses that compiler's LLVM tools and reports
only `bindings/python/src/`, excluding dependencies from `TOTAL`. Python adapter
code and native Rust tests are outside this measurement. It leaves an
instrumented debug extension installed; restore the release build using the
Python build step above afterwards.