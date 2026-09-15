# tools

Unless stated otherwise, run these commands from the qodec directory.
[The parent build workflow](../../.github/workflows/build.yaml) uses `check.py`
for qodec's verification gates; the other scripts are optional helpers.

- `check.py` runs verification gates using the invoking Python interpreter.
- `binding-coverage.sh` measures the PyO3 wrapper crate.
- The `format_*.py` scripts are cosmetic YAML formatters. They do not change
  the meaning of any artifact.

## `check.py`

Use the selected Python executable; the runner does not select or install an
environment. It works from any working directory and stops on the first failure.
Use an environment with the documented compiler and Python tool prerequisites.

```bash
python tools/check.py python
python tools/check.py all --dry-run
```

Scopes are `rust`, `python`, `docs`, `coverage`, `examples`, `packaging`, and `all`. The Python scope
rebuilds the extension, verifies the import location, then runs mypy, stubtest,
and pytest. Use it before `docs` when the installed extension needs rebuilding.
Rust and rustdoc commands select only `qodec`, `qodec-python`, and `qodec-c`.
Clippy uses the parent's `-D clippy::pedantic` policy. The C library is built
before its tests. Coverage selects only the Rust core, with an 88% line floor.
`examples` audits every retained example with `qdk.ec.audit`, allowing only the
[documented unsupported-verification warnings](../examples/README.md#correctness-tests).
It requires a compatible QDK build with its `ec` dependencies; missing analysis
support fails the check. CI does not yet supply that build, so it runs the other
scopes individually. `all` includes every scope. Commands and working
directories are printed; a dry run never reports gates as passed.

From the qdk-ec repository root, use `python qodec/tools/check.py <scope>`.
No local editor configuration is needed. Compiler/linker setup must already be
available; the runner preserves those settings and never installs tools or
replaces compilers.
See [check prerequisites](../../.github/instructions/qodec-checks.instructions.md).

The root [VS Code tasks](../../.vscode/tasks.json) provide `qodec: check ...`
commands using the selected Python interpreter. Local interpreter settings stay
ignored; the task definitions contain no machine-specific paths.

Runner tests: `python -m unittest discover -s tools -p 'test_check.py'`.
They require PyYAML and run in the `packaging` scope along with the formatter tests.

## Packaging Checks

`python tools/check.py packaging` checks the Cargo file inventory, builds an
sdist, rebuilds a wheel outside the checkout, and imports it with site-packages
disabled. It checks archive contents, metadata, type stubs, and a model round
trip without changing the selected installation.

The same suite builds a temporary extension with `test-support` enabled to
test native-thread parser callbacks, native/Python replacement, and closed
callbacks. These private helpers are absent from normal wheels. This is a local
artifact check, not a publishing job; [the wheels workflow](../../.github/workflows/qodec-wheels.yaml)
checks the other native platforms.

From the qdk-ec repository root, `python qodec/tools/check_wheel.py target/wheels`
installs the single wheel in that directory into a temporary environment and
checks its imports, stubs, metadata, and model round trip outside the checkout.
Native-wheel validation uses it.
It uses the invoking interpreter's pip; `ensurepip` is not required.

The [parent Azure build stage](../../.ado/stages/build.yaml) builds qodec into
`target/qodec-wheels`, runs this probe on each native host, then copies the
verified wheel into the platform's shared artifact. Linux x86_64 adds the sdist.
The [ESRP publisher](../../.ado/stages/publish_python.yaml) selects only qodec
artifacts when `publishQodecPython` is enabled and rejects incomplete collections.
It is not part of the local check runner.

## `binding-coverage.sh`

Coverage for `bindings/python/src/`, which `cargo llvm-cov` cannot reach: those
wrappers only ever run inside the extension module Python loads, so the
workspace report scores them 0%. This builds the extension instrumented, drives
it with pytest, and reads the profile back against that exact `.so`.

```bash
tools/binding-coverage.sh
```

It leaves an instrumented **debug** extension installed. Restore the normal one
before benchmarking or shipping:

```bash
cd bindings/python && maturin develop --release
```

## `format_gadget_yaml.py`

Prefer compact flow-style lists (e.g. `[0, 1, 2]`) for gadget YAML, falling
back to block style when a list is long or the line would get too wide.

```bash
python tools/format_gadget_yaml.py [paths...]   # default: examples tests
python tools/format_gadget_yaml.py --check      # report, don't write
```

Options: `--max-items` (max entries kept on one line), `--max-line` (max line
width), `--check`.

## `format_isa_yaml.py`

Normalize instruction-set YAML, emitting multi-line string scalars in the
readable block (`|`) style.

```bash
python tools/format_isa_yaml.py [paths...]      # default: examples
python tools/format_isa_yaml.py --check         # report, don't write
```

Options: `--width` (max line width), `--check`.
