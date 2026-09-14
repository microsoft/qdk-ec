---
description: 'Use when changing or reviewing qodec Python bindings, PyO3 wrappers, .pyi stubs, exports, Python value semantics, binding tests, or Python API documentation.'
applyTo: 'qodec/bindings/python/**'
---

# qodec Python Bindings

The PyO3 wrappers are in [qodec/bindings/python/src/](../../qodec/bindings/python/src);
the package and hand-written stubs are in
[qodec/bindings/python/python/qodec/](../../qodec/bindings/python/python/qodec).
Keep wrappers thin. Read [qodec-model.instructions.md](qodec-model.instructions.md)
for changes to model behavior, even if only binding files are edited.

## Signatures and Exports

- `.pyi` stubs are the canonical Python signature source. Update them alongside
  binding changes, together with tests and docstrings.
- Each stub mirrors its own module. The top-level stub covers curated top-level
  exports; the `codes`, `gadgets`, and `instructions` stubs describe their own
  modules. Do not declare a name at the top level unless it resolves there at
  runtime. The `qodec.instructions` module name remains unchanged.
- PyO3 constructors are `__new__`, not `__init__`; PyO3 classes are `@final`.
- Optional constructor arguments are keyword-only: use `*` in the PyO3 signature.
- Value types implement structural `__eq__` and are then unhashable under PyO3.
  Follow the separate model-navigation contract for `Node` identity and hashing.
- Name-keyed collections such as `Layer.gadgets` and
  `InstructionSet.instructions` accept a `list` or `dict` and return a `dict`
  keyed by mnemonic.

## Verification

- Read [qodec-checks.instructions.md](qodec-checks.instructions.md) for commands
  and working directories. Rebuild the extension before runtime checks whenever
  Rust core or binding changes affect it; an installed binary can be stale.
- Keep the package's PEP 561 `py.typed` marker. Without it, stub checking can
  report success without comparing the shipped stubs.
- Run `stubtest` for binding or signature changes. It checks signatures, not
  runtime values; test value-shape changes such as `int` to `bool` with `pytest`.
- Keep [stubtest-allowlist.txt](../../qodec/bindings/python/stubtest-allowlist.txt)
  minimal. It is for type-only aliases with no runtime counterpart by design,
  not for hiding binding/stub mismatches.
- Python API docs use Sphinx AutoAPI over `.pyi` files. Check generated docs and
  doctests when changing stubs or usage documentation; do not commit the output.