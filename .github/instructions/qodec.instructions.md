---
description: 'qodec: shared architecture, ownership, compatibility, and task routing.'
applyTo: 'qodec/**'
---

# qodec

qodec models quantum error-correction protocols: codes, logical instructions, and
gadgets that lower a source instruction set to a target instruction set. The
qodec package includes the Rust core, Python bindings, and C bindings. It is
pre-1.0; every 0.x minor may break the API and on-disk format.

## Shared Rules

- The Rust model in `qodec/src/` is canonical. Implement behavior there; keep the
  Python and C bindings thin translation layers.
- qodec preserves declarations; `qdk.ec.audit` checks protocol correctness. Retain
  only operation-specific guards against ambiguity, information loss, or
  unavailable interpretation. Do not add a second general diagnostic API.
- For model changes, check the Rust types, JSON schemas, Python wrappers and
  stubs, C ABI and generated header, documentation, examples, and tests together.
- The Rust FFI source and `cbindgen` configuration own the generated
  [C header](../../qodec/bindings/c/include/qodec.h); never edit it by hand. Keep shared
  model concepts aligned across Rust, Python, and C while using C-native shapes.
  Preserve explicit ABI version, size, and offset tests, and regenerate the header
  with [regenerate.sh](../../qodec/bindings/c/regenerate.sh) after source changes.
- Core loaders and parsers return `Result` with typed errors. Reserve `panic!`
  and `unreachable!` for tests and genuinely unreachable invariants.
- Use `Layer.instruction_set`, `Circuit.instruction_set`, and `Circuit.calls`
  (Rust `calls()`). An instruction set has sibling `blocks` and `instructions`.
  Do not restore the old `isa` field or use `Circuit.instructions` for calls.
- Breaking on-disk changes bump `schema_version` under the compatibility contract
  in [CHANGELOG.md](../../qodec/CHANGELOG.md). Additive changes and fixes rejecting
  already-invalid documents do not. Update affected schemas and documentation.
- Each qodec crate declares its own `[package].version`; all three belong to
  the `qodec` cargo-release group, independently of `schema_version`. Use the
  scoped `cargo release version` commands in [RELEASING.md](../../qodec/RELEASING.md),
  never hand-edit version strings or bump sibling qdk-ec packages.
- Private design drafts are not part of this public source tree. Do not copy
  them into documentation or release artifacts without publication approval.
- Do not commit generated documentation. Language guidance belongs in rustdoc,
  Python stubs and usage docs, or the [C binding guide](../../qodec/bindings/c/README.md).
- Before comparing qodec with QDK, deq, Stim, or another package, verify the
  claimed behavior against current source or documentation for the version under
  discussion. Do not infer another package's behavior from an absent qodec API or
  concept.
- In introductory and ecosystem documentation, explain what the reader can do
  and why before cataloging fields. Introduce qodec terms when they are needed,
  with a concrete example; do not assume knowledge of an older format or of the
  distinctions among blocks, operands, parameters, arguments, encodings, and
  readouts.
- When the user approves a batch of review findings, track every accepted item
  through implementation and relevant checks, then complete any requested
  independent review pass before reporting. Pause only for a blocker, scope
  expansion, or an unresolved design decision.

## Read Before the Relevant Task

Read only the guidance required by the task. These rules also apply to reviews
and cross-language work, even when the edited file does not match a scoped glob.

- Before changing model behavior, loading, validation, circuit calls, references,
  navigation, schemas, or model examples/docs, read
  [qodec-model.instructions.md](qodec-model.instructions.md). This includes model
  changes made through a Python or C binding.
- Before Python binding, stub, export, or binding-test work, read
  [qodec-python.instructions.md](qodec-python.instructions.md).
- Before builds, tests, lint, coverage, documentation checks, or pre-merge
  verification, read [qodec-checks.instructions.md](qodec-checks.instructions.md).
- Before version or release work, read [RELEASING.md](../../qodec/RELEASING.md), including
  its prerelease rules. Releases go through a PR; tag the merged commit.

Keep the relevant instruction file current when a convention or CI check changes.