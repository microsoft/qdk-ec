# Changelog

## Unreleased

### Validation tightened

These reject documents no consumer could interpret, so `schema_version` is
unchanged under the compatibility contract below.

- A slice selector may select at most 1048576 positions. Every consumer that
  expands a selector allocates one reference per position, so an unbounded
  slice such as `circuit.readouts[0:18446744073709551615]` exhausted memory in
  the C projection, Python's `expand()`, and `Reference::parse_many`.
- `frames` keys are parsed with the reference grammar, like every equation term.
  A misspelled target such as `ou[0].z[0]` now fails to load instead of round-tripping.
- A Pauli token may not carry an operand prefix. `target.Z_0` is rejected; code
  qubits are addressed directly, as in `Z_0`.
- Unknown action-step fields and unknown fields inside a rotation are rejected
  instead of being silently discarded during serialization.

### Fixed

- Removing a layer's gadgets no longer restores old code definitions on save.
  Slices retain code bindings for their retained layers, including unused codes.
- Rust slices preserve metadata, explicit schema version, and manifest filename,
  matching Python. Source locations and stored artifact maps are not copied.
- C loading rejects strings containing NUL instead of deleting bytes, preventing
  name collisions and altered source text. JSON-escaped metadata remains lossless.
- C metadata fields always contain JSON object text, including `{}` when empty.
- Schemas accept the loader's empty action drafts, reference whitespace, and
  instruction-set filenames containing `#`. Code Pauli token spelling agrees
  with loading; numeric limits and slice arithmetic remain parser checks.
- Directory saves retain unchanged references to files outside the original
  manifest's directory. Edits are copied locally; external files are never
  overwritten. Reused files are checked for changes before writing. Bundles
  copy current values without reusing external files. Generated names cannot
  redirect output paths. A manifest filename deliberately pointing above the
  destination still raises the output root.
- The loader reports a conflicting artifact kind and a directory path as typed
  errors rather than as `io::Error`.
- `Display` no longer panics on a draft it cannot serialize, and no longer
  presents a code name where a block type belongs.
- The C header's `logical_count` and `observe_count` describe what a draft can
  actually contain. Circuit-qubit identifiers are `uint64_t` everywhere and
  counts are `size_t`.

### API Changes

- Python owned collections are live mutable views rather than detached containers.
  Instructions are shared mutable definitions with read-only mnemonics; loaded
  gadgets share their layer's instruction object. Mapping keys must match
  instruction or gadget mnemonics. Action/condition collections are immutable.
  Derived protocol indexes are read-only live mappings. Parsed calls remain
  standalone objects and do not edit circuit source.
- Python model objects implement `__copy__`, `__deepcopy__`, and `__replace__`.
  Deep copies preserve internal sharing and loaded history while isolating mutable
  children. There are no Rust API, serialization, schema-version, or C ABI changes.
- Rust and Python layers expose `codes`, a sparse map from block type to code
  definition. Python accepts it as a keyword-only constructor argument.
  `Qodec.codes` includes explicitly bound codes without gadgets. Rust `Layer`
  literals must supply the new field; an empty map lets encodings supply bindings.
- `Circuit.qubits` is renamed to `Circuit.blocks` in Rust and Python; Rust's
  `qubits_with` is renamed to `blocks_with`. The result is still distinct circuit
  block labels in first-appearance order, not flattened physical qubits. Python
  retains a property. The old names are not aliases.
- Rust `Qodec::save` and `save_bundle` return the written manifest `PathBuf`.
  Python `Qodec.save` returns `pathlib.Path`. Pass the returned path to `load`;
  destination directories, source sidecars, and relative-path behavior are unchanged.

## 0.1.0 - Initial Release

The first release uses package version `0.1.0`, on-disk `schema_version: 1`,
and C ABI revision `1`. These versions are independent.

### Model

- Codes, instruction sets, and gadgets describe a lowering chain from logical
  instructions to physical operations. Instruction sets declare sibling
  `blocks` and `instructions`; circuits contain instruction calls.
- Instructions describe stabilizations, observations, Clifford operations,
  rotations, conditions, and declared parameters. Gadgets supply circuits,
  boundary encodings, parameter bindings, checks, readouts, and sparse frames.
- Parity equations contain property-path references and integer bits `0` or `1`.
  References support indices, slices, and unions. Frames describe additional
  output-sign corrections; omission does not reset incoming frames.

### Persistence and Interpretation

- Load and save referenced YAML or JSON artifacts and multi-document YAML
  bundles. Reference fields determine artifact types; filenames are unrestricted.
  Saves serialize the current model, including edits and metadata.
- Loading and saving preserve incomplete protocol drafts and uninterpreted
  circuit source. They enforce preservation preconditions, not protocol
  correctness. Protocol checks belong to `qdk.ec.audit`.
- YAML circuit parsing is built in. Rust and Python register source parsers in
  one Rust-owned registry; the latest registration wins. Python's optional
  `qodec[parsers]` extra supplies a Stim adapter with bounded repeat expansion
  and noise-free constant readouts. Parsing is separate from persistence.

### Language APIs

- Rust and Python support construction, editing, structural equality, model
  navigation, source locations, and persistence.
- Python requires 3.11 or newer and includes type stubs. Its native extension
  uses CPython's stable ABI starting at Python 3.11.
- C provides a read-only projection with explicit ownership, error reporting,
  and an ABI-version check. C bindings are distributed as source.
- JSON Schemas, language guides, and example protocols describe the format and
  its use. Example audit limits are listed in [examples/README.md](examples/README.md).

## Compatibility Contract

qodec is pre-1.0. Each `0.x` minor release may break the public API or on-disk
format; patch releases preserve compatibility. Pin an exact package version
for reproducible work.

- `schema_version` is an optional non-negative integer on the manifest. When
  present, it must equal the loader's `CURRENT_SCHEMA_VERSION`. Omission means
  the loader's current version; declare it explicitly in persistent datasets.
- Increment `schema_version` by exactly one when a previously valid artifact
  would fail to load, change meaning, need a new required field, or have its
  references or bundle entries interpreted differently. Adding optional fields
  or rejecting already-invalid documents does not require an increment.
- There is no compatibility window for explicit schema versions. Changing the
  number alone does not migrate a document; its contents must match the target
  format. A package release may leave the schema version unchanged.
- The C ABI revision is independent of the schema and package versions. Bump it
  for a breaking change to a C symbol, signature, struct layout, or calling
  convention. Callers must compare `qodec_abi_version()` with the header's
  `QODEC_ABI_VERSION` before accessing projected structs.