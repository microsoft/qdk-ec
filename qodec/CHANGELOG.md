# Changelog

## Unreleased

### 0.2.0 Development

- Collection getters use standard `MutableSequence` and `MutableMapping`
  annotations, without custom stub-only container types. Typed item edits use
  normalized values; constructors and whole-property setters accept shorthand.
  Runtime shorthand item edits remain supported. Code operators and frame-map
  keys are strings on read; checks and frame equations contain References and bits.
- Python instruction construction, replacement, and live list edits share Rust
  parameter/flag uniqueness guards. Failed edits are atomic; draft-loading rules
  are unchanged.
- `Reference.Slice` takes optional `step` by keyword. Native segment wrapping
  reuses parser validation instead of formatting and parsing a new reference.
- Copy protocols explicitly register mutable types for distinct outer copies
  and immutable values for identity-preserving copies; missing mutable helpers
  no longer silently select shared identity.
- Python `Reference` and `ReferenceLike` now live at the package root, alongside
  `Node`. Import them from `qodec`; `qodec.gadgets` no longer re-exports them.
- `Reference` accepts general model addresses. Rust and Python `Qodec.resolve`
  and `Node.resolve` accept strings or parsed references; `Gadget.resolve` adds
  standalone gadget lookup. Slices and unions return selection nodes, preserving
  order and duplicates and rejecting any missing member.
- Model paths use `in`/`out` instead of `inputs`/`outputs`. Encoding paths expose
  `stabilizers`, `x`, and `z` directly; Python object properties keep their names.
  Standalone nodes use gadget identity and have no source locations. Circuit
  parsing remains explicit. General addresses cannot be used in parity equations
  or frame keys. Existing parity spelling, schema version, and C ABI are unchanged.
- `Reference.segments` exposes general path structure as immutable Python
  `Reference.Field`, `Key`, `Index`, `Slice`, and `Union` values. Rust exposes
  `Reference::segments()` and `ReferenceSegment`. Parity-specific Reference
  inspection attributes are removed, along with Rust's public `ReferenceTarget`,
  `GadgetBoundary`, and `EncodingPropertyKind`. Parity consumers interpret the
  structural segments; `ParityTerm::validate()` checks parity syntax separately.
  Python construction accepts only strings and references; arbitrary objects do
  not convert through `str()`.
- Reference equality, ordering, and hashing use normalized addresses. Numeric
  spelling, selector whitespace, JSON escapes, and known encoding-operator aliases
  do not affect identity. Selection shape, order, and duplicates remain significant.
  Python references no longer compare equal to strings; convert strings explicitly
  for equality. `ReferenceLike` inputs still accept strings. Authored `.path` text
  and serialization are preserved; `expand()` always produces canonical references.
- Frame assignments and loading reject duplicate equivalent targets instead of
  silently dropping equations. Live frame lookup accepts equivalent reference
  spellings and retains the stored key spelling. Save detects spelling-only edits
  even when reference values compare equal. No schema or C ABI change is required.
- Live frame operations validate supplied keys before matching equivalent spellings.
  `frames.update()` applies entries in order with the last value winning; invalid
  keys or final equations leave the map unchanged. Bulk updates index existing
  references once instead of scanning them for every incoming entry.
- Python `Node.value()` extracts scalars and model objects only. Use
  `as_sequence()` and `as_mapping()` for collections, including selections.
  Parameter-binding mappings support the same live key lookup and enumeration.
- Union expansion and selection enumeration avoid copying the full selector for
  every member. Parsed references keep one structural representation.
- Python frame keys accept strings or references for construction, whole-map
  assignment, and runtime live mapping edits. Iteration preserves authored string
  keys. Typed item edits use `MutableMapping[str, Check]`; whole-map setters retain
  broader input types. At runtime, item assignment, `update`, and `setdefault`
  also accept sequences of string or parsed-reference terms and literal bits;
  reads return immutable normalized tuples. Supply an equation to `setdefault`.
- Circuit-readout arguments accept slices selecting exactly one position.
  YAML and Python parser callbacks normalize these to the same record index;
  empty and multi-position selections are rejected. Circuit source is preserved.
- Rust `InstructionSet::resolve(mnemonic)` is renamed to `instruction(mnemonic)`;
  it still returns a copy of the declaration. Model-path `resolve` returns nodes.
- Reference syntax errors describe the general model-path grammar. A valid
  address used illegally in parity data reports `ReferenceParseError::NotParity`.

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