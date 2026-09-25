# Changelog

## 0.2.1

- Python 3.14t and Python 3.15+ `abi3t` wheels alongside existing `abi3` wheels.

## 0.2.0

Breaking Rust and Python API changes. On-disk schema and C ABI versions remain `1`.

### Added

- General model references with typed segments, slices, unions, and standalone
  gadget navigation.
- Live Python collections, shared mutable instructions, and standard
  copy/deepcopy/replace protocols. Action values and parity equations stay immutable.
- Explicit layer code bindings. Save operations return the written manifest path.

### Migration

| Previous API | Replacement |
| --- | --- |
| Python `qodec.gadgets.Reference`, `ReferenceLike` | Import from `qodec` |
| Parity-specific reference attributes and Rust target enums | `Reference.segments` / `ReferenceSegment` |
| Model paths `inputs` / `outputs` | `in` / `out`; Python properties are unchanged |
| Python `Node.as_sequence()` / `as_mapping()` | `sequence_nodes()` / `mapping_nodes()` |
| Python `Node.as_action()` / `is_none` | `value()` / `value() is None` |
| `Circuit.qubits` / Rust `qubits_with` | `blocks` / `blocks_with` |
| Rust `InstructionSet::resolve(mnemonic)` | `instruction(mnemonic)` |
| Rust `Reference::parse_many(text)` | `Reference::parse(text)?.expand().collect::<Vec<_>>()` |
| Rust `ReadoutSpec::terms()` / `Readout::terms()` | `equation.iter()` |

- References compare normalized addresses, not strings; authored text is preserved.
- Python `Node.value()` returns ordinary getter values. Its optional type check
  uses `isinstance`, so a Boolean also satisfies `int`.
- Rust `Layer` literals require `codes`; an empty map permits inferred bindings.
- Python collection edits write through. Use `list`, `dict`, or `copy` for snapshots;
  mapping iterators snapshot entries while nested metadata values remain live.

### Fixes

- Preserve external artifacts, spelling-only edits, code bindings, and slice metadata
  during saves; never overwrite external inputs.
- Make live collection edits atomic and preserve unchanged child sharing.
- Reject malformed or ambiguous references, invalid action fields, and NUL-containing
  C strings. Limit slices to 1,048,576 selected positions.
- Align schemas with loading, improve error reporting, and remove unnecessary
  allocations from navigation, reference expansion, and C projection.

See [model paths](docs/concepts/paths.md) and
[Python usage](bindings/python/docs/usage.rst) for ownership and navigation details.

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