# Model paths and nodes

A model path identifies one occurrence in a resolved qodec. It can address a
layer, a gadget, a collection, or one scalar value. For example:

```text
layers[0].gadgets["measure_xx"].readouts[0].equation[1]
```

This names the second reference in the first declared readout of that gadget.
It does not evaluate the reference or parse the circuit. Paths work on constructed
models as well as loaded ones; source locations are optional additional information.

## Exact lookup

`Qodec.resolve(path)` returns a `Node`. An empty path selects the root.
`Node.resolve(path)` follows the same grammar relative to that node and returns
a node whose `path` is relative to the qodec root.

| Syntax | Meaning |
| --- | --- |
| `layers`, `.readouts` | A documented field name; ASCII letter or underscore followed by letters, digits, or underscores. |
| `[0]` | A zero-based, nonnegative sequence index. |
| `["measure_xx"]` | A literal mapping key, quoted and escaped as a JSON string. |

No whitespace outside quoted keys, negative indices, slices, unions, wildcards,
filters, method calls, or attribute discovery. Mapping keys use brackets, even
when they look like identifiers. Leading zeroes on indices are accepted and
removed in the canonical path; equivalent JSON escapes are normalized.

Malformed syntax is a Rust `PathError::Syntax` or Python `ValueError`.
A nonexistent field, key, index, or traversal through the wrong kind of value is
`PathError::Missing` or `LookupError`. A present optional field containing `None`
is not a missing path. No implicit fallback to an ancestor occurs.

## Model shape

Paths follow resolved declarations, not YAML envelopes, file references, or the
implementation language's container types. Both languages use this shape:

| Object | Fields |
| --- | --- |
| Qodec | `name`, `description`, `schema_version`, `manifest_filename`, `metadata`, `layers`, `instruction_sets`, `codes` |
| Layer | `instruction_set`, `codes`, `gadgets` |
| InstructionSet | `name`, `description`, `metadata`, `blocks`, `instructions` |
| Instruction | `mnemonic`, `description`, `metadata`, `inputs`, `outputs`, `parameters`, `flags`, `action` |
| Code | `name`, `description`, `metadata`, `stabilizers`, `x`, `z` |
| Gadget | `implements`, `circuit`, `inputs`, `outputs`, `parameter_bindings`, `checks`, `readouts`, `frames`, `metadata` |
| Circuit | `instruction_set`, `source`, `format` |
| Encoding | `code`, `support`, `block_types` |
| Block | `name`, `encodes` |
| BlockOperand | `block`, `is_variadic` |
| Parameter | `name`, `kind` |
| Parameter.Kind | `value` |
| Action | `condition`; variant-specific `operators`, `observables`, `generators`, `operator`, `pauli`, `angle` |
| Condition | `predicates`, `invert` |
| Readout | `position`, `name`, `is_flag`, `equation` |
| Reference | `path` |

Root `instruction_sets` and `codes` are maps keyed by name. Layer `codes` maps
block types to explicitly bound code definitions, including unused codes.
`instructions` and
`gadgets` are maps keyed by mnemonic. `blocks`, `parameters`, boundaries, actions,
checks, and readouts are ordered sequences. Metadata contains JSON values and
string-keyed maps. Check, readout, and frame equations contain `Reference` values
and integer literal bits. `frames` is a map keyed by the authored target path;
for example, `frames["out[0].z[0]"][0]` selects its first term.
Pauli strings are strings, not parsed Pauli expressions.

Computed circuit calls, qubit lists, circuit readout records, effective formats,
and code dimensions are outside this navigation surface. Read the existing typed
accessors explicitly when these computations are needed. Model paths are also
distinct from gadget parity `Reference`: the latter addresses bits and signs
relative to a gadget, with its own closed grammar and meaning.

## Typed nodes

```python
node = protocol.resolve('layers[0].gadgets["measure_xx"]')
gadget = node.value(Gadget)
readouts = node.resolve("readouts").as_sequence()
equation = readouts[0].resolve("equation").as_sequence()
```

Python asks for one expected type with `node.value(expected)`; Rust has one
`as_*` accessor per type. Neither converts a value. Rust returns `Option`;
Python raises `TypeError` on mismatch. `value(int)` excludes booleans, and
`value(float)` excludes integers.
Rust's `as_int` returns `i128`, accommodating signed and unsigned JSON integers.
`is_none` tests an absent optional value or JSON null. `as_action` returns an
`ActionStep` reference in Rust and the existing action-type union in Python;
it stays a named accessor because no single expected type describes that union.

`as_sequence` returns every child in order. `as_mapping` returns every key and
child; it is total for the selected mapping, not a filtered result. Python returns
a tuple and a read-only mapping respectively. Nodes deliberately have no indexing,
length, or iteration special methods. They also have no assignment-through-path API.

Python nodes are live handles. Each value access follows the current model;
replacing a layer changes the object returned by its node. Removing a path makes
value access raise `LookupError`. Normal qodec getter sharing and copying rules
still apply. Rust nodes borrow the model, preventing mutation while in use.

Lookup selects only the requested children from borrowed data. It does not
construct a second model or convert unselected actions. Python and Rust share
path parsing and selection for native model values; Python returns existing
objects where the usual getters promise sharing. Enumerating a collection
creates its child handles, not copies of their descendants. A conditional
observation can therefore expose its condition and observable strings even
when extracting the whole Python action is unavailable.

Equality and hashing compare the owning qodec's identity and the canonical path,
not model contents. Two paths to the same shared object are different occurrences.
Node hashes remain unchanged when Python targets change. Compare extracted values
for structural equality. Python `str(node)` and Rust `Display` return its path;
`repr`/`Debug` show its path and type without dumping the model. Python repr labels
removed targets as missing. Python truth tests raise `TypeError`: choose `value(bool)`,
`is_none`, or a typed collection accessor explicitly.

## Source locations

`node.source_location` in Python, or `node.source_location()` in Rust, returns an
output-only `SourceLocation` with the actual file `path` and 1-based `line`, or
no location. A bundle location points into the outer file. External equations
point into their own referenced files. Source information is captured from the
text used for loading; it is not stored in YAML or included in model equality.
Neither `Node` nor `SourceLocation` has a public constructor; obtain them through
model lookup and `source_location`.

The map is sparse. Constructed values, in-memory bundle text, slices, defaults,
and derived fields may have no location. Position parsing is best-effort and
does not make otherwise loadable drafts invalid; YAML aliases currently omit
positions for their document. Locations describe the loaded revision, not later
filesystem edits. Saving does not relocate the in-memory model to the output.

Rust clears locations before handing out mutable layers or metadata and when
changing manifest values. Python compares the current model against its loaded
snapshot and suppresses all locations while they differ. This conservative rule
also catches changes through shared children. Load the saved file again to obtain
locations for an edited model.
