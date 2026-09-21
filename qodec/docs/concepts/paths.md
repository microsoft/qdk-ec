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

`Qodec.resolve(path)` returns a `Node`. It accepts a string or a parsed
`Reference`; both use the same grammar. An empty path selects the root.
`Node.resolve(path)` follows the same grammar relative to that node and returns
a node whose `path` is relative to its owner.

| Syntax | Meaning |
| --- | --- |
| `layers`, `.readouts` | A documented field name; ASCII letter or underscore followed by letters, digits, or underscores. |
| `[0]` | A zero-based, nonnegative sequence index. |
| `[0:3:2]` | An exclusive-stop slice with a positive step (here, positions 0 and 2). |
| `[2,0,2]` | A selection preserving the listed order and duplicates. |
| `["measure_xx"]` | A literal mapping key, quoted and escaped as a JSON string. |

Whitespace is allowed around slice and union indices, but not around fields.
Negative indices, empty selections, wildcards, filters, method calls, and
attribute discovery are not supported. Mapping keys use brackets, even
when they look like identifiers. Leading zeroes on indices are accepted and
removed in the canonical path; equivalent JSON escapes are normalized.

Slices and unions return a selection node, even when they select one entry.
Python `sequence_nodes()` and Rust `as_sequence()` return the selected nodes
with their individual model paths. Python `value()` returns a tuple of selected
values, preserving order and duplicates without unwrapping singletons.
There is no field broadcasting: select a member before following its fields.
Every selected index must exist; slices do not silently truncate at the end.
Missing members fail the whole lookup. A slice can select at most 1048576 positions.

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
| Instruction | `mnemonic`, `description`, `metadata`, `in`, `out`, `parameters`, `flags`, `action` |
| Code | `name`, `description`, `metadata`, `stabilizers`, `x`, `z` |
| Gadget | `implements`, `circuit`, `in`, `out`, `parameter_bindings`, `checks`, `readouts`, `frames`, `metadata` |
| Circuit | `instruction_set`, `source`, `format` |
| Encoding | `code`, `support`, `block_types`, `stabilizers`, `x`, `z` |
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

Boundary path fields are `in` and `out`. Python object properties are
`inputs` and `outputs`.

For example, `gadget_node.resolve("in[0].stabilizers[1]")` returns the second
stabilizer declaration of the first input encoding. `in[0].code` selects its
code object. The returned Pauli text uses code-local indices, not circuit labels.
Resolution does not evaluate the stabilizer's sign or place the operator.

Computed circuit calls, block lists, circuit readout records, effective formats,
and code dimensions are outside this navigation surface. Read the typed
accessors explicitly when these computations are needed. In particular,
`circuit.readouts[i]` is a valid parity reference but cannot be resolved
as a model declaration. General model addresses such as `metadata["description"]`
are valid `Reference` values but invalid in parity equations and frame keys.

## Reference structure

```python
reference = Reference('layers[0].gadgets["measure_z"]')
assert reference.segments == (
	Reference.Field("layers"), Reference.Index(0),
	Reference.Field("gadgets"), Reference.Key("measure_z"),
)
```

`Reference` describes an address; `Node` represents its selection in an owner.
The same reference can be resolved against different protocols or gadgets.
`segments` is an immutable tuple of immutable, hashable values supporting Python
pattern matching. Field names and literal mapping keys are distinct.
`Reference.Slice(start, stop, *, step=1)` retains a compact exclusive-stop slice;
`Reference.Union(indices)` holds an immutable tuple preserving order and duplicates.
`Reference.Index(value)` holds one sequence position. Rust exposes the same five
forms as `ReferenceSegment` through `Reference::segments()`.

`Reference("checks[01]")` equals `Reference("checks[1]")` and has the same hash.
Equality compares normalized addresses, not authored spelling or resolved values.
Numeric spelling, selector spacing, and JSON key escapes do not affect identity.
Within gadget-local paths, `out[0].code.z[1]` aliases `out[0].z[1]`; the same
rule applies to input encodings, X and stabilizer fields, and paths prefixed by
`layers[n].gadgets["name"]`. Other roots need model context to identify aliases;
arbitrary metadata fields are not rewritten. References contain no owner.

Selection shape, order, and duplicates remain significant: `checks[1]` is a
single node, whereas `checks[1:2]` is a selection node. References do not compare
equal to raw strings; explicitly construct a `Reference` for address comparisons.
APIs accepting `ReferenceLike` still accept strings and convert them at the boundary.

`path`, `str`, and serialization preserve authored text. `segments` preserves the
parsed structure. `expand()` expands only the final index selector, preserving
order and duplicates, and returns canonical references even for singleton
selections. Earlier selections are retained; no fields are broadcast and no
model is inspected.

Parity consumers interpret permitted segment patterns. Rust's
`ParityTerm::validate()` checks gadget-local parity syntax without checking bounds
or protocol correctness. Python gadget fields perform that check on assignment.
Construction accepts only strings or existing References, not arbitrary objects.

## Typed nodes

```python
node = protocol.resolve('layers[0].gadgets["measure_xx"]')
gadget = node.value(Gadget)
readouts = node.resolve("readouts").value()
equation = node.resolve("readouts[0].equation").value(tuple)
readout_nodes = node.resolve("readouts").sequence_nodes()
```

Python `node.value()` returns what ordinary field access returns, including live
collection views, immutable equations, shared model objects, scalars, and `None`.
For example, `gadget.resolve("checks[1]").value(tuple)` returns the same value
as `gadget.checks[1]`. The optional positional type uses `isinstance` without
conversion, copying, or element validation. A mismatch raises `TypeError`.
Use runtime types such as `Sequence`, not `Sequence[Reference]`; runtime-checkable
protocols are accepted. Normal Python subclass rules apply, so `value(int)`
accepts booleans. Static typing returns `object` without an expected type and
the requested type when supplied.

A selection's value is a tuple of selected values, not a writable selection view.
Its elements keep their ordinary ownership rules. Rust retains one typed `as_*`
accessor per model type, returning `Option` without conversion. Rust's `as_int`
returns `i128`, accommodating signed and unsigned JSON integers. Rust `is_none`
tests an absent optional value or JSON null, and `as_action` returns an
`ActionStep` reference. Python uses `node.value() is None` and `node.value()`
or `node.value(ExpectedActionType)` respectively.

Rust's `InstructionSet::instruction(mnemonic)` is a separate lookup returning
a copy of one instruction declaration. It takes a mnemonic, not a model path;
`resolve` is reserved for navigation returning a `Node`.

Python `sequence_nodes()` returns every child node in order; `mapping_nodes()`
returns every key and child node. Rust calls these `as_sequence()` and
`as_mapping()`. Mapping enumeration is total, not filtered. Python returns
a tuple and a read-only mapping respectively. These methods enumerate nodes
without extracting their values. Nodes deliberately have no indexing,
length, or iteration special methods. They also have no assignment-through-path API.

Python nodes are live handles. Each value access follows the current model;
replacing a layer changes the object returned by its node. Removing a path makes
value access raise `LookupError`. Values follow qodec's getter sharing and copying
rules. A previously returned view stays bound to its original owner; reading the
node again follows the current path. Getter failures propagate. Rust nodes borrow
the model, preventing mutation while in use.

Lookup selects only the requested children from borrowed data. It does not
construct a second model or convert unselected actions. Python and Rust share
path parsing and selection for native model values; Python returns existing
objects where the usual getters promise sharing. Enumerating a collection
creates its child handles, not copies of their descendants. A conditional
observation can therefore expose its condition and observable strings even
when extracting the whole Python action is unavailable.

Equality and hashing compare owner identity and canonical path,
not model contents. Two paths to the same shared object are different occurrences.
Node hashes remain unchanged when Python targets change. Compare extracted values
for structural equality. Python `str(node)` and Rust `Display` return its path;
`repr`/`Debug` show its path and type without dumping the model. Python repr labels
removed targets as missing. Python truth tests raise `TypeError`: choose `value(bool)`,
`value() is None`, or a collection value explicitly.

`Gadget.resolve` also accepts strings and references. Its nodes use the gadget
as their owner and gadget-relative paths. They remain distinct from
protocol-owned occurrences of the same object, and follow the same live
mutation rules in Python. Standalone gadget nodes have no source locations.

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
