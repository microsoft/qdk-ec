# Validation

qodec preserves resolved declarations. Audit checks whether those declarations
describe a valid protocol. Loading is not certification: incomplete protocols,
out-of-range parity references, and invalid circuits can be loaded and saved.

qodec rejects an operation when proceeding would lose information, require
guessing, or violate that operation's preconditions. Missing artifact files
and ambiguous definitions still fail: there is no unresolved-document model.
[Additional requirements](#additional-requirements) and
[gadget verification](#gadget-verification) below cover work left to other tools.

## Checking current objects

`Qodec::validate()` checks preservation preconditions and returns
`Result<(), String>`. Python's `Qodec.validate()` returns `None` or raises
`ValueError`. Neither reads files, changes objects, nor caches a result.

Named definitions must be unambiguous. Gadget instructions and targets must
agree with their layers, because saving reconstructs these links. Encodings
must align with operands and bind each layer block type to one code. Readout
positions and roles must agree with their list positions because they are
derived on reload. The bottom layer cannot have gadgets without a target layer.

Rust `Code::validate()` checks Pauli token syntax needed for dimensions and
default support, not equal X/Z counts. `InstructionSet::validate()` checks
unique names used in maps, not action uses. These are preservation checks,
not general-purpose diagnostic APIs.

Support-type checks are shared with serialization. All occurrences of a circuit
label on one input or output boundary must agree on its block type. The same
label may have different types on opposite boundaries. Explicit type lists
must match the support length and name declared circuit block types. An empty
list infers the type when the circuit instruction set has exactly one block
type. Otherwise validation leaves the unspecified types unresolved, but
serialization requires explicit types.

Loading resolves file references, then uses the same consistency checks as
`Qodec::validate()`. Errors identify the failing component, so loading can add
its source path without repeating validation. A referenced code unused by any
gadget is checked separately with `Code::validate()`; its layer binding is
retained when saving. Missing files, conflicting document references, and a
bottom-layer gadget that cannot be resolved remain loading errors.

All Rust and Python save methods check preservation before writing. Empty and
single-layer drafts can round-trip if the remaining objects satisfy these
requirements. Audit reports that they are incomplete protocols. Loaded file
paths are writing hints, not a separate model to validate or save.

Audit owns protocol diagnostics. Failed declarations block dependent analysis,
not unrelated artifacts. Direct analysis and interpreting accessors still
enforce their own preconditions, whether or not audit has run.

## Structural (per artifact, no cross-references)

The schemas describe artifact shapes. The Rust loader parses field types,
required fields, and reference syntax; it does not run a JSON Schema validator.
Schema checks in the test suite compare the documented shapes with example data.

### Instruction sets

These are protocol requirements checked by audit. Unique map keys also remain
qodec preconditions because duplicate entries could be silently discarded.

- **Names.** Instruction mnemonics and block-type names are unique within the ISA.
- **Parameters.** Parameter names are unique. Guards name declared `bit`
  parameters or outcomes of preceding `observe` steps, never later outcomes
  or flags. Rotation angles name `number` or `integer` parameters. Pauli
  positions can name declared `pauli` parameters; their supplied operators
  remain subject to the caller's concrete index bounds.
- **Block references.** Each block type in an instruction's `in`/`out` lists is
  declared in the ISA's `blocks` map.
- **Flat index space.** Each operand contributes its block type's `encodes` count
  of consecutive indices, in operand order. The larger of the input and output
  capacities defines the boundary range, not a limit on temporary qubits.
- **Temporary qubits.** An unconditional `stabilize` can introduce indices
  outside the boundary range. Later steps may use those indices, but cannot use
  them before preparation. Only the indices actually named are introduced;
  gaps remain unavailable. Temporary qubits do not carry between instructions.
- **Conditional preparation.** A conditional `stabilize` can use boundary
  qubits and already introduced temporary qubits, but cannot introduce new ones.
- **Action index checks.** These rules are checked when both boundaries are
  statically sized. For variadic boundaries, the consumer must determine the
  boundary range from the actual call and apply the same preparation rules.

These are rules for an ISA definition. Block liveness for concrete calls
depends on the source language and its consumer.

### Inline YAML calls

These rules apply when interpreting calls. Loading and saving preserve inline
source lists and explicitly tagged text without validating individual calls.

Argument values can be booleans (`true` or `false`), numbers, strings, lists of
non-negative integers, or lists of strings. Empty lists are accepted; Boolean
lists, nulls, maps, and mixed-type lists are rejected. Booleans are distinct from
integers, and quoted `'true'` and `'false'` remain strings.

Booleans are not block operands or selection bits. Block operands are
non-negative integers or strings. `select` is always a list of patterns whose
values are the integers `0` or `1`, not booleans. See
[Inline YAML calls](../representations/yaml.md#inline-yaml-calls) for both call forms.

### Parity references

Check, readout, and frame equations contain references and integer bits `0` or
`1`. Rust uses `ParityTerm`; Python uses `Reference` objects and integers.
YAML writes references as strings and constants as integers. Loading parses each
reference and retains its spelling. Invalid
paths, empty or reversed slices, and zero strides are rejected at construction
or deserialization, including in external check and readout files. Python gadget
constructors and setters apply the same syntax checks; a failed setter leaves
the existing equations unchanged.

Reference construction checks syntax. Audit checks that selected encodings,
operators, and gadget-readout positions exist. Circuit-readout positions are
checked when the circuit has a supported parser.
This does not prove an equation physically correct. Empty whole equations
remain valid and have parity zero.

A complete readout list has one entry per declared outcome and flag. qodec
also preserves partial or excessive lists; audit reports them as errors.
Omission leaves equations for derivation, but audit reports missing observable
and flag equations as errors. An empty equation
inside a supplied list declares the constant-zero bit, not an omitted binding.

## Cross-reference (between artifacts in the qodec)

File loading follows references from the selected manifest: layer ISAs, codes,
and YAML gadget documents, then the gadgets' external components. The referencing
field determines the expected type. Unreferenced files and bundle entries are
ignored, not validated or rejected as orphan gadgets. See
[Loading](../representations/yaml.md#loading) for path and bundle rules.

The artifacts fit together:

- Every reference (an ISA a gadget targets, a code a layer binds, a circuit
  source, an external check or readout list) resolves to content of the expected
  type. Missing referenced files or entries are load errors.
- Each resolved path identifies one artifact type. The manifest path is reserved
  for the manifest.
- Every listed layer resolves. Audit requires two or more layers for a complete
  lowering protocol; persistence permits fewer.
- Every gadget implements an instruction in the source ISA (the layer above the
  gadget's circuit target). It is supplied by the layer that lists it, or stated
  explicitly via `implements` and then checked to agree with the layer.
- Every gadget's circuit targets an ISA in the layer below, and its `in`/`out`
  entries align with the implemented instruction's `in`/`out` positions.
- **Logical capacity (audit).** A layer-bound code's `x` and `z` lists have equal lengths,
  matching the logical-qubit count its block type declares.
- **Code-qubit index bounds (audit).** Code Pauli indices are non-negative and must fit
  the encoding's support. For a circuit ISA with a single block type of size
  `stride`, each support entry contributes `stride` code qubits, and indices are
  checked against `len(support) * stride`. Position `k` selects qubit `k % stride`
  within circuit operand `support[k // stride]`.
- **Codes are bound once, on the layer.** Each layer's manifest binds every source
  block type to a single code, in its `codes:` map. A gadget's `in`/`out` entries
  carry only support, never a code, so two gadgets can't disagree about how a block
  type is encoded in a loaded layer. In-memory authoring can create conflicting
  bindings, which validation rejects. Matching block types let gadgets share an
  encoding, but a consumer still has to connect their supports and transport
  their frames. Code switching uses different block types bound to different codes.

### Circuit sources

Loading and saving do not interpret circuit calls. Source text and inline
YAML lists are preserved. An explicit `format` marks a string as inline text,
including `yaml` and languages without a qodec parser. Without a format tag,
a string is a file reference.

`Circuit.calls()`, `Circuit.blocks`, and `Circuit.readouts` interpret source on
request and may fail on a loadable draft. The C projection retains source and
records parse errors per circuit. Python's `Observe` cannot represent a
condition: reading such an action raises rather than silently dropping it,
while the stored declaration remains saveable. A conditional observation also
cannot produce a statically sized `Circuit.readouts` result.

Audit checks that each argument name names a parameter declared by the called
instruction and its value must have a compatible type. Forwarded parameters
retain their declared types; integers can supply number parameters, but not
the reverse. Homogeneous integer and string lists retain their supported
argument forms and are checked by element type. A `circuit.readouts[i]` argument binds only a `bit` parameter and
must address a preceding call's record. Each `select` key must name a declared
flag or a valid position in the instruction's flag list.

YAML interpretation is built in; other formats use registered parsers or an
optional adapter. Python's default Stim adapter needs Stim, included in `qodec[parsers]`; see
[source formats](../representations/source-formats.md). Registration does not
change what loading and saving preserve.

Sources without a configured parser remain opaque. Audit reports analysis limits
instead of certifying their circuit behavior or record bounds.

## Algebraic (commutation of declared Pauli operators)

Audit, not qodec loading or saving, checks these requirements:

- Stabilizers in a code definition pairwise commute.
- For each logical qubit $i$, its logical $X$ and logical $Z$ anticommute. All other
  pairs of logical generators commute, and every logical generator commutes with
  every stabilizer.
- For each Clifford action in an ISA, the declared generator-image map
  preserves commutation relations over the complete generator basis, including
  implicit identity images of unmentioned generators. Checking only explicitly
  supplied pairs would miss invalid maps such as `X_0 -> Z_0` with `Z_0`
  otherwise left unchanged.

## Additional requirements

Analysis routines check the mathematical preconditions they rely on even when
called without audit. The code analysis also requires:

- **Logical independence.** Logical operators must be non-trivial and independent
  modulo the stabilizer group. qodec does not separately check these properties.

## Gadget verification

Verifying that a gadget's circuit realizes its implemented instruction belongs
to tools that interpret the circuit, not to qodec's structural validation.
Under noiseless stabilizer execution, a verifier should check:

- **Observables.** Each observable readout realizes the corresponding instruction
  Pauli, including its sign, matched by `observe`-outcome position.
- **Checks.** Each check's XOR sum is deterministically zero.
- **Flags.** Each flag readout is deterministically zero.
- **Dependencies.** Readout equations have consistent, uniquely determined
  values. A reference cycle is allowed when the equations have a unique solution.
- **Output stabilizers.** Valid declared equations determine the output stabilizer
  signs; merely mentioning the signs is insufficient.
- **Logical action and frames.** Compare the circuit's action, including explicit
  `frames` deltas, with the instruction's independent action. Frame values may
  use circuit bits and literal bits, or acyclic readout aliases resolving only
  to those terms. Input and output encoding signs are invalid delta inputs.
  Missing frame entries mean no additional correction, not an inferred one.

Parity verification includes arbitrary incoming Pauli-frame signs. A relation
that works only when those signs are zero is not generally correct.

The flag condition rules out always-rejecting flags and flags that leak logical
information through input-state dependence.

Computing code distance and constructing a decoder's fault model are also tasks
for other tools; qodec defines neither a distance claim nor a fault model.

## Where stabilizers live

Each layer binds its own codes. For example, C6 code qubits are logical qubits
of C4 blocks below them; the C6 stabilizers and the C4 stabilizers belong to
different layer bindings. Each layer's gadgets describe the corresponding
checks, readouts, and frames. A consumer composes these across lowering steps;
it must not interpret an upper-layer code index as a bottom-layer physical label.

> **Consumer note.** How a *consumer* turns a multi-layer qodec into a concrete
> artifact (for example, a single flat Stim circuit), including which edges'
> decoding surfaces it composes, is an implementation detail of that consumer, not
> a property of the qodec model. A well-behaved target should handle the full
> multi-layer decoding surface. Some targets may have limitations and should say so
> in their own documentation. Do not over-fit a qodec's layering to one consumer's
> current limitations.
