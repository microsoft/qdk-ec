---
description: 'Use when changing or reviewing qodec model behavior, loading, persistence, validation, circuit calls, parity references, navigation, schemas, examples, or documentation, including through Python or C bindings.'
applyTo: 'qodec/src/**,qodec/schemas/**,qodec/tests/**,qodec/examples/**,qodec/docs/concepts/**'
---

# qodec Model

Read the owning Rust implementation before changing its contract. Loading and
resolution live in [qodec/src/qodec/](../../qodec/src/qodec); operation-specific guards live
in [validation.rs](../../qodec/src/validation.rs).

## Loading and Persistence

- Rust `Qodec::load`, Python `Qodec.load`, and C `qodec_load` take an explicit
  manifest or bundle file path, never a directory. Save takes a destination
  directory and returns the written manifest path; pass that result to load.
- Directory saves reuse unchanged filesystem artifacts outside the original
  manifest directory, after checking their loaded text before any writes.
  Edited artifacts and documents needing new references get local copies.
  External input files must never be overwritten. Project-local files copy
  normally; bundles copy all current values and do not reuse external files.
  Bundle entries are not filesystem origins. Loading records are comparison
  baselines, never substitutes for current values. No symlink or race protection
  is promised. Keep generated artifact names from redirecting output paths.
- Reference fields determine artifact types: layer `instruction_set`, `codes`,
  and `gadgets`, then each gadget's external `checks`, `readouts`, and circuit
  `source`. Read only referenced content. Do not scan directories, classify
  artifacts by suffix, or reject unreferenced files and bundle entries.
- Resolve paths relative to the containing manifest or gadget document, including
  its key inside a bundle. Preserve leading `..` components. Example directory
  layouts and artifact filename suffixes are conventions, not requirements.
- A bundle starts with a single-entry `{arbitrary-key: manifest}` envelope. Its
  key is unrestricted; remaining entries are looked up by referenced path.
  Circuit-source entries may contain raw text.
- Manifest `gadgets` values reference YAML gadget documents, never raw circuits.
  A circuit-only gadget is a document containing, for example,
  `circuit: ./idle.stim`.
- Read `CURRENT_SCHEMA_VERSION` for the current version. Update explicit versions
  in fixtures and generators when the format changes; retain old versions only
  in deliberate compatibility-rejection tests. Keep fixture paths explicit.

## Circuit Source and Calls

- Loading and saving preserve circuit source without interpreting calls,
  including inline YAML lists, invalid text, and unknown languages. Explicit
  `format` tags inline text; untagged strings are file references.
- Language identification is separate from artifact typing. Preserve the source
  extension and inline `format` rules; do not imply that arbitrary languages can
  be parsed. `Circuit.calls`, `blocks`, and `readouts` parse on request and may
  fail on a loadable draft. Audit owns call validity. The C projection retains
  source and reports parse failures per circuit.
- `Circuit.blocks` returns distinct block labels in first-appearance order,
  without expanding multi-qubit blocks. It remains a read-only Python property.
- Python `Circuit.calls()` is a method with a keyword-only `parser` override.
  `qodec.register(parser, format="tag")` installs a PyO3 callback wrapper in the
  shared Rust registry; no Python registry exists. Latest registration wins
  across both languages in the same linked Rust instance. YAML is registered
  directly in Rust. Python callbacks get an ISA snapshot and return complete
  call values, require a live interpreter, and are released at shutdown.
  The optional Stim adapter is registered at module load only if Stim is available.
  Optional Python Stim uses the official parser. Rust has `register` and
  `*_with` overrides but no default Stim parser. Configuration is not model data.
- Before changing YAML call parsing, read
  [inline_yaml_parser.rs](../../qodec/src/inline_yaml_parser.rs). The object form is
  `{mnemonic: {operands: [...], arguments: {...}, select: [...]}}`. All three
  fields default to empty; reject unknown fields. The list form is shorthand
  for block operands followed by named arguments, with no reserved argument
  names. Per-call selection exists only in the object form. Save preserves the
  authored form.
- Scalar `true` and `false` are `Argument::Boolean(bool)`, distinct from integers,
  in both call forms. Quoted values remain strings. Reject booleans in arrays,
  block operands, and selection bits. `select` is an OR list of patterns, never
  a bare map; each pattern is an AND of integer `0`/`1` flag values.

## References and Navigation

- Before changing parity or gadget code or documentation, read the module docs
  in [parity.rs](../../qodec/src/parity.rs) for the reference grammar and readout roles.
  Equations are flat lists of references to bits and signs relative to the gadget
  root; an equation does not by itself assert that its parity is zero.
- Encoding references are positional, for example `in[0].stabilizers[1]`.
  Never author the rejected named-operand form `in.target.stabilizers[0]` or
  `in: {target: ...}`. Reference slices and unions expand to multiple atoms.
- `Reference` is a general owner-independent model address, not a parity descriptor.
  Its `segments` expose Field, Key, Index, Slice, and Union (Rust `ReferenceSegment`;
  immutable nested Python `Reference` types). Do not restore parity-specific
  reference attributes. Parity consumers interpret permitted segment patterns;
  Rust `ParityTerm::validate` and Python gadget assignment enforce parity syntax.
  Equality/hashing ignore authored spelling and normalize encoding-operator aliases
  in gadget-local and root layer/gadget paths. Other roots need model context;
  selection shape/order/duplicates stay significant. Python equality is Reference-only,
  while ReferenceLike input boundaries explicitly convert strings. Authored path text
  and serialization remain unchanged. Expansion canonicalizes spelling and aliases
  and expands only the final selector. Persistence must compare serialized documents
  when deciding artifact reuse so spelling-only edits are not lost.
- Before changing `Qodec.resolve`, `Node`, or source locations, read
  [paths.md](../../qodec/docs/concepts/paths.md). Model navigation follows resolved
  declarations and never implicitly parses circuits. `Qodec.resolve`,
  `Node.resolve`, and `Gadget.resolve` accept strings or `Reference` values.
  Paths use `in`/`out`, not `inputs`/`outputs`; encoding operators are directly
  addressable. General model addresses remain invalid as parity terms or frame
  keys. Selections preserve order and duplicates and fail on any missing member.
  Standalone gadget nodes use gadget identity and have no source locations.
  Keep `Node` opaque, with no collection dunders; Python truth tests raise.
  Python `Node.value(expected=object)` returns ordinary getter values with their
  normal ownership and mutability. The optional positional type uses `isinstance`
  without conversion or element validation. Selections return tuples of values.
  `sequence_nodes` and `mapping_nodes` enumerate child nodes without extracting
  their values. Rust retains its typed accessors. Getter errors propagate;
  do not materialize entire parent collections to navigate to one child.
  Source locations are optional loaded-revision points, not protocol data.

## Validation Boundaries

- Keep field/reference syntax, code Pauli syntax needed for dimensions, map-name
  uniqueness, resolved artifacts, layer bindings, encoding alignment, and derived
  readout roles as operation-specific guards. Bottom-layer gadgets require a
  target layer.
- Empty/single-layer drafts, unequal X/Z lists, partial readouts, out-of-bounds
  parity references, invalid parameter uses/actions, and invalid circuit text
  can persist. Algebra and protocol correctness belong to `qdk.ec.audit`.
  Analysis routines retain their own preconditions.
- Do not export `ValidationIssue` or `validation_issues()` as a general diagnostic
  API. Do not reject loadable drafts merely because audit would report errors.
- Omitted observable and flag equations are undefined, not implicit zero bindings;
  both are audit errors. An explicit `[]` equation declares zero. A supplied
  partial list of readout equations is also an audit error.