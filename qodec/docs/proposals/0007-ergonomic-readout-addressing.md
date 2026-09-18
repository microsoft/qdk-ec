# 0007 - Ergonomic readout addressing

**Status:** proposed

## Context

Consider a preparation circuit with six sub-preparation flags followed by six
verification measurements. Its measurement-dependent flag uses positions 6-11:

```yaml
readouts:
  - reject_x: ["circuit.readouts[6]", "circuit.readouts[8]", "circuit.readouts[10]"]
```

Inserting a flag-producing call before the measurements shifts each referenced
position. The references still parse but now select different bits. The
[circuit readout model](../concepts/gadget.md) defines one ordered record of
outcomes and flags; absolute indices depend on every preceding call.

Existing `{name: [terms]}` labels are not reference identifiers.

An edit to one call can therefore require changes to equations elsewhere in
the gadget. A missed update may select an existing but unrelated bit, so a
bounds check will not catch it. Better addresses could preserve the author's
intended reference through such edits. They belong in the format only if tools
need to retain that intent; a generator that reliably recomputes all indices
may be sufficient otherwise.

## Benefits

- Reduce manual renumbering and the risk of silently selecting the wrong bit
  when editing a circuit.

## Drawbacks

- Several spellings would address the same bit.
- Views and call positions survive only some kinds of edits.
- Names need scope and collision rules.
- Circuit-level lookup requires parsing, unlike loading a draft.

## Proposal

Keep one positional record and the meanings of existing `circuit.readouts[i]`
references. Evaluate two filtered views as the preferred candidate: one selects
only observe outcomes and one selects only flags. All indices are zero-based
within their named view and resolve to positions in the complete record.

Names and call-local references are alternatives, not additional requirements.
Adding a name must not change a readout's position or role. None of these forms
changes output order or makes loading depend on circuit parsing.

### Filtered views

The candidate spellings are `circuit.measurements[i]` for the i-th observe
outcome and `circuit.flags[i]` for the i-th flag. An outcome index is unchanged
when only flags are inserted before it. Inserting an earlier outcome still
changes it. This is proposed syntax:

```yaml
readouts:
  - reject_x: ["circuit.measurements[0]", "circuit.measurements[2]", "circuit.measurements[4]"]
```

These views do not create additional readout records or model-navigation fields.

## Alternatives

### Named readouts

A call could name the bit it emits (`as: my_flag`), referenced as
`circuit.readouts.my_flag`. The name would be stable under insertion of unrelated
calls. This syntax is proposed, not supported by the current call parser.

Named addressing could cover two additional scopes:

- **Gadget outputs:** allow an existing named entry to be referenced as
  `readouts.syndrome`, with `readouts[i]` remaining its canonical position.
  Reuse the existing `{name: [terms]}` entry shape.
- **Instruction outcomes:** consider optional `{name: Pauli}` observe entries
  and names in action guards. Decide this separately from circuit-call names;
  the declaration and the caller's record are different scopes.

Matching instruction outcomes to gadget readouts remains positional. Names are
local aliases, not linking keys. Existing duplicate labels must not silently
become ambiguous references or invalid drafts.

### Per-call addressing

Reference a bit by its emitting call and a local index — `circuit[<call>].readouts[<i>]`,
or by mnemonic + occurrence (`measure_x_all[2].readouts[0]`). This avoids global
readout offsets, but is not generally insertion-stable: call indices shift when
calls are inserted, and occurrence counts shift when matching calls are inserted.

### Other options

An end-relative index such as `circuit.readouts[-k]` survives prepends but not
appends. It would add a second index convention to the current absolute grammar.
An authoring tool could instead compute absolute indices from its own stable
identifiers and emit the current format. Prefer that option if consumers do not
need the authoring addresses in persisted artifacts.

## Discussion

### Names and scope

The view candidate adds two reference properties. `measurements` names observe
results; `outcomes` is an alternative already used by action guards, but in a
different index space. `flags` reuses the existing role; `heralds` would include
signals that need not be ideally zero. The named-call alternative adds `as`;
`name` could instead be mistaken for a call label. Binding APIs are unspecified.

Lookup must report missing parsers, indices, or ambiguous names, not skip them.
New grammar requires parser, schema, and binding support. Label disagreements
belong in authoring diagnostics or audit. Existing positions and labels must
retain their meanings.

[Typed readout fields](0003-typed-readout-fields.md) are separate: they attach
information to a readout rather than choose how to address it.

## Open Questions

- Do persisted filtered views solve a problem that an authoring tool cannot
  solve by emitting absolute indices? Test representative circuit edits.
- Can views and names coexist (a named *measurement*), or do we pick one?
- Which scopes support addressable names: gadget outputs, instruction outcomes,
  circuit-call outputs, or all three?
- Do guards use a bare outcome name or an explicit `readouts.<name>` reference?
  How are collisions with parameter and flag names handled?
- How should duplicate existing readability labels and disagreements between an
  instruction's labels and its gadget's labels be handled?
