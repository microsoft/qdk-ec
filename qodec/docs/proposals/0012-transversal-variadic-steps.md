# 0012 - Transversal action steps over variadic block groups

**Status:** proposed

## Context

A caller supplies a list of qubit blocks and wants H applied to each one. The
same instruction should work for one block or a hundred. A fixed action list
cannot grow with that count; a variadic operand alone does not repeat its action.

Proposed syntax, assuming block type `qubit`:

```yaml
mnemonic: h_all
description: Transversal Hadamard on a variable number of qubits.
in:  [[qubit]]
out: [[qubit]]
action:
  - clifford: {Z_0: X_0, X_0: Z_0}
    over: qubit
```

Passing a Pauli parameter over a variadic group is different: `observe: p`
measures one supplied operator, rather than repeating an action per block.

A fixed-count instruction family must grow whenever a caller uses a new group
size. Expanding into calls outside qodec avoids that family, but leaves the
meaning of a whole-register operation with the generator rather than its
instruction definition. A declared repetition lets implementations for different
sizes share that definition, including the requirement that paired groups
advance together.

## Benefits

- Reuse one instruction definition across register sizes instead of generating
  and maintaining a fixed-count family.

## Drawbacks

- Consumers need group-binding and expansion rules.
- The same template applies to every block; this is not a general loop.
- Repeated observations are excluded because their output count would vary.
- A short declaration can expand to many operations.

## Proposal

### The `over:` modifier

- Any unitary step (`clifford`, `pauli`, `rotate`) — and `stabilize` — may carry an
  `over:` field naming one or more **variadic** operand groups.
- The operators describe one block from each selected group. Repeat the step
  once per zero-based block position, advancing all selected groups together.
- `over:` takes a block-type name (`over: qubit`) or a list for a multi-group,
  lockstep tiling (`over: [register_row, interaction_row]`).

For two groups, repeat a state swap between paired register and interaction
blocks. Both groups persist; this is not allocation or destruction. The example
uses one-qubit blocks to show the complete template:

```yaml
mnemonic: move_rows
in:  [[register_row], [interaction_row]]
out: [[register_row], [interaction_row]]
action:
  - clifford:
      in[0].X_0: out[1].X_0
      in[0].Z_0: out[1].Z_0
      in[1].X_0: out[0].X_0
      in[1].Z_0: out[0].Z_0
    over: [register_row, interaction_row]
```

Multi-group templates address blocks with the operand-qualified atoms of
[0011](0011-parameterized-pauli-indices.md) (`in[k].`, `out[k].`) so the second
group's base need not be written as a count-dependent flat literal (see Semantics).

### Semantics

- For N blocks of logical width w, block k starts at offset kw within its group.
  Group bases follow [flat logical indexing](../concepts/instruction-set.md).
- Apply the single-block template N times with those offsets.
- For several groups, repetition k uses block k from each group. Positional
  qualifiers avoid writing count-dependent group bases in the template.
- The repetition index is not an authored variable; there is no per-block expression.

### Validation

- Check the Clifford template once: repeating a symplectic map on disjoint blocks
  preserves that property. Parameterized indices still need 0011's binding checks.
- Each group must resolve unambiguously to a declared variadic operand.
- Bound groups must have equal lengths. Reject mismatches; do not truncate.
- Template indices must stay within one block from each selected group.

These are consumer checks. Loading does not expand groups or parse circuits.

### Composition with 0011

`over` varies the block count; [0011](0011-parameterized-pauli-indices.md) varies
indices within a block. They can combine without adding arithmetic or general loops.

## Alternatives

- Generate fixed-count instructions: current grammar, more declarations.
- Repeat calls in the consumer when repetition need not be a declared action.
- Add `foreach`: permits per-block variation but needs a larger expression language.

## Discussion

### Names and public surface

One field is added: `over`, naming the groups the action covers. `repeat` suggests
an independent count; `foreach` suggests a loop variable. It accepts a list or
a scalar for one group. Positional group selectors remain an open alternative.

Absent `over` retains the current action. Store the template rather than its
expansion, with matching Rust, schema, and binding support. Consumers may expand
it or operate directly on the template if the result is the same. Apply the
shared compatibility rule to the added field; document required consumer support.

## Open Questions

- Select groups by block type or by operand position?
- Can `if`/`unless` guard the whole repetition? Are per-block guards out of scope?
- How should unequal group sizes be reported after binding?
- Do zero bound blocks mean identity? What does an empty `over` list mean?
- Should repeated `observe` be a separate proposal with variable readout counts?
