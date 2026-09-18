# 0011 - Parameterized Pauli indices in instruction actions

**Status:** proposed

## Context

A register move swaps two selected sites. Literal indices require a separate
Clifford declaration for each pair. Letting indices refer to `integer` parameters
would keep one declaration. The register remains the block operand; site indices
are arguments. Proposed syntax:

```yaml
mnemonic: move
description: Swap the states at two sites in a register.
in: [zone]
out: [zone]
parameters: {source: integer, destination: integer}
action:
  - clifford:
      X_source: X_destination
      Z_source: Z_destination
      X_destination: X_source
      Z_destination: Z_source
```


The number of site pairs grows quadratically with register width. Maintaining
a separate instruction for each pair repeats the same operation and makes
routine routing choices part of the instruction inventory. Index arguments
would let the author specify the allowed operation once while the caller chooses
its sites. Unlike supplying an entire Clifford, the caller would not be free
to replace the operation with a different one.

## Benefits

- Avoid a separate instruction definition for every site pair while keeping
  the permitted operation fixed.

## Drawbacks

- Bounds and index collisions need checking after arguments are bound.
- Operator parsers and tools must understand parameter names inside Pauli strings.
- This does not add index arithmetic or variable-size repetition.

## Proposal

### Syntax

- A parameter of kind `integer` may be referenced in any flat-index position of an
  action operator, written as a bare identifier in place of the numeric index.
- The subscript disambiguates lexically: after `_`, a run of digits is a literal
  index (`X_0`) and a letter-led identifier is a parameter reference (`X_source`), the
  same convention `angle: theta` already uses for a whole field.
- It is accepted in `pauli`, `clifford` (both keys and values), `observe`,
  `stabilize`, and the `rotate` axis.

CZ between two caller-named sites:

```yaml
mnemonic: cz_pair
in: [zone]
out: [zone]
parameters: {first: integer, second: integer}
action:
  - clifford:
      X_first: X_first Z_second
      X_second: Z_first X_second
```

The identifier-in-index-position rule extends the Pauli-atom grammar and is
specified here, without changing the separate gadget parity-reference grammar.

### Operand-relative indices

For operands of widths 8 and 4, the second operand occupies flat indices 8-11.
A caller should be able to select its site 1 without computing flat index 9.

Allow `in[k].` or `out[k].` on each Pauli factor. Here k is the zero-based
operand position, and the trailing index is relative to that operand.
`in[1].X_site` resolves to `base(in[1]) + site`, where the base is the sum of
preceding operand widths. Factors may name different operands and sides.

```yaml
mnemonic: move_between_zones
in: [register_zone, interaction_zone]
out: [register_zone, interaction_zone]
parameters: {register: integer, interaction: integer}
action:
  - clifford:
      in[0].X_register: out[1].X_interaction
      in[0].Z_register: out[1].Z_interaction
      in[1].X_interaction: out[0].X_register
      in[1].Z_interaction: out[0].Z_register
```

With `register=3` and `interaction=1`, `in[0].X_register` resolves to input
flat index 3 and `out[1].X_interaction` to output flat index 9. Unqualified
indices keep their whole-side meaning. Relative bounds use the selected operand's
width, so `interaction` must be in `[0, 4)`.

Quote bracketed factors in YAML flow mappings, for example
`"in[1].X_interaction"`. The block mapping above needs no quotes.

### Semantics

- At a call site each index-parameter binds to a concrete non-negative integer; the
  action is the template with those indices substituted.
- A referenced name must identify an `integer` parameter on this instruction.
  An interpreting consumer reports missing, wrong-type, negative, or out-of-range
  arguments; it must not clip indices or substitute a default.
- The instruction's flat index-space size *n* is still fixed by `in`/`out`.
  A substituted flat index must lie in `[0, n)`; an operand-qualified offset
  (`in[k].`/`out[k].`) is instead bounded by that operand's size (see
  [Operand-relative indices](#operand-relative-indices)).
- `observe: Z_q` still produces one outcome. Choosing its index does not change
  the readout count.

### Validation

After binding, check bounds, required distinctness, and symplectic preservation.
A symbolic proof can be reused only when its assumptions hold, including
parameter/literal collisions. Whether coincident indices mean identity or an
invalid call remains open.

These are interpretation and audit checks, not conditions for preserving a
draft. Existing `parameter_bindings` forward parameters into gadget circuits;
conformance is checked against the bound action.

## Alternatives

- Supply an entire Clifford as an argument: more general, but the caller chooses
  the operation rather than just its sites.
- Generate fixed-index instructions: no new grammar, but more declarations.
- Route in a source-language adapter: adequate if the formal action need not
  express the site choice.
- Add `permute`: unnecessary for the example because a permutation is a Clifford.

## Discussion

### Names and public surface

No fields or types are added by the text-based candidate. It reuses `integer`
rather than adding `index`; `X_source` follows `X_0` without new braces or sigils.
`in[k]` and `out[k]` reuse positional boundaries; block-type names would be
ambiguous when repeated. Example parameter names are not built-ins.

Keep indices in Pauli text initially. A structured literal-or-parameter type,
like `Scalar` for angles, would affect more public value types. Either choice
needs parser, schema, substitution, and value tests; text storage is not enough.
Resolve operand offsets before checking collisions. Arithmetic selections must
be bound as concrete indices, not written as expressions inside operators.

Existing [Pauli helpers](../../bindings/python/python/qodec/codes.pyi) can already
construct `"X_source"`; interpreting it is the new work. Gadget parity references
use a separate grammar. Bound indices can also serve
[Choi actions](0010-choi-stabilizer-actions.md), while
[0012](0012-transversal-variadic-steps.md) varies repetition count.

## Open Questions

- Is the bare-name syntax clear when literal and parameter indices are mixed?
- Is `integer` sufficient, or does an index need its own constrained type?
- Require distinct bound indices, or define the valid coincident cases too?
- Are positional qualifiers sufficient, or do unambiguous type aliases help?
