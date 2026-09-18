# 0008 - Triggers for adaptive syndrome extraction and error correction

**Status:** proposed

## Context

An error-correction protocol may start with a short verification round, then
use its results to decide whether to extract more syndrome information or
perform recovery. Always running the extra work costs time and operations;
discarding every flagged run does not implement recovery and continuation.

qodec can pass a prior readout to an instruction's `bit` parameter, but each
gadget flag is currently one parity. Some decisions depend on several parities:
for example, perform more checks if any verification test fails, or use a
particular recovery only for selected combinations of results. Those decisions
need more than a single XOR. `select` describes post-selection, not a new
trigger bit or a conditional extraction schedule.

This proposal asks how a gadget should declare a trigger that its callers can
use consistently. The initial candidate is an OR of ideally zero parities for
the common "any verification test failed" decision. It is not a complete
language for adaptive protocols or a claim that every decision belongs in a flag.

## Benefits

- Let an adaptive runtime reserve extra extraction or recovery for runs that
  need it, using the gadget's declared decision rather than a caller-specific rule.

## Drawbacks

- Nonlinear triggers require more than parity propagation in consumers.
- One trigger bit loses the result pattern a decoder may need to choose a recovery.
- A trigger definition does not supply conditional scheduling or solve variable
  readout counts.

## Proposal

Start with a flag that aggregates verification results. For "any test failed,"
OR is sufficient: unlike XOR, it remains one when two tests fail together.
Use the following candidate shapes at a flag's readout position:

- A flat list of parity terms means XOR, as today.
- A list of parity lists means OR of their XOR results. This is proposed syntax.

The outer list is one flag definition, not multiple output bits. Its position
in `readouts` is fixed by the instruction's outcome count and declared flags.
Circuit indices inside each parity remain zero-based positions in the full
record, including outcomes and flags from called gadgets.

- Only flag positions permit this aggregate; checks and observable readouts remain parities.
- Each constituent references the lower circuit record, not a peer flag, and
  must be ideally zero. Audit establishes that property; syntax does not.
- The output uses ordinary [bit-argument threading](../representations/yaml.md#threading-readouts-into-bit-parameters).
  No call-level control flow is added.
- Reuse parity terms inside each list. Reject mixed scalar/nested lists.
  DNF and CNF are alternatives below, not additional forms in this candidate.

### Example: request recovery

`extract` declares the flag; its gadget realizes it in `readouts:` as an OR of three
single-bit checks:

```yaml
readouts:
  - triggered:
      - ["circuit.readouts[1]"]
      - ["circuit.readouts[3]"]
      - ["circuit.readouts[5]"]
```

Assume `extract` reports only the flag `triggered`. As the first call, it puts
that flag at circuit record position zero. A later `correct` instruction uses
its `bit` parameter `c` to guard a fixed correction:

```yaml
- extract: [0]
- correct: [0, c: "circuit.readouts[0]"]
```

This demonstrates passing the decision, not choosing the correction from a
syndrome. A protocol needing different recoveries must retain the relevant
results and specify how the consumer chooses among them.

## Alternatives

### Keep the decision in the consumer

Expose separate parity flags and let the decoder or runtime choose the next
step. This needs no format change and preserves the individual results. Prefer
it when the decision is a decoder policy rather than part of the gadget's
declared output. Use `select` only when rejecting the run is the intended response.

### DNF: OR of AND conditions

A protocol might act only when at least two of three tests fail. For test bits
$s_0,s_1,s_2$, that trigger is

$$
(s_0 \land s_1) \lor (s_0 \land s_2) \lor (s_1 \land s_2).
$$

This is disjunctive normal form (DNF): an OR of AND conditions. It suits a list
of result patterns that each require action. Negated tests allow conditions
such as "the first test failed and the second passed." The tests could be
named bits or parities, but that choice and their reference scope need a contract.

DNF covers decisions that OR-of-parities cannot generally express, but is not
needed just to detect any failed test. It also introduces conjunction and
possibly negation, and enumerating acceptable patterns can become large.
`select` already uses an OR-of-AND shape for post-selection; reusing that shape
would still require a separate meaning for computing a trigger.

### CNF: AND of OR conditions

The same two-of-three trigger can be written

$$
(s_0 \lor s_1) \land (s_0 \lor s_2) \land (s_1 \lor s_2).
$$

This is conjunctive normal form (CNF): every OR clause must hold. It suits
decisions stated as several requirements, each with acceptable alternatives.
It is less direct than DNF when an author wants to enumerate recovery cases.

With negation, both DNF and CNF can express any Boolean function. Neither is
always compact, and converting between them can expand the expression
exponentially. Supporting both would add two spellings without adding expressive
power. Choose one only if a concrete protocol needs more than the initial trigger;
do not make normal-form conversion part of ordinary load/save.

### Why not start with a general Boolean form?

For ideally zero test parities, "do more work if any failed" needs only OR.
Richer forms matter when particular combinations select different actions.
That may be a decoder or scheduling program, not a single gadget flag.

DNF, CNF, and negation are not inherently incompatible with flags: the examples
above are zero when all test bits are zero. Conversely, a formula such as
$\lnot s_0$ is one in that case. Audit must establish that a proposed flag is
zero under the protocol's faultless execution, regardless of expression form.
Checking the all-zero input alone is insufficient if the constituent results
can vary ideally. Nonzero ideal outcomes need a different readout role.

## Discussion

### Names and representation

No field names are added. `any_of` is an alternative to nesting; it names the
operation, whereas `checks` would suggest a separate check collection.
Extend the readout representation rather than creating another flag API.
Binding names remain unspecified.

### A trigger is not a schedule

Repeated syndrome extraction may change the number of measurements in a run.
Existing bit arguments can control supported instruction actions, but do not
make conditional observations compatible with a statically sized readout record.
An end-to-end adaptive extraction example must define a fixed readout interface
or a separate consumer scheduling contract. This proposal does not silently
add variable-length calls, loops, or conditional measurement support.

### Candidate applications

These are protocols to examine, not verified consumers of the proposed form.
For each, identify the measured results, the decision, and the next operation
before choosing OR, DNF, CNF, or consumer-only logic:

| Application | Candidate response |
| --- | --- |
| [Reichardt flag fault tolerance](https://arxiv.org/abs/1804.06995), [Chao-Reichardt](https://arxiv.org/abs/1705.02329) | Perform full extraction when a verification trigger fires. |
| [Adaptive Steane/Knill error correction](https://arxiv.org/abs/quant-ph/0312190) | Perform correction or another extraction on a nonzero syndrome. |
| [Chamberland-Beverland flag fault tolerance](https://arxiv.org/abs/1708.02246) | Choose an additional extraction from flag results. |
| [Erasure-biased codes](https://arxiv.org/abs/2201.03540) | Replace or otherwise handle a located erasure. |
| [Fusion-based computation](https://arxiv.org/abs/2101.09310) | Adapt the route to a reported fusion outcome. |
| [Leakage handling](https://arxiv.org/abs/2102.06132) | Reset or replace a qubit after detecting leakage. |

A fusion or preparation can fail without faults. Such a herald is not an
ideal-zero flag and needs a different readout role.

### Consumer and format support

Update Rust, schemas, bindings, and the [gadget guide](../concepts/gadget.md).
Unsupported consumers must report an error, not interpret the nested list as
XOR or omit the flag.

## Open Questions

- Which adaptive extraction or error-correction protocol needs a declared
  trigger, rather than consumer-local logic? Show its decision and next operation.
- Is "any failed test" sufficient, or do selected result combinations require
  conjunction or negation? Compare OR-of-parities, DNF, and CNF on that example.
- Does the adaptive step retain a fixed readout interface? If not, what separate
  scheduling and result-count contract is needed?
- Which individual results must remain available after computing the trigger?
- Is nesting clear enough, or should an explicit operation key distinguish it?
- What do an empty outer list and empty constituent parities mean? Define these
  cases without changing the existing meaning of an empty parity.
- Can integer terms be used in constituents under the ideal-zero requirement?
- If a Boolean form is needed, what are its input scope, size limits, and
  evaluation rules, and which decisions should remain in the decoder?
