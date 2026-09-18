# 0002 - Decode-side fault models

**Status:** deferred

## Context

Suppose two faults flip the same check bits, but only one also flips a logical
output. The decoder must choose between these explanations using their
probabilities. A gadget's check equations identify the syndrome bits; they do
not describe these candidate faults or their logical effects.

A circuit analyzer can derive a fault model, but qodec has no artifact for
storing it against the gadget's checks, readouts, and output encodings.

Without a shared description, each decoder must repeat the circuit analysis
or depend on the analyzer's private format. That makes it harder to compare
decoders on exactly the same faults. A qodec fault model would connect the
analyzer's result to the protocol definitions it describes, without requiring
qodec to perform the analysis itself.

## Benefits

- Compare decoders on the same faults without repeating circuit analysis.
- Compose gadget fault models without counting a shared event as independent faults.

## Drawbacks

- Fault models become stale when their circuit or noise assumptions change.
- Pauli flips do not describe every kind of noise.
- Incorrect composition can double-count events or lose correlations.
- qodec would need another artifact type; an external format might suffice.

## Proposal

Add a fault-model artifact referencing one gadget. Several models may describe
the same gadget. `*.faults.yaml` is a filename convention, not a type tag.
Proposed syntax:

### Artifact and fault effects

`faults` is a list of events, not a map of all possible errors. `constants` is
a sparse map from local parameter names to numeric values. Each event supplies
its probability and the bits or signs it flips:

```yaml
gadget: ./idle.gadget.yaml
constants:
  p_correlated: 0.001
faults:
  - flips: ["checks[0]"]
    probability: 0.001
  - flips: ["checks[0]", "checks[1]", "out[0].x[0]"]
    probability: p_correlated
```

The second event flips two check bits and a logical-X sign. These are three
effects of one event, not three independent faults.

| Proposed target | Meaning |
| --- | --- |
| `checks[i]` | Flip the i-th check result. |
| `readouts[i]` | Flip the i-th gadget readout. |
| `out[entry].x[i]` / `out[entry].z[i]` | Flip an output logical sign. |

All indices are zero-based and local to the referenced gadget. `entry` indexes
its output encodings; the final index selects the check, readout, or logical
operator in the named collection. Omission from `flips` means no effect from
that event, not a claim that the omitted quantity is immune to every fault.

A residual $\bar Z_i$ flips the logical-X sign, a residual $\bar X_i$
flips the logical-Z sign, and $\bar Y_i$ flips both. Stabilizer disturbances
are described through their check effects initially; whether an inter-gadget
summary also needs an output syndrome state is an open composition question.
Fault-effect targets are a new contract, not the existing parity grammar.

### Probability

Candidate probability forms are a numeric literal, a parameter, a scaled
parameter such as `{parameter: p_data, scale: 0.333}`, or a sum of terms.
Interpreters must know whether such sums combine mutually exclusive events
or approximate a distribution; the representation cannot silently choose that.

Conditional rates could describe an erasure herald:

```yaml
constants: {p_erase: 0.5}
faults:
  - flips: ["checks[0]"]
    probability:
      - {parameter: p_erase, if: ["circuit.readouts[0]"]}
```

Here the referenced bit is assumed to be the herald. `if` and `unless` would
test XOR predicates; their conditional-probability interpretation must be
specified, including when the herald is available. Access to an attached field
such as `circuit.readouts[0].lost` would additionally depend on
[typed readout fields](0003-typed-readout-fields.md).

### Correlated faults across gadgets

Explore splitting one event into parts identified by
`correlation: {id, index, size, via?}`. Positional boundary ports route parts
across the composition graph:

```yaml
faults:
  - flips: ["checks[0]"]
    probability: p_boundary
    correlation: {id: bdy_x, index: 1, size: 2, via: ["in[0]"]}
  - flips: ["checks[0]"]
    probability: p_boundary
    correlation: {id: bdy_x, index: 0, size: 2, via: ["out[0]"]}
```

A composition tool would match connected parts and form one event, without
multiplying its probability once per part. Candidate consistency rules include
unique `(id, index)` pairs locally, agreement on `size`, and routing through
declared boundary entries. Repeated gadget instances and incomplete boundary
events need an explicit scoping and truncation policy before this is a contract.

## Alternatives

- Keep fault models in decoder tooling: no new qodec artifact, but tools must
  agree on an external format and its references.
- Recompute from the circuit and noise inputs: avoids stale summaries but
  repeats the analysis for each consumer.
- Use [noise actions](0001-mix-stochastic-action-step.md): describes operations,
  not their propagated effects on gadget checks and outputs.
- Store a decoder-specific graph: convenient for that decoder, but may lose
  effects or correlations another decoder needs.

## Discussion

### Names and scope

The sketch has 14 field names; binding APIs remain unspecified.

| Name | Naming reason |
| --- | --- |
| `gadget` | Existing artifact term; `source` means circuit text or a file. |
| `constants` | Local values; `calibration` would imply their origin. |
| `faults` | Events; `errors` could mean the resulting state. |
| `flips` | Bit/sign changes; `targets` omits the effect. |
| `probability` | Event chance; `weight` need not be a probability. |
| `parameter` | Existing named-input term; no separate `variable` term. |
| `scale` | Multiplier; not the resulting `probability`. |
| `if` | Existing parity condition; no new `when` spelling. |
| `unless` | Existing complementary condition; no `not_if` alias. |
| `correlation` | Shared event; `group` does not imply dependence. |
| `id` | Event identity; `name` could mean a label. |
| `index` | Part position; `offset` suggests a circuit record index. |
| `size` | Part count; `weight` is overloaded in decoding. |
| `via` | Connecting ports; `support` already names encoding placement. |

An analyzer could produce a model once for several decoders or gadget instances.
Composition must account for boundary correlations; a local summary is not
automatically sufficient for a concatenated or adaptive protocol.

Tie the model to its circuit, encodings, equation order, and noise inputs.
Report stale references rather than omit events. Audit checks agreement and
completeness; loading does not prove either. Residuals must compose with existing
[frame corrections](../concepts/gadget.md#frames) without applying a correction twice.

## Open Questions

- Which analyzer and decoder will demonstrate the artifact?
- Should qodec own the artifact, or should it be an external format referencing
  a qodec? How is it explicitly bound into a persisted experiment?
- Which event independence, exclusivity, conditioning, and approximation rules
  must the first version support?
- What provenance is sufficient to detect stale fault effects?
- Are output logical residuals sufficient for the chosen composition workflow,
  or must it also describe output syndrome and correlated boundary state?
- How are event identities scoped across repeated gadgets and incomplete parts?
- How does a summary interact with `mix`, typed heralds, and frame corrections?