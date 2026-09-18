# 0001 - `mix`: a stochastic action step

**Status:** proposed

## Context

An ideal circuit cannot tell us whether error correction reduces errors.
That depends on which faults occur and how often. For example, two qubits
flipping together with probability p can defeat a code that handles independent
flips well. Leaving out that distinction leaves out part of the experiment.

An instruction cannot currently say "apply X with probability p." Noise must
be expressed in the circuit's source language or supplied by a consumer.
`mix` would let the qodec itself state the noisy operation, so a simulator and
a fault-analysis tool need not rely on separate descriptions of it. This belongs
in qodec when the operation's probability is part of what the author wants to
specify; it need not replace external noise models for ideal protocols.

## Benefits

- Share noisy or deliberately randomized experiments without separate scripts
  defining their probabilities for each tool.

## Drawbacks

- Simulators and audit tools need support for stochastic actions.
- Putting noise in a circuit ties it to a noise scenario; an external noise
  model may be easier when comparing many scenarios.
- A noisy action is not equivalent to its ideal action. Tools must distinguish
  those contracts.
- `mix` does not describe noise with memory or report which branch occurred.

## Proposal

Add `mix`: a list of actions with probabilities. Initially allow Pauli,
Clifford, and rotation branches that emit no readouts. Proposed syntax,
assuming a one-qubit block type `qubit`:

```yaml
mnemonic: depolarize1
description: Apply each nonidentity Pauli with probability p.
in: [qubit]
out: [qubit]
parameters: {p: number}
action:
  - mix:
      - {pauli: X_0, probability: p}
      - {pauli: Y_0, probability: p}
      - {pauli: Z_0, probability: p}
```

### Channel semantics

For branch channels $\mathcal E_j$ with probabilities $p_j$,

$$
\mathcal M(\rho) = \sum_j p_j\mathcal E_j(\rho)
  + \left(1-\sum_j p_j\right)\rho.
$$

Each probability is finite and nonnegative, and their sum is at most one.
The remaining probability applies identity. The example requires
$0 \le p \le 1/3$: `p` is each Pauli's probability, not the total error
probability. Interpretation must reject invalid probabilities rather than
renormalize them or omit a branch.

- One invocation makes one mutually exclusive draw. Separate `mix` steps are
  independent unless another explicitly specified model introduces correlation.
- `probability` is legal only inside a branch. A numeric literal or a named
  numeric parameter supplies its value; arbitrary expressions are not proposed.
- Branches act on the same live quantum interface and produce no readouts.
  `observe` is excluded, so the mixture cannot make output-bit arity stochastic.
- A branch condition uses the ordinary action guard semantics. Draw first;
  if its condition is false, that draw applies identity. The probability itself
  does not become an implicit conditional expression.
- The selected branch is unobserved. A heralded channel needs an explicit
  outcome contract; `mix` does not expose a sampled branch as a flag.

## Alternatives

- Supply noise externally: no format change, but the qodec does not describe
  the experiment's noise.
- Use source-language noise: no new action type, but tools must interpret that
  language to know the instruction's noisy behavior.
- Couple to an environment and discard it: a mathematical description of the
  channel, but less direct than giving branch probabilities. Consumer support
  for constructing or analyzing such a dilation is not assumed.
- Use [fault summaries](0002-fault-models.md): these describe propagated error
  effects, not the quantum operation that causes them.

## Discussion

### Names and scope

Two field names are proposed. `mix` names a probabilistic mixture; `noise`
would exclude intentional randomness. `probability` gives a branch's chance;
`weight` would wrongly suggest automatic normalization. Binding APIs are unspecified.

Setting rates to zero recovers the ideal action in the example. Tools must not
make that substitution implicitly. Calibration and parameter sweeps remain
consumer inputs unless fixed in the artifact.

Implementation needs the new branch representation in Rust, schemas, and bindings,
plus consumer support for probability checks and sampling. The shared loading
and compatibility rules in the [README](0000-README.md#status) still apply.

## Open Questions

- Which experiment and consumers will demonstrate the need over external noise injection?
- Are unitary branches sufficient initially? Nonunitary branches and nested
  mixtures need explicit interface and interpretation rules before inclusion.
- How should a consumer declare support for mixtures and non-Clifford branches?
- How should tools pair ideal and noisy actions for conformance checking?
- What provenance ties a bound probability set to a calibration or benchmark?
- Which cross-call correlations should remain outside `mix` entirely?