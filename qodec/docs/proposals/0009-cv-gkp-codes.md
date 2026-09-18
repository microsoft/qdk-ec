# 0009 - Continuous-variable (GKP) codes

**Status:** deferred

## Context

An outer qubit code may encode each of its physical qubits in an oscillator.
Correcting that inner encoding requires a continuous measurement and a
displacement chosen from its result. A qodec can describe the outer qubit code,
but its Pauli operators and binary readouts do not describe those oscillator
operations.

This proposal considers [Gottesman-Kitaev-Preskill encoding](https://arxiv.org/abs/quant-ph/0008040)
(GKP) beneath qubit layers. Broader continuous-variable (CV) support is not assumed.
It is deferred until a worked oscillator-level protocol is in scope.

Leaving the oscillator implementation outside qodec is sufficient if a tool
only needs abstract qubit operations. It is insufficient when the protocol must
specify the inner correction: which continuous quantity is measured, which
displacement follows, and what information reaches the outer decoder. Describing
that boundary would let tools assess the complete encoding rather than assume
the oscillator behaves like an ideal physical qubit.

## Benefits

- Analyze the oscillator correction and outer qubit code together, rather than
  assume the inner encoding behaves like an ideal qubit.

## Drawbacks

- The model needs operators and readouts beyond Paulis and bits.
- Simulation and decoding need different algorithms.
- The oscillator layer must define how its measurements become logical bits.

## Proposal

### Bottom-layer scope

Evaluate an oscillator code and its lowering beneath the qubit layers:

- a `mode` block type and a **displacement** operator alongside `pauli`;
- real-valued readouts confined to GKP gadgets;
- a consumer computes the displacement from the real syndrome. Unlike the
  Boolean aggregation in [0008](0008-optional-or-of-parities-flags.md), this
  requires a runtime real-valued argument, not just a `bit` parameter.

Keep the qubit layers unchanged if the oscillator-to-qubit interface permits it.
That interface is not yet specified.

Illustrative only (**not** current grammar):

```yaml
code:
  block: mode
  stabilizers: ["D(q: 2*sqrt(pi))", "D(p: 2*sqrt(pi))"]
  x: ["D(q: sqrt(pi))"]
  z: ["D(p: sqrt(pi))"]
```

## Alternatives

- Keep the oscillator implementation outside qodec: no new algebra in the model.
- Generalize operators and readouts throughout qodec: supports more CV protocols,
  but changes code and parity contracts beyond the bottom layer.

## Discussion

### Names and units

Two candidate names are shown: `mode`, the quantum-optics term for an oscillator,
and `D`, the displacement symbol. `qubit` and `pauli` would imply the existing
binary model. No binding API or compatible schema is specified.

The quadratures `q` and `p` need units, a commutator, and a displacement sign
convention. The example's lattice spacing depends on these choices.

### Interpretation changes

| qodec today | CV/GKP needs |
|---|---|
| operators: Pauli, GF(2) symplectic | displacements, **real** symplectic |
| readouts: bits (GF(2)) | **reals** (homodyne), `mod √π` |
| Boolean ideal-zero flags | continuous syndrome values need a separate interpretation before producing Boolean flags |
| checks: XOR-to-zero parities | real-linear combinations mod a lattice |
| decoder: distance / GF(2) | closest-point under Gaussian noise |
| corrections: predicated Paulis | displacements by `f(real)` |

Starting examples include [GKP error correction](https://arxiv.org/abs/1908.03579),
a [photonic architecture](https://arxiv.org/abs/2010.02905), and an
[experimental encoding](https://arxiv.org/abs/1907.12487).

## Open Questions

- Does the first worked example need bottom-layer support only or a general
  real-valued operator model?
- What types distinguish real-valued readouts and displacements from binary
  parity terms, and how does the boundary yield the qubit layer's bits?
- Does qodec model the Gaussian channel at all, or only the code + lowering (decoder and
  noise stay external, as for qubit codes)?
- Is the right unit a single mode, or also multi-mode (analog) stabilizers used in some GKP
  variants?
- Which concatenated GKP example and consumer establish the required scope?
