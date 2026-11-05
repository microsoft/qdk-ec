# Instruction Set Architecture

An **instruction set architecture** (ISA) declares the operations available at one layer of a qodec: their semantics, their typed operands, and the classical bits they consume and produce. An ISA is an abstract object. A qodec references one ISA per layer.

> **Representations.** See [YAML](../representations/yaml.md) for the file format and the [language APIs](model.md#representations) for in-memory access.

## Quick Start

A minimal ISA file with one instruction:
```yaml
name: Minimal
blocks: {qubit: 1}
instructions:
  - mnemonic: measure_z
    description: Measure Z and consume the qubit.
    in: [qubit]
    action:
      - observe: Z_0
```

| Field          | Meaning                                                              |
| -------------- | -------------------------------------------------------------------- |
| `name`         | Human-readable name for this instruction set                         |
| `description`  | Optional prose description of the ISA                                |
| `blocks`       | Block types: a name and the number of qubits it encodes      |
| `instructions` | List of instruction declarations, described below |
| `metadata`     | Optional [annotations](model.md#metadata) on the ISA |

Within each instruction:

| Field          | Meaning |
| -------------- | ------- |
| `mnemonic`     | Name used by circuit calls |
| `description`  | Required prose description; may be empty |
| `in` / `out`   | Positional operand lists: `in` declares inputs, `out` declares outputs. Each entry's position fixes its place in a flat index space. |
| `parameters`   | Classical parameter declarations: a name-to-type map. Calls supply arguments of these types. |
| `action`       | What the instruction does: a sequence of steps (`clifford`, `pauli`, `observe`, `rotate`, `stabilize`) |
| `flags`        | Names of additional reported bits, expected to be zero without faults |
| `metadata`     | Optional [annotations](model.md#metadata) on the instruction |

This instruction measures a qubit in Z and consumes it: the block appears in
`in` but not `out`. The measurement produces one classical bit, called an
outcome: 0 denotes eigenvalue +1 and 1 denotes eigenvalue -1.

## Overview

An ISA file declares two things:

1. **Blocks**: the block types in play, with their qubit counts.
2. **Instructions**: each with a mnemonic, typed operands, and an action.

The **action** specifies what the instruction does; a gadget supplies its
implementation. Tools can analyze supported action steps without recognizing
the instruction's mnemonic. Support varies by tool: storing an action does not
mean the current audit engine can verify it.

A more complete instruction, showing measurement and predicated correction:

```yaml
  mnemonic: measure_and_reset
  description: Measure qubit in Z basis and reset to |0⟩. Outputs 1 bit.
  in: [qubit]
  out: [qubit]
  action:
    - observe: "Z_0"
    - pauli: "X_0"
      if: ["outcomes[0]"]
```

This instruction entry uses the `qubit` block declared above. It measures Z,
applies X when the result is 1, and keeps the qubit in $|0\rangle$. The guard
names the first observe outcome as `outcomes[0]`; the outcome remains visible
to the caller.

## Concepts

### Blocks

An instruction set declares the block types it uses as a map from block type name to its qubit count:
```yaml
blocks: {carbon: 2}
```

This says: the carbon block type encodes 2 logical qubits per block. Instructions operate on blocks, meaning named instances of a block type, not individual qubits. The ISA lives at the logical level, so how a block is physically implemented is irrelevant.

Blocks play the part registers play in a classical ISA: the named things instructions read and write, addressed by the instruction rather than by hardware location. The difference is lifetime. A register file is fixed, whereas blocks are created by a preparation and destroyed by a measurement, so an instruction's `in`/`out` lists have to say which blocks survive it.

The `blocks` section is deliberately minimal: a name and a qubit count. The ISA trusts the author's action definitions. Verifying that a Clifford tableau actually corresponds to a valid transversal gate on a specific code requires the [code definition](code.md) (stabilizers, logical operators), which is a separate artifact.
The distinction between blocks and codes is intentional.  It allows a single ISA to be reused with different codes. See the [c4c6](../../examples/c4c6/) example.
### Instruction Structure

An instruction declares its quantum inputs (`in`), quantum outputs (`out`), classical parameters (`parameters`), and its semantics (`action`):
```yaml
  mnemonic: rotate_z0
  description: Rotate and measure Z, then apply the selected X correction.
  in: [carbon]
  out: [carbon]
  parameters: {correction: bit, theta: number}
  action:
    - rotate: {pauli: "Z_0", angle: theta}
    - observe: "Z_0"
    - pauli: "X_0"
      if: [correction, "outcomes[0]"]
```

**`in`**: Quantum inputs, as a positional list of block types: `[carbon]` declares one input block of type `carbon`. `in: [carbon, carbon]` assigns indices 0 and 1 to the first two-qubit block, then 2 and 3 to the second. That is the rule: each entry's qubits occupy the next range, giving one flat index space `0..n-1` across the side. Actions address that flat space directly (`Z_0`, `X_3`).

**`out`**: Quantum outputs, a positional list of block types (assigned its own flat indices, starting from 0) declaring which quantum state survives the instruction.

Both sides number their qubits from 0, so comparing the two capacities says what happens to the state. There are three cases:

- **`out` capacity equals `in`** — the same boundary positions survive. `in: [carbon]` with `out: [carbon]` transforms those qubits in place. The action need not be unitary: it may measure, reset, or use temporary qubits.
- **`out` capacity exceeds `in`** — the indices past the input range are fresh. Their state is whatever the action declares: a `stabilize` step forces the +1 eigenspace of the operators it names, so `stabilize: Z_0` prepares the encoded zero state and `stabilize: X_0` the encoded plus state. `in: [carbon]` with `out: [carbon, carbon]` keeps the first block at flat 0–1 and allocates a second at flat 2–3. An instruction with no `in` at all is a pure preparation.
- **`out` capacity is less than `in`** — the input indices past the output range are consumed: they are traced out at the end of the action. This does not itself produce a measurement outcome. If `out` is omitted it defaults to `[]`, so every input block is consumed.

The larger of the two capacities gives the **boundary** index range, not a limit
on the action's qubits. An unconditional `stabilize` can introduce temporary
qubits outside that range. Later steps can use those indices; no other step can
introduce them. Temporary indices may be sparse: preparing `Z_7` does not
introduce indices 1 through 6. They are local to one instruction invocation and
are traced out at its end, along with every other qubit absent from `out`.

Conditional `stabilize` can act on boundary qubits and already introduced
temporary qubits, but cannot introduce a new temporary qubit. Prepare it
unconditionally first, so its existence does not depend on a measurement result.

Changing block types does not create fresh logical indices when the boundary
positions are retained. The action must specify any state transfer or reduction;
the gadget must implement it between the actual encodings. See the merge
example under [Gate-based](#gate-based-surface-codes-color-codes).

A variadic operand is written
`[[carbon]]`: a one-element list naming the block type. Its number of blocks
comes from the call; each block contributes the next consecutive indices.

**`parameters`**: Classical input declarations, as a map from name to type. A call supplies an argument for each parameter it binds. A `bit` parameter accepts a runtime classical bit, such as a prior measurement result, and is referenced by name in `if`/`unless` guards. Other parameters accept literal arguments; the action uses their names in expressions such as `angle: theta` or `observe: p`:

| Type      | Meaning                                                  |
| --------- | -------------------------------------------------------- |
| `bit`     | A runtime classical input bit (condition-eligible)       |
| `number`  | A floating-point value (e.g., rotation angle)            |
| `integer` | An integer value                                         |
| `boolean` | A boolean value                                          |
| `string`  | A string value                                           |
| `pauli`   | A Pauli operator with named block references             |

**`flags`**: Extra classical bits an instruction reports alongside its outcomes. An array of names: `flags: [reject]` declares one flag named `reject`. By convention `flags:` follows `action:`, mirroring the gadget's `readouts` order (the `observe` outcomes first, then the flags).

Flags report evidence of faults. Under noiseless execution each flag must be
zero. A nonzero flag does not prescribe a response: a consuming tool may use it
for decoding, selection, or another policy. The instruction's action describes
its ideal behavior, not that fault-response policy.

The error-correction sense of the word fits as well as the classical one: there a *flag* is the bit from an ancilla placed to catch faults that would otherwise spread into correlated errors, after Chao and Reichardt's [Quantum error correction with only two extra qubits](https://arxiv.org/abs/1705.02329).

The instruction declares only the flag's name. The gadget supplies its equation.
See [Readouts](gadget.md#readouts) for the distinction between observable and flag
outputs, and [Checks](gadget.md#checks) for decoder-internal parities.

### Actions

An action is a sequence of steps, executed in array order. Steps are sequential by definition. The schema describes *what* happens, not *when* or *whether operations can overlap*. Parallelism, pipelining, and instruction-level timing are gadget-specific implementation concerns. The five step types can express any finite-qubit quantum channel: prepare temporary qubits, apply a unitary built from Pauli rotations and Clifford steps, and trace out the qubits absent from `out`. `observe` and `if`/`unless` additionally describe reported measurements and classical conditioning.

| Step      | What it does                                    | Example                         |
| --------- | ----------------------------------------------- | ------------------------------- |
| `clifford`| Apply a Clifford unitary (stabilizer tableau)   | `clifford: {Z_0: X_0, X_0: Z_0}` |
| `pauli`   | Apply a Pauli operator                          | `pauli: "X_0"`               |
| `observe` | Measure Pauli observables (produces outcomes) | `observe: ["Z_0", "Z_1"]` |
| `rotate`  | Pauli rotation: $\exp(-i \cdot \theta/2 \cdot P)$ | `rotate: {pauli: Z_0, angle: 0.785}` |
| `stabilize` | Prepare or reset into the +1 eigenspace of a Pauli operator | `stabilize: ["Z_0", "Z_1"]` |

Observables within a single `observe` or `stabilize` step are applied sequentially in array order. All unitaries (`clifford`, `pauli`, `rotate`) are defined up to global phase.

A size-one `observe` or `stabilize` step may drop the array brackets: `observe: Z_0` is shorthand for `observe: ["Z_0"]`, and `stabilize: Z_0` for `stabilize: ["Z_0"]`.

`observe` outcomes use the convention: **0 = eigenvalue +1, 1 = eigenvalue -1**.
Entries are Pauli strings, not named maps. Later guards address outcomes as
`outcomes[i]`, counted across all observe steps in the action. Repeated
measurements are distinct positions: `observe: [Z_0, Z_0]` produces two bits.

`observe` steps cannot be predicated. Predication would make the number of outcomes data-dependent, breaking the static guarantee that an instruction's outcome count is determined by its ISA definition alone. To conditionally measure, use an `observe` step and discard the result at the program level.

Most steps have straightforward semantics. `stabilize` deserves a note: whenever the step runs, it forces the state into the +1 eigenspace of the given operator. The implementation handles this internally (e.g., measure and correct). No outcome is produced, and the ISA guarantees success. It can prepare temporary qubits as well as reset existing ones. When the caller needs visibility into the outcome, for example in state distillation where failure discards the output, use `observe` instead and let the caller branch on the outcome.

### Temporary qubits and channels

This instruction measures whether two logical qubits are both $|1\rangle$.
It returns one bit and keeps both data qubits. Unlike measuring them separately
and taking the AND, it preserves superpositions within the subspace spanned by
$|00\rangle$, $|01\rangle$, and $|10\rangle$.

```yaml
name: JointMeasurements
blocks: {qubit: 1}
instructions:
  - mnemonic: measure_both_one
    description: Measure whether both qubits are one, without measuring each separately.
    in: [qubit, qubit]
    out: [qubit, qubit]
    action:
      - stabilize: Z_2
      - rotate: {pauli: Y_2, angle: 0.7853981633974483}
      - rotate: {pauli: Z_0 Y_2, angle: -0.7853981633974483}
      - rotate: {pauli: Z_1 Y_2, angle: -0.7853981633974483}
      - rotate: {pauli: Z_0 Z_1 Y_2, angle: 0.7853981633974483}
      - observe: Z_2
```

`stabilize: Z_2` prepares temporary qubit 2 in $|0\rangle$. The four commuting
rotations apply $R_Y(\pi)$ to it only when both data qubits are $1$, leaving it
unchanged otherwise. The angles are $\pm\pi/4$, written to floating-point
precision. Measuring `Z_2` reports the predicate, and qubit 2 is then traced out
because it is absent from `out`.

The two outcomes have projectors $\Pi_1=|11\rangle\langle11|$ and
$\Pi_0=I-\Pi_1$. This is not a Pauli measurement: `observe: Z_0 Z_1` would report
parity, which cannot distinguish $|00\rangle$ from $|11\rangle$.

These temporary qubits are part of the mathematical action. They do not require
a gadget to allocate corresponding physical ancillas or use the same circuit.
Only the declared input/output operation must agree. A preparation such as
`stabilize: Z_1 Z_2` constrains a joint eigenspace but does not select a unique
state within it; an action that needs a particular state must specify enough
preparation conditions.

### Pauli and Clifford indices

Pauli operators use a sparse format over the flat index space: `X_0 Z_2` means $X$ on flat qubit 0, $Z$ on flat qubit 2, identity elsewhere. Indices are 0-based across the whole instruction. For example, if an instruction has two carbon blocks (2 qubits each, occupying flat indices 0–1 and 2–3), `Z_0 Z_2` is ZZ between the first qubit of each block.

Clifford unitaries use stabilizer tableau format. A Hadamard on flat qubit 0 is `{Z_0: X_0, X_0: Z_0}` (Z maps to X, X maps to Z). The available qubits are the boundary indices plus temporary indices introduced by earlier `stabilize` steps, not just the indices named in the tableau. Omitted mappings are taken to be identity. The tableau is a YAML map from input Pauli generator to its image:
```yaml
# Inline (flow) form
clifford: {Z_0: X_0, X_0: Z_0}
```

```yaml
# Block form — only non-trivial mappings, identities inferred
clifford:
  Z_0: X_0
  X_0: Z_0
```

### Classical Bit Flow

`observe` steps produce outcomes in declaration order, and the calling program binds those bits to its own classical state. The ISA schema does not prescribe a program representation. For illustration only, an SSA-like rendering might look like:

```
(%0) = measure_z %a             # observe produces 1 bit and consumes %a
(%1) = rotate_z0 %b %0 1.5708    # %0 supplies the correction parameter
```

Inside an action, the referencing rule is the one from [Actions](#actions): an `observe` outcome is `outcomes[<i>]` (positional, across all `observe` steps), and a parameter is referenced by name. Outcomes a guard never references are simply visible to the caller.

### Predication

Any step except `observe` can be conditionally executed using `if` or `unless`:
```yaml
- pauli: "X_0"
  if: [correction, "outcomes[0]"]
```

This applies $X_0$ on flat qubit 0 when the parity (XOR) of the `bit` parameter `correction` and outcome `outcomes[0]` is 1. Use `unless` for even parity (XOR is 0).

A guard may name only `bit` parameters and earlier outcomes, the latter as
`outcomes[i]`. It cannot name one of the instruction's flags. At a call site,
a prior call's flag is a recorded bit that may be supplied to a `bit` parameter;
that is distinct from referring to the current instruction's flag by name.

Predication uses parity (XOR) semantics exclusively. This is a deliberate restriction: XOR is linear, which means Pauli corrections conditioned on XOR of measurement outcomes can be efficiently propagated through a stabilizer simulator without branching. In practice, a conditional Pauli with `if: [a, b]` can be absorbed into the Pauli frame, so the correction is tracked symbolically rather than applied physically. This is how Pauli frame tracking works in stabilizer simulation, and XOR predication is both necessary and sufficient for it. More complex classical logic (AND, majority vote, decoder-dependent corrections) belongs at the program level, not inside a single instruction.

### Variadic Blocks

Some instructions accept a variable number of blocks. Wrap the block type name in an array. The canonical example is `MPP`, Stim's measure-Pauli-product instruction, which takes an arbitrary Pauli over an arbitrary number of qubits:

```yaml
  mnemonic: mpp
  description: Measure an arbitrary Pauli product across N blocks (Stim MPP).
  in: [[carbon]]
  out: [[carbon]]
  parameters: {p: pauli}
  action:
    - observe: p
```

`in: [carbon]` declares one block; `in: [[carbon]]` declares a variable-length
group. The caller supplies the actual blocks. They share the flat index space,
so indices in the supplied Pauli are 0-based across the group. Fixed entries
can name different block types.

Example call sites (illustrative pseudocode):
```
(%0) = mpp [%a, %b] "Z_0 Z_2"
(%1) = mpp [%a, %b, %c] "X_0 X_2 X_4"
```

For variadic instructions with caller-supplied Paulis, the boundary range depends on the number of blocks actually passed. If the caller passes 2 carbon blocks (4 qubits, indices 0–3), index 4 is outside that range and may be used only after an unconditional `stabilize` introduces it. The `mpp` action above has no such preparation, so its supplied Pauli cannot reference index 4. The schema cannot check the actual call statically; consumers must validate its indices and temporary-qubit preparation order.

## A Complete Example: the Carbon Code

The [[12,2,4]] carbon code encodes 2 logical qubits per block. Here is an instruction set for it:
```yaml
name: Carbon
description: Instruction set for the [[12,2,4]] carbon code

blocks: {carbon: 2}

instructions:
  - mnemonic: prepare_zero
    description: Prepare a carbon block in |00⟩
    out: [carbon]
    action:
      - stabilize: ["Z_0", "Z_1"]

  - mnemonic: measure_z
    description: Measure both qubits in Z, destroy block. Outputs 2 bits.
    in: [carbon]
    action:
      - observe: ["Z_0", "Z_1"]

  - mnemonic: transversal_cx
    description: Transversal CNOT between two carbon blocks
    in: [carbon, carbon]
    out: [carbon, carbon]
    action:
      - clifford:
          X_0: X_0 X_2
          X_1: X_1 X_3
          Z_2: Z_0 Z_2
          Z_3: Z_1 Z_3

  - mnemonic: t0
    description: T gate (π/4 rotation) on first qubit
    in: [carbon]
    out: [carbon]
    action:
      - rotate: {pauli: "Z_0", angle: 0.7853981633974483}

  - mnemonic: rotate_z0
    description: Parameterized Z rotation on first qubit. Outputs 1 bit.
    in: [carbon]
    out: [carbon]
    parameters: {correction: bit, theta: number}
    action:
      - rotate: {pauli: "Z_0", angle: theta}
      - observe: "Z_0"
      - pauli: "X_0"
        if: [correction, "outcomes[0]"]

  - mnemonic: mpp
    description: Measure an arbitrary Pauli product. Outputs 1 bit.
    in: [[carbon]]
    out: [[carbon]]
    parameters: {p: pauli}
    action:
      - observe: p
```

(A real ISA would carry more, such as `reset`, `hadamard` and `measure_zz`, but each just repeats one of the patterns above: a `stabilize`, a `clifford`, or an `observe`.)

A program using this ISA (pseudocode, since a formal program representation is a separate concern not addressed by this schema):
```
prepare_zero %a
prepare_zero %b
transversal_cx %a %b
(%0, %1) = measure_z %b
(%2) = rotate_z0 %a %0 1.5708
t0 %a
(%3, %4) = measure_z %a
```

The `t0` instruction specifies a Z rotation by pi/4, up to global phase. It
does not claim that the code has a transversal T gate. A gadget might implement
it using gate teleportation and a magic state. Expose intermediate measurements
in the instruction's action only when those bits are part of its public
contract; otherwise they belong to the gadget's circuit and frame handling.

## A Physical-Level Example: Stim Compatibility

The schema handles physical-level circuits too. A block type declared as `{physical: 1}` gives you individual qubits, and stim's gate set maps directly:
```yaml
name: Stim
description: Physical-level ISA matching stim's gate set

blocks: {physical: 1}

instructions:
  - mnemonic: H
    description: Hadamard gate
    in: [physical]
    out: [physical]
    action:
      - clifford:
          Z_0: X_0
          X_0: Z_0

  - mnemonic: CX
    description: Controlled-X (CNOT)
    in: [physical, physical]
    out: [physical, physical]
    action:
      - clifford:
          X_0: X_0 X_1
          Z_1: Z_0 Z_1

  - mnemonic: M
    description: Measure in Z basis, destroy qubit. Outputs 1 bit.
    in: [physical]
    action:
      - observe: "Z_0"

  - mnemonic: R
    description: Reset qubit to |0⟩
    in: [physical]
    out: [physical]
    action:
      - stabilize: "Z_0"

  - mnemonic: MPP
    description: Measure an arbitrary Pauli product. Outputs 1 bit.
    in: [[physical]]
    out: [[physical]]
    parameters: {p: pauli}
    action:
      - observe: p
```

(`S`, `CZ`, `MX`, and `MR` round out the gate set, each another `clifford`, `observe`, or `observe`-then-`stabilize`.)

This illustrates instruction definitions, not the optional Python Stim adapter.
The adapter supports a [documented subset](../representations/source-formats.md#stim-parser-limits);
declaring a mnemonic such as `MPP` does not add source-parser support for it.

## Quantum Computing Paradigms

The schema's action steps (`clifford`, `observe`, `pauli`, `rotate`, `stabilize`) are paradigm-agnostic. They describe operations on encoded state, not the physical mechanisms that implement them. This section shows how different quantum computing paradigms map to the schema.

### Gate-based (surface codes, color codes)

The most direct mapping. Transversal and lattice-surgery operations on persistent code blocks.
```yaml
name: PatchOperations
blocks: {patch: 1, merged_patch: 1}
instructions:
  - mnemonic: controlled_x
    description: Controlled X from the first patch to the second.
    in: [patch, patch]
    out: [patch, patch]
    action:
      - clifford: {X_0: X_0 X_1, Z_1: Z_0 Z_1}
  - mnemonic: merge_zz
    description: Measure joint Z parity and retain one logical qubit.
    in: [patch, patch]
    out: [merged_patch]
    action:
      - observe: "Z_0 Z_1"
      - clifford: {X_0: X_0 X_1, Z_1: Z_0 Z_1}
```

Blocks persist across instructions and are passed between them positionally. The `in`/`out` declarations track which qubits survive (and, via the merge, which code now hosts them).

### Measurement-based (cluster states, one-way QC)

Computation proceeds by preparing entangled resource states and consuming them via single-qubit measurements. Each measurement outcome can be supplied as an argument to a `bit` parameter to condition later measurements.
```yaml
name: ClusterOperations
blocks: {qubit: 1}
instructions:
  - mnemonic: prepare_cluster_3
    description: Prepare a three-qubit linear cluster state.
    out: [qubit, qubit, qubit]
    action:
      - stabilize: ["X_0 Z_1", "Z_0 X_1 Z_2", "Z_1 X_2"]
  - mnemonic: measure_adaptive_x
    description: Measure X after the selected Z correction.
    in: [qubit]
    parameters: {feedforward: bit}
    action:
      - pauli: "Z_0"
        if: [feedforward]
      - observe: "X_0"
```

The feedforward pattern is captured by `bit` parameters (classical inputs from prior measurements) driving `if`/`unless` guards on later steps. The cluster's entanglement structure is encoded in the `stabilize` generators.

### Fusion-based (FBQC)

Resource-state preparation and joint measurements can use the same action
vocabulary. This example prepares a GHZ state and defines a destructive joint
XX/ZZ measurement on two qubits:
```yaml
name: ResourceOperations
blocks: {qubit: 1}
instructions:
  - mnemonic: prepare_ghz_4
    description: Prepare a four-qubit GHZ state.
    out: [qubit, qubit, qubit, qubit]
    action:
      - stabilize: ["X_0 X_1 X_2 X_3", "Z_0 Z_1", "Z_0 Z_2", "Z_0 Z_3"]
  - mnemonic: fusion_xx_zz
    description: Measure joint XX and ZZ and consume both qubits.
    in: [qubit, qubit]
    action:
      - observe: ["X_0 X_1", "Z_0 Z_1"]
```

The measurement consumes its two input qubits, not the whole resource from
which they came. Other resource qubits remain outside that call. A probabilistic
fusion implementation must also describe its failure outputs; this example
specifies only the ideal joint measurement.

### Bosonic / cat qubits

An instruction set can describe the logical qubits encoded in oscillator modes.
Its Pauli actions still act on finite-dimensional logical qubits; the qodec
code format does not describe oscillator wavefunctions or photon-number operators.
```yaml
name: CatLogicalOperations
blocks: {cat: 1}
instructions:
  - mnemonic: controlled_x
    description: Logical controlled X from the first block to the second.
    in: [cat, cat]
    out: [cat, cat]
    action:
      - clifford: {X_0: X_0 X_1, Z_1: Z_0 Z_1}
  - mnemonic: measure_x
    description: Destructive logical X measurement.
    in: [cat]
    action:
      - observe: "X_0"
```

This records logical intent. Which operations are practical, their error bias,
and how their physical implementations work are separate questions for the
target backend; the mnemonic alone makes no performance or noise guarantee.

### Physical level (stim compatibility)

A block type declared with a capacity of 1 gives bare physical qubits, and the schema becomes a typed wrapper around stim's gate set. See [the Stim-compatibility example](#a-physical-level-example-stim-compatibility) above.

### What varies across paradigms

The examples share an instruction format, not a physical model. Block lifetimes
and available operations belong in the instruction set. Geometry, noise, timing,
and hardware-specific failure mechanisms require additional information. A tool
must support that information before it can make claims about a platform.

## Design Principles

- **The action is authoritative.** Block names, descriptions, and mnemonics are for humans. The action is for machines.
- **Observable outputs follow the action.** Observe steps determine the observable bits. Output blocks and flag names are declared separately.
- **No physical implementation details.** The schema describes logical semantics. An action may use temporary qubits to specify a channel, but physical ancilla strategies, gate decompositions, and syndrome extraction circuits remain the gadget's responsibility.
- **Static where possible, parameterized where necessary.** Most instructions have fixed Pauli operators in their actions. When needed, `pauli`-typed parameters allow caller-supplied operators for general instructions like `MPP`.

## Under Consideration

The instruction format does not specify routing, scheduling, noise models,
decoding algorithms, or oscillator-level dynamics. Metadata can preserve
annotations, but does not give them shared semantics.

Circuits already represent ordered calls; see [Inline YAML](../representations/yaml.md#inline-yaml-calls).
General loops and classical processing beyond parity guards are not part of
that call format. Rotation approximation costs and precision targets belong to
the tool implementing the action, not to the exact rotation it specifies.

Parameters can postpone analysis until a call supplies their values. This
applies to rotation angles and bit conditions as well as Pauli arguments.

See [Validation](validation.md) for ISA validation rules.

## Appendix A: Formal Step Semantics

Each action step maps a quantum state $\rho$ and a classical bit vector $\mathbf{c}$ to a new state and bit vector. The quantum state includes the boundary qubits and any temporary qubits introduced so far. Let $n$ denote their current count; sparse temporary indices are labels, not additional intervening qubits. A qubit introduced by `stabilize` has the state that preparation specifies, with no implicit initial $|0\rangle$ state for unconstrained degrees of freedom.

At the end of the action, retain the quantum indices in `out` and trace out all
others. Keep the reported observe outcomes. Tracing out a qubit does not append
an outcome or condition on a measurement result.

**clifford** ($U$): Apply a Clifford unitary.

$$\rho \mapsto U \rho U^\dagger, \quad \mathbf{c} \text{ unchanged}$$

$U$ is specified by its action on the $n$-qubit Pauli generators (the stabilizer tableau). Defined up to global phase.

**pauli** ($P$): Apply a Pauli operator as a unitary.

$$\rho \mapsto P \rho P^\dagger, \quad \mathbf{c} \text{ unchanged}$$

Since $P$ is Hermitian and unitary, $P^\dagger = P$, so this is $\rho \mapsto P \rho P$.

**observe** ($[P_1, P_2, \ldots, P_k]$): Measure Pauli observables sequentially.

For each $P_i$ in order, measure and record:

$$\rho \mapsto \frac{(I + (-1)^{b_i} P_i)}{2} \rho \frac{(I + (-1)^{b_i} P_i)}{2} \cdot \frac{1}{\text{tr}(\cdots)}$$

where $b_i \in \{0, 1\}$ is sampled with probability $\text{tr}\!\left(\frac{I + (-1)^{b_i} P_i}{2} \rho\right)$. The bit is appended to the outcome vector:
**0 = eigenvalue +1, 1 = eigenvalue -1**. Later guards refer to it by position
as `outcomes[i]`. Named observe entries are not supported.

**rotate** ($P$, $\theta$): Apply a Pauli rotation.

$$\rho \mapsto e^{-i\theta P/2} \, \rho \, e^{+i\theta P/2}, \quad \mathbf{c} \text{ unchanged}$$

Defined up to global phase. When $\theta = \pi/4$ and $P = Z$, this is the T gate up to global phase.

**stabilize** ($[P_1, P_2, \ldots, P_k]$): Force the state into the +1 eigenspace, sequentially.

An unconditional step introduces each referenced nonidentity qubit index outside
the boundary range that has not appeared before. These qubits then remain
available to later steps until the instruction ends. A conditional step cannot
introduce temporary qubits. On newly prepared qubits the step specifies the
output eigenspace; on existing qubits it resets into that eigenspace. Partial
constraints do not choose a state or a recovery within the unconstrained space.

For each $P_i$ in order, the output state satisfies $P_i |\psi\rangle = +|\psi\rangle$. One possible implementation measures $P_i$, obtains outcome $b_i$, and, if $b_i = 1$, applies some recovery $C_i$ that anticommutes with $P_i$ ($\{C_i, P_i\} = 0$):

$$\rho \mapsto \frac{(I + P_i)}{2} \rho \frac{(I + P_i)}{2} \cdot \frac{1}{\text{tr}(\cdots)} \quad \text{if } b_i = 0$$
$$\rho \mapsto C_i \frac{(I - P_i)}{2} \rho \frac{(I - P_i)}{2} C_i^\dagger \cdot \frac{1}{\text{tr}(\cdots)} \quad \text{if } b_i = 1$$

In both cases, the resulting state is in the +1 eigenspace of $P_i$. The action does not distinguish recoveries that satisfy that guarantee. No outcome is produced, and $b_i$ is consumed internally. The implementation may use measurement and correction, or any other mechanism that achieves the same effect.

**Predication** (`if` / `unless`): When a step carries an `if: [r_1, r_2, \ldots]` condition, it is executed only when $r_1 \oplus r_2 \oplus \cdots = 1$. For `unless`, it is executed when the parity is 0. If the condition is not met, the step is a no-op ($\rho$ and $\mathbf{c}$ are unchanged).

## Appendix B: Lattice Surgery Implementation Example

This appendix illustrates a logical joint measurement and a possible intermediate
block grouping. The declarations alone do not specify or verify fault-tolerant
lattice surgery; that requires codes and physical gadgets.

Consider a logical joint ZZ measurement between two surface code patches:

**Logical ISA (surface code):**
```yaml
name: SurfaceCode
description: Logical instruction set for rotated surface code

blocks: {surface: 1}

instructions:
  - mnemonic: measure_zz
    description: Joint ZZ measurement between two patches. Outputs 1 bit.
    in: [surface, surface]
    out: [surface, surface]
    action:
      - observe: "Z_0 Z_1"
```

This says: measure the ZZ parity of two qubits. The ISA does not specify *how*. It could be lattice surgery, a transversal gate with an ancilla, or something else entirely.

**Patch-level ISA (intermediate abstraction):**

The intermediate ISA below groups two logical qubits into one block after
measuring their joint Z parity. Here `merged_patch` deliberately retains two
logical indices. This differs from a merge that retains only one logical qubit,
such as `merge_zz` in [the gate-based example](#gate-based-surface-codes-color-codes):
```yaml
name: PatchLevel
description: Logical regrouping around a joint parity measurement

blocks: {patch: 1, merged_patch: 2}

instructions:
  - mnemonic: merge_zz
    description: >
      Measure joint ZZ and group the two logical qubits into one block.
    in: [patch, patch]
    out: [merged_patch]
    action:
      - observe: "Z_0 Z_1"

  - mnemonic: split
    description: >
      Split a merged patch back into two individual patches.
    in: [merged_patch]
    out: [patch, patch]
    action: []
```

Both instructions retain flat indices 0 and 1. The first measures ZZ and changes
their block grouping; `split` restores the grouping with an identity action.
That states the logical requirement. It does not perform a physical code change
or show that a chosen pair of codes supports one.

**Implementation circuit** (pseudocode):
```
# Implement logical measure_zz on patches %a, %b:
(%0) = merge_zz %a %b → %merged    # consume %a, %b; produce %merged; output ZZ parity
split %merged → %a2 %b2            # consume %merged; produce %a2, %b2
```

A physical implementation must supply the stabilizer measurements, boundary
changes, and repeated rounds needed by the chosen codes. Its gadget must state
how measurement results produce the logical parity and any frame correction.
A decoding algorithm may use the gadget's checks, but is not encoded by the
instruction's XOR guards.

Each intermediate instruction can have its own gadget in the next layer.
Verifying these implementations requires the [code definitions](code.md),
encodings, and equations as well as the instruction actions. The
[surface-code example](../../examples/surface/) supplies a concrete protocol;
this appendix is only a grouping illustration.

## Appendix C: String Format Grammar

Pauli operators are strings. Clifford maps use those strings as keys and values.
The forms below are recommended for authored documents. qodec preserves action
strings without interpreting their algebra; consumers check the syntax and
semantics they support.

### Pauli Operator (Sparse Format)

A Pauli operator is a tensor product of single-qubit Paulis ($I$, $X$, $Y$, $Z$) with an optional sign ($+$ or $-$). The sparse format lists only the non-identity terms.

**Grammar (EBNF):**
```
pauli       = [sign] , [term_list] ;
sign        = "+" | "-" ;
term_list   = term , { ws , term } ;
term        = pauli_char , "_" , index ;
pauli_char  = "X" | "Y" | "Z" ;
index       = digit , { digit } ;
digit       = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
ws          = " " , { " " } ;
```

Each term names a flat index. Terms are separated by one or more spaces. The empty string (no terms, no sign) represents the identity operator $I$. When a sign is present with no terms, it represents a scalar (`"-"` is $-I$). The `+` sign is optional, so `X_0` and `+X_0` denote the same operator.

Use real signs ($+1$, $-1$) for these Hermitian operators. An imaginary phase
would make an observable or rotation axis non-Hermitian. Text preservation by
qodec does not certify that an expression is a valid operator.

**Examples:**

| String            | Operator                                |
| ----------------- | --------------------------------------- |
| `X_0`             | $X$ on flat qubit 0                     |
| `X_0 Z_1`         | $X_0 \otimes Z_1$ across qubits 0, 1     |
| `-Y_0 X_1`        | $-Y_0 \otimes X_1$                      |
| `Z_0 Z_1`         | $ZZ$ between qubits 0 and 1             |
| (empty string)    | $I$ (identity)                          |

**Rules:**
- Flat indices address the instruction's input/output capacities or temporary
  qubits introduced by preceding unconditional `stabilize` steps.
- Indices are non-negative integers, 0-based across the instruction. Audit
  checks bounds and preparation order; qodec preserves these declarations.
- Write each flat index at most once; do not rely on a consumer simplifying products on repeated indices.
- The ordering of terms does not matter: `X_0 Z_1` and `Z_1 X_0` denote the same operator.

### Clifford Unitary (Stabilizer Tableau Format)

A Clifford unitary on $n$ qubits is specified by its action on the $2n$ Pauli generators $Z_0, X_0, Z_1, X_1, \ldots, Z_{n-1}, X_{n-1}$. Each generator maps to a (possibly multi-qubit) Pauli operator under conjugation by the unitary: $U P U^\dagger$.

The tableau is a YAML or JSON **mapping**, not a string or an array of strings:

```yaml
clifford: {Z_0: X_0, X_0: Z_0}
```

**Identity mappings may be omitted.** Any generator not listed maps to itself.
Available indices come from the instruction's boundary capacities and temporary
qubits introduced by earlier unconditional `stabilize` steps, not from the map.

**Examples:**

| Mapping                                                    | Unitary  |
| ---------------------------------------------------------- | -------- |
| `{Z_0: X_0, X_0: Z_0}` | Hadamard |
| `{X_0: Y_0}` | S gate (Z identity image omitted) |
| `{X_0: X_0 X_1, Z_1: Z_0 Z_1}` | CNOT (identity images omitted) |

**Rules:**
- Each generator (left side of `:`) must be a weight-1 Pauli ($X_i$ or $Z_i$).
- The image (right side of `:`) is a Pauli operator in sparse format.
- Use normal YAML or JSON mapping syntax; strings containing colons are not mappings.
- The complete images, including implicit identities, must preserve Pauli
  commutation relations. Audit checks the mapping and index use; qodec stores
  the declaration without interpreting it. Analysis routines enforce their
  own preconditions before computing with the map.

### Named References

Pauli and numeric parameters are referenced by name within the action. For example, if an instruction declares `parameters: {p: pauli}`, then `observe: p` uses the Pauli argument supplied at the call. If it declares `parameters: {theta: number}`, then `angle: theta` uses the supplied angle argument.

In `if`/`unless` conditions, a `bit` parameter name refers to its supplied argument. An earlier `observe` outcome is referenced by position as `outcomes[i]`.

Use the parameter's declared name exactly. Outcome indices are non-negative
integers and refer only to measurements that precede the guarded step.

## Appendix D: Common Gate Tableaux

For readers unfamiliar with stabilizer tableau notation, this appendix shows how common quantum gates are represented as Clifford unitaries. The tableau describes how each Pauli generator transforms under conjugation by the gate: $U P U^\dagger$.

The intuition: a Clifford unitary is fully determined by how it transforms the Pauli generators. For example, the Hadamard swaps $X \leftrightarrow Z$, so $HXH^\dagger = Z$ and $HZH^\dagger = X$. The tableau `Z_0: X_0, X_0: Z_0` encodes exactly this.

### Single-Qubit Gates

All examples select a single qubit at flat index 0.

| Gate      | Matrix                                                                           | Tableau                                    | What changes                    |
| --------- | -------------------------------------------------------------------------------- | ------------------------------------------ | ------------------------------- |
| Hadamard  | $\frac{1}{\sqrt{2}}\begin{pmatrix}1 & 1 \\ 1 & -1\end{pmatrix}$                | `Z_0: X_0, X_0: Z_0`              | $X \leftrightarrow Z$           |
| S         | $\begin{pmatrix}1 & 0 \\ 0 & i\end{pmatrix}$                                   | `Z_0: Z_0, X_0: Y_0`              | $X \mapsto Y$, $Z$ unchanged   |
| S†        | $\begin{pmatrix}1 & 0 \\ 0 & -i\end{pmatrix}$                                  | `Z_0: Z_0, X_0: -Y_0`             | $X \mapsto -Y$, $Z$ unchanged  |
| X (Pauli) | $\begin{pmatrix}0 & 1 \\ 1 & 0\end{pmatrix}$                                   | `Z_0: -Z_0, X_0: X_0`             | $Z \mapsto -Z$, $X$ unchanged  |
| Z (Pauli) | $\begin{pmatrix}1 & 0 \\ 0 & -1\end{pmatrix}$                                  | `Z_0: Z_0, X_0: -X_0`             | $X \mapsto -X$, $Z$ unchanged  |
| Y (Pauli) | $\begin{pmatrix}0 & -i \\ i & 0\end{pmatrix}$                                  | `Z_0: -Z_0, X_0: -X_0`            | $Z \mapsto -Z$, $X \mapsto -X$ |

### Two-Qubit Gates

Examples select two qubits at flat indices 0 and 1.

| Gate      | Tableau (non-identity mappings only)                             | What changes                                               |
| --------- | ---------------------------------------------------------------- | ---------------------------------------------------------- |
| CNOT      | `X_0: X_0 X_1, Z_1: Z_0 Z_1`                      | Control X spreads to target; target Z spreads to control   |
| CZ        | `X_0: X_0 Z_1, X_1: Z_0 X_1`                      | Each qubit's X picks up the other's Z                      |
| SWAP      | `Z_0: Z_1, X_0: X_1, Z_1: Z_0, X_1: X_0`     | All generators swap between the two qubits                 |

Flat indices make multi-qubit operations unambiguous: `X_0: X_0 X_1` shows that control-X from qubit 0 spreads to qubit 1. For a block type with a capacity of 2, like `carbon`, two blocks occupy flat indices 0–1 and 2–3, so a transversal CNOT between them spreads across the block boundary with `X_0: X_0 X_2, …` (see the [Carbon example](#a-complete-example-the-carbon-code)).

Identity mappings (e.g., `Z_0: Z_0` for CNOT) are omitted above for brevity. In the schema, they may be included explicitly or left out, and the effect is the same.
