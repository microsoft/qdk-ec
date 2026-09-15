# Code

A **code definition** lists stabilizer generators and the physical Pauli
operators representing each logical X and Z. An encoding places those operators
on the blocks used by a gadget's circuit.

See [Validation](validation.md) for code and encoding requirements.

> **Representations.** See [YAML](../representations/yaml.md) for the file format and the [language APIs](model.md#representations) for in-memory access.

## Quick Start

A minimal code definition, the 3-qubit repetition code:
```yaml
name: repetition_3
stabilizers:
  - "Z_0 Z_1"
  - "Z_1 Z_2"
x:
  - "X_0 X_1 X_2"
z:
  - "Z_0"
```

| Field         | Meaning                                                              |
| ------------- | -------------------------------------------------------------------- |
| `name`        | Identifier for this code instance                                    |
| `stabilizers` | Stabilizer generators in Pauli sparse format                         |
| `x`           | One logical $X$ operator per logical qubit (Pauli sparse format)     |
| `z`           | One logical $Z$ operator per logical qubit (Pauli sparse format)     |

This defines a `[[3,1,1]]` stabilizer code: three physical qubits and one logical
qubit. It corrects one bit flip, but a single Z is a logical error, so its
distance against arbitrary Pauli errors is one. `x[i]` and `z[i]` name the
logical operator pair at position $i$. Here logical X acts on all three qubits;
logical Z can be represented by Z on any one of them.

## What This Captures

The [instruction set](instruction-set.md) can declare a block type `surface`
with `blocks: {surface: 1}` without choosing a code. A layer binds that type to a code,
such as a distance-five surface code. Gadgets then place its code qubits on
circuit blocks through their `in` and `out` encodings. At an intermediate
layer, those circuit blocks may themselves be encoded.

## Fields

### Required

**`name`** (string): Identifier for this code definition. The layer's `codes`
map binds an instruction-set block type to the code artifact; the code's name
need not match the block type.

**`stabilizers`** (array of strings): Generators in [Pauli sparse format](instruction-set.md#pauli-operator-sparse-format).
For a valid code, they generate the stabilizer group $S$, whose common +1
eigenspace is the code space. Dependent generators are allowed and preserved.
An analysis tool may compute an independent basis; qodec does not remove
generators or renumber references during loading. These operators are distinct
from a gadget's measurement-based parity checks.

**`x`** (array of strings): One logical $X$ operator per logical qubit, in Pauli sparse format. The array index determines the logical qubit index: `x[0]` belongs to logical qubit 0, `x[1]` to logical qubit 1, and so on.

**`z`** (array of strings): One logical Z operator per logical qubit.
For a valid code, `z[i]` anticommutes with `x[i]`, and the arrays have equal
lengths. Audit checks these requirements; unfinished drafts may have unequal lists.

> **Why two parallel arrays?** Splitting them into separate top-level lists matches the [property-path references](gadget.md#parity-equations-property-path-references) used in gadget parity equations, where `in[<entry>].x[<i>]` and `in[<entry>].z[<i>]` name a specific operator independently of its partner.

### Optional

**`description`** (string): Human-readable description. Code distance, if known, can be mentioned here in prose: distance is intentionally not a field of this format. A declared-but-unverified distance tends to drift out of sync with the actual code, so tools that need distance should compute it from the stabilizers.


**`metadata`**: Optional [annotations](model.md#metadata).

## Validation

qodec checks field types and Pauli syntax used for dimension inference.
Stabilizers and logical operators use unsigned Paulis: `Z_0` is accepted, but
`+Z_0` and `-Z_0` are rejected. Codes cannot declare a negative operator sign;
instruction actions may still use signed Paulis. An index must be less than
the host's `usize::MAX` so that its dimension fits in a machine-sized integer.

X/Z lists can be edited independently and need not have matching lengths while drafting.
These mismatches and algebraic mistakes do not prevent loading or saving. Audit checks:

- **Stabilizers commute.** All pairs of stabilizer generators must commute: $[S_i, S_j] = 0$ for all $i, j$.
- **Logical operators are symplectic.** For each logical qubit $i$: $\{X_i, Z_i\} = 0$ (anticommute). For $i \neq j$: $[X_i, X_j] = 0$, $[Z_i, Z_j] = 0$, $[X_i, Z_j] = 0$ (commute).
- **Logical operators commute with stabilizers.** $[L, S] = 0$ for every logical operator $L$ and stabilizer generator $S$.

Analysis routines must enforce the mathematical preconditions they rely on:

- **Logical operators are non-trivial.** No logical operator is in the stabilizer group. Equivalently, the logical operators are independent modulo the stabilizer group.

Audit also checks these declaration requirements:
- **Qubit indices.** Non-negative indices name positions within the encoded
  block, not circuit labels. For a uniform target block size `stride`, position
  `k` selects offset `k % stride` within `support[k // stride]`. Audit checks
  that nonidentity operators fit the encoding's support; mixed block types
  contribute their respective sizes. See [Gadget](gadget.md).
- **Consistency with the block type.** The code's `len(x)` and `len(z)` must
  equal the logical-qubit count declared for its block type in the instruction
  set's `blocks` map (`Block.encodes` in the API).

## Examples

### Steane `[[7,1,3]]` Code

The smallest color code, encoding 1 logical qubit in 7 physical qubits at distance 3:
```yaml
name: steane
description: "Steane [[7,1,3]] color code (distance 3)"

stabilizers:
  - "X_0 X_1 X_2 X_3"
  - "X_1 X_2 X_4 X_5"
  - "X_2 X_3 X_5 X_6"
  - "Z_0 Z_1 Z_2 Z_3"
  - "Z_1 Z_2 Z_4 Z_5"
  - "Z_2 Z_3 Z_5 Z_6"

x:
  - "X_0 X_1 X_2 X_3 X_4 X_5 X_6"
z:
  - "Z_0 Z_1 Z_2 Z_3 Z_4 Z_5 Z_6"
```

Six stabilizer generators, one logical qubit, seven physical qubits. Logical X
and Z act on all seven qubits. The Steane code supports transversal Clifford
operations, not a transversal T gate. See the [Steane example](../../examples/steane/steane.qodec.yaml)
for explicit instruction actions and their circuits.

### `[[4,2,2]]` Code (Two Logical Qubits)

A four-qubit error-detecting code with two logical qubits:
```yaml
name: C4
description: "[[4,2,2]] error-detecting code (Knill's C4, distance 2)"

stabilizers:
  - "X_0 X_1 X_2 X_3"
  - "Z_0 Z_1 Z_2 Z_3"

x:
  - "X_0 X_1"
  - "X_0 X_2"
z:
  - "Z_0 Z_2"
  - "Z_0 Z_1"
```

Two stabilizer generators, two logical qubits, four physical qubits. If a layer
binds a block type to this code, logical `X_0` on that block is represented by
`X_0 X_1` on its code qubits, and logical `Z_1` by `Z_0 Z_1`. The gadget's support
determines where those code qubits sit in the circuit.

## Pauli String Format

Code operators use the sparse terms defined in the ISA's [Appendix C](instruction-set.md#pauli-operator-sparse-format), without its optional sign. In a code definition, qubit indices refer to physical-qubit *positions* of an encoding that uses the code. They have no absolute meaning until interpreted against an encoding's `support` array.

## Scope

This schema defines **code instances**: a specific code at a specific distance with specific stabilizers and logical operators. It does not define:

- **Code families.** The relationship between `surface_d3` and `surface_d5` (same construction, different parameters) is not captured. A code library generates instances, and this format captures the output.
- **Physical layout.** Qubit geometry, connectivity, and spatial coordinates are relevant for routing and placement but are not part of the code's algebraic structure.
- **Noise model.** Error rates, noise channels, and bias are properties of the physical implementation, not the code.
- **Decoder.** How to decode syndromes into corrections is an implementation choice, not a code property.

Those concerns belong to consuming tools and their data formats. A qodec code
definition supplies the stabilizer and logical-operator algebra they use.
