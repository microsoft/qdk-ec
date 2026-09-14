# Walkthrough

Suppose you have a circuit that measures three physical qubits. What logical
operation does it perform? Which code does it expect, and how do you read its
answer? qodec records those connections alongside the circuit.

We will follow `measure_z` from the
[repetition3 example](../examples/repetition3/): one logical measurement becomes
three physical measurements and equations for interpreting them. Then we will
compare a different circuit against the same instruction, showing what can be
reused and what must be checked again.

For the catalog of artifact kinds, see the [README](../README.md#artifacts).

The YAML snippets below show parts of the example, not separate runnable
files. The committed example stores them together in one
[bundle](../examples/repetition3/repetition3.qodec.yaml). The Python comparison
later in this guide runs from the qodec source directory containing `examples/`.

## 1. Block types are declared in the instruction set architecture

An instruction set declares operations and the block types they act on. The
repetition3 **instruction set architecture**, or ISA, declares one block type:

```yaml
# repetition3.isa.yaml
name: repetition3
blocks: {repetition3: 1}
```

A block of type `repetition3` holds one logical qubit. The declaration does not
specify its encoding; the code in
[section 3](#3-code-definitions-describe-how-a-block-is-encoded) will do that.

Instructions refer to block types in their input and output lists. A preparation
can produce a block without consuming one; a destructive measurement consumes
a block without returning one. Those lists describe the quantum state present
before and after each instruction.

## 2. Instructions act on those blocks

Now an instruction. `measure_z` takes one block in and declares one outcome:

```yaml
# repetition3.isa.yaml (continued)
instructions:
  - mnemonic: measure_z
    description: Destructive Z-basis measurement of the logical Z observable.
    in: [repetition3]           # one input block of type `repetition3`
    action:
      - observe: Z_0            # ← single outcome: Z of flat qubit 0
```

`action` specifies the operation independently of its name. Here
`observe: Z_0` requires a measurement of Pauli Z on qubit 0. A checker can
compare a candidate circuit with this requirement, provided it understands the
declared actions and circuit language. See
[gadget verification](concepts/validation.md#gadget-verification).

The index in `Z_0` addresses the logical qubit held by this instruction's input
block. It does not address a physical circuit qubit. For instructions with
several blocks, see the
[instruction-set guide](concepts/instruction-set.md) for the index rules.

## 3. Code definitions describe how a block is encoded

The block type says how many logical qubits a block holds. A **code definition**
describes how those qubits are encoded in the next layer:

```yaml
# repetition3.code.yaml
name: repetition3
stabilizers:
  - "Z_0 Z_1"
  - "Z_1 Z_2"
x: ["X_0 X_1 X_2"]
z: ["Z_0"]
```

`stabilizers` are the operators that read +1 on every valid codeword. This is
how syndrome measurements detect errors without measuring the encoded logical
value. Here both operators compare neighboring qubits: `Z_0 Z_1` has value +1
on states with even Z parity on physical qubits 0 and 1.

`x` and `z` are parallel arrays with one entry per logical qubit, giving the
physical operators that play the part of that qubit's logical X and Z. This code
holds one logical qubit, so each array has one entry.

qodec preserves these declarations and checks their Pauli syntax. Code-algebra
checks, such as stabilizer commutation and logical X/Z anticommutation, belong
to `qdk.ec.audit`, not loading. For the stabilizer formalism itself, see Gottesman's
[Stabilizer Codes and Quantum Error Correction](https://arxiv.org/abs/quant-ph/9705052).

The two uses of `Z_0` refer to different qubits. In
[section 2](#2-instructions-act-on-those-blocks), it is the instruction's
logical qubit. In the code definition, it is physical qubit 0, chosen as a
representative of that logical Z. The gadget places the code's qubits on its
circuit; the code's `z` list tells it which physical operator represents the
logical measurement.

A code definition does not name the ISA or gadget that uses it. A layer binds
a block type to a code, and its gadgets use that binding.

## 4. A gadget implements an instruction with a circuit

A **gadget** is one instruction's implementation: the circuit that carries it
out, written in the instruction set of the layer below.

For `measure_z` the entire implementation is three measurements:

```yaml
# measure_z.gadget.yaml
circuit: {format: stim, source: "M 0 1 2"}
```

Measuring all three qubits in the Z basis destroys the encoded state and
produces three bits. The checks and readout equations below complete the gadget
by declaring how to interpret those bits.

What pairs an instruction with its implementation is the manifest, by mnemonic.
It is also where you can watch the two layers meet:

```yaml
# repetition3.qodec.yaml
layers:
  - instruction_set: repetition3.isa.yaml     # the instructions being implemented
    codes:
      repetition3: repetition3.code.yaml
    gadgets:
      measure_z: measure_z.gadget.yaml
  - instruction_set: stim+rz.isa.yaml         # the ISA the circuits are written in
```

The `codes` entry binds the logical block to its encoding. This gadget omits
an explicit circuit input map, so qodec uses the code's default support:
physical qubits 0, 1, and 2. Gadgets that place blocks elsewhere can specify
their boundary support explicitly.

`format: stim` identifies the source language and makes `source` inline text.
Without a format tag, a string source is a file path. A YAML list of calls is
inline source without a tag. Loading preserves source without interpreting it;
parsed accessors such as `Circuit.calls` support inline YAML and a subset of
Stim. Other language tags, including `openqasm`, can be stored for consumers
that understand them. See [source formats](representations/source-formats.md).

## 5. Readouts tie circuit outcomes back to logical bits

The circuit measured three bits. The instruction promised one. Something has to
say how you get from three to one, and that is `readouts`:

```yaml
# measure_z.gadget.yaml (continued)
readouts:
  - ["circuit.readouts[0]", "in[0].z[0]"]   # ← XOR of physical bit 0 with the input's logical-Z sign
```

`readouts` is a positional list with one entry per outcome the instruction
declares. `measure_z` declares one, so there is one entry, and that entry is a
parity equation: XOR its terms together and you have the logical bit.

Both terms are paths, read from the gadget. `circuit.readouts[0]` is the first
bit the circuit measured, counting in program order. `in[0].z[0]` walks to the
gadget's first input, then to entry 0 of that block's `z` list, which is the
`z: ["Z_0"]` you saw in
[section 3](#3-code-definitions-describe-how-a-block-is-encoded).

The second term is not another measurement. It is a **sign**. A block arrives
carrying the signs of its stabilizers and logical operators, often called a
Pauli frame, and XOR-ing that sign in is what turns a raw measurement into a
logical value. Every parity equation has this shape: an XOR over sign-valued
terms drawn from the input encoding, the circuit's measurement record, and, for
a gadget that hands a block back, the output encoding. See
[parity equations](concepts/gadget.md#parity-equations-property-path-references).

Position pairs each readout equation with its declared outcome: the first
equation describes the first `observe`, and so on. Optional readout names are
labels, not a different way to address the bits.

The same list can also carry **flags**, extra bits an instruction declares in its
ISA `flags:` field, such as a `reject` from a fault-tolerant preparation. See
[Outputs of an instruction: observables, flags, and checks](concepts/gadget.md#outputs-of-an-instruction-observables-flags-and-checks).

## 6. Checks declare what should be deterministic

A gadget also declares `checks`. These are parity equations too, the same shape
as a readout, but with a different job. Each one is a combination of measurement
outcomes and stabilizer signs that comes out zero whenever nothing has gone
wrong. A decoder watches them, and a non-zero check is an error to locate.

```yaml
# measure_z.gadget.yaml (continued)
checks:
  - ["circuit.readouts[0]", "circuit.readouts[1]", "in[0].stabilizers[0]"]   # XOR of outcomes 0,1 with stabilizer 0 (Z_0 Z_1)
  - ["circuit.readouts[1]", "circuit.readouts[2]", "in[0].stabilizers[1]"]   # XOR of outcomes 1,2 with stabilizer 1 (Z_1 Z_2)
```

In the first equation, the XOR of measurement bits 0 and 1 equals the incoming
sign bit of `Z_0 Z_1`. A +1 sign is represented by bit 0; a -1 sign by bit 1.
Including the sign bit makes the check zero under noiseless execution even
when the incoming Pauli frame is not zero. If the check produces 1, the declared
relation was violated. A decoder uses the checks and a fault model to infer
which errors could explain that violation.

## Putting it together

A consumer tool reading this gadget gets, in one bundle:

- **The instruction** (`measure_z` of the repetition3 ISA): what logical operation this implements.
- **The circuit** (a stim circuit with one input encoding wiring physical qubits 0–2): how to actually run it.
- **The observables**: how to extract the logical readouts from the physical measurement record.
- **The checks**: what to verify for error detection.

These declarations organize the parts needed to implement the instruction.
An execution tool supplies the backend, and a decoding tool adds a fault model
and decoder. Neither needs to guess which logical instruction this gadget is
supposed to implement.

## Compare two implementations

The instruction need not change when its circuit changes. For `measure_z`,
consider undoing the repetition encoding with two CNOTs before measuring.
Both circuits should measure the same logical Z, but their physical operations
and check equations differ.

Run this from the qodec directory containing `examples/`, with qodec and Stim
installed. It constructs
a separate candidate without modifying the loaded protocol:

```python
import qodec
import stim
from qodec.gadgets import Circuit

protocol = qodec.Qodec.load("examples/repetition3/repetition3.qodec.yaml")
logical, physical = protocol.layers
direct = logical.gadgets["measure_z"]

decoded = qodec.Gadget(
  implements=direct.implements,
  circuit=Circuit(
    physical.instruction_set, "CX 0 1 0 2\nM 0 1 2", format="stim"
  ),
  inputs=direct.inputs,
  checks=[
    ["circuit.readouts[1]", "in[0].stabilizers[0]"],
    ["circuit.readouts[1]", "circuit.readouts[2]", "in[0].stabilizers[1]"],
  ],
  readouts=[["circuit.readouts[0]", "in[0].z[0]"]],
)

assert direct.implements == decoded.implements
assert direct.inputs == decoded.inputs
assert direct.readouts == decoded.readouts
assert direct.checks != decoded.checks

for gadget, first_check_bits in ((direct, [0, 1]), (decoded, [1])):
  circuit = stim.Circuit(gadget.circuit.source)
  assert circuit.has_flow(stim.Flow(input=stim.PauliString("Z__"), measurements=[0]))
  assert circuit.has_flow(stim.Flow(input=stim.PauliString("ZZ_"), measurements=first_check_bits))
  assert circuit.has_flow(stim.Flow(input=stim.PauliString("_ZZ"), measurements=[1, 2]))
```

Here Stim checks that the named measurement parities reproduce the input
logical Z (`Z__`) and stabilizers (`ZZ_` and `_ZZ`). The equations include the
incoming sign bits, so both gadgets' checks are zero for any incoming Pauli
frame under noiseless execution. Each circuit measures
all three qubits, leaving no quantum output, as the instruction requires.
The qodec declarations hold the requirement and candidate implementations;
Stim performs this circuit-level check.

| Part | What happens when the circuit changes? |
| --- | --- |
| Instruction and its logical action | Reused unchanged |
| Physical ISA, input code, and encoding | Reused unchanged |
| Circuit and check equations | Replaced together |
| Logical readout equation | Unchanged in this example; must still be verified |
| Fault model and decoding performance | Must be assessed for the new circuit |

**The same noiseless action does not mean the same fault protection.** Suppose
only measurement bit 0 is flipped after the gates. In the direct gadget, that
flips both the logical readout and the first check. In the decoded gadget, it
flips the logical readout but neither check. The replacement therefore loses
protection against that error, despite passing the noiseless action checks.
The extra gates also introduce fault locations to assess. This is a comparison
of candidates, not a recommendation to use the decoded circuit.

The comparison uses a fixed logical requirement and two implementations with
different error-detection behavior. A decoding adapter would also need the
chosen circuit's fault model. See
[qodec and deq](ecosystem.md#deq-the-decoding-surface) for that next step.

## From one instruction to a protocol

We followed a single instruction the whole way down. A working qodec holds many,
and repetition3 is no exception. Its ISA declares four, and the manifest gives
each one a gadget:

| Instruction | Blocks | What it does |
| ----------- | ------ | ------------ |
| `prepare_z` | out 1 | Prepare the encoded zero state |
| `idle` | in 1, out 1 | One syndrome-extraction round, no logical action |
| `measure_z` | in 1 | Destructive Z-basis measurement, the one traced above |
| `rotate_z` | in 1, out 1 | Logical Z rotation by `theta`, non-Clifford in general |

A standard memory experiment strings them together:

```
prepare_z · idle · idle · … · idle · measure_z
```

Each `idle` is one syndrome-extraction round. Its circuit measures the code's
two stabilizers using a pair of ancilla qubits and otherwise leaves the encoded
state alone, which is why the ISA gives it no action at all. Its checks are
written against the block's boundary, and that is what lets consecutive rounds
link up:

```yaml
# idle.gadget.yaml
checks:
  - ["circuit.readouts[0]", "in[0].stabilizers[0]"]    # this round against the incoming sign
  - ["circuit.readouts[1]", "in[0].stabilizers[1]"]
  - ["circuit.readouts[0]", "out[0].stabilizers[0]"]   # and on to the next round
  - ["circuit.readouts[1]", "out[0].stabilizers[1]"]
```

The `in` checks compare this round's measurement against the sign the block
arrived carrying, which is the previous round's measurement of the same
stabilizer. The `out` checks hand the sign onward, so the next gadget, `idle` or
otherwise, reads it as its own `in[0].stabilizers[0]`. See
[carry-forward across gadgets](concepts/gadget.md#carry-forward-across-gadgets).

This is also why `idle` declares an `out:` block and `measure_z` does not. A
gadget that hands a block back can pass signs forward. A destructive one ends
the chain.

## Where to go from here

- **Inspect it.** The whole example is one file,
  [repetition3.qodec.yaml](../examples/repetition3/repetition3.qodec.yaml).
  Load it with the [Python API](../bindings/python/README-python.md) and inspect the
  objects this walkthrough described.
- **See the model whole.** [The qodec model](concepts/model.md) covers the
  lowering chain and the artifact kinds in one place, and the
  [glossary](concepts/glossary.md) is the one-page vocabulary.
- **Find out what is checked.** [Validation](concepts/validation.md) separates
  qodec's preservation checks from the protocol and algebra checks in audit.
- **Write your own.** [The YAML representation](representations/yaml.md)
  documents the on-disk form, both the directory layout and the single-file
  bundle used here.
- **Read a bigger one.** [The examples](../examples/README.md) run from
  `steane` and `iceberg` to `c4c6`, which stacks encoding layers with several
  logical qubits per block.
