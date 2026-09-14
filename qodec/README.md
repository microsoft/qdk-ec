# qodec

> **A formal model of quantum error correction protocols.**

QEC tools provide codes, circuits, simulators, and decoders. What is missing
is a common description of **how those parts fit together**: which logical
instruction a circuit implements, how its qubits are encoded, and what its
measurements mean. These connections often live ad-hoc in papers, generator scripts,
and conventions, leaving each tool to reconstruct them.

qodec records instruction sets, codes, and gadgets in one formal description.
You choose the logical and physical instruction sets. Each instruction's
meaning is defined formally, so a tool can check that a gadget realizes the
instruction it claims to, and a protocol written against one instruction set can
be lowered onto a different physical one without rewriting what it means.

A qodec is a sequence of layers. At each layer, gadgets implement a source ISA (instruction set architecture)
using operations from a target ISA:

```mermaid
flowchart LR
  L(["`**Source ISA**`"]) --> G["`**Gadgets**`"] --> P(["`**Target ISA**`"])
  classDef isa fill:#eff6ff,stroke:#60a5fa,color:#1e3a8a,stroke-width:1.25px
  classDef gadget fill:#fff1f2,stroke:#fda4af,color:#9f1239,stroke-width:1.25px
  class L,P isa
  class G gadget
```

Usually, the source ISA is logical, the target ISA is physical, and the gadgets
describe fault-tolerant circuits.

Use qodec to share a protocol, verify its circuits, compare implementations,
and reuse it across simulation and decoding experiments. Start with one
instruction and its gadget; a complete architecture is not required.
See [Using qodec with QDK, deq, and Stim](docs/ecosystem.md) for the connections
between these tools.

## Example

The smallest working qodec in this repository is [`repetition3.qodec.yaml`](examples/repetition3/repetition3.qodec.yaml), the three-qubit bit-flip repetition code lowered to a stim-compatible "physical" layer.

### Layers

A qodec is a stack of abstraction layers, each layer pairs an instruction set architecture (ISA) with gadgets that lower it to the next. Here there are two layers, a `repetition3` logical layer and a `stim+rz` physical layer. Any number of layers is possible in general.

```yaml
name: repetition3
layers:
  - instruction_set: repetition3.isa.yaml
    codes:
      repetition3: repetition3.code.yaml
    gadgets:
      prepare_z: prepare_z.gadget.yaml
      idle: idle.gadget.yaml
      measure_z: measure_z.gadget.yaml
      rotate_z: rotate_z.gadget.yaml
  - instruction_set: stim+rz.isa.yaml
```

### Instructions

The repetition3 ISA declares the user-visible instructions: `prepare_z`, `idle`, `measure_z`, and a non-Clifford `rotate_z`. Each instruction defines its operands and parameters _and_ its formal action:

```yaml
- mnemonic: measure_z
  description: Destructive Z-basis measurement of the logical Z observable.
  in: [repetition3]
  action: [observe: Z_0]
```

### Gadgets

A gadget lowers one such instruction to the next layer. It provides a `circuit` that realizes the instruction, parity `checks` that XOR to zero (unless there are errors), and `readouts` that relate the instruction's outcomes to circuit measurements. A decoding tool, e.g. [deq](https://github.com/microsoft/qdk-ec/tree/main/deq), combines these relationships with a fault model. The gadget below implements the `measure_z` instruction from above.

```yaml
circuit: {format: stim, source: "M 0 1 2"}
checks:
  - ["circuit.readouts[0]", "circuit.readouts[1]", "in[0].stabilizers[0]"]
  - ["circuit.readouts[1]", "circuit.readouts[2]", "in[0].stabilizers[1]"]
readouts: [["circuit.readouts[0]", "in[0].z[0]"]]
```

Here `in[0]` refers to the input encoding, the instruction's first input block. In this case, the repetition3 code on physical qubits 0–2.

The [walkthrough](docs/walkthrough.md) traces the same example end to end. See the [examples/](examples/) folder for more.

## Features

- **Choose your physical ISA.** Native gates, rotations, and measurements; no fixed gate set.
- **Universal, not just Clifford.** Include Toffoli and parameterized Pauli rotations.
- **Separate meaning from implementation.** Verify circuits against declared actions, not gate names.
- **Mix, switch, and concatenate codes.** Keep every encoding level explicit.
- **Bring your own circuits.** Retain your existing [source formats](docs/representations/source-formats.md); analysis needs a compatible parser.
- **Keep measurement meanings.** Checks and readouts travel with the circuit, independently of fault models and decoders.
- **Check structure and correctness separately.** [Schemas](schemas/) define file structure; [audit](docs/concepts/validation.md) checks supported circuits and actions.

## Artifacts

A qodec contains the following artifact types.

**Top-level artifacts**, the building blocks of a qodec:

| Artifact                                     | Role                                                                                |
| -------------------------------------------- | ----------------------------------------------------------------------------------- |
| [Instruction Set](docs/concepts/instruction-set.md)   | Defines available operations per layer                                     |
| [Gadget](docs/concepts/gadget.md)                     | Per-instruction lowering rule: implements, circuit, encodings, checks, readouts, and frames |
| [Code](docs/concepts/code.md)                         | Stabilizers, logicals                                                      |

**Parity checks and readouts** are normally inlined inside a gadget, but can be factored out into their own file and referenced by relative path (useful for large generated lists). A circuit is always inline; what it may reference is its source text:

| Component                                             | Role                                                       |
| ----------------------------------------------------- | ---------------------------------------------------------- |
| [Circuit](docs/concepts/gadget.md#circuit-object)        | Realization circuit/program source                         |
| [Parity checks](docs/concepts/gadget.md#checks)          | Deterministic-check list                                   |
| [Readouts](docs/concepts/gadget.md#readouts)              | Output-bit list: observables then flags (parity equations)  |

### Instruction Sets

The manifest's `layers` field orders instruction set architectures (**ISA**s) from logical to physical. That order is the lowering chain. Each ISA declares instructions that operate on blocks, named groups of qubits such as a `[[4,2,2]]` code block or a single physical qubit.

### Gadgets

A **gadget** is a per-instruction lowering rule. Each gadget pairs one source-ISA instruction (the one it implements) with a target-ISA realization (the circuit, plus its boundary encodings) plus the decoding surface needed to reason about it under noise: parity checks and readouts.

**Checks** are deterministic combinations of measurement outcomes (and optionally stabilizers from the input/output encodings) whose parity is zero in the absence of errors. **Readouts** are the bits the gadget exposes. Its logical observables come first, each a combination of measurement outcomes that produces a logical output, followed by any flags (heralding bits the decoder passes through).

### Codes

In the repetition-code example, three physical qubits store one logical qubit.
Its **code definition** lists the stabilizers that define valid encoded states
and the physical operators that represent logical X and Z.

A layer chooses a code for each kind of block. Each gadget specifies which
blocks at the layer below carry each encoded block at its input and output. You can
change the code and its gadgets without changing what the instruction means.

## Representations

A qodec is an **abstract object, not a file format**: the forms below represent the same underlying qodec. The Rust model is the source of truth for the schemas and bindings.

- **In-memory domain types**: the [Rust crate](src/) provides a typed value for every artifact, with serde support throughout. The [Python package](bindings/python/) is a thin PyO3 layer over it, exposing the same data model. See [`bindings/python/`](bindings/python/) for the full API and worked examples.
- **C ABI**: [`bindings/c/`](bindings/c/) provides a read-only C interface. It is distributed as source; build a shared or static library and use the provided [`qodec.h`](bindings/c/include/qodec.h) header.
- **Referenced YAML files**: a manifest and its artifacts, linked by relative path, with no required directory layout or filename suffix. Diff-friendly and hand-authorable.
- **Single-file YAML bundle**: one multi-document YAML stream whose first document contains the manifest in a single-entry envelope with an unrestricted key. Remaining entries are looked up by referenced path.

## Installation

### Python

Python 3.11 or newer is required. Follow the
[source build instructions](bindings/python/README-python.md#build-and-install-from-source)
to install from a qdk-ec checkout.

Use `--extras parsers` with `maturin develop` to install optional external
parsers, currently Stim. YAML interpretation is built in, and the base package can load and save
Stim source without the extra. See
[parser registration](docs/representations/source-formats.md#registering-a-parser)
for other languages and custom parsers.

The example files are not installed with the Python package. Run the following
from the qodec source directory containing [examples/](examples/). Outside the checkout,
obtain [the repetition3 bundle](examples/repetition3/repetition3.qodec.yaml)
and replace the path passed to `load` with its location.

```python
import qodec

protocol = qodec.Qodec.load("examples/repetition3/repetition3.qodec.yaml")
print(protocol)

gadget = protocol.layers[0].gadgets["measure_z"]
print(gadget.circuit.source)
```

See the [Python guide](bindings/python/docs/usage.rst) for creating, editing,
and saving protocols.

## Documentation

Start with the [walkthrough](docs/walkthrough.md) to follow a logical instruction
through its implementation. The remaining docs describe the
[model](docs/concepts/model.md) and its
[representations](docs/representations/yaml.md).

| Document                                       | Content                                              |
| ---------------------------------------------- | ---------------------------------------------------- |
| [Walkthrough](docs/walkthrough.md)             | Worked example: one instruction from declaration to physical execution |
| [The qodec model](docs/concepts/model.md)      | The abstract object: lowering chain, artifact kinds, gadgets versus compilation |
| [Instruction Set](docs/concepts/instruction-set.md) | ISA: blocks, instructions, actions               |
| [Code](docs/concepts/code.md)                  | Stabilizers and logical operators                      |
| [Gadget](docs/concepts/gadget.md)              | Instruction implementation, encodings, checks, readouts, and frames |
| [Validation](docs/concepts/validation.md)      | What loading preserves and what audit checks          |
| [Using qodec with QDK, deq, and Stim](docs/ecosystem.md) | Existing connections, shared workflows, and future compatibility |
| [YAML representation](docs/representations/yaml.md) | On-disk forms, loading, source formats, shorthands |
| [Python API](bindings/python/README-python.md) | Usage guide and stub-generated API reference |
| [Rust API](src/lib.rs) | Crate-level guide and rustdoc reference |
| [C API](bindings/c/README.md) | Build/linking guide, ownership, and generated header |

See [Building the documentation](CONTRIBUTING.md#documentation) for local HTML
builds of the Rust and Python API reference.

## FAQ

**Where does the name come from?**
From [codec](https://en.wikipedia.org/wiki/Codec), which encodes and decodes
digital information. A qodec describes the quantum counterpart: encoding
logical qubits into physical qubits and decoding physical measurements into
logical results.

**Why not use deq?**
Use [deq](https://github.com/microsoft/qdk-ec/tree/main/deq) to model faults and
run decoding experiments. Both tools describe codes and gadgets. qodec also
specifies what each instruction must do, separately from its circuit, and lets
you choose the physical instructions and encoding levels. You can compare
circuits against that specification while changing fault models or decoders.
Using both tools requires [gadget translation and a fault model](docs/ecosystem.md#deq-the-decoding-surface).

**Why not use Stim?**
Use Stim to simulate and analyze stabilizer circuits. qodec records which
logical instruction a circuit implements, how its qubits are encoded, and what
its measurements mean. That lets you compare or replace circuits without
reconstructing their purpose. The Stim circuit remains the gadget's source.

**Why not use QASM or QIR?**
They describe a program to run. qodec adds a separate specification of the
logical instruction, how its qubits are encoded, and what its measurements
mean. The program remains the gadget's circuit source.

**Why YAML?**
YAML is easy to read, edit, and compare in version control. Tools in different
languages can read the same files. Files are optional: use the Rust or Python
API to build qodec objects in memory. The C API is read-only.

**Does qodec do synthesis, compilation, or decoding?**
No. qodec defines the protocol; other tools compile it, simulate it, decode
measurements, or estimate resources. See [the model guide](docs/concepts/model.md#the-lowering-chain)
for how those tools use qodec.

**Is qodec stable?**
Not yet. Any pre-1.0 minor release may change the API or file format. Pin an
exact package version for reproducible work. The manifest's `schema_version`
tracks file-format changes separately from the package version. Loading
requires a matching schema version; if omitted, the loader assumes its current
version. See [CHANGELOG.md](CHANGELOG.md) for the compatibility rules.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for qodec's development workflow and required checks.
The parent repository owns the [code of conduct](../CODE_OF_CONDUCT.md),
[security reporting policy](../SECURITY.md), and [support policy](../SUPPORT.md).
Do not report security vulnerabilities through public issues.

## License

Licensed under the MIT License. See [`LICENSE`](LICENSE).

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft trademarks or logos is subject to and must follow [Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general). Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship. Any use of third-party trademarks or logos are subject to those third-party's policies.
