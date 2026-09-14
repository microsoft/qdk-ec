# The qodec model

A **qodec** organizes a quantum-error-correction protocol: which operations it
offers, how its blocks are encoded, and which circuits implement those
operations. Checks and readout equations connect the circuit's measurements to
the protocol's logical results; frames declare additional output-sign corrections.
These relationships are part of the model,
not conventions a tool must recover from a circuit or generator script.

The protocol is a stack of layers. Each layer has an instruction set
architecture (ISA), codes, and gadgets that implement its instructions using
the next layer's operations. Instruction meanings are separate from their
implementations, so you can compare candidate gadgets against the same
requirement. See the [walkthrough](../walkthrough.md) for a worked example and
[Using qodec with QDK, deq, and Stim](../ecosystem.md) for how these tools use
the model.

This page is about the model itself. How you write it down, as YAML on disk, as
Rust values, as Python objects, or over the read-only C ABI, is the subject of
the [representation guides](#representations).

## The lowering chain

A protocol can have a logical layer and a physical layer, or intermediate
layers for routing or concatenated encoding. The manifest orders them from
most abstract to most concrete:

```mermaid
flowchart LR
  L(["Source ISA"]) --> G["Gadgets"] --> P(["Target ISA"])
```

Each gadget specifies a circuit for one instruction, using operations from
the next layer. The bottom layer has an ISA but no gadgets because it has no
target layer. A complete lowering protocol needs at least two layers; qodec
can also load and save shorter, unfinished drafts.

You define the bottom ISA as well as the layers above it. It can describe the
physical gates and measurements your target provides; it need not be Stim's
instruction set. Above it, a code can be built from already encoded blocks.
Both the physical operations and each encoding level remain explicit instead
of being hidden inside a flattened circuit. A consuming tool must understand
the chosen operations to analyze or execute them.

A gadget describes a replacement; it does not run a compilation pass. A
consuming tool chooses gadgets for the calls in a program and combines their
circuits. A layer may omit an explicit gadget when a tool can build one, but
qodec itself does not fill that gap.

For a three-layer stack, the top layer's gadgets use the middle ISA, and the
middle layer's gadgets use the bottom ISA. A tool reaches the bottom by
composing those two lowerings. Reordering layers changes those relationships,
so the affected gadget bindings must be checked again.

An ISA's instructions operate on **blocks**, named groups of qubits. A block
type says how many qubits it holds and says nothing about how they are encoded.
Each ISA declares its block types. A layer's `codes` map binds a type to one
code when that type is encoded by its gadgets. Drafts may also retain unused
bindings and types that no gadget yet encodes.

The shared binding means gadgets agree on the code for a given block type.
Composing them still requires aligning their circuit supports and carrying
measurement and frame information forward. qodec records those declarations;
a consuming tool performs the composition. See [validation.md](validation.md).

## The artifact kinds

A qodec is made of four kinds of artifact.

| Artifact            | Role                                                                          |
| ------------------- | ----------------------------------------------------------------------------- |
| **Manifest**        | Orders the layer stack, top (abstract) to bottom (concrete), plus metadata.   |
| **Instruction set** | The operations available at one layer: blocks, instructions, and their actions. See [instruction-set.md](instruction-set.md). |
| **Code**            | A code's stabilizers and logical operators. See [code.md](code.md).           |
| **Gadget**          | One per-instruction lowering rule, belonging to the layer it lowers. See [gadget.md](gadget.md). |

A gadget carries checks for decoding and readout equations for its observables
and flags. Its optional `frames` map states local output-sign corrections in
addition to the frame propagation implied by its action. See the
[gadget guide](gadget.md#frames) for a concrete correction and the
[glossary](glossary.md) for the terms.

## Metadata

The manifest, instruction sets, code definitions, gadgets, and individual
instructions accept an optional `metadata` mapping. qodec preserves it through
load and save without interpreting it. Metadata participates in structural
equality.

Use it for tool- or author-specific annotations such as provenance, durations,
native-gate mappings, and classification tags. Gadgets have no `description`
field, so their annotations belong in `metadata`.

```yaml
metadata:
  calibration_run: "2026-05-31T09:14Z"
  acme:
    duration_ns: 800
    native_gate: rz
```

qodec reserves no keys and imposes no size or nesting limit. Grouping keys by
tool, as with `acme` above, is recommended to avoid collisions, not required.

## Representations

A qodec has the representations below. The Rust model is the source of truth
for the schemas and bindings. The C ABI reads a qodec without building one.

| Representation                 | Guide                                            |
| ------------------------------ | ------------------------------------------------ |
| On-disk YAML (referenced files or single-file bundle) | [representations/yaml.md](../representations/yaml.md) |
| In-memory Python objects       | [Python API](../../bindings/python/README-python.md) |
| In-memory Rust values          | [Rust API](../../src/lib.rs) |
| Read-only C ABI                | [C API](../../bindings/c/README.md) |

In YAML, the referencing field determines the artifact type, not the filename.
Loading follows those references from an explicit manifest or bundle file path;
it requires no directory layout. See [Loading](../representations/yaml.md#loading).

See [Building the documentation](../../CONTRIBUTING.md#documentation) for local
Rust and Python API reference builds.

The concept pages in this directory use YAML as their canonical illustrative
form, but the semantics they describe belong to the model, not to any one
representation.

For validation and verification rules, see [Validation](validation.md).
