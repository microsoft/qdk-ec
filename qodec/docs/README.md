# qodec documentation

qodec organizes the parts of a quantum error-correction protocol and records
how they fit together: the instructions, codes, circuits, and measurement
relationships. The same description can be used to compare implementations,
build encoding layers, or share a protocol with another tool.

Start with the project [README](../README.md) for the overview, then:

1. [Walkthrough](walkthrough.md): follow a logical measurement through its
	circuit and readout equations, then compare a different implementation.
2. [The model](concepts/model.md): how instruction sets, codes, and gadgets form
	a protocol. The concept pages below explain each part in detail.
3. [Using qodec with QDK, deq, and Stim](ecosystem.md): choose analysis and
	decoding tools, with the requirements and limits of their connections.

## Concepts

These pages explain the model using YAML examples. The same definitions apply
to Rust and Python objects and to the C projection.

| Document                                | What it covers                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| [The qodec model](concepts/model.md)    | The abstract object: the lowering chain and the artifact kinds.                 |
| [Model paths](concepts/paths.md) | Exact node lookup, typed access, occurrence identity, and loaded source locations. |
| [Instruction set](concepts/instruction-set.md) | Blocks, instructions, actions, and the formal step semantics.            |
| [Code](concepts/code.md)                | Stabilizers and logical operators.                                  |
| [Gadget](concepts/gadget.md)            | Instruction implementations, encodings, checks, readouts, and output frame corrections. |
| [Validation](concepts/validation.md)    | Preservation checks in qodec, protocol checks in audit, and the limits of each. |
| [Glossary](concepts/glossary.md)        | One-page reference for the artifact kinds and decoding-surface vocabulary.      |

## Representations

Rust owns the model; Python exposes it for inspection and editing. Both can
load and save YAML. The C ABI exposes a read-only projection, not a writer or
an independent authoring model. Circuit source is preserved; qodec does not
translate it between languages.

| Document                                | What it covers                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| [YAML](representations/yaml.md)         | Reference paths, bundle envelopes, source formats, and shorthands. |
| [Python API](../bindings/python/README-python.md) | Source-build instructions, usage guide, and local Sphinx reference build. |
| [Rust API](../src/lib.rs)               | Crate-level guide and rustdoc reference. |
| [C API](../bindings/c/README.md)        | Building, linking, ownership, and the generated header. |

The language references build locally; see
[Building the documentation](../CONTRIBUTING.md#documentation).

## Other

| Document                                | What it covers                                                                 |
| --------------------------------------- | ------------------------------------------------------------------------------ |
| [Source formats](representations/source-formats.md) | Stored source, available parsers, and limits of the Stim subset. |
