# Glossary

Terms used in the model and its representations. See the [gadget guide](gadget.md)
for worked parity examples and the [ecosystem guide](../ecosystem.md) for how
other tools use this data.

## Artifact kinds

These filenames are conventions, not type tags. Reference fields determine
which artifact a path contains. Artifacts may instead be entries in one bundle.

| File              | Role                                                          |
| ----------------- | ------------------------------------------------------------ |
| `qodec.yaml`      | Top-level manifest: the ordered layer stack (and metadata).  |
| `*.isa.yaml`      | Instruction interface: block, flag, and readout contracts.   |
| `*.code.yaml`     | A code's stabilizers and logical operators.                  |
| `*.gadget.yaml`   | The lowering of one ISA instruction.                         |

External sibling files (`*.checks.yaml`, `*.readouts.yaml`) factor large
generated sections out of a gadget that names them by path. Small lists
usually stay inline.

## Structural terms

- **layer**: one level of the abstraction stack: an instruction set
  architecture (ISA) together with the gadgets that lower it to the next layer
  down. The ISA is the layer's interface. The gadgets show how it runs one
  level lower. The bottom (most concrete) layer has no layer beneath it and so
  carries an empty gadget set. A qodec is an ordered stack of layers, top
  (abstract) to bottom (concrete).
- **lowering**: rewriting an operation in a layer's ISA into a circuit in the
  layer immediately below, via that layer's gadgets. Lowering is strictly
  pairwise. Multi-level lowering composes the per-layer steps in order.
- **operand**: a block an instruction acts on, declared by position in `in`/`out`
  and bound to a circuit block at a call.
- **parameter**: a named classical input declared by an instruction, with a type.
- **argument**: a value supplied for a parameter at a call.

## Decoding-surface terms

- **checks**: parity assertions a decoder watches. Each check is an XOR of
  references and literal bits whose parity is zero in noiseless execution,
  with the incoming frame accounted for.
  A non-zero check is an error syndrome. Called *detectors* in deq and stim.
- **readouts**: the bits a gadget exposes to the layer above, as one positional list: the
  implemented instruction's `observe` outcomes first (the *observables*), then its
  `flags:` flags. Each entry is one parity equation, optionally named. Within
  its gadget it is `readouts[i]`; a caller receives it in that caller's
  `circuit.readouts` record.
- **observable readout**: a gadget output corresponding to one `observe`
  outcome in its implemented instruction. Its equation may include circuit
  bits and incoming frame signs.
- **flag**: a `flags:`-field readout: a parity the gadget publishes to the
  caller (e.g. a
  `reject` from a fault-tolerant preparation). Same shape as a check, and known
  to be zero on a faultless run, but reported rather than spent on correction.
  An OR over several flags lives in the
  consumer's `select` post-selection.
- **frame**: classical signs used to interpret the encoded state. A gadget's
  `frames` map supplies additional output logical-sign corrections from circuit
  bits and literal bits. Missing entries mean no additional correction, not a
  reset of the incoming frame. See [Frames](gadget.md#frames).
- **circuit.readouts[\<i\>]**: a circuit output bit at position `i`. Each call
  contributes its observe outcomes followed by its flags. It is distinct from
  the containing gadget's declared `readouts[i]`.

## Boundary terms

- **implements**: the abstract ISA instruction a gadget realizes, written
  `<instruction_set>#<mnemonic>`. Optional on a gadget. The layer that lists it supplies the
  instruction otherwise.
- **block**: a named group of qubits an instruction operates on (e.g. a
  `[[4,2,2]]` code block or a single physical qubit). The register of a qodec
  ISA, except that blocks are created and destroyed rather than always there.
- **encoding / support**: which blocks at the layer below carry one block
  above. The `support` lists those blocks in code-qubit order.
- **reference**: a property-path string (e.g.
  `in[0].stabilizers[0]`, `circuit.readouts[2]`) used inside parity
  equations. A selector can address several bits.
- **parity term**: either a reference or the integer literal `0` or `1`.
  Terms combine by XOR; an empty equation is zero.
