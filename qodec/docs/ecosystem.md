# Using qodec with QDK, deq, and Stim

Keep the protocol's instruction meanings, codes, encodings, and measurement
equations in qodec. Choose the consuming tool for the question you need to answer:

- **Does this circuit implement its declared instruction?** Use QDK's
	`qdk.ec.audit` for supported actions and circuits.
- **How do faults affect decoding?** Use deq with translated gadget
	descriptions and a fault model.
- **What does this Stim circuit do?** Use Stim directly for simulation and
	circuit-level checks.

## QDK: working with protocols

**Start with QDK when you want to build or audit qodec objects.** Its
error-correction tools operate on those objects directly: `build_qodec` builds
protocols, `filled` returns copies with derived relations for supported circuits,
and `audit` checks declarations against available analysis.

This describes the `qdk.ec` development integration. It requires a compatible
QDK build exposing those functions and its `ec` dependencies; installing
qodec alone does not provide them. Check the QDK release you use rather than
assume every published version contains this API.

With that environment, run from the qodec source directory:

```python
import qodec
import qdk.ec

protocol = qodec.Qodec.load("examples/repetition3/repetition3.qodec.yaml")
report = qdk.ec.audit(protocol, promote_warnings=True)
print(report)
```

This example reports one unsupported analysis, for its rotation action; its
conserved stabilizer signs are verified.
Its supported checks pass, but it is not warning-free. An unsupported analysis is
unverified, not a proof of an incorrect circuit. `promote_warnings=True` makes
both errors and warnings fail `report.ok`; informational diagnostics remain
non-failing. A clean audit still does not prove code distance or fault tolerance.

The [worked comparison](walkthrough.md#compare-two-implementations) checks two
measurement circuits against the same logical instruction using Stim. It also
shows why matching noiseless behavior is not enough to compare fault protection.

## deq: the decoding surface

**qodec describes the fault-tolerance protocol. deq defines the decoding
surface, including the fault model.** They overlap in gadget descriptions.
qodec specifies the computation and its encoded implementation; deq specifies
how faults affect the checks and logical results used in decoding.

| Feature | qodec | deq |
| --- | :---: | :---: |
| User-defined physical ISA | ✓ | ✗ |
| Instruction specification independent of gadgets | ✓ | ✗ |
| General quantum actions, including non-Clifford | ✓ | ✗ |
| Multiple encoding levels, each with its own ISA | ✓ | ✗ |
| Codes, gadgets, and boundary encodings | ✓ | ✓ |
| Checks and logical readouts | ✓ | ✓ |
| Declared logical Clifford transformations | ✓ | ✓ |
| Derive Pauli propagation and measurement effects from supported circuits | ✗ | ✓ |
| Fault model with error probabilities and effects | ✗ | ✓ |
| Decoder integration and runtime Pauli-frame tracking | ✗ | ✓ |

✓ Supported directly. ✗ Not provided by the model or tool itself.

- **Actions:** deq's automatic circuit analysis uses Stim-compatible operations.
	Its dynamic decoding can support non-Clifford protocols, but its Pauli
	propagation relations do not specify general non-Clifford quantum actions.
- **Encoding levels:** deq can nest horizontal gadget compositions. This does
	not add a new encoding level with its own blocks, codes, and ISA.
- **External tools:** qodec supplies the protocol specification. Consuming tools
	perform circuit analysis and supply fault models and decoding workflows.

[deq](https://github.com/microsoft/qdk-ec/tree/main/deq) connects error
mechanisms and their probabilities to the checks and logical results they
affect. It can derive those relationships from supported circuits or accept
explicit declarations, then compose gadgets and manage decoding.

The connection requires **gadget translation**, not a call passing a qodec
object straight to a decoder. An adapter must preserve encodings, measurement
references, and logical readout meanings, and supply a fault model separately.
This guide does not provide a public adapter installation or an end-to-end
qodec-to-deq command. Treat that translation as a prerequisite, not as a feature
obtained by installing qodec and deq side by side.

For protocol experiments, use decoders such as
[PyMatching](https://pymatching.readthedocs.io/en/stable/), relay-bp, and
tesseract through deq. This keeps decoder integration and Pauli-frame tracking
in the decoding workflow. Direct decoder use remains useful for decoder
research.

qodec retains the instruction specification and encoding levels while you
change noise assumptions or decoders. It also lets you choose the physical ISA
and compare replacement gadgets against the same intended action. A new gadget
still needs its own fault analysis; matching the noiseless action does not
establish the same fault protection.

## Stim: concrete circuits

[Stim](https://github.com/quantumlib/Stim) simulates and analyzes stabilizer
circuits. It samples measurements and derives detector error models from suitable
noisy, annotated circuits.

**Use Stim when working with a gadget's concrete circuit.** A Stim-format
source can be passed directly to `stim.Circuit`; qodec keeps the logical
instruction and encodings around it.

Run this from the qodec directory containing `examples/`, with qodec and Stim
installed:

```python
import qodec
import stim

protocol = qodec.Qodec.load("examples/repetition3/repetition3.qodec.yaml")
measurement = protocol.layers[0].gadgets["measure_z"]
circuit = stim.Circuit(measurement.circuit.source)

assert circuit.num_measurements == 3
assert len(measurement.readouts) == 1
```

The circuit measures three physical qubits; the gadget exposes one logical readout.

qodec keeps the encodings and parity equations connecting those two descriptions;
passing only the circuit text to Stim leaves them behind. Translating them into
an experiment is an adapter's job. qodec's parser supports a subset of Stim
syntax, while its instruction model also describes non-Clifford operations that
Stim cannot simulate. See [source formats](representations/source-formats.md).

Use qodec when you need to retain the instruction requirement, encoding context
and measurement meanings while comparing circuits or changing decoders. A
standalone circuit or decoding input does not carry all of those declarations.
The shared model avoids reconstructing them for every experiment; it does not
remove the need for compatible parsers, analysis tools and adapters.