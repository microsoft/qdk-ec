# qodec examples

These examples show how to organize different QEC protocols in the same model.
Each connects instruction meanings, codes, circuits, and measurement equations.
They also exercise the model with code switching, multiple encoding levels,
and non-Clifford operations.

- **Compare implementations:** the [walkthrough](../docs/walkthrough.md#compare-two-implementations)
  keeps one repetition-code instruction fixed while changing its circuit and
  checking the resulting measurement relationships and fault behavior.
- **Build another encoding level:** [C4/C6](c4c6/qodec.yaml) uses the same ISA
  at two levels, with a different code and gadgets at each.
- **Choose the physical ISA:** [C422/C832](c422-c832-arch/qodec.yaml) supplies
  its own physical instruction set, including non-Clifford operations, and
  connects two codes through teleportation gadgets.

## Correctness tests

From the qdk-ec repository root or its qodec directory, check that every example
has layers, gadgets, and codes, and survives saving and reloading in both layouts:

```bash
cargo test -p qodec --test round_trip_test
```

Loading checks preservation preconditions, not code algebra or protocol
correctness. For protocol audit, use an environment containing this qodec
checkout, pytest, and a compatible QDK with the `ec` dependencies and
`qdk.ec.audit`. Run from the qodec root:

```bash
python -m pytest examples/tests -q
```

Each committed example manifest has a named test that calls
`qdk.ec.audit(protocol)` and asserts `report.ok`. Every error fails the test.
Warnings must match the per-example rule, reason, and maximum count in
[`UNSUPPORTED_WARNINGS`](tests/test_examples.py); new or increased warnings fail.
No audit rules are disabled. Missing dependencies or example files fail rather
than skip, and an inventory test covers every `*qodec.yaml` example.

The ten retained examples have no audit errors. Six have no warnings:
`bacon-shor`, `c4c6`, `honeycomb`, `iceberg`, `steane`, and `surface`. The remaining
29 warnings cover these limitations:

| Example | Allowed Unsupported Verification |
| --- | --- |
| `c422-c832-arch` | CCZ rotations and readout verification for teleport actions mixing preparation and measurement (10 warnings). |
| `distillation-15` | Non-Clifford action and flag verification (2 warnings). |
| `reed-muller-15` | Transversal T action, checks, and output-frame verification (16 warnings). |
| `repetition3` | Parameterized rotation action (1 warning); its conserved stabilizer signs are verified. |

These warnings identify unverified behavior, not proved mismatches. This release
policy does not claim those checks passed. It requires a compatible QDK build
with the conditional-Pauli, flag-record, commuting-rotation, signed-observable,
and logical-phase fixes used by these tests; qodec alone does not provide the
audit engine. Parity verification uses each called instruction's noiseless
contract, including zero flags. Lower-layer implementations are audited separately.
An analysis failure under `check-mismatch` or `flag-mismatch` means the equation
could not be evaluated. A proven nonzero noiseless parity is an error, not an
allowed warning.
Neither a clean audit nor a successful load proves code distance or fault
tolerance; those require separate analysis.

## Loading

Pass an explicit manifest or bundle file path, such as
[`repetition3/repetition3.qodec.yaml`](repetition3/repetition3.qodec.yaml) or
[`c4c6/qodec.yaml`](c4c6/qodec.yaml), to `Qodec.load`. The loader follows the
manifest's references; it does not scan the example directory. Unreferenced
files and bundle entries are ignored.

The filenames and directories shown here describe these examples, not format
requirements. Reference fields determine artifact types. Every manifest
`gadgets` entry names a YAML gadget document, even for a circuit-only gadget
whose whole document is `circuit: ./idle.stim`. Source-language identification
is separate from artifact naming. See the
[loading rules](../docs/representations/yaml.md#loading).

## Shared physical ISA

Most examples reuse [`stim.isa.yaml`](stim.isa.yaml) as their physical ISA
(`- instruction_set: ../stim.isa.yaml`) instead of repeating its instruction definitions.
It includes Stim gates and conditional Pauli instructions used by inline YAML
circuits. This is a convenience, not a required bottom ISA: repetition3 adds a
parameterized rotation in its own ISA, and examples needing non-Clifford T
operations declare those separately. A layer `instruction_set:` path resolves
relative to the manifest document (its key within a bundle), preserving the
leading `..`. This file is read because the layer references it, not because
of its name or location.

## Protocol examples

Start with repetition3 for the smallest bundle, C4/C6 for concatenated layers,
or the code family closest to your experiment. The audit status and limitations
above apply to all entries below.

| Example | Code or arrangement | What to inspect |
| --- | --- | --- |
| [repetition3](repetition3/) | Three-qubit bit-flip repetition code | Destructive readout, checks, and a parameterized Z rotation in one bundle. |
| [C4/C6](c4c6/) | Two encoded levels sharing an ISA | Different code bindings at each level and explicit output frame corrections. |
| [C422/C832](c422-c832-arch/) | Teleportation between two codes | Code changes, local frame corrections, and a custom physical ISA. |
| [Steane](steane/) | [[7,1,3]] | Transversal Clifford operations and syndrome extraction with flag ancillas. |
| [Surface](surface/) | Distance-3 rotated patches and a merged patch | Repeated checks and a two-input, one-output joint Z measurement. |
| [Reed-Muller](reed-muller-15/) | [[15,1,3]] | A declared transversal T implementation; non-Clifford verification remains limited. |
| [Distillation](distillation-15/) | 15-to-1 over a trivial base code | Preparation, consumption of magic states, and declared rejection flags. |
| [Bacon-Shor](bacon-shor/) | [[9,1,3]] subsystem code | Weight-two gauge measurements combined into stabilizer checks. |
| [Iceberg](iceberg/) | [[k+2,k,2]] detection family | Several logical qubits per block and flags for detection. |
| [Honeycomb](honeycomb/) | Six data qubits, two logical qubits | Three code definitions for successive measurement rounds. |

The [iceberg builder](iceberg/iceberg.py) exposes `build_iceberg(k)` and the
[surface builder](surface/surface.py) exposes `build_surface_code(d)`. The
committed bundles use `k = 4` and `d = 3`. Consult each script for parameter
requirements and the objects it returns. Generated sizes need their own audit
and fault analysis; the committed snapshot's test result does not cover a family.

## Reading the declarations

**Checks connect rounds.** A memory gadget can relate a measurement to an input
stabilizer sign, an output stabilizer sign, or both through separate equations.
Those references preserve the information needed when rounds are composed;
they are not just detector annotations on an isolated circuit.

**Flags leave the gadget.** Iceberg detection flags can support a discard
policy. Steane's `idle_ft` uses flag ancillas to report faults that can spread
through a syndrome circuit. The same `flags` field serves both purposes; the
consumer decides how to use the bits. Noiseless-zero verification alone does
not establish the fault response or the code distance.

**A code may change between rounds.** Honeycomb measures XX, YY, and ZZ edges
in successive rounds. Each round has a different stabilizer group, represented
by a separate code and block type. The round gadgets declare the logical action
and frame corrections between those encodings. Bacon-Shor uses a different
approach: its gauge measurements appear in the circuit and check equations,
since the code artifact has no gauge-generator field.

**Distillation separates a gadget from a factory policy.** The example follows
[Fowler et al., Figure 33a](https://arxiv.org/abs/1208.0928): prepare a Bell pair,
encode one half, apply transversal T-dagger operations, then measure that half.
The gadget declares the retained output and rejection flags. A consumer decides
whether to keep that output and retry; qodec does not run that loop. The current
audit engine cannot fully verify this non-Clifford action or its flag equations,
as recorded above.

**A block change is not itself an implementation.** Surface `merge_zz` has two
fixed input blocks and one output block, not a variadic input. Its action states
the joint measurement and retained logical state. The code, circuit, encodings,
and equations supply the physical implementation. Declaring the new block name
alone would not describe lattice surgery.
