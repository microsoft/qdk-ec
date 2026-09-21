# Multi-Port Gadgets: The Transversal CNOT

In the [language basics chapter](language-basics.md), every gadget had at most one
`INPUT` and one `OUTPUT` port. But logical gates that act on multiple code blocks —
like a transversal CNOT — need **multiple ports**. This chapter introduces multi-port
gadgets and shows how the transpiler automatically captures cross-block error propagation.

---

## The TransversalCNOT Gadget

A transversal CNOT applies CX gates between corresponding physical qubits of two code
blocks. In the `.deq` language, this is expressed with two `INPUT` and two `OUTPUT` ports:

[TransversalCNOT gadget](../examples/multi-port/snippet_cnot_gadget.deq)
<!-- deq-highlight-begin: ../examples/multi-port/snippet_cnot_gadget.deq -->
```deq
GADGET TransversalCNOT {
    INPUT RepetitionCode 0 1 2      # control block
    INPUT RepetitionCode 3 4 5      # target block
    CX 0 3 1 4 2 5                  # transversal CX
    OUTPUT RepetitionCode 0 1 2     # control out
    OUTPUT RepetitionCode 3 4 5     # target out
}
```
<!-- deq-highlight-end: ../examples/multi-port/snippet_cnot_gadget.deq -->

| Line                          | Meaning                                                     |
| ----------------------------- | ----------------------------------------------------------- |
| `INPUT RepetitionCode 0 1 2`  | First input port (control block) on physical qubits 0, 1, 2 |
| `INPUT RepetitionCode 3 4 5`  | Second input port (target block) on physical qubits 3, 4, 5 |
| `CX 0 3 1 4 2 5`              | Three parallel CX gates: (0→3), (1→4), (2→5)                |
| `OUTPUT RepetitionCode 0 1 2` | First output port (control out)                             |
| `OUTPUT RepetitionCode 3 4 5` | Second output port (target out)                             |

The gadget has **0 physical measurements** — it's a unitary gate with no syndrome
extraction. But as we'll see, it has a non-trivial effect on the decoding graph.

---

## What the Transpiler Produces: Cross-Block Unfinished Checks

The most interesting part is the TransversalCNOT's transpiler output. Even though it has
**0 measurements**, it has **4 unfinished checks** — two per output port:

[Full noiseless example](../examples/multi-port/01_noiseless.deq)
<!-- deq-highlight-begin: ../examples/multi-port/01_noiseless.deq -->
```deq
# Transversal CNOT on the repetition code — no noise.
# Demonstrates a multi-port gadget with 2 INPUT and 2 OUTPUT ports.

CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}

GADGET TransversalCNOT {
    INPUT RepetitionCode 0 1 2      # control block
    INPUT RepetitionCode 3 4 5      # target block
    CX 0 3 1 4 2 5                  # transversal CX
    OUTPUT RepetitionCode 0 1 2     # control out
    OUTPUT RepetitionCode 3 4 5     # target out
}
```
<!-- deq-highlight-end: ../examples/multi-port/01_noiseless.deq -->

The transpiler derives 4 unfinished checks — let's look at the annotated output:

[Noiseless TransversalCNOT annotated](../examples/multi-port/snippet_01_noiseless_cnot.deq)
<!-- deq-highlight-begin: ../examples/multi-port/snippet_01_noiseless_cnot.deq -->
```deq
@GTYPE(1)
@CHECKS("manual", verify=0)
GADGET TransversalCNOT {
    INPUT RepetitionCode 0 1 2
    INPUT RepetitionCode 3 4 5
    CX 0 3 1 4 2 5
    OUTPUT RepetitionCode 0 1 2
    OUTPUT RepetitionCode 3 4 5
    CHECK OUT0.S0 IN0.S0
    CHECK OUT0.S1 IN0.S1
    CHECK OUT1.S0 IN1.S0 IN0.S0
    CHECK OUT1.S1 IN1.S1 IN0.S1
    PROPAGATE OUT0.LZ0 FROM IN0.LZ0 IN1.LZ0
    PROPAGATE OUT0.LX0 FROM IN0.LX0
    PROPAGATE OUT1.LZ0 FROM IN1.LZ0
    PROPAGATE OUT1.LX0 FROM IN0.LX0 IN1.LX0

    # --- statistics ---
    # finished checks: 0
    # unfinished checks: 4
    #   weight distribution: { 2:2, 3:2 }
    # errors: 0
}
```
<!-- deq-highlight-end: ../examples/multi-port/snippet_01_noiseless_cnot.deq -->

The first `OUTPUT` (control) has two checks that reference only control input virtuals:
- `rec[-4] rec[-8]` = control output virtual ⊕ control input virtual → same-port check
- `rec[-3] rec[-7]` = same pattern for the second stabilizer

The second `OUTPUT` (target) has two checks that reference **both** input ports:
- `rec[-2] rec[-6] rec[-8]` = target output virtual ⊕ target input virtual ⊕ control input virtual
- `rec[-1] rec[-5] rec[-7]` = same for the second stabilizer

**Why the cross-block references?** The CX gate transforms Z stabilizers:
$Z_{\text{target}} \to Z_{\text{control}} \otimes Z_{\text{target}}$. So the target's
output stabilizer $Z_3 Z_4$ (after CX) equals the product of the control's input
stabilizer $Z_0 Z_1$ and the target's input stabilizer $Z_3 Z_4$. This shows up as
`in0[0] ⊕ in1[0]` in the unfinished check — a check that references virtual measurements
from **both** input ports.

This is exactly how the JIT compiler constructs cross-block dependencies in the decoding
graph: when the TransversalCNOT is instantiated at runtime, the unfinished checks from
the predecessor gadgets on both the control and target lines are XOR'd with these
cross-port checks to produce the finished checks.

---

## The correction_propagation Matrix

The `correction_propagation` matrix for TransversalCNOT is $4 \times 5$:

$$
\begin{pmatrix}
1 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 1 & 0
\end{pmatrix}
$$

Reading this matrix:
- **Rows** = output observables: $[X_c, Z_c, X_t, Z_t]$ (2 per output port)
- **Columns** = input observables + constant: $[X_c, Z_c, X_t, Z_t, \text{const}]$

Each column records which output observables are flipped when the corresponding
input observable's *symplectic partner* (correction) is propagated through the circuit:
- Column 0 ($X_c$): propagate $Z_c$ through CX. $Z_c \to Z_c$ — unchanged, so only
  $X_c^{\text{out}}$ (row 0) is flipped
- Column 1 ($Z_c$): propagate $X_c$ through CX. $X_c \to X_c X_t$ — X propagates
  control→target, flipping $Z_c^{\text{out}}$ (row 1) and $Z_t^{\text{out}}$ (row 3)
- Column 2 ($X_t$): propagate $Z_t$ through CX. $Z_t \to Z_c Z_t$ — Z propagates
  target→control, flipping $X_c^{\text{out}}$ (row 0) and $X_t^{\text{out}}$ (row 2)
- Column 3 ($Z_t$): propagate $X_t$ through CX. $X_t \to X_t$ — unchanged, so only
  $Z_t^{\text{out}}$ (row 3) is flipped

This encodes the standard CNOT conjugation rule: $X_c \to X_c X_t$ (X propagates
control→target) and $Z_t \to Z_c Z_t$ (Z propagates target→control). The
`correction_propagation` matrix records how Pauli frame corrections propagate forward
through the gadget via the symplectic partner walk.

---

## Adding Noise: Cross-Block Error Propagation

With noise, the error structure reveals the CNOT's impact on decoding:

[Noisy TransversalCNOT annotated](../examples/multi-port/snippet_02_noisy_cnot.deq)
<!-- deq-highlight-begin: ../examples/multi-port/snippet_02_noisy_cnot.deq -->
```deq
@GTYPE(1)
@CHECKS("manual", verify=0)
GADGET TransversalCNOT {
    INPUT RepetitionCode 0 1 2
    INPUT RepetitionCode 3 4 5
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2 3 4 5
    ERROR(0.01) C0 C2 OUT0.LX0 OUT1.LX0  # E0
    ERROR(0.01) C0 C1 C2 C3 OUT0.LX0 OUT1.LX0  # E1
    ERROR(0.01) C1 C3 OUT0.LX0 OUT1.LX0  # E2
    ERROR(0.01) C2 OUT1.LX0  # E3
    ERROR(0.01) C2 C3 OUT1.LX0  # E4
    ERROR(0.01) C3 OUT1.LX0  # E5
    CX 0 3 1 4 2 5
    OUTPUT RepetitionCode 0 1 2
    OUTPUT RepetitionCode 3 4 5
    CHECK OUT0.S0 IN0.S0
    CHECK OUT0.S1 IN0.S1
    CHECK OUT1.S0 IN1.S0 IN0.S0
    CHECK OUT1.S1 IN1.S1 IN0.S1
    PROPAGATE OUT0.LZ0 FROM IN0.LZ0 IN1.LZ0
    PROPAGATE OUT0.LX0 FROM IN0.LX0
    PROPAGATE OUT1.LZ0 FROM IN1.LZ0
    PROPAGATE OUT1.LX0 FROM IN0.LX0 IN1.LX0

    # --- statistics ---
    # finished checks: 0
    # unfinished checks: 4
    #   weight distribution: { 2:2, 3:2 }
    # errors: 6
    #   check-weight distribution: { 1:2, 2:3, 4:1 }
}
```
<!-- deq-highlight-end: ../examples/multi-port/snippet_02_noisy_cnot.deq -->

The error patterns show two distinct behaviors:

**Control-side X errors** (qubits 0, 1, 2) propagate to **both** blocks:
- `ERROR(0.01) C0 C2 OUT0.LX0 OUT1.LX0` — X on qubit 0 triggers both C0 (control check)
  and C2 (target check), and leaves an X-type logical residual on both logical qubits

**Target-side X errors** (qubits 3, 4, 5) stay in the **target** block only:
- `ERROR(0.01) C2 OUT1.LX0` — X on qubit 3 triggers only C2 (target check) and has an
  X-type residual on only the target's logical qubit

This asymmetry is exactly the CNOT error propagation rule: X errors propagate from
control to target, while Z errors (not present in this X-only noise model) would
propagate from target to control.

---

## Summary

| Aspect                 | Single-port (Idle) | Multi-port (TransversalCNOT)         |
| ---------------------- | ------------------ | ------------------------------------ |
| INPUT/OUTPUT ports     | 1 each             | 2 each (control + target)            |
| Physical measurements  | 2 (ancillae)       | 0 (unitary gate)                     |
| Finished checks        | 2 (time-like)      | 0                                    |
| Unfinished checks      | 2 (same-port)      | 4 (2 same-port + 2 cross-port)       |
| Error propagation      | Within one block   | X: control→target; Z: target→control |
| correction_propagation | 2×3 identity       | 4×5 with off-diagonal entries        |

**Key insight:** even a gadget with 0 measurements has non-trivial unfinished checks when
it has multiple ports. The cross-port checks (`in0[k] ⊕ in1[k]`) are what enable the JIT
compiler to construct the correct decoding graph across entangling gates — linking the
check structures of the control and target code blocks.
