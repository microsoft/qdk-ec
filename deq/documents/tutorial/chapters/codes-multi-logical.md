# Codes with Multiple Logical Qubits ($k > 1$)

In the [language basics chapter](language-basics.md), we defined the repetition code
$[[3,1,1]]$ which encodes a single logical qubit ($k=1$). Many practical QEC codes encode
**multiple logical qubits** in a single code block — for instance, the $[[15,7,3]]$ quantum
Hamming code encodes 7 logical qubits in 15 physical qubits with distance 3.

This chapter shows how to define, compile, and simulate such codes in deq, using the
$[[15,7,3]]$ quantum Hamming code as a running example. Along the way, we introduce the
`MPP` (Measure Pauli Product) instruction for ancilla-free syndrome extraction.

---

## The $[[15,7,3]]$ Quantum Hamming Code

The quantum Hamming code is a CSS code constructed from the classical $[15,11,3]$ Hamming
code. Its parameters are:

| Parameter | Value | Meaning                                         |
| --------- | ----- | ----------------------------------------------- |
| $n$       | 15    | physical qubits                                 |
| $k$       | 7     | logical qubits                                  |
| $d$       | 3     | code distance (corrects any single-qubit error) |

Because it is a CSS code, all stabilizers are either pure-$X$ or pure-$Z$. There are
$n - k = 8$ independent stabilizer generators: 4 of type $Z$ and 4 of type $X$. Each
stabilizer has weight 8 (acts on half the qubits).

---

## Defining the Code: Multiple `LOGICAL` Lines

For a code with $k$ logical qubits, the `CODE` block contains **$k$ `LOGICAL` declarations**,
each specifying an $X$ and $Z$ logical operator pair for one logical qubit.

[Code definition](../examples/multi-logical/snippet_code.deq)
<!-- deq-highlight-begin: ../examples/multi-logical/snippet_code.deq -->
```deq
CODE QuantumHamming [[15,7,3]] {
    LOGICAL X0*X1*X2 Z2*Z4*Z5
    LOGICAL X1*X7*X9 Z4*Z5*Z8*Z9
    LOGICAL X0*X1*X7*X10 Z4*Z6*Z8*Z10
    LOGICAL X3*X7*X11 Z4*Z8*Z11
    LOGICAL X0*X3*X7*X12 Z4*Z5*Z6*Z8*Z12
    LOGICAL X1*X3*X7*X13 Z6*Z8*Z13
    LOGICAL X0*X1*X3*X7*X14 Z5*Z8*Z14
    STABILIZER Z0*Z2*Z4*Z6*Z8*Z10*Z12*Z14
    STABILIZER Z1*Z2*Z5*Z6*Z9*Z10*Z13*Z14
    STABILIZER Z3*Z4*Z5*Z6*Z11*Z12*Z13*Z14
    STABILIZER Z7*Z8*Z9*Z10*Z11*Z12*Z13*Z14
    STABILIZER X0*X2*X4*X6*X8*X10*X12*X14
    STABILIZER X1*X2*X5*X6*X9*X10*X13*X14
    STABILIZER X3*X4*X5*X6*X11*X12*X13*X14
    STABILIZER X7*X8*X9*X10*X11*X12*X13*X14
}
```
<!-- deq-highlight-end: ../examples/multi-logical/snippet_code.deq -->

Each `LOGICAL` line declares one logical qubit's $\bar{X}$ and $\bar{Z}$ operators. For
instance, logical qubit 0 has $\bar{X}_0 = X_0 X_1 X_2$ and $\bar{Z}_0 = Z_2 Z_4 Z_5$.

deq validates the commutation relations during transpilation:

1. **Each pair anticommutes**: $\bar{X}_i$ and $\bar{Z}_i$ must anticommute (they define
   conjugate observables of the same logical qubit).
2. **Different qubits commute**: All logical operators of different logical qubits must
   pairwise commute — $\bar{X}_i$ with $\bar{X}_j$, $\bar{X}_i$ with $\bar{Z}_j$, and
   $\bar{Z}_i$ with $\bar{Z}_j$ for $i \neq j$ (logical qubits are independent).
3. **All stabilizers commute** with all logical operators.
4. **Stabilizers commute pairwise**.

If any of these conditions is violated, the transpiler raises a clear error message
identifying the offending operators.

> **Contrast with $k = 1$:** In the repetition code, there was a single `LOGICAL` line.
> With $k = 7$, we write 7 `LOGICAL` lines. The rest of the `CODE` block (`STABILIZER`
> lines, `[[n,k,d]]` parameters) works the same way.

---

## Syndrome Extraction with `MPP`

The `MPP` (Measure Pauli Product) instruction measures a multi-qubit Pauli operator
**directly**, without ancilla qubits. This is particularly convenient for codes with
high-weight stabilizers like the quantum Hamming code, where each stabilizer acts on
8 qubits.

A single `MPP` instruction can measure **multiple stabilizers** — each Pauli product
(separated by spaces) produces one measurement result:

[Idle gadget](../examples/multi-logical/snippet_idle.deq)
<!-- deq-highlight-begin: ../examples/multi-logical/snippet_idle.deq -->
```deq
GADGET Idle {
    INPUT QuantumHamming 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    X_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    Z_ERROR(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    MPP(0.001) Z0*Z2*Z4*Z6*Z8*Z10*Z12*Z14 Z1*Z2*Z5*Z6*Z9*Z10*Z13*Z14 Z3*Z4*Z5*Z6*Z11*Z12*Z13*Z14 Z7*Z8*Z9*Z10*Z11*Z12*Z13*Z14
    MPP(0.001) X0*X2*X4*X6*X8*X10*X12*X14 X1*X2*X5*X6*X9*X10*X13*X14 X3*X4*X5*X6*X11*X12*X13*X14 X7*X8*X9*X10*X11*X12*X13*X14
    OUTPUT QuantumHamming 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
}
```
<!-- deq-highlight-end: ../examples/multi-logical/snippet_idle.deq -->

The first `MPP` line measures all 4 Z-type stabilizers simultaneously; the second measures
the 4 X-type stabilizers. Together, these 8 measurements extract the full syndrome.

The optional argument `(0.001)` specifies the measurement flip probability — each
measurement result is independently flipped with this probability. In a noiseless
simulation, `MPP` (without an argument) performs perfect measurements.

> **Note:** `MPP` is generally not a native operation on real quantum hardware — in
> practice, stabilizer measurements are decomposed into ancilla-based circuits. However,
> `MPP` is very useful for evaluating a code under **phenomenological noise**, where the
> only error sources are data qubit errors (`X_ERROR`, `Z_ERROR`) and pure measurement
> flip errors (`MPP(p)`). This noise model isolates the code's intrinsic error-correcting
> capability from the details of a specific syndrome extraction circuit.

> **Compared to ancilla-based extraction:** In the language basics chapter, syndrome
> extraction used `R` (reset ancilla), `CX` (entangle), and `M` (measure ancilla) — one
> ancilla per stabilizer. `MPP` replaces this entire sequence with a single instruction.
> Both approaches produce the same check structure in the compiled output.

---

## Multiple `READOUT` Statements

When the code encodes $k > 1$ logical qubits, the `MeasureZAll` gadget needs **$k$
`READOUT` statements** — one per logical qubit. Each `READOUT` XORs the physical
measurement results corresponding to one logical $\bar{Z}$ operator.

[Measurement gadget](../examples/multi-logical/snippet_measure.deq)
<!-- deq-highlight-begin: ../examples/multi-logical/snippet_measure.deq -->
```deq
GADGET MeasureZAll {
    INPUT QuantumHamming 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    M(0.001) 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    READOUT rec[-13] rec[-11] rec[-10]              # L0: Z2*Z4*Z5
    READOUT rec[-11] rec[-10] rec[-7] rec[-6]       # L1: Z4*Z5*Z8*Z9
    READOUT rec[-11] rec[-9] rec[-7] rec[-5]        # L2: Z4*Z6*Z8*Z10
    READOUT rec[-11] rec[-7] rec[-4]                # L3: Z4*Z8*Z11
    READOUT rec[-11] rec[-10] rec[-9] rec[-7] rec[-3]  # L4: Z4*Z5*Z6*Z8*Z12
    READOUT rec[-9] rec[-7] rec[-2]                 # L5: Z6*Z8*Z13
    READOUT rec[-10] rec[-7] rec[-1]                # L6: Z5*Z8*Z14
}
```
<!-- deq-highlight-end: ../examples/multi-logical/snippet_measure.deq -->

After `M 0 1 2 ... 14` measures all 15 physical qubits, each `READOUT` computes the parity
of the qubits in one logical $\bar{Z}$ operator. For example, the first `READOUT`
(`rec[-13] rec[-11] rec[-10]`) computes $M_2 \oplus M_4 \oplus M_5$, which is the
eigenvalue of $\bar{Z}_0 = Z_2 Z_4 Z_5$.

> **Contrast with $k = 1$:** The repetition code had a single `READOUT rec[-3] rec[-2] rec[-1]`
> combining all 3 physical measurements. With $k = 7$, each `READOUT` selects a
> different subset of measurements matching the logical operator's support.

---

## Multiple `ASSERT_EQ` Statements

The `PROGRAM` block defines the logical error criterion. With $k = 7$ logical qubits,
we need 7 assertions — one per readout:

[Program](../examples/multi-logical/snippet_program.deq)
<!-- deq-highlight-begin: ../examples/multi-logical/snippet_program.deq -->
```deq
PROGRAM MemoryExperiment {
    PrepareZAll 0
    Idle 0
    MeasureZAll 0
    ASSERT_EQ rec[-7] 0     # logical qubit 0
    ASSERT_EQ rec[-6] 0     # logical qubit 1
    ASSERT_EQ rec[-5] 0     # logical qubit 2
    ASSERT_EQ rec[-4] 0     # logical qubit 3
    ASSERT_EQ rec[-3] 0     # logical qubit 4
    ASSERT_EQ rec[-2] 0     # logical qubit 5
    ASSERT_EQ rec[-1] 0     # logical qubit 6
}
```
<!-- deq-highlight-end: ../examples/multi-logical/snippet_program.deq -->

A **logical error** occurs if *any* of the 7 readouts differs from its expected value.
The decoder must correct all 7 logical qubits simultaneously.

---

## Compiling and Running

### Noiseless verification

First, verify the circuit is correct with a noiseless sample:

```bash
deq sample multi-logical.deq \
    --program MemoryExperiment --noiseless --interpret --shots 10 --seed 42
```

Every shot should show all-zero syndromes and all-zero readouts.  The last
two lines of each shot's output confirm this:

```text
Syndrome: 0b0000000000000000
Readout:  0b0000000
```

### Noisy simulation

Compile the JIT library and run a noisy simulation:

```bash
# compile
deq transpile multi-logical.deq \
    --out multi-logical.deq.jit --program MemoryExperiment

# simulate with RelayBP decoder (100,000 shots)
deq server \
    --decoder black-box-relay-bp \
    --coordinator monolithic \
    --controller jit \
    --controller-config '{"filepath":"multi-logical.deq.jit"}' \
    --simulator jit-static \
    --simulator-config '{
        "filepath": "multi-logical.stim",
        "jit_library_filepath": "multi-logical.deq.jit",
        "shots": 10000000,
        "errors": 100,
        "seed": 123
    }'
```

At $p = 0.001$, the logical error rate is below the physical error rate,
confirming that the code is functioning correctly.

```
=== Simulation Complete ===
  Shots: 177011/10000000
  Logical errors: 100/100
  Error rate: 5.649366e-4 ± 1.11e-4
  Decoding time: 84.528s (4.775e-4s per shot)
```

---

## Summary

| Aspect                        | $k = 1$ (repetition code) | $k = 7$ (quantum Hamming)         |
| ----------------------------- | ------------------------- | --------------------------------- |
| `LOGICAL` lines               | 1                         | 7 (one per logical qubit)         |
| `STABILIZER` generators       | $n - k = 2$               | $n - k = 8$ (4 Z-type + 4 X-type) |
| Virtual measurements per port | 2                         | 8                                 |
| `READOUT` lines               | 1                         | 7 (one per logical $\bar{Z}$)     |

The key principle: **everything scales with $k$**. Each logical qubit contributes one
`LOGICAL` declaration, one `READOUT` in the measurement gadget, and one `ASSERT_EQ` in
the program. The transpiler, noise builder, and decoder handle the rest automatically.
