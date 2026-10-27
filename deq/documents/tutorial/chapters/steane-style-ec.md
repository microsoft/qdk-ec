# Steane-style Syndrome Extraction

Traditional syndrome extraction uses stabilizer measurements (e.g., `MPP`) that
produce checks spanning two consecutive rounds — the current measurement and
the previous one. This creates cross-round dependencies that force the window
decoder to look at multiple gadgets simultaneously (`buffer_radius ≥ 1`).

Steane-style syndrome extraction eliminates these cross-round dependencies
entirely. Each syndrome gadget prepares fresh ancilla blocks, entangles them
with the data, and measures destructively. The **output is a fresh logical
block** (the logical ancilla), so the checks depend only on local measurements —
enabling fully time-isolated decoding with `buffer_radius = 0`.

This chapter walks through a concrete implementation on the Steane [[7,1,3]]
code and demonstrates how `@DECODE_ONLY` phantom noise restores full
distance scaling under buffer=0 window decoding.

## The Steane [[7,1,3]] code

The Steane code encodes 1 logical qubit in 7 physical qubits with distance 3.
It has 6 stabilizer generators — 3 Z-type and 3 X-type:

[Steane code definition](../examples/steane-ec/snippet_code_definition.deq)
<!-- deq-highlight-begin: ../examples/steane-ec/snippet_code_definition.deq -->
```deq
CODE SteaneCode [[7,1,3]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z3*Z4*Z5*Z6 Z1*Z2*Z5*Z6 Z0*Z2*Z4*Z6
    STABILIZER X3*X4*X5*X6 X1*X2*X5*X6 X0*X2*X4*X6
}
```
<!-- deq-highlight-end: ../examples/steane-ec/snippet_code_definition.deq -->

The logical operators are $\bar{X} = X_0 X_1 X_2$ and $\bar{Z} = Z_0 Z_1 Z_2$.

## The `SteaneSyndrome` gadget

The syndrome extraction gadget uses a teleportation-based approach:

[SteaneSyndrome gadget](../examples/steane-ec/snippet_syndrome_gadget.deq)
<!-- deq-highlight-begin: ../examples/steane-ec/snippet_syndrome_gadget.deq -->
```deq
GADGET SteaneSyndrome {
    INPUT SteaneCode 0 1 2 3 4 5 6

    # prepare 1st ancilla logical block in |0_L> state
    R 7 8 9 10 11 12 13
    MPP X10*X11*X12*X13 X8*X9*X12*X13 X7*X9*X11*X13
    CZ rec[-3] 8 rec[-3] 9 rec[-3] 11
    CZ rec[-2] 8 rec[-2] 9 rec[-2] 11 rec[-2] 12
    CZ rec[-1] 8 rec[-1] 9
    DEPOLARIZE1(${p}) 7 8 9 10 11 12 13

    # prepare 2nd ancilla logical block in |+_L> state
    RX 14 15 16 17 18 19 20
    MPP Z17*Z18*Z19*Z20 Z15*Z16*Z19*Z20 Z14*Z16*Z18*Z20
    CX rec[-3] 17
    CX rec[-2] 15 rec[-2] 16 rec[-2] 17 rec[-2] 20
    CX rec[-1] 15 rec[-1] 16
    DEPOLARIZE1(${p}) 14 15 16 17 18 19 20

    # CNOT data to 1st ancilla
    CX 0 7 1 8 2 9 3 10 4 11 5 12 6 13
    DEPOLARIZE2(${p}) 0 7 1 8 2 9 3 10 4 11 5 12 6 13

    # CNOT 2nd ancilla to 1st ancilla
    CX 14 7 15 8 16 9 17 10 18 11 19 12 20 13
    DEPOLARIZE2(${p}) 14 7 15 8 16 9 17 10 18 11 19 12 20 13

    # measure data in X basis, measure 1st ancilla in Z basis
    MX(${pm}) 0 1 2 3 4 5 6
    MZ(${pm}) 7 8 9 10 11 12 13

    OUTPUT SteaneCode 14 15 16 17 18 19 20
}
```
<!-- deq-highlight-end: ../examples/steane-ec/snippet_syndrome_gadget.deq -->

The circuit proceeds in five stages:

| Step | Operations                                | Purpose                                           |
| ---- | ----------------------------------------- | ------------------------------------------------- |
| 1    | `R 7–13`, `MPP X`-stab, `CZ` correction   | Prepare 1st ancilla in $\lvert 0_L \rangle$       |
| 2    | `RX 14–20`, `MPP Z`-stab, `CX` correction | Prepare 2nd ancilla in $\lvert +_L \rangle$       |
| 3    | `CX data→anc1`                            | Propagate X errors from data into ancilla 1       |
| 4    | `CX anc2→anc1`                            | Entangle ancilla 2 with ancilla 1 (teleportation) |
| 5    | `MX data`, `MZ anc1`                      | Destructive measurements yield syndrome           |
| 6    | `OUTPUT anc2`                             | Fresh ancilla becomes the new data block          |

The key insight: because ancilla 2 is prepared **fresh** inside this gadget,
the output block carries no history from the previous round. Any errors on the
output come only from the current gadget's noise.

## Check structure: time-isolated syndrome

Running `deq annotate` on this gadget reveals the check structure:

[Annotated SteaneSyndrome](../examples/steane-ec/snippet_syndrome_annotated.deq)
<!-- deq-highlight-begin: ../examples/steane-ec/snippet_syndrome_annotated.deq -->
```deq
@GTYPE(3)
@CHECKS("manual", verify=0)
GADGET SteaneSyndrome {
    INPUT SteaneCode 0 1 2 3 4 5 6
    R 7 8 9 10 11 12 13
    MPP X10*X11*X12*X13 X8*X9*X12*X13 X7*X9*X11*X13
    CZ rec[-3] 8 rec[-3] 9 rec[-3] 11
    CZ rec[-2] 8 rec[-2] 9 rec[-2] 11 rec[-2] 12
    CZ rec[-1] 8 rec[-1] 9
    @SIMULATE_ONLY
    DEPOLARIZE1(0) 7 8 9 10 11 12 13
    RX 14 15 16 17 18 19 20
    MPP Z17*Z18*Z19*Z20 Z15*Z16*Z19*Z20 Z14*Z16*Z18*Z20
    CX rec[-3] 17
    CX rec[-2] 15 rec[-2] 16 rec[-2] 17 rec[-2] 20
    CX rec[-1] 15 rec[-1] 16
    @SIMULATE_ONLY
    DEPOLARIZE1(0) 14 15 16 17 18 19 20
    CX 0 7 1 8 2 9 3 10 4 11 5 12 6 13
    @SIMULATE_ONLY
    DEPOLARIZE2(0) 0 7 1 8 2 9 3 10 4 11 5 12 6 13
    CX 14 7 15 8 16 9 17 10 18 11 19 12 20 13
    @SIMULATE_ONLY
    DEPOLARIZE2(0) 14 7 15 8 16 9 17 10 18 11 19 12 20 13
    MX(0) 0 1 2 3 4 5 6
    MZ(0) 7 8 9 10 11 12 13
    CHECK M19 M18 M17 M16 IN0.S0
    CHECK M19 M18 M15 M14 IN0.S1
    CHECK M19 M17 M15 M13 IN0.S2
    CHECK M12 M11 M10 M9 IN0.S3
    CHECK M12 M11 M8 M7 IN0.S4
    CHECK M12 M10 M8 M6 IN0.S5
    OUTPUT SteaneCode 14 15 16 17 18 19 20
    CHECK OUT0.S0
    CHECK OUT0.S1
    CHECK OUT0.S2
    CHECK OUT0.S3
    CHECK OUT0.S4
    CHECK OUT0.S5
    PROPAGATE OUT0.LZ0 FROM IN0.LZ0 M6 M7 M8
    PROPAGATE OUT0.LX0 FROM IN0.LX0 M13 M14 M15

    # --- statistics ---
    # finished checks: 6
    #   weight distribution: { 5:6 }
    # unfinished checks: 6
    #   weight distribution: { 1:6 }
    # errors: 0
}
```
<!-- deq-highlight-end: ../examples/steane-ec/snippet_syndrome_annotated.deq -->

The gadget has **6 finished checks** (syndrome) and **6 unfinished checks**
(output stabilizer tracking). Let us examine them in detail.

### Unfinished checks: output is a +1 eigenstate

The 6 unfinished checks appear after the `OUTPUT` statement:

```
CHECK OUT0.S0
CHECK OUT0.S1
CHECK OUT0.S2
CHECK OUT0.S3
CHECK OUT0.S4
CHECK OUT0.S5
```

Each is a **weight-1** check referencing a single output-virtual stabilizer
(`OUT0.S<s>`). This means the output state is a $+1$ eigenstate of every
stabilizer, **regardless of the input state**. The fresh ancilla block's
stabilizer values are deterministically 0.

This is a striking contrast with traditional repeated-round syndrome
extraction, where unfinished checks have weight $\geq 2$ because the output
stabilizer value must be expressed as a combination of the current round's
measurement and the previous round's value.

Because the unfinished checks are weight-1 and do not expand into additional
physical measurements, the previous gadget's output stabilizer values are
simply constants. This is the key to what happens next.

### Finished checks: purely local syndrome

The 6 finished checks appear before the `OUTPUT` statement:

```
CHECK M19 M18 M17 M16 IN0.S0
CHECK M19 M18 M15 M14 IN0.S1
CHECK M19 M17 M15 M13 IN0.S2
CHECK M12 M11 M10 M9 IN0.S3
CHECK M12 M11 M8 M7 IN0.S4
CHECK M12 M10 M8 M6 IN0.S5
```

Each finished check combines **one input-virtual stabilizer** (`IN0.S<s>`)
with **four physical measurements** (`M<i>`) from the current gadget. Since
the previous gadget's unfinished checks are weight-1 (the output is always a
$+1$ eigenstate), the input-virtual stabilizer contribution is a known
constant — it does not expand into measurements from a previous round.

This means the finished checks depend **only on the current gadget's local
physical measurements**. The decoder can compute the full syndrome without
consulting any neighboring gadgets. This is the defining property of
Steane-style EC that enables `buffer_radius = 0`.

## Gate noise and phantom errors

### The problem

When the `CX` gates between ancilla 2 and ancilla 1 have noise
(`DEPOLARIZE2`), a Z error on an ancilla-2 qubit:

1. Stays on ancilla 2 (which becomes the output)
2. Does **not** affect ancilla 1 (Z does not propagate through CX on the
   control side)
3. Has **no local check** in the current gadget (ancilla 2 is not measured)

The window coordinator correctly drops this error from the decoder — it is a
commit-region error that triggers external checks.
Such errors should never be selected to avoid syndrome conflicts.
But the error propagates to
the next gadget's input, and the next gadget (e.g., `MeasureZ`) has no error
models to explain it.

### The solution: `@DECODE_ONLY` phantom noise

The `@DECODE_ONLY` decorator marks an instruction as visible only to the
decoder (not included in the simulation). By adding phantom noise to the
receiving gadget's input, we give the decoder error models for carried-over
errors:

[MeasureZ with phantom noise](../examples/steane-ec/snippet_measure_phantom.deq)
<!-- deq-highlight-begin: ../examples/steane-ec/snippet_measure_phantom.deq -->
```deq
GADGET MeasureZ {
    INPUT SteaneCode 0 1 2 3 4 5 6

%if has_phantom:
    @DECODE_ONLY
    DEPOLARIZE1(${p}) 0 1 2 3 4 5 6
%endif

    M 0 1 2 3 4 5 6
    READOUT rec[-7] rec[-6] rec[-5]
}
```
<!-- deq-highlight-end: ../examples/steane-ec/snippet_measure_phantom.deq -->

The `@DECODE_ONLY DEPOLARIZE1` instruction:
- Is **included** in the JIT library (decoder sees error models for input errors)
- Is **excluded** from the `.stim` file (simulation does not double-count noise)
- Gives the window decoder the ability to explain syndrome defects caused by
  the previous gadget's output errors

Note that `@SIMULATE_ONLY` is **not** needed on the `DEPOLARIZE2` after the CX
gates. The window coordinator automatically drops commit-region errors with
external checks, so these error models are harmless — they simply get discarded.

## Simulation results

We compare logical error rates for the `Minimal` program
(`PrepareZ → SteaneSyndrome → MeasureZ`) at physical error rate $p = 10^{-3}$:

```text
Steane [[7,1,3]] code — LER sweep (100 target errors per data point)
================================================================================

                                          p = 1e-3
monolithic                                3.0e-4
buffer=0 (no phantom)                     3.3e-3
buffer=0 (with @DECODE_ONLY phantom)      2.4e-4
```

|                         | LER at p = 10⁻³ | Ratio to monolithic   |
| ----------------------- | --------------- | --------------------- |
| monolithic              | 3.0 × 10⁻⁴      | 1×                    |
| buffer=0 (no phantom)   | 3.3 × 10⁻³      | **11× worse**         |
| buffer=0 (with phantom) | 2.4 × 10⁻⁴      | **0.8×** (comparable) |

Without phantom noise, `buffer=0` produces a logical error rate **11×
worse** than monolithic — the decoder has no error model for errors carried
over from the previous gadget, so it misattributes syndrome defects and
applies wrong corrections.

With `@DECODE_ONLY` phantom noise, `buffer=0` performance matches
monolithic. The phantom `DEPOLARIZE1` gives the receiving gadget's decoder
error models for carried-over errors, restoring full decoding accuracy.

## Summary

| Property                 | Traditional MPP syndrome      | Steane-style EC                           |
| ------------------------ | ----------------------------- | ----------------------------------------- |
| Finished checks span     | 2 rounds (current + previous) | 1 round (current only)                    |
| Unfinished checks weight | ≥ 2                           | 1                                         |
| Minimum `buffer_radius`  | ≥ 1                           | 0                                         |
| Cross-round dependencies | Yes                           | No                                        |
| Requires phantom noise   | No                            | Yes (`@DECODE_ONLY` on receiving gadgets) |

Steane-style syndrome extraction enables **fully time-isolated window
decoding**: each gadget is decoded independently with `buffer_radius = 0`.
The only annotation needed is `@DECODE_ONLY` phantom noise on gadgets that
receive teleported data, giving their decoders error models for carried-over
errors. This dramatically simplifies the decoding system — no overlapping
windows, no cross-round syndrome dependencies, and no temporal buffering.
