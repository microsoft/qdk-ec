# Composing Gadgets with COMPOSE

The [language chapter](language-basics.md) showed how to define small gadgets — such as
a single syndrome extraction round — and how the transpiler automatically derives checks
and errors for each one. But what happens when you need **multiple rounds**?

You have two options:

1. **Write a single large gadget** with multiple rounds of syndrome extraction inlined.
   The transpiler auto-derives checks for the entire circuit, but the resulting error
   model can have errors that span **all rounds simultaneously** — producing a dense,
   unstructured decoding hypergraph.

2. **Use `COMPOSE`** to chain small per-round gadgets. Each sub-gadget is analyzed
   independently, then the COMPOSE mechanism chains their matrices and accumulates
   their checks. The result: errors are **always local to individual sub-gadgets**,
   producing a sparse, well-structured hypergraph — by construction.

This chapter demonstrates the difference with our running d=3 repetition code example.

## Private Helpers

Mark an internal `GADGET` or `COMPOSE` with argument-free `@PRIVATE` when it
must only be called from another `COMPOSE`. For example:

```deq
CODE Qubit [[1,1,1]] { LOGICAL X0 Z0 }

GADGET PrepareZero {
    R 0
    OUTPUT Qubit 0
}

@PRIVATE
GADGET FilterZero {
    INPUT Qubit 0
    M 0
    PRESELECT rec[-1]
    OUTPUT Qubit 0
}

COMPOSE PrepareFilteredZero {
    PrepareZero 0
    FilterZero 0
    OUTPUT Qubit 0
}
```

A `PROGRAM` may call `PrepareFilteredZero`, but cannot call `FilterZero`
directly, including through a `REPEAT` or a subprogram. Private definitions
remain available to compositions across imports; `@PRIVATE` is not file-local
visibility. Private gadget types are used internally to compile compositions,
but are **not included in generated JIT libraries**, including program-only
builds. Their behavior is incorporated into the public composed gadgets;
there is no standalone private runtime gadget type to call.

Annotation preserves the private source definitions and marker for inspection
and round trips. Generated JIT metadata retains only their names for clear
program-call diagnostics. Rebuild older JIT libraries after changing visibility.

Only the private helper's input-isolation warning is deferred. Invalid
measurement references, port ordering, and other structural errors are still
rejected. Every public composition is checked after recursive expansion, in
both simulation and decoder views: operations before its last `PRESELECT` must
not touch the composition's live input qubits. For the example, retrying
`PrepareFilteredZero` recreates its qubit from scratch. A public wrapper that
instead passes an existing input qubit into `FilterZero` still emits the unsafe
retry warning. `@PRIVATE` does not by itself make preselection safe.

---

## The Problem: Flat Multi-Round Gadgets

Let's write a single gadget with 3 rounds of syndrome extraction:

[Flat 3-round gadget](../examples/compose/01_flat_3idle.deq)
<!-- deq-highlight-begin: ../examples/compose/01_flat_3idle.deq -->
```deq
# Flat 3-round syndrome extraction: all 3 rounds in a single gadget
# This demonstrates the problem with auto-derived checks spanning all rounds

CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}

GADGET PrepareZ {
    R 0 1 2
    X_ERROR(0.01) 0 1 2
    OUTPUT RepetitionCode 0 1 2
}

# A single gadget with 3 rounds of syndrome extraction inlined
GADGET Flat3Idle {
    INPUT RepetitionCode 0 2 4
    REPEAT 3 {
        X_ERROR(0.01) 0 2 4
        R 1 3
        CX 0 1 2 3
        CX 2 1 4 3
        X_ERROR(0.01) 1 3
        M 1 3
    }
    OUTPUT RepetitionCode 0 2 4
}

GADGET MeasureZ {
    INPUT RepetitionCode 0 1 2
    M(0.01) 0 1 2
    READOUT rec[-3] rec[-2] rec[-1]
}

PROGRAM Simulation {
    PrepareZ 0
    Flat3Idle 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
```
<!-- deq-highlight-end: ../examples/compose/01_flat_3idle.deq -->

The circuit is physically identical to running the Idle gadget 3 times. Running
`annotate` reveals the derived checks and errors:

[Annotated flat 3-round gadget](../examples/compose/01_flat_3idle.annotated.deq)
<!-- deq-highlight-begin: ../examples/compose/01_flat_3idle.annotated.deq -->
```deq
@PTYPE(1)
CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1  # generator S0, destabilizer DS0=X1*X2
    STABILIZER Z1*Z2  # generator S1, destabilizer DS1=X0*X1
}

@GTYPE(1)
@CHECKS("manual", verify=0)
GADGET PrepareZ {
    R 0 1 2
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C0 OUT0.LX0  # E0
    ERROR(0.01) C0 C1 OUT0.LX0  # E1
    ERROR(0.01) C1 OUT0.LX0  # E2
    OUTPUT RepetitionCode 0 1 2
    CHECK OUT0.S0
    CHECK OUT0.S1
    PROPAGATE OUT0.LZ0 FROM
    PROPAGATE OUT0.LX0 FROM

    # --- statistics ---
    # finished checks: 0
    # unfinished checks: 2
    #   weight distribution: { 1:2 }
    # errors: 3
    #   check-weight distribution: { 1:2, 2:1 }
}

@GTYPE(2)
@CHECKS("manual", verify=0)
GADGET Flat3Idle {
    INPUT RepetitionCode 0 2 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 2 4
    ERROR(0.01) C0 OUT0.LX0  # E0
    ERROR(0.01) C0 C1 OUT0.LX0  # E1
    ERROR(0.01) C1 OUT0.LX0  # E2
    R 1 3
    CX 0 1 2 3
    CX 2 1 4 3
    @SIMULATE_ONLY
    X_ERROR(0.01) 1 3
    ERROR(0.01) C2  # E3
    ERROR(0.01) C3  # E4
    M 1 3
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 2 4
    ERROR(0.01) C0 C2 OUT0.LX0  # E5
    ERROR(0.01) C0 C1 C2 C3 OUT0.LX0  # E6
    ERROR(0.01) C1 C3 OUT0.LX0  # E7
    R 1 3
    CX 0 1 2 3
    CX 2 1 4 3
    @SIMULATE_ONLY
    X_ERROR(0.01) 1 3
    ERROR(0.01) C4  # E8
    ERROR(0.01) C5  # E9
    M 1 3
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 2 4
    ERROR(0.01) C0 C2 C4 OUT0.LX0  # E10
    ERROR(0.01) C0 C1 C2 C3 C4 C5 OUT0.LX0  # E11
    ERROR(0.01) C1 C3 C5 OUT0.LX0  # E12
    R 1 3
    CX 0 1 2 3
    CX 2 1 4 3
    @SIMULATE_ONLY
    X_ERROR(0.01) 1 3
    ERROR(0.01) C0 C2 C4 C6  # E13
    ERROR(0.01) C1 C3 C5 C7  # E14
    M 1 3
    CHECK M4 IN0.S0
    CHECK M5 IN0.S1
    CHECK M4 M0
    CHECK M5 M1
    CHECK M4 M2
    CHECK M5 M3
    OUTPUT RepetitionCode 0 2 4
    CHECK OUT0.S0 M4
    CHECK OUT0.S1 M5
    PROPAGATE OUT0.LZ0 FROM IN0.LZ0
    PROPAGATE OUT0.LX0 FROM IN0.LX0

    # --- statistics ---
    # finished checks: 6
    #   weight distribution: { 2:6 }
    # unfinished checks: 2
    #   weight distribution: { 2:2 }
    # errors: 15
    #   check-weight distribution: { 1:6, 2:3, 3:2, 4:3, 6:1 }
}

@GTYPE(3)
@CHECKS("manual", verify=0)
GADGET MeasureZ {
    INPUT RepetitionCode 0 1 2
    @SIMULATE_ONLY
    M(0.01) 0 1 2
    @DECODE_ONLY
    M 0 1 2
    ERROR(0.01) C0 R0  # E0
    ERROR(0.01) C0 C1 R0  # E1
    ERROR(0.01) C1 R0  # E2
    READOUT rec[-3] rec[-2] rec[-1]  # IN0.LX0
    CHECK M1 M0 IN0.S0
    CHECK M2 M1 IN0.S1

    # --- statistics ---
    # finished checks: 2
    #   weight distribution: { 3:2 }
    # unfinished checks: 0
    # errors: 3
    #   check-weight distribution: { 1:2, 2:1 }
}

PROGRAM Simulation {
    PrepareZ 0
    Flat3Idle 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
```
<!-- deq-highlight-end: ../examples/compose/01_flat_3idle.annotated.deq -->

Look at the **checks** in the Flat3Idle gadget — they all involve `M4` or `M5`
(the measurements from **round 3**):

```
CHECK M4 IN0.S0   # round 3 vs input virtual
CHECK M4 M0       # round 3 vs round 1
CHECK M4 M2       # round 3 vs round 2
```

Although these checks are weight-2 (each references only 2 measurements), they are
**not** the natural round-to-round comparisons. Instead, every check involves the last
round's measurement — the auto-derivation chose a valid but non-local basis. Minimizing
the check weight alone does not guarantee a good decoding hypergraph structure.

Now look at the **errors**:

```
ERROR(0.01) C0 C2 C4 OUT0.LX0              # data error in round 1: triggers 3 checks + logical
ERROR(0.01) C0 C1 C2 C3 C4 C5 OUT0.LX0     # data error in round 1 (qubit 1): triggers 6 checks + logical
ERROR(0.01) C0 C2 C4 C6                    # measurement error in round 3: triggers 4 checks
```

A single data qubit error in round 1 triggers checks from **all 3 rounds** simultaneously
(C0, C1, C2, C3, C4, C5) plus the logical observable — a weight-6 hyperedge in the
decoding graph. This happens because the flat gadget's error analysis propagates each
error through the **entire** remaining circuit: a data error in round 1 propagates
through the round-2 and round-3 CNOT gates, accumulating check triggers along the way.

The checks themselves are weight-2, but because they are non-locally defined (all
involving the last round), the errors that trigger them span multiple rounds. The
combination of non-local checks and high-weight errors makes the decoding hypergraph
dense and difficult to decode efficiently.

---

## The Solution: COMPOSE

The `COMPOSE` block chains small gadgets at the **logical level**, preserving each
sub-gadget's error locality:

[COMPOSE 3-round gadget](../examples/compose/02_compose_3idle.deq)
<!-- deq-highlight-begin: ../examples/compose/02_compose_3idle.deq -->
```deq
# COMPOSE version: 3 rounds of syndrome extraction via composition
# Demonstrates well-structured checks by construction

CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}

GADGET PrepareZ {
    R 0 1 2
    X_ERROR(0.01) 0 1 2
    OUTPUT RepetitionCode 0 1 2
}

GADGET Idle {
    INPUT RepetitionCode 0 2 4
    X_ERROR(0.01) 0 2 4
    R 1 3
    CX 0 1 2 3
    CX 2 1 4 3
    X_ERROR(0.01) 1 3
    M 1 3
    OUTPUT RepetitionCode 0 2 4
}

GADGET MeasureZ {
    INPUT RepetitionCode 0 1 2
    M(0.01) 0 1 2
    READOUT rec[-3] rec[-2] rec[-1]
}

COMPOSE Idle3 {
    INPUT RepetitionCode 0
    REPEAT 3 {
        Idle 0
    }
    OUTPUT RepetitionCode 0
}

PROGRAM Simulation {
    PrepareZ 0
    Idle3 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
```
<!-- deq-highlight-end: ../examples/compose/02_compose_3idle.deq -->

The key part is the `COMPOSE` block:

[COMPOSE Idle3 block](../examples/compose/snippet_compose_idle3.deq)
<!-- deq-highlight-begin: ../examples/compose/snippet_compose_idle3.deq -->
```deq
COMPOSE Idle3 {
    INPUT RepetitionCode 0
    REPEAT 3 {
        Idle 0
    }
    OUTPUT RepetitionCode 0
}
```
<!-- deq-highlight-end: ../examples/compose/snippet_compose_idle3.deq -->

| Element                   | Meaning                                |
| ------------------------- | -------------------------------------- |
| `INPUT RepetitionCode 0`  | Declare an input port on code block 0  |
| `REPEAT 3 { ... }`        | Unroll 3 copies of the body            |
| `Idle 0`                  | Apply the Idle gadget on code block 0  |
| `OUTPUT RepetitionCode 0` | Declare an output port on code block 0 |

Note: inside a `COMPOSE` block, the numbers are **code block indices** (not physical
qubit indices). Each code block is an instance of a `CODE` type and may contain multiple
logical qubits when $k > 1$.

Running `annotate` on the COMPOSE version produces a flattened `GADGET` block:

[Annotated COMPOSE 3-round gadget](../examples/compose/02_compose_3idle.annotated.deq)
<!-- deq-highlight-begin: ../examples/compose/02_compose_3idle.annotated.deq -->
```deq
@PTYPE(1)
CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1  # generator S0, destabilizer DS0=X1*X2
    STABILIZER Z1*Z2  # generator S1, destabilizer DS1=X0*X1
}

@GTYPE(1)
@CHECKS("manual", verify=0)
GADGET PrepareZ {
    R 0 1 2
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C0 OUT0.LX0  # E0
    ERROR(0.01) C0 C1 OUT0.LX0  # E1
    ERROR(0.01) C1 OUT0.LX0  # E2
    OUTPUT RepetitionCode 0 1 2
    CHECK OUT0.S0
    CHECK OUT0.S1
    PROPAGATE OUT0.LZ0 FROM
    PROPAGATE OUT0.LX0 FROM

    # --- statistics ---
    # finished checks: 0
    # unfinished checks: 2
    #   weight distribution: { 1:2 }
    # errors: 3
    #   check-weight distribution: { 1:2, 2:1 }
}

@GTYPE(2)
@CHECKS("manual", verify=0)
GADGET Idle {
    INPUT RepetitionCode 0 2 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 2 4
    ERROR(0.01) C0 OUT0.LX0  # E0
    ERROR(0.01) C0 C1 OUT0.LX0  # E1
    ERROR(0.01) C1 OUT0.LX0  # E2
    R 1 3
    CX 0 1 2 3
    CX 2 1 4 3
    @SIMULATE_ONLY
    X_ERROR(0.01) 1 3
    ERROR(0.01) C0 C2  # E3
    ERROR(0.01) C1 C3  # E4
    M 1 3
    CHECK M0 IN0.S0
    CHECK M1 IN0.S1
    OUTPUT RepetitionCode 0 2 4
    CHECK OUT0.S0 M0
    CHECK OUT0.S1 M1
    PROPAGATE OUT0.LZ0 FROM IN0.LZ0
    PROPAGATE OUT0.LX0 FROM IN0.LX0

    # --- statistics ---
    # finished checks: 2
    #   weight distribution: { 2:2 }
    # unfinished checks: 2
    #   weight distribution: { 2:2 }
    # errors: 5
    #   check-weight distribution: { 1:2, 2:3 }
}

@GTYPE(3)
@CHECKS("manual", verify=0)
GADGET MeasureZ {
    INPUT RepetitionCode 0 1 2
    @SIMULATE_ONLY
    M(0.01) 0 1 2
    @DECODE_ONLY
    M 0 1 2
    ERROR(0.01) C0 R0  # E0
    ERROR(0.01) C0 C1 R0  # E1
    ERROR(0.01) C1 R0  # E2
    READOUT rec[-3] rec[-2] rec[-1]  # IN0.LX0
    CHECK M1 M0 IN0.S0
    CHECK M2 M1 IN0.S1

    # --- statistics ---
    # finished checks: 2
    #   weight distribution: { 3:2 }
    # unfinished checks: 0
    # errors: 3
    #   check-weight distribution: { 1:2, 2:1 }
}

@GTYPE(4)
@CHECKS("manual", verify=0)
GADGET Idle3 {
    INPUT RepetitionCode 0 1 2
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C0 OUT0.LX0  # E0
    ERROR(0.01) C0 C1 OUT0.LX0  # E1
    ERROR(0.01) C1 OUT0.LX0  # E2
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 3 4
    ERROR(0.01) C0 C2  # E3
    ERROR(0.01) C1 C3  # E4
    M 3 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C2 OUT0.LX0  # E5
    ERROR(0.01) C2 C3 OUT0.LX0  # E6
    ERROR(0.01) C3 OUT0.LX0  # E7
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 3 4
    ERROR(0.01) C2 C4  # E8
    ERROR(0.01) C3 C5  # E9
    M 3 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C4 OUT0.LX0  # E10
    ERROR(0.01) C4 C5 OUT0.LX0  # E11
    ERROR(0.01) C5 OUT0.LX0  # E12
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 3 4
    ERROR(0.01) C4 C6  # E13
    ERROR(0.01) C5 C7  # E14
    M 3 4
    OUTPUT RepetitionCode 0 1 2
    CHECK IN0.S0 M0
    CHECK IN0.S1 M1
    CHECK M0 M2
    CHECK M1 M3
    CHECK M2 M4
    CHECK M3 M5
    CHECK M4 OUT0.S0
    CHECK M5 OUT0.S1
    PROPAGATE OUT0.LZ0 FROM IN0.LZ0
    PROPAGATE OUT0.LX0 FROM IN0.LX0

    # --- statistics ---
    # finished checks: 6
    #   weight distribution: { 2:6 }
    # unfinished checks: 2
    #   weight distribution: { 1:2 }
    # errors: 15
    #   check-weight distribution: { 1:6, 2:9 }
}

PROGRAM Simulation {
    PrepareZ 0
    Idle3 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
```
<!-- deq-highlight-end: ../examples/compose/02_compose_3idle.annotated.deq -->

The composed `Idle3` gadget appears as:

[Composed Idle3 gadget](../examples/compose/snippet_idle3_annotated.deq)
<!-- deq-highlight-begin: ../examples/compose/snippet_idle3_annotated.deq -->
```deq
@GTYPE(4)
@CHECKS("manual", verify=0)
GADGET Idle3 {
    INPUT RepetitionCode 0 1 2
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C0 OUT0.LX0  # E0
    ERROR(0.01) C0 C1 OUT0.LX0  # E1
    ERROR(0.01) C1 OUT0.LX0  # E2
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 3 4
    ERROR(0.01) C0 C2  # E3
    ERROR(0.01) C1 C3  # E4
    M 3 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C2 OUT0.LX0  # E5
    ERROR(0.01) C2 C3 OUT0.LX0  # E6
    ERROR(0.01) C3 OUT0.LX0  # E7
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 3 4
    ERROR(0.01) C2 C4  # E8
    ERROR(0.01) C3 C5  # E9
    M 3 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C4 OUT0.LX0  # E10
    ERROR(0.01) C4 C5 OUT0.LX0  # E11
    ERROR(0.01) C5 OUT0.LX0  # E12
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 3 4
    ERROR(0.01) C4 C6  # E13
    ERROR(0.01) C5 C7  # E14
    M 3 4
    OUTPUT RepetitionCode 0 1 2
    CHECK IN0.S0 M0
    CHECK IN0.S1 M1
    CHECK M0 M2
    CHECK M1 M3
    CHECK M2 M4
    CHECK M3 M5
    CHECK M4 OUT0.S0
    CHECK M5 OUT0.S1
    PROPAGATE OUT0.LZ0 FROM IN0.LZ0
    PROPAGATE OUT0.LX0 FROM IN0.LX0

    # --- statistics ---
    # finished checks: 6
    #   weight distribution: { 2:6 }
    # unfinished checks: 2
    #   weight distribution: { 1:2 }
    # errors: 15
    #   check-weight distribution: { 1:6, 2:9 }
}
```
<!-- deq-highlight-end: ../examples/compose/snippet_idle3_annotated.deq -->

Every check references only **adjacent measurement pairs** — each spans exactly 2
positions in the measurement record. And crucially, the Idle gadget's errors remain
unchanged:

```
ERROR(0.01) C0 LX0    # data error: 1 check + logical
ERROR(0.01) C0 C2     # measurement error: 2 checks
```

Each error triggers at most **2 checks** — weight-2 edges, perfectly matchable. The
COMPOSE mechanism achieves this because it processes each sub-gadget independently:
errors are resolved within their originating gadget and never propagate through subsequent
sub-gadgets' circuits.

### Why the errors differ

In the flat gadget, a data qubit error in round 1 propagates through the round-2 and
round-3 CNOT gates. The transpiler sees this full propagation and reports the error as
triggering checks from all subsequent rounds.

In the COMPOSE version, the same data qubit error is analyzed within a single Idle
instance. It triggers one finished check and has a residual (it flips the output
observable). When the next Idle instance runs, the residual is carried through the
correction propagation matrix — but this is handled by the **Pauli frame**, not by
additional check triggers. The error's effect on the decoding hypergraph remains local.

---

## REPEAT Inside COMPOSE

The `REPEAT N { ... }` block inside COMPOSE unrolls N copies of the body at transpile
time. It is equivalent to writing the body N times:

[REPEAT equivalence](../examples/compose/repeat_equivalent.deq)
<!-- deq-highlight-begin: ../examples/compose/repeat_equivalent.deq -->
```deq
# These are equivalent:
COMPOSE Idle3 {
    INPUT RepetitionCode 0
    REPEAT 3 {
        Idle 0
    }
    OUTPUT RepetitionCode 0
}

COMPOSE Idle3 {
    INPUT RepetitionCode 0
    Idle 0
    Idle 0
    Idle 0
    OUTPUT RepetitionCode 0
}
```
<!-- deq-highlight-end: ../examples/compose/repeat_equivalent.deq -->

The `REPEAT` form is preferred for clarity and to avoid repetitive code, especially when
the number of rounds is large.

---

## Nested COMPOSE

A COMPOSE gadget is itself a gadget — it can be used inside other COMPOSE blocks. This
enables hierarchical composition:

[Nested COMPOSE: Idle4 = Idle3 + Idle](../examples/compose/03_nested_compose.deq)
<!-- deq-highlight-begin: ../examples/compose/03_nested_compose.deq -->
```deq
# Nested COMPOSE: Idle4 = Idle3 + Idle
# Demonstrates composing composed gadgets

CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}

GADGET PrepareZ {
    R 0 1 2
    X_ERROR(0.01) 0 1 2
    OUTPUT RepetitionCode 0 1 2
}

GADGET Idle {
    INPUT RepetitionCode 0 2 4
    X_ERROR(0.01) 0 2 4
    R 1 3
    CX 0 1 2 3
    CX 2 1 4 3
    M(0.01) 1 3
    OUTPUT RepetitionCode 0 2 4
}

GADGET MeasureZ {
    INPUT RepetitionCode 0 1 2
    M(0.01) 0 1 2
    READOUT rec[-3] rec[-2] rec[-1]
}

COMPOSE Idle3 {
    INPUT RepetitionCode 0
    REPEAT 3 {
        Idle 0
    }
    OUTPUT RepetitionCode 0
}

COMPOSE Idle4 {
    INPUT RepetitionCode 0
    Idle3 0
    Idle 0
    OUTPUT RepetitionCode 0
}

PROGRAM Simulation {
    PrepareZ 0
    Idle4 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
```
<!-- deq-highlight-end: ../examples/compose/03_nested_compose.deq -->

The key definitions:

[Nested COMPOSE definitions](../examples/compose/snippet_nested_compose.deq)
<!-- deq-highlight-begin: ../examples/compose/snippet_nested_compose.deq -->
```deq
COMPOSE Idle3 {
    INPUT RepetitionCode 0
    REPEAT 3 {
        Idle 0
    }
    OUTPUT RepetitionCode 0
}

COMPOSE Idle4 {
    INPUT RepetitionCode 0
    Idle3 0
    Idle 0
    OUTPUT RepetitionCode 0
}
```
<!-- deq-highlight-end: ../examples/compose/snippet_nested_compose.deq -->

`Idle4` uses `Idle3` (a COMPOSE) and `Idle` (a GADGET) as sub-gadgets. This works because
COMPOSEs are processed in declaration order — by the time `Idle4` is processed, `Idle3`
is already available as a `JitGadgetType`.

The annotated output for `Idle4` shows 8 measurements (6 from Idle3 + 2 from Idle) and
10 checks (8 from Idle3 + 2 from Idle):

[Annotated nested COMPOSE](../examples/compose/03_nested_compose.annotated.deq)
<!-- deq-highlight-begin: ../examples/compose/03_nested_compose.annotated.deq -->
```deq
@PTYPE(1)
CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1  # generator S0, destabilizer DS0=X1*X2
    STABILIZER Z1*Z2  # generator S1, destabilizer DS1=X0*X1
}

@GTYPE(1)
@CHECKS("manual", verify=0)
GADGET PrepareZ {
    R 0 1 2
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C0 OUT0.LX0  # E0
    ERROR(0.01) C0 C1 OUT0.LX0  # E1
    ERROR(0.01) C1 OUT0.LX0  # E2
    OUTPUT RepetitionCode 0 1 2
    CHECK OUT0.S0
    CHECK OUT0.S1
    PROPAGATE OUT0.LZ0 FROM
    PROPAGATE OUT0.LX0 FROM

    # --- statistics ---
    # finished checks: 0
    # unfinished checks: 2
    #   weight distribution: { 1:2 }
    # errors: 3
    #   check-weight distribution: { 1:2, 2:1 }
}

@GTYPE(2)
@CHECKS("manual", verify=0)
GADGET Idle {
    INPUT RepetitionCode 0 2 4
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 2 4
    ERROR(0.01) C0 OUT0.LX0  # E0
    ERROR(0.01) C0 C1 OUT0.LX0  # E1
    ERROR(0.01) C1 OUT0.LX0  # E2
    R 1 3
    CX 0 1 2 3
    CX 2 1 4 3
    @SIMULATE_ONLY
    M(0.01) 1 3
    @DECODE_ONLY
    M 1 3
    ERROR(0.01) C0 C2  # E3
    ERROR(0.01) C1 C3  # E4
    CHECK M0 IN0.S0
    CHECK M1 IN0.S1
    OUTPUT RepetitionCode 0 2 4
    CHECK OUT0.S0 M0
    CHECK OUT0.S1 M1
    PROPAGATE OUT0.LZ0 FROM IN0.LZ0
    PROPAGATE OUT0.LX0 FROM IN0.LX0

    # --- statistics ---
    # finished checks: 2
    #   weight distribution: { 2:2 }
    # unfinished checks: 2
    #   weight distribution: { 2:2 }
    # errors: 5
    #   check-weight distribution: { 1:2, 2:3 }
}

@GTYPE(3)
@CHECKS("manual", verify=0)
GADGET MeasureZ {
    INPUT RepetitionCode 0 1 2
    @SIMULATE_ONLY
    M(0.01) 0 1 2
    @DECODE_ONLY
    M 0 1 2
    ERROR(0.01) C0 R0  # E0
    ERROR(0.01) C0 C1 R0  # E1
    ERROR(0.01) C1 R0  # E2
    READOUT rec[-3] rec[-2] rec[-1]  # IN0.LX0
    CHECK M1 M0 IN0.S0
    CHECK M2 M1 IN0.S1

    # --- statistics ---
    # finished checks: 2
    #   weight distribution: { 3:2 }
    # unfinished checks: 0
    # errors: 3
    #   check-weight distribution: { 1:2, 2:1 }
}

@GTYPE(4)
@CHECKS("manual", verify=0)
GADGET Idle3 {
    INPUT RepetitionCode 0 1 2
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C0 OUT0.LX0  # E0
    ERROR(0.01) C0 C1 OUT0.LX0  # E1
    ERROR(0.01) C1 OUT0.LX0  # E2
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    M(0.01) 3 4
    @DECODE_ONLY
    M 3 4
    ERROR(0.01) C0 C2  # E3
    ERROR(0.01) C1 C3  # E4
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C2 OUT0.LX0  # E5
    ERROR(0.01) C2 C3 OUT0.LX0  # E6
    ERROR(0.01) C3 OUT0.LX0  # E7
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    M(0.01) 3 4
    @DECODE_ONLY
    M 3 4
    ERROR(0.01) C2 C4  # E8
    ERROR(0.01) C3 C5  # E9
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C4 OUT0.LX0  # E10
    ERROR(0.01) C4 C5 OUT0.LX0  # E11
    ERROR(0.01) C5 OUT0.LX0  # E12
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    M(0.01) 3 4
    @DECODE_ONLY
    M 3 4
    ERROR(0.01) C4 C6  # E13
    ERROR(0.01) C5 C7  # E14
    OUTPUT RepetitionCode 0 1 2
    CHECK IN0.S0 M0
    CHECK IN0.S1 M1
    CHECK M0 M2
    CHECK M1 M3
    CHECK M2 M4
    CHECK M3 M5
    CHECK M4 OUT0.S0
    CHECK M5 OUT0.S1
    PROPAGATE OUT0.LZ0 FROM IN0.LZ0
    PROPAGATE OUT0.LX0 FROM IN0.LX0

    # --- statistics ---
    # finished checks: 6
    #   weight distribution: { 2:6 }
    # unfinished checks: 2
    #   weight distribution: { 1:2 }
    # errors: 15
    #   check-weight distribution: { 1:6, 2:9 }
}

@GTYPE(5)
@CHECKS("manual", verify=0)
GADGET Idle4 {
    INPUT RepetitionCode 0 1 2
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C0 OUT0.LX0  # E0
    ERROR(0.01) C0 C1 OUT0.LX0  # E1
    ERROR(0.01) C1 OUT0.LX0  # E2
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    M(0.01) 3 4
    @DECODE_ONLY
    M 3 4
    ERROR(0.01) C0 C2  # E3
    ERROR(0.01) C1 C3  # E4
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C2 OUT0.LX0  # E5
    ERROR(0.01) C2 C3 OUT0.LX0  # E6
    ERROR(0.01) C3 OUT0.LX0  # E7
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    M(0.01) 3 4
    @DECODE_ONLY
    M 3 4
    ERROR(0.01) C2 C4  # E8
    ERROR(0.01) C3 C5  # E9
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C4 OUT0.LX0  # E10
    ERROR(0.01) C4 C5 OUT0.LX0  # E11
    ERROR(0.01) C5 OUT0.LX0  # E12
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    M(0.01) 3 4
    @DECODE_ONLY
    M 3 4
    ERROR(0.01) C4 C6  # E13
    ERROR(0.01) C5 C7  # E14
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C6 OUT0.LX0  # E15
    ERROR(0.01) C6 C7 OUT0.LX0  # E16
    ERROR(0.01) C7 OUT0.LX0  # E17
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    M(0.01) 3 4
    @DECODE_ONLY
    M 3 4
    ERROR(0.01) C6 C8  # E18
    ERROR(0.01) C7 C9  # E19
    OUTPUT RepetitionCode 0 1 2
    CHECK IN0.S0 M0
    CHECK IN0.S1 M1
    CHECK M0 M2
    CHECK M1 M3
    CHECK M2 M4
    CHECK M3 M5
    CHECK M4 M6
    CHECK M5 M7
    CHECK M6 OUT0.S0
    CHECK M7 OUT0.S1
    PROPAGATE OUT0.LZ0 FROM IN0.LZ0
    PROPAGATE OUT0.LX0 FROM IN0.LX0

    # --- statistics ---
    # finished checks: 8
    #   weight distribution: { 2:8 }
    # unfinished checks: 2
    #   weight distribution: { 1:2 }
    # errors: 20
    #   check-weight distribution: { 1:8, 2:12 }
}

PROGRAM Simulation {
    PrepareZ 0
    Idle4 0
    MeasureZ 0
    ASSERT_EQ rec[-1] 0
}
```
<!-- deq-highlight-end: ../examples/compose/03_nested_compose.annotated.deq -->

[Composed Idle4 gadget](../examples/compose/snippet_idle4_annotated.deq)
<!-- deq-highlight-begin: ../examples/compose/snippet_idle4_annotated.deq -->
```deq
@GTYPE(5)
@CHECKS("manual", verify=0)
GADGET Idle4 {
    INPUT RepetitionCode 0 1 2
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C0 OUT0.LX0  # E0
    ERROR(0.01) C0 C1 OUT0.LX0  # E1
    ERROR(0.01) C1 OUT0.LX0  # E2
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    M(0.01) 3 4
    @DECODE_ONLY
    M 3 4
    ERROR(0.01) C0 C2  # E3
    ERROR(0.01) C1 C3  # E4
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C2 OUT0.LX0  # E5
    ERROR(0.01) C2 C3 OUT0.LX0  # E6
    ERROR(0.01) C3 OUT0.LX0  # E7
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    M(0.01) 3 4
    @DECODE_ONLY
    M 3 4
    ERROR(0.01) C2 C4  # E8
    ERROR(0.01) C3 C5  # E9
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C4 OUT0.LX0  # E10
    ERROR(0.01) C4 C5 OUT0.LX0  # E11
    ERROR(0.01) C5 OUT0.LX0  # E12
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    M(0.01) 3 4
    @DECODE_ONLY
    M 3 4
    ERROR(0.01) C4 C6  # E13
    ERROR(0.01) C5 C7  # E14
    @SIMULATE_ONLY
    X_ERROR(0.01) 0 1 2
    ERROR(0.01) C6 OUT0.LX0  # E15
    ERROR(0.01) C6 C7 OUT0.LX0  # E16
    ERROR(0.01) C7 OUT0.LX0  # E17
    R 3 4
    CX 0 3 1 4
    CX 1 3 2 4
    @SIMULATE_ONLY
    M(0.01) 3 4
    @DECODE_ONLY
    M 3 4
    ERROR(0.01) C6 C8  # E18
    ERROR(0.01) C7 C9  # E19
    OUTPUT RepetitionCode 0 1 2
    CHECK IN0.S0 M0
    CHECK IN0.S1 M1
    CHECK M0 M2
    CHECK M1 M3
    CHECK M2 M4
    CHECK M3 M5
    CHECK M4 M6
    CHECK M5 M7
    CHECK M6 OUT0.S0
    CHECK M7 OUT0.S1
    PROPAGATE OUT0.LZ0 FROM IN0.LZ0
    PROPAGATE OUT0.LX0 FROM IN0.LX0

    # --- statistics ---
    # finished checks: 8
    #   weight distribution: { 2:8 }
    # unfinished checks: 2
    #   weight distribution: { 1:2 }
    # errors: 20
    #   check-weight distribution: { 1:8, 2:12 }
}
```
<!-- deq-highlight-end: ../examples/compose/snippet_idle4_annotated.deq -->

Every check still spans exactly 2 adjacent measurements — the hierarchical composition
preserves the per-round-pair structure at every level.

---

## How It Works

The COMPOSE mechanism performs three key operations for each sub-gadget application:

1. **Measurement offsetting**: Each sub-gadget's measurements are assigned global indices
   in sequence. Idle instance 0 gets measurements 0–1, instance 1 gets 2–3, instance 2
   gets 4–5. This prevents index collisions and preserves round identity.

2. **Matrix chaining**: The sub-gadget's `correction_propagation` matrix is multiplied
   with the accumulated state to propagate observable corrections through the chain. This
   is how the Pauli frame tracks logical observable transformations.

3. **Check accumulation**: Each sub-gadget's finished checks are added to a global list
   with their measurement indices offset. Errors reference these global check indices.
   Because each sub-gadget's errors only reference its own checks, the resulting error
   model is inherently structured.

The result is a single `JitGadgetType` — indistinguishable from a hand-written gadget in
the `.deq.jit` format. The decoder sees it as one gadget with well-structured checks.

---

## COMPOSE vs PROGRAM

Both `COMPOSE` and `PROGRAM` can chain gadgets, but they serve different purposes:

| Aspect                | `COMPOSE`                                  | `PROGRAM`                              |
| --------------------- | ------------------------------------------ | -------------------------------------- |
| **Output**            | One `JitGadgetType` (flattened)            | Separate `JitInstruction`s (preserved) |
| **When processed**    | Transpile time                             | Runtime (JIT compiler)                 |
| **Checks**            | Merged into one check model                | Separate per-gadget check models       |
| **Windowed decoding** | Not applicable (single gadget)             | Enables parallel windowed decoding     |
| **Use case**          | Structuring checks for a logical operation | Defining the execution plan            |

**Rule of thumb**: Use `COMPOSE` to build structured multi-round gadgets (e.g., fault-tolerant
operations). Use `PROGRAM` to define the sequence of these composed gadgets for decoding.

A typical pattern:

[COMPOSE vs PROGRAM pattern](../examples/compose/compose_vs_program.deq)
<!-- deq-highlight-begin: ../examples/compose/compose_vs_program.deq -->
```deq
COMPOSE FTIdle {
    INPUT Code 0
    REPEAT 10 { SyndromeExtraction 0 }
    OUTPUT Code 0
}

PROGRAM MemoryExperiment {
    PrepareZ OUT(0)
    FTIdle 0
    FTIdle 0
    MeasureZ IN(0)
    ASSERT_EQ rec[-1] 0
}
```
<!-- deq-highlight-end: ../examples/compose/compose_vs_program.deq -->

The `FTIdle` gadget has well-structured checks (thanks to COMPOSE). The `PROGRAM` keeps
each `FTIdle` instance separate, enabling the windowed decoder to process them in parallel.

---

## Summary

| Concept                | Purpose                                                           |
| ---------------------- | ----------------------------------------------------------------- |
| `COMPOSE Name { ... }` | Chain sub-gadgets into a single gadget with structured checks     |
| `REPEAT N { ... }`     | Unroll N copies inside COMPOSE (syntactic sugar)                  |
| `Idle 0`               | Apply sub-gadget on code block 0                                  |
| Nested COMPOSE         | Use a COMPOSE gadget inside another COMPOSE                       |
| Error locality         | COMPOSE preserves per-sub-gadget error locality by construction   |
| vs PROGRAM             | COMPOSE flattens at transpile time; PROGRAM preserves for runtime |
