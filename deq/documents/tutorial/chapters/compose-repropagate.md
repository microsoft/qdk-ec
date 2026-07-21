# Logical Teleportation in COMPOSE: the `@REPROPAGATE` Decorator

The [COMPOSE chapter](compose-gadgets.md) showed that chaining sub-gadgets through
`COMPOSE` keeps each sub-gadget's checks and errors **local** by construction. The
mechanism that makes that locality work is **the JIT compiler** (the same Rust pipeline
that the runtime decoder uses): it assembles the composed gadget from individual
sub-gadget pieces and inherits each sub-gadget's local check structure verbatim. It is
not tied to any particular way of computing the *propagation matrices* that go
alongside the checks.

By default the COMPOSE pipeline computes those propagation matrices by **matrix
composition** of the sub-gadgets' individual propagation matrices. That is the natural
choice because it mirrors what happens at runtime: the runtime decoder chains the
same matrices step by step as instances of these gadgets stream in. But matrix
composition has one blind spot: it cannot invent classical feed-forward. If the
intended logical channel depends on a mid-circuit measurement outcome being XOR'd
back into the output Pauli frame, no chain of sub-gadget matrices — none of which
individually sees both the measurement *and* the output — can ever record that
dependence. Matrix composition happily returns *a* propagation row for the affected
output logical, but the row is **missing the classical correction**, and the composed
gadget silently implements the wrong channel.

The textbook example is **logical teleportation**: the input state is recovered on a
different code block only after a classical-feed-forward Pauli correction conditioned
on a mid-circuit measurement. Without an explicit `CONDITIONAL` in the COMPOSE body
or an `@REPROPAGATE` decorator that re-derives propagation from the flat inlined
circuit, matrix composition drops that correction and the resulting binary is *not*
the intended logical identity.

Crucially, `deq annotate` does not fail on the broken COMPOSE.
The bug is only visible if you **read the emitted `PROPAGATE` rows**. An empty
right-hand side on an output-logical row that should preserve its input observable is
the diagnostic. This chapter walks through that pattern: the plain-COMPOSE
teleportation, what its emitted `PROPAGATE` reveals, and how `@REPROPAGATE` (or an
explicit `CONDITIONAL`) restores the missing dependence.

---

## A logical teleportation COMPOSE

A [[4,1,2]] code block can be initialised in $|0\rangle_L$ by `PrepareZero`
(initialise the data qubits in $|0\rangle$, then measure the code stabilizer
$X_0 X_1 X_2 X_3$).  Composing that with a transversal `CNOT` and an `X`-basis
measurement of the first block implements logical teleportation from port 0 to
port 1:

[Teleportation COMPOSE — without `@REPROPAGATE`](../examples/compose-repropagate/01_teleport_logical.deq)
<!-- deq-highlight-begin: ../examples/compose-repropagate/01_teleport_logical.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># Logical teleportation, attempted with the default COMPOSE build path.</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000"># ***This file is a NEGATIVE example.***  It compiles and annotates</span></span>
<span class="line"><span style="color:#008000"># without error, but the resulting composed gadget is not actually a</span></span>
<span class="line"><span style="color:#008000"># logical identity from port 0 to port 1: matrix composition drops the</span></span>
<span class="line"><span style="color:#008000"># classical feed-forward that teleportation requires.  Compare the</span></span>
<span class="line"><span style="color:#008000"># `PROPAGATE OUT0.LZ0 FROM` (empty) row emitted for `Teleport` in</span></span>
<span class="line"><span style="color:#008000"># `01_teleport_logical.annotated.deq` with the informative</span></span>
<span class="line"><span style="color:#008000"># `PROPAGATE OUT0.LZ0 FROM IN0.LZ0 M1 M3` row emitted for the</span></span>
<span class="line"><span style="color:#008000"># `@REPROPAGATE` variant in `02_teleport_repropagate.annotated.deq`.</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000"># Code layout:  4 physical qubits per logical qubit.</span></span>
<span class="line"><span style="color:#008000">#     0   1</span></span>
<span class="line"><span style="color:#008000">#   Z   X   Z</span></span>
<span class="line"><span style="color:#008000">#     2   3</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> Code</span><span style="color:#000000"> [[</span><span style="color:#098658">4</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">2</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z3</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#000000">*</span><span style="color:#0000FF">X3</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZero</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#000000">*</span><span style="color:#0000FF">X3</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> CNOT</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 4</span><span style="color:#098658"> 1</span><span style="color:#098658"> 5</span><span style="color:#098658"> 2</span><span style="color:#098658"> 6</span><span style="color:#098658"> 3</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureX</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> M0</span><span style="color:#001080"> M2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000"># Logical teleportation: |psi> in port 0, |0_L> prepared on port 1,</span></span>
<span class="line"><span style="color:#008000"># transversal CNOT, measure X on port 0 -> the input logical state</span></span>
<span class="line"><span style="color:#008000"># should end up on port 1 (possibly up to a conditional logical Z).</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000"># Without @REPROPAGATE (or an explicit CONDITIONAL), the COMPOSE</span></span>
<span class="line"><span style="color:#008000"># pipeline composes the propagation matrices of the sub-gadgets.</span></span>
<span class="line"><span style="color:#008000"># Matrix composition cannot invent classical feed-forward, so the</span></span>
<span class="line"><span style="color:#008000"># composed row for `OUT0.LZ0` comes out empty: no input logical</span></span>
<span class="line"><span style="color:#008000"># operator (and no measurement bit) propagates to the output LZ.</span></span>
<span class="line"><span style="color:#008000"># Since the LZ operator is what flips the X observable, the input's</span></span>
<span class="line"><span style="color:#008000"># X observable correction is discarded rather than teleported.  The `LX`</span></span>
<span class="line"><span style="color:#008000"># operator still propagates cleanly (input LX -> output LX, both</span></span>
<span class="line"><span style="color:#008000"># flip the Z observable), so the Z observable correction does survive — but</span></span>
<span class="line"><span style="color:#008000"># a gadget that only teleports one basis is not the identity.</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000"># See 02_teleport_repropagate.deq for the @REPROPAGATE fix.</span></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> Teleport</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    PrepareZero</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    CNOT</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    MeasureX</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZero</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Teleport</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureX</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose-repropagate/01_teleport_logical.deq -->

The COMPOSE block on its own is what we care about:

[Teleport COMPOSE block](../examples/compose-repropagate/snippet_teleport_compose.deq)
<!-- deq-highlight-begin: ../examples/compose-repropagate/snippet_teleport_compose.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> Teleport</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    PrepareZero</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    CNOT</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    MeasureX</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose-repropagate/snippet_teleport_compose.deq -->

| Step          | Effect                                                              |
| ------------- | ------------------------------------------------------------------- |
| `INPUT Code 0`   | Logical state $|\psi\rangle_L$ arrives on code block 0           |
| `PrepareZero 1`  | Prepare $|+\rangle_L$ on code block 1                            |
| `CNOT 0 1`       | Transversal CNOT — entangles the two blocks                      |
| `MeasureX 0`     | Measure code block 0 in the $X$ basis (Bell-style projection)    |
| `OUTPUT Code 1`  | Logical output is now on code block 1                            |

Mathematically this implements $\bar{I}$ from port 0 to port 1, but the equality is
*conditional*: depending on the parity of the `MeasureX 0` outcome, the output state on
port 1 may differ from the input by a logical $\bar{Z}$. In a real quantum circuit you
either apply a corrective $\bar{Z}$ classically or absorb it into the Pauli frame. Either
way, the relationship between input and output observables is not pure matrix
composition — it depends on a measurement outcome that only the composed circuit, not
any individual sub-gadget, has access to. Matrix composition of the three sub-gadgets'
propagation matrices can still produce *a* propagation row (the runtime would compute
it the same way), but the resulting row no longer matches what static analysis of the
inlined flat circuit would derive. The next section shows exactly that mismatch.

---

## What goes wrong

Run the annotator on this file:

```sh
deq annotate 01_teleport_logical.deq
```

There is no error. The command silently writes
`01_teleport_logical.annotated.deq`, and re-transpilation confirms round-trip
equivalence. But the annotated output for the composed `Teleport` gadget contains
the diagnostic:

[Annotated Teleport GADGET — plain COMPOSE](../examples/compose-repropagate/snippet_teleport_plain_annotated.deq)
<!-- deq-highlight-begin: ../examples/compose-repropagate/snippet_teleport_plain_annotated.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">4</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Teleport</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X5</span><span style="color:#000000">*</span><span style="color:#0000FF">X6</span><span style="color:#000000">*</span><span style="color:#0000FF">X7</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 4</span><span style="color:#098658"> 1</span><span style="color:#098658"> 5</span><span style="color:#098658"> 2</span><span style="color:#098658"> 6</span><span style="color:#098658"> 3</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S2</span><span style="color:#001080"> M0</span><span style="color:#001080"> M1</span><span style="color:#001080"> M2</span><span style="color:#001080"> M3</span><span style="color:#001080"> M4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S0</span><span style="color:#267F99"> OUT0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> IN0.S1</span><span style="color:#267F99"> OUT0.S1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span><span style="color:#267F99"> OUT0.S2</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> M1</span><span style="color:#001080"> M3</span><span style="color:#008000">  # IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 1</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 6:1 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 3</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 1:3 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose-repropagate/snippet_teleport_plain_annotated.deq -->

Look at the two `PROPAGATE` rows.  A `PROPAGATE` row traces the *forward
Heisenberg propagation of an input logical Pauli operator through the gadget*.
`PROPAGATE OUT0.LX0 FROM IN0.LX0` says the input logical $\bar{X}$ operator
propagates through the gadget to reappear as the output logical $\bar{X}$
operator (both are the operator that flips their frame's $\bar{Z}$ observable) —
so the $\bar{Z}$ observable on port 1 tracks the input's $\bar{Z}$ observable on
port 0 and that half of the teleport works.

But **`PROPAGATE OUT0.LZ0 FROM` has an empty right-hand side**: no input operator
(and no XOR with any mid-circuit measurement bit) propagates to the output logical
$\bar{Z}$ operator. Because $\bar{Z}$ is the operator that flips the frame's
$\bar{X}$ observable, the runtime has no expression for the output $\bar{X}$
observable in terms of the input — the input's $\bar{X}$ correction is discarded
rather than teleported.

To confirm, look at the compiled `correction_propagation` (cp) and
`physical_correction` (pc) matrix rows for `OUT0.LZ0` in the two variants:

| Variant                                 | cp row (input logical operators) | pc row (mid-circuit measurements) |
| --------------------------------------- | -------------------------------- | --------------------------------- |
| Plain `COMPOSE` (this file)             | `{}` (empty)                     | `{}` (empty)                      |
| `@REPROPAGATE COMPOSE` (see next file)  | `{IN0.LZ0}`                      | `{M1, M3}`                        |

The `@REPROPAGATE` version records the correct propagation
`OUT0.LZ0 = IN0.LZ0 ⊕ M1 ⊕ M3`: the input $\bar{Z}$ operator propagates to the
output $\bar{Z}$ operator, up to a classical XOR with the parity of two
mid-circuit measurements — exactly the feed-forward correction that teleportation
requires. The plain version records nothing on either side.

Why does matrix composition produce the empty row? Trace the sub-gadgets:

* `PrepareZero 1` initialises port 1 in $|0\rangle_L$ (a +1 eigenstate of $\bar{Z}$).
  It has no INPUT ports, so its propagation matrix has *no* $\bar{Z}$-operator
  source at all — nothing to feed into an output $\bar{Z}$ column.
* `CNOT 0 1` propagates $\bar{Z}_1 \to \bar{Z}_1$ (Z on target stays on target),
  so any $\bar{Z}$ operator at port 1's output can only come from port 1's *input*
  $\bar{Z}$ operator — which `PrepareZero` never supplied.
* `MeasureX 0` reads out port 0 but has no output port, so nothing propagates
  through it either.

Nowhere in this chain does port 0's input $\bar{Z}$ operator meet port 1's output
$\bar{Z}$ operator except via the mid-circuit measurement outcome — and that
meeting is exactly the classical-feed-forward step that matrix composition cannot
invent.

**The reader's tool for spotting this bug is the annotated `PROPAGATE` row.**
Whenever your COMPOSE is supposed to preserve some input logical operator on some
output port, the emitted `PROPAGATE OUT<p>.L<P>` should list that input operator
on its right-hand side (possibly XOR'd with some `M<i>` bits for the classical
correction). An empty right-hand side on a row whose input operator should have
propagated forward means matrix composition has quietly dropped a classical
correction.

Two ways to add the correction back:

1. `@REPROPAGATE` — swap the propagation strategy to circuit-flow analysis on the
   flat inlined body. The next section shows this in full.
2. Write an explicit `CONDITIONAL rec[-k] <pauli> <wire>` inside the COMPOSE body.
   The canonicalizer's merge pass (step 9) folds that CONDITIONAL into
   cp/pc, producing the same binary as `@REPROPAGATE`. Concretely, replacing the
   plain COMPOSE with

   ```text
   COMPOSE Teleport {
       INPUT Code 0
       PrepareZero 1
       CNOT 0 1
       MeasureX 0
       CONDITIONAL rec[-1] Z0 1
       OUTPUT Code 1
   }
   ```

   makes the emitted `PROPAGATE OUT0.LZ0` show `FROM IN0.LZ0 M1 M3` too. The
   trade-off: `CONDITIONAL` requires you to name every classical correction
   yourself; `@REPROPAGATE` derives them from the flat circuit automatically.

---

## The fix: `@REPROPAGATE`

The corrected file adds a single decorator line on top of the COMPOSE block:

[Teleport COMPOSE block — with `@REPROPAGATE`](../examples/compose-repropagate/snippet_teleport_compose_repropagate.deq)
<!-- deq-highlight-begin: ../examples/compose-repropagate/snippet_teleport_compose_repropagate.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@REPROPAGATE</span></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> Teleport</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    PrepareZero</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    CNOT</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    MeasureX</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose-repropagate/snippet_teleport_compose_repropagate.deq -->

Full file:

[Teleportation COMPOSE — with `@REPROPAGATE`](../examples/compose-repropagate/02_teleport_repropagate.deq)
<!-- deq-highlight-begin: ../examples/compose-repropagate/02_teleport_repropagate.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># Logical teleportation realised with a COMPOSE block — fixed version.</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000"># Identical to 01_teleport_logical.deq except that ``@REPROPAGATE`` is</span></span>
<span class="line"><span style="color:#008000"># attached to the COMPOSE block.  The decorator tells the transpiler to</span></span>
<span class="line"><span style="color:#008000"># recompute the propagation matrices from circuit flow on the inlined</span></span>
<span class="line"><span style="color:#008000"># body so the conditional logical Pauli that teleportation implies can</span></span>
<span class="line"><span style="color:#008000"># be derived automatically.  ``deq annotate`` then verifies cleanly.</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000"># Code layout:  4 physical qubits per logical qubit.</span></span>
<span class="line"><span style="color:#008000">#     0   1</span></span>
<span class="line"><span style="color:#008000">#   Z   X   Z</span></span>
<span class="line"><span style="color:#008000">#     2   3</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> Code</span><span style="color:#000000"> [[</span><span style="color:#098658">4</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">2</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z3</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#000000">*</span><span style="color:#0000FF">X3</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZero</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#000000">*</span><span style="color:#0000FF">X3</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> CNOT</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 4</span><span style="color:#098658"> 1</span><span style="color:#098658"> 5</span><span style="color:#098658"> 2</span><span style="color:#098658"> 6</span><span style="color:#098658"> 3</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureX</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> M0</span><span style="color:#001080"> M2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">@REPROPAGATE</span></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> Teleport</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    PrepareZero</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    CNOT</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    MeasureX</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Simulation</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZero</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Teleport</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureX</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose-repropagate/02_teleport_repropagate.deq -->

`deq annotate` now succeeds:

```sh
deq annotate 02_teleport_repropagate.deq
# Wrote 02_teleport_repropagate.annotated.deq
# Verifying annotated output is equivalent to original (pass --no-verify to skip)...
# Verification passed.
```

The annotated COMPOSE renders as a flat `GADGET Teleport` block:

[Annotated Teleport GADGET](../examples/compose-repropagate/snippet_teleport_annotated.deq)
<!-- deq-highlight-begin: ../examples/compose-repropagate/snippet_teleport_annotated.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">4</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Teleport</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X5</span><span style="color:#000000">*</span><span style="color:#0000FF">X6</span><span style="color:#000000">*</span><span style="color:#0000FF">X7</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 4</span><span style="color:#098658"> 1</span><span style="color:#098658"> 5</span><span style="color:#098658"> 2</span><span style="color:#098658"> 6</span><span style="color:#098658"> 3</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-4]</span><span style="color:#001080"> rec[-2]</span><span style="color:#008000">  # IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M4</span><span style="color:#001080"> M3</span><span style="color:#001080"> M2</span><span style="color:#001080"> M1</span><span style="color:#001080"> M0</span><span style="color:#267F99"> IN0.S2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Code</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S0</span><span style="color:#267F99"> IN0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S1</span><span style="color:#267F99"> IN0.S1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S2</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span><span style="color:#001080"> M1</span><span style="color:#001080"> M3</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 1</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 6:1 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 3</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:3 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/compose-repropagate/snippet_teleport_annotated.deq -->

The decisive line is

```text
PROPAGATE OUT0.LZ0 FROM IN0.LZ0 M1 M3
```

The `IN0.LZ0` token is what was missing from the plain-COMPOSE emission — the
row now says the input logical $\bar{Z}$ operator *does* propagate forward to
the output logical $\bar{Z}$ operator, so the input state is preserved rather
than discarded. The trailing `M1 M3` are internal-measurement references that
encode the conditional correction: when the parity of those two measurements is
$1$, the propagated output $\bar{Z}$ operator is flipped in the Pauli frame.
`@REPROPAGATE` derives all three tokens directly from the inlined circuit (it
can see the `MX 0 1 2 3` and trace the resulting Pauli frame forwards) — the
same derivation the verifier would also run, so the emitted `PROPAGATE` rows
come out as the natural-Heisenberg form of the composed body.

---

## What `@REPROPAGATE` keeps vs. changes

A common worry: "if `@REPROPAGATE` recompiles from a flat circuit, do I lose the
structural benefits that made me choose `COMPOSE` in the first place?" No. The check
structure is produced by the JIT compiler regardless of which propagation strategy is
in use; `@REPROPAGATE` only swaps the propagation-derivation strategy on the *side*,
from matrix composition to circuit-flow analysis. The local check structure — the
exact reason to use `COMPOSE` over a flat GADGET, as the
[COMPOSE chapter](compose-gadgets.md) explains in detail — is preserved verbatim.

| Aspect                                             | Plain `COMPOSE`                       | `@REPROPAGATE COMPOSE`                  |
| -------------------------------------------------- | ------------------------------------- | --------------------------------------- |
| Finished / unfinished `CHECK`s                     | From the JIT compiler                 | Same — from the JIT compiler           |
| Measurements, readouts, input/output ports         | From the JIT compiler                 | Same — from the JIT compiler           |
| `correction_propagation`, `physical_correction`    | Matrix-composed from sub-gadgets      | **Recomputed from inlined circuit flow** |
| `ERROR(p) ...` rows derived from noise             | From the JIT compiler                 | **Recomputed against the new propagation** |

Only the bottom two rows change. The JIT compiler's check structure encodes the
sub-gadget composition — e.g., for multi-round syndrome extraction it produces the
weight-2 round-to-round comparison checks decoders rely on, not those non-local
checks. `@REPROPAGATE` keeps those checks verbatim and only patches the
propagation/error side, which is the side that could not handle the conditional Pauli.

---

## When to reach for it

Reach for `@REPROPAGATE` whenever a COMPOSE block implements a logical operation
that **depends on a measurement outcome via classical feed-forward**, including:

- logical teleportation (the example above);
- lattice surgery with conditional logical Pauli corrections;
- any other pattern where the input→output Pauli flow has a row that is only
  determined after looking at internal measurement outcomes.

Because `deq annotate` does **not** raise an error when the classical correction is
missing, the reliable diagnostic recipe is to inspect the emitted `PROPAGATE` rows:

1. Write the `COMPOSE` block first, **without** `@REPROPAGATE`.
2. Run `deq annotate` and open the resulting `.annotated.deq`.
3. Locate the `GADGET <ComposeName>` block. For every output logical operator
   your COMPOSE is supposed to preserve, check that the corresponding
   `PROPAGATE OUT<p>.L<P>` line has the matching input operator on its
   right-hand side. Also check that any classical corrections you expect are
   reflected as `M<i>` tokens.
4. If a state-preserving row has an empty (or otherwise unexpected) right-hand
   side, matrix composition has dropped a classical correction. Add
   `@REPROPAGATE` to the COMPOSE, or write the correction explicitly as a
   compose-level `CONDITIONAL rec[-k] <pauli> <wire>` statement. Either choice
   folds the correction back into cp/pc; the two produce canonically equivalent
   binaries.

---

## Summary

| Concept                            | Purpose                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| Check locality in `COMPOSE`        | Comes from the **JIT compiler**, independent of how propagation matrices are derived          |
| Default propagation strategy       | Matrix composition of sub-gadget propagation matrices (mirrors runtime composition)           |
| Matrix composition's blind spot    | Cannot invent classical feed-forward — a missing correction shows up as an empty (or unexpected) `PROPAGATE` row in the annotated GADGET |
| `@REPROPAGATE COMPOSE Name { ... }` | Swap the propagation strategy to circuit-flow analysis on the inlined body                   |
| Compose-level `CONDITIONAL rec[-k] <pauli> <wire>` | Alternative fix: name the correction explicitly; canonicalizer folds it into cp/pc  |
| What changes with `@REPROPAGATE`   | Only `correction_propagation`, `physical_correction`, and the noise-derived `ERROR` rows      |
| What stays the same                | Checks, measurements, readouts, ports — all still produced by the JIT compiler               |
| How to diagnose a broken COMPOSE   | Read the emitted `PROPAGATE OUT<p>.L<P>` rows; if an input logical operator that should propagate forward is missing from the corresponding right-hand side, matrix composition dropped a classical correction |
