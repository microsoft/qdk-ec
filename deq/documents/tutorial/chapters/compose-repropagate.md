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
composition is a *convenient default*, not a fundamental property of COMPOSE. As soon
as matrix composition produces a row that the static verifier cannot reproduce on the
inlined flat circuit, we need a different way to fill in that row — *without* giving up
the JIT compiler's check structure.

The textbook example where this happens is **logical teleportation**: the input state
is recovered on a different code block only after a classical-feed-forward Pauli
correction conditioned on a mid-circuit measurement. The correction lives at the
*global* circuit level — no individual sub-gadget can see it, so matrix composition
produces a propagation row that flat-circuit analysis cannot derive on its own.

The `@REPROPAGATE` decorator is the fix. It swaps just the propagation-derivation
strategy from matrix composition to circuit-flow analysis on the inlined body, while
leaving the JIT compiler's check structure untouched. This chapter shows what goes
wrong without it, why, and exactly which pieces the decorator changes.

---

## A logical teleportation COMPOSE

A [[4,1,2]] code block can be initialised in $|+\rangle_L$ by `PrepareZero` (initialise
the data qubits in $|0\rangle$, then measure $X_0 X_1 X_2 X_3$). Composing that with a
transversal `CNOT` and an `X`-basis measurement of the first block implements logical
teleportation from port 0 to port 1:

[Teleportation COMPOSE — without `@REPROPAGATE`](../examples/compose-repropagate/01_teleport_logical.deq)
<!-- deq-highlight-begin: ../examples/compose-repropagate/01_teleport_logical.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># Logical teleportation realised with a COMPOSE block.</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000"># This file is the *negative* example: the COMPOSE has no @REPROPAGATE</span></span>
<span class="line"><span style="color:#008000"># decorator, so `deq annotate` will fail at the verification step.  See</span></span>
<span class="line"><span style="color:#008000"># 02_teleport_repropagate.deq for the working version.</span></span>
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
<span class="line"><span style="color:#008000"># Logical teleportation: |psi> in port 0, |+_L> prepared on port 1,</span></span>
<span class="line"><span style="color:#008000"># transversal CNOT, measure X on port 0 -> the input logical state ends</span></span>
<span class="line"><span style="color:#008000"># up on port 1 (possibly up to a conditional logical Z).</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000"># Without @REPROPAGATE, the COMPOSE pipeline composes the</span></span>
<span class="line"><span style="color:#008000"># *propagation matrices* of PrepareZero, CNOT and MeasureX, which</span></span>
<span class="line"><span style="color:#008000"># cannot represent the conditional logical Pauli correction that</span></span>
<span class="line"><span style="color:#008000"># teleportation implicitly requires.  `deq annotate` therefore</span></span>
<span class="line"><span style="color:#008000"># rejects the rendered PROPAGATE statements during verification.</span></span>
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

## What goes wrong without `@REPROPAGATE`

Run the annotator on this file:

```sh
deq annotate 01_teleport_logical.deq
```

After writing the annotated output, `deq annotate` re-transpiles it to verify
round-trip equivalence — and that verification fails:

```text
ValueError: in GADGET 'Teleport': PROPAGATE for output row 0 (OUT0.LZ0) does not lie in the basis-freedom span of that row; the spec differs from the canonical flow-derived value by 3 bit(s) that cannot be expressed as any XOR of input-stabilizers, output-stabilizer joint rows, or finished-check parities.
  Hint: if 'Teleport' was generated by 'deq annotate' from a COMPOSE block, add the @REPROPAGATE decorator to that COMPOSE.  @REPROPAGATE switches the COMPOSE build to the flat-circuit pipeline so its propagation matrices come from actual circuit flow on the inlined body, not from sub-gadget matrix composition.
```

(The exact text is captured into
[`01_teleport_annotate_error.txt`](../examples/compose-repropagate/01_teleport_annotate_error.txt)
by the chapter's generator script, so the build catches any drift.)

The failing check is the `PROPAGATE` statement for `OUT0.LZ0` (the logical $\bar{Z}$
column of port 0's output frame). At COMPOSE build time the JIT compiler chained the
three sub-gadgets' propagation matrices and produced a `PROPAGATE OUT0.LZ0 FROM ...`
row whose right-hand side includes contributions from internal measurements — a faithful
representation of the conditional correction. When `deq annotate` rewrites the COMPOSE
as a flat `GADGET` and the verifier re-transpiles it, the only information the verifier
has is the inlined circuit; it cannot recover the matrix-composed row from circuit flow
alone, and reports that 3 bits of the spec "cannot be expressed as any XOR of
input-stabilizers, output-stabilizer joint rows, or finished-check parities".

In other words: matrix composition and circuit-flow analysis are two *different* ways
of producing a propagation matrix. They agree on most COMPOSEs — which is why the
default matrix-composition path works almost everywhere — but for teleportation-style
operations the two strategies produce rows that the verifier knows are equivalent only
if you can already see the underlying measurement-conditioned Pauli, and the
flat-circuit pipeline cannot.

The hint at the bottom of the error message points at the fix: add `@REPROPAGATE` to
the COMPOSE.

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

The trailing `M1 M3` are internal-measurement references that encode the conditional
logical $\bar{Z}$: when the parity of those two measurements is `1`, the output frame's
$\bar{Z}$ column is flipped. `@REPROPAGATE` derives this directly from the *inlined*
circuit (it can see the `MX 0 1 2 3` and trace the resulting Pauli frame forwards),
which is exactly the same derivation the verifier runs — so build and verifier now
agree.

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
weight-2 round-to-round comparison checks decoders rely on, not weight-1 single-shot
checks. `@REPROPAGATE` keeps those checks verbatim and only patches the
propagation/error side, which is the side that could not handle the conditional Pauli.

---

## When to reach for it

Use `@REPROPAGATE` whenever a COMPOSE block implements a logical operation that
**depends on a measurement outcome via classical feed-forward**, including:

- logical teleportation (the example above);
- gate teleportation of Clifford or non-Clifford gates;
- lattice surgery with conditional logical Pauli corrections;
- magic-state injection followed by a conditional Clifford fix-up;
- any other pattern where the input→output Pauli flow has a row that is only
  determined after looking at internal measurement outcomes.

A reliable diagnostic recipe:

1. Write the `COMPOSE` block first, **without** `@REPROPAGATE`.
2. Run `deq annotate`. If verification passes, the default matrix-composition strategy
   was sufficient for this COMPOSE — you are done.
3. If verification fails with
   ```
   PROPAGATE for output row ... does not lie in the basis-freedom span
   ```
   add `@REPROPAGATE` to the COMPOSE. The error message itself names the decorator.

---

## Summary

| Concept                            | Purpose                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| Check locality in `COMPOSE`        | Comes from the **JIT compiler**, independent of how propagation matrices are derived          |
| Default propagation strategy       | Matrix composition of sub-gadget propagation matrices (mirrors runtime composition)           |
| `@REPROPAGATE COMPOSE Name { ... }` | Swap the propagation strategy to circuit-flow analysis on the inlined body                   |
| What changes                       | Only `correction_propagation`, `physical_correction`, and the noise-derived `ERROR` rows      |
| What stays the same                | Checks, measurements, readouts, ports — all still produced by the JIT compiler               |
| When you need it                   | Logical operations whose Pauli flow depends on classical feed-forward (e.g. teleportation)    |
| How to diagnose                    | If `deq annotate` rejects a PROPAGATE row's basis-freedom span, add `@REPROPAGATE`            |
