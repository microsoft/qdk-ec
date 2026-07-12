# Conditional Pauli Corrections: the `CONDITIONAL` Statement

The [`@REPROPAGATE` chapter](compose-repropagate.md) showed one way to handle a
`COMPOSE` block whose logical output depends on a mid-circuit measurement: switch the
propagation strategy from matrix composition of sub-gadgets to circuit-flow analysis
on the inlined body, so the natural Heisenberg propagation through the inlined
circuit derives the measurement-conditioned Pauli frame for free.

`CONDITIONAL` is the second way: instead of relying on circuit flow to derive the
correction, you **write it down as a logical-level statement** inside the `COMPOSE`
(or `PROGRAM`) body:

```
CONDITIONAL rec[-2] Z0 2   # apply logical Z on wire 2 if logical readout rec[-2] = 1
```

The transpiler injects a synthesized empty identity gadget that carries the
correction, and the merge() canonicalizer folds the readout's measurement set into
the affected output observable's measurement deps — giving the same final
`correction_propagation` / `physical_correction` matrices as the `@REPROPAGATE`
pathway.

This chapter walks through Bell-pair logical teleportation — the canonical
measurement-based logical operation — and shows three equivalent ways to express
its Pauli frame correction: `@REPROPAGATE`, COMPOSE-level `CONDITIONAL`, and
PROGRAM-level `CONDITIONAL`.  We then quantify each variant against physical noise
to show that the choice is purely about *how the correction is expressed in source*;
the runtime behavior is identical.

---

## Bell-pair logical teleportation in one figure

Three distance-3 rotated surface-code patches — the input patch carrying
$|\psi\rangle_L$ on wire 0 plus two ancillary patches on wires 1 and 2 — interact
as follows:

![Stim timeline of the Bell-pair teleportation: Bell prep on q1/q2, transversal CNOT and H on q0/q1, two measurements, then `X rec[1]` and `Z rec[0]` feedforward on q2](../examples/conditional-correction/teleport_timeline.png)

Each `q0` / `q1` / `q2` line in the diagram stands in for a full 9-qubit
distance-3 surface-code patch; the diagram is rendered at the logical-qubit
level so the Bell-pair preparation, Bell-basis measurement, and feedforward
corrections dominate the view rather than the syndrome-extraction noise inside
each patch.

In words:

1. The input patch `|ψ⟩_A` is on wire 0.
2. `PrepareBell 1 2` puts wires 1 and 2 into the logical Bell state
   $|\Phi^+\rangle_L = (|0_L 0_L\rangle + |1_L 1_L\rangle)/\sqrt{2}$ by preparing wire 1 in
   $|+_L\rangle$, wire 2 in $|0_L\rangle$, and applying a transversal logical CNOT
   from wire 1 to wire 2.
3. `MeasureBell 0 1` destructively measures wires 0 and 1 in the logical Bell basis
   by applying another transversal CNOT and then measuring wire 0 in X (giving
   $m_{XX} = \langle \bar X_0 \bar X_1\rangle$) and wire 1 in Z (giving
   $m_{ZZ} = \langle \bar Z_0 \bar Z_1\rangle$).
4. The post-measurement state on wire 2 is $X^{m_{ZZ}} Z^{m_{XX}} |\psi\rangle_L$ —
   the input state up to a Pauli frame that depends on the random Bell-measurement
   outcomes.

Step 4 is the measurement-conditioned Pauli frame: we need to apply $Z$ on wire 2
if $m_{XX} = 1$ and $X$ on wire 2 if $m_{ZZ} = 1$ for the gadget to act as logical
identity (i.e. wire 2 carries the same logical state that wire 0 had on input).

The Bell-pair building blocks are reusable, so we factor them out into the chapter's
shared library:

[`PrepareBell` and `MeasureBell` in the shared library](../examples/conditional-correction/snippet_prepare_bell.deq)
<!-- deq-highlight-begin: ../examples/conditional-correction/snippet_prepare_bell.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> PrepareBell</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareX</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    TransversalCNOT</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/conditional-correction/snippet_prepare_bell.deq -->

[`MeasureBell` destructively reads the Bell basis](../examples/conditional-correction/snippet_measure_bell.deq)
<!-- deq-highlight-begin: ../examples/conditional-correction/snippet_measure_bell.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> MeasureBell</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    TransversalCNOT</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    MeasureX</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/conditional-correction/snippet_measure_bell.deq -->

---

## Three ways to express the correction

Once the frame-correction bits are available, there are three idiomatic ways to absorb
them into the gadget so downstream code sees a clean logical-identity teleport.

All three variants below rely on a small but crucial deq feature — **concatenated
COMPOSE**: once you have declared a `COMPOSE` block, its name becomes callable from
inside any *later* `COMPOSE` (or `PROGRAM`) body just like a `GADGET`, so you can
build layered abstractions without inlining everything by hand. For example,
`PrepareBell` and `MeasureBell` are themselves `COMPOSE` blocks assembled
from lower-level gadgets, and the three teleport variants below invoke them by name
in the same way you would invoke a hand-written GADGET.  This lets each teleport
variant express the *whole* logical operation in five lines while the underlying
Bell-pair mechanics live once in the shared library.

### Variant 1 — `@REPROPAGATE`: let the natural Heisenberg derive the correction

The transversal CNOT in `MeasureBell` already propagates the input Pauli operators
through to the measurement record — `LZ` of the input gets folded into the wire-1
`MeasureZ` outcome, and `LX` gets folded into the wire-0 `MeasureX` outcome.  The
`@REPROPAGATE` decorator tells the compose builder to rebuild the propagation matrix
from the inlined flat circuit instead of matrix-composing the sub-gadgets'
propagation matrices.  The natural circuit flow then automatically encodes
"output = input XOR (some measurement outcomes)", and no explicit CONDITIONAL needs to
be written.

[`TeleportRepropagate` via `@REPROPAGATE`](../examples/conditional-correction/snippet_teleport_repropagate.deq)
<!-- deq-highlight-begin: ../examples/conditional-correction/snippet_teleport_repropagate.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@REPROPAGATE</span></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> TeleportRepropagate</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    PrepareBell</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    MeasureBell</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/conditional-correction/snippet_teleport_repropagate.deq -->

This is the most compact form — the user just writes the COMPOSE body and
decorates it.  But `@REPROPAGATE` cannot resolve a more fundamental issue with
measurement-based logical operations: **the same physical circuit realizes many
inequivalent logical actions**, and deq deliberately refuses to guess which one
the user wants.  `CONDITIONAL` (and its cousin `VIRTUAL`) are how the user
*picks* one such reading.  The lattice-surgery joint-$\bar Z$ merge `MZZ` in
[`tests/circuit/surface_code/lattice_surgery_d3.deq`](../../../tests/circuit/surface_code/lattice_surgery_d3.deq) is
the canonical example — see the [lattice-surgery chapter](lattice-surgery.md)
for the full walkthrough of the ambiguity and how `CONDITIONAL` and
hand-written `PROPAGATE` rows resolve it.

### Variant 2 — COMPOSE-level `CONDITIONAL`

The `CONDITIONAL` statement applies a logical Pauli on a wire of the COMPOSE block,
conditioned on a previous logical readout:

[`TeleportConditional` with COMPOSE-level `CONDITIONAL`](../examples/conditional-correction/snippet_teleport_conditional.deq)
<!-- deq-highlight-begin: ../examples/conditional-correction/snippet_teleport_conditional.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> TeleportConditional</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    PrepareBell</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    MeasureBell</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    CONDITIONAL</span><span style="color:#001080"> rec[-2]</span><span style="color:#267F99"> Z0</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    CONDITIONAL</span><span style="color:#001080"> rec[-1]</span><span style="color:#267F99"> X0</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/conditional-correction/snippet_teleport_conditional.deq -->

Reading the body line by line:

| Line                              | Effect                                                                                                                  |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `PrepareBell 1 2`                 | prepare $|\Phi^+\rangle_L$ on wires 1 and 2                                                                              |
| `MeasureBell 0 1`                 | destructively measure wires 0 and 1 in the Bell basis (2 logical readouts: $m_{XX}$, $m_{ZZ}$)                          |
| `CONDITIONAL rec[-2] Z0 2`        | apply logical $Z$ on logical qubit 0 of wire 2 iff $m_{XX} = 1$                                                          |
| `CONDITIONAL rec[-1] X0 2`        | apply logical $X$ on logical qubit 0 of wire 2 iff $m_{ZZ} = 1$                                                          |
| `OUTPUT SurfaceCode 2`            | wire 2 carries the teleported logical state                                                                              |

There is no mention of physical qubits anywhere in the body — the correction is
expressed purely at the logical level (which Pauli, which logical qubit, which wire).
The transpiler synthesizes a one-port identity gadget carrying a
`remote_conditional_correction` modifier for each CONDITIONAL; the canonicalizer then
folds each readout's measurement set into the affected output observable's
measurement deps.  The merged `logical_correction` matrix ends up empty (every
conditional contribution has been absorbed into `correction_propagation` /
`physical_correction`), and the runtime decoder still sees the readouts it needs to
apply the actual frame correction at decode time.

### Variant 3 — PROGRAM-level `CONDITIONAL`

The exact same statements can also live directly in a `PROGRAM` body, without a
wrapping `COMPOSE`:

[PROGRAM-level CONDITIONAL inline](../examples/conditional-correction/snippet_teleport_program_conditional.deq)
<!-- deq-highlight-begin: ../examples/conditional-correction/snippet_teleport_program_conditional.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> TeleportProgramConditionalMemoryZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    PrepareBell</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    MeasureBell</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    CONDITIONAL</span><span style="color:#001080"> rec[-2]</span><span style="color:#267F99"> Z0</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    CONDITIONAL</span><span style="color:#001080"> rec[-1]</span><span style="color:#267F99"> X0</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/conditional-correction/snippet_teleport_program_conditional.deq -->

This is convenient for one-off circuits or for orchestrating conditional Pauli fix-ups
between unrelated gadgets, where you don't want to introduce a new named COMPOSE for
the sole purpose of carrying the correction.  Operationally, however, a PROGRAM-level
CONDITIONAL is *not* equivalent to its COMPOSE-level cousin: COMPOSE behaves like a
gadget body and the transpiler flattens each CONDITIONAL into the merged gadget's
propagation matrix offline, whereas PROGRAM keeps each gadget invocation as a separate
instruction and dispatches the conditional through the runtime system rather than
folding it into a matrix at transpile time.  See the
[`compose-gadgets`](compose-gadgets.md) chapter for the full COMPOSE vs PROGRAM
distinction.

---

## How the three variants compare

| Aspect                                | `@REPROPAGATE` (Variant 1)                                  | COMPOSE-level `CONDITIONAL` (Variant 2) | PROGRAM-level `CONDITIONAL` (Variant 3) |
| ------------------------------------- | ----------------------------------------------------------- | --------------------------------------- | --------------------------------------- |
| Where the correction is declared      | Implicit — derived from circuit flow                        | Inside the `COMPOSE` body                | Inside the `PROGRAM` body                |
| Lines of code in the user-facing block | Smallest (no CONDITIONAL needed)                          | Two `CONDITIONAL` statements              | Two `CONDITIONAL` statements             |
| Does the user name physical qubits?  | No                                                          | No                                        | No                                       |
| Reusable as a sub-gadget?             | Yes, via the wrapping COMPOSE                                | Yes, via the wrapping COMPOSE             | No — lives at the top of the program     |
| Requires a transversal-gate path     | **Yes** — the inlined body must carry the measured Pauli operator to the output qubits | No                                  | No                                       |
| Where the absorption happens          | Flat-circuit Heisenberg on the inlined body                  | `merge()` step 9 absorption pass         | Not absorbed offline — runtime dispatches the conditional per invocation |
| Final `correction_propagation` matrix | Folded at transpile time (matches COMPOSE-level)             | Folded at transpile time                  | Unfolded — handled at runtime instead    |

All three produce a COMPOSE or PROGRAM that acts as logical identity on the teleported
state once the runtime applies the frame correction.  Variants 1 and 2 are merged
offline into byte-identical `correction_propagation` matrices, while Variant 3 keeps
the conditional as a runtime instruction; the *logical-level* end-to-end behavior is
the same across all three.

---

## End-to-end verification

The COMPOSE-level CONDITIONAL fixture comes with a memory program that prepares the
input patch in $|0_L\rangle$, teleports it, then measures the output patch in the Z
basis:

[`TeleportConditionalMemoryZ` PROGRAM](../examples/conditional-correction/02_teleport_compose_conditional.deq)
<!-- deq-highlight-begin: ../examples/conditional-correction/02_teleport_compose_conditional.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># Variant 2 — COMPOSE-level CONDITIONAL.</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000"># Same Bell-pair teleportation, but the Pauli frame correction is</span></span>
<span class="line"><span style="color:#008000"># expressed as an explicit pair of ``CONDITIONAL`` statements at the</span></span>
<span class="line"><span style="color:#008000"># logical level:</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000">#     CONDITIONAL rec[-2] Z0 2   # if m_XX = 1, apply Z to output patch</span></span>
<span class="line"><span style="color:#008000">#     CONDITIONAL rec[-1] X0 2   # if m_ZZ = 1, apply X to output patch</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000"># No ``@REPROPAGATE`` decorator is needed.  The transpiler injects a</span></span>
<span class="line"><span style="color:#008000"># synthesized identity gadget carrying a</span></span>
<span class="line"><span style="color:#008000"># ``remote_conditional_correction`` modifier for each statement; the</span></span>
<span class="line"><span style="color:#008000"># canonicalizer folds the readout's measurement set into the affected</span></span>
<span class="line"><span style="color:#008000"># output observable's measurement deps, yielding the same</span></span>
<span class="line"><span style="color:#008000"># ``correction_propagation`` / ``physical_correction`` matrices as</span></span>
<span class="line"><span style="color:#008000"># ``TeleportRepropagate``.</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">IMPORT</span><span style="color:#A31515"> "00_teleportation_library.deq"</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> TeleportConditional</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    PrepareBell</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    MeasureBell</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    CONDITIONAL</span><span style="color:#001080"> rec[-2]</span><span style="color:#267F99"> Z0</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    CONDITIONAL</span><span style="color:#001080"> rec[-1]</span><span style="color:#267F99"> X0</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> TeleportConditionalMemoryZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    TeleportConditional</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/conditional-correction/02_teleport_compose_conditional.deq -->

After applying the conditional correction the teleport is logical identity on Z, so
the final `MeasureZ` reads `0` deterministically.  Running 20 noiseless shots
captures the random Bell-measurement outcomes and the always-zero terminal
measurement:

```sh
deq sample 02_teleport_compose_conditional.deq \
    --program TeleportConditionalMemoryZ \
    --shots 20 --noiseless --interpret --seed 42
```

The first few shots of `teleport_conditional_sample.txt` (excerpted):

```text
Readouts:
  TeleportConditional: READOUT m0 m1 m2 = 0  (m24 ⊕ m25 ⊕ m26)
  TeleportConditional: READOUT m0 m3 m6 = 1  (m33 ⊕ m36 ⊕ m39)
  MeasureZ: READOUT m0 m3 m6 = 0  (m33 ⊕ m36 ⊕ m39 ⊕ m42 ⊕ m45 ⊕ m48)
```

The two `TeleportConditional` readouts ($m_{XX}$ and $m_{ZZ}$ from `MeasureBell`)
flip randomly across shots, but the final `MeasureZ` reads `0` on every single shot:
the CONDITIONAL absorbed the frame correction into the canonical readout's
measurement set, so the deterministic logical bit comes out unchanged regardless of
which way the random bits fell.

The end-to-end runtime check is `deq simulate ler` (decoder + classical correction
applied for real):

```sh
deq simulate ler 02_teleport_compose_conditional.deq \
    --program TeleportConditionalMemoryZ \
    --shots 20 --batch-size 20 --seed 42
```

Output (excerpt of `teleport_conditional_simulate.txt`):

```text
=== Simulation Results ===
  Shots:          20
  Logical errors: 0
```

Zero logical errors over 20 noiseless shots — the runtime side of the pipeline
agrees with the canonicalizer-side absorption.

---

## Logical error rate under noise

The noiseless run above proves the gadget is *semantically* correct.  To show it is
also a useful *error-correcting* operation, we sweep the physical error rate of a
Stim SI1000 noise model and measure the surviving logical error rate (LER) of the
same `TeleportConditionalMemoryZ` program.

### One-command pipeline

`deq inject si1000` adds depolarizing/measurement noise of strength `p` to every
gate in a `.deq` file; `deq simulate ler` then runs the full transpile → compile →
runtime pipeline against a black-box relay-BP decoder:

```sh
# 1) Inject noise into the primitives fixture — the only file that
#    actually contains physical gates.  ``00_teleportation_library.deq``
#    and ``02_teleport_compose_conditional.deq`` contain only GADGET /
#    COMPOSE / PROGRAM invocations, so `deq inject si1000` has nothing
#    to attach noise to; we just ``cp`` them and rewrite the IMPORT
#    chain to redirect to the noisy fixture.  All three ``*.noisy.deq``
#    outputs are gitignored — regenerate them on demand.
deq inject si1000 ../../../../tests/circuit/surface_code/surface_code_d3.deq \
    --p 1e-4 --out surface_code_d3.noisy.deq
cp 00_teleportation_library.deq        00_teleportation_library.noisy.deq
cp 02_teleport_compose_conditional.deq 02_teleport_compose_conditional.noisy.deq
sed -i 's|"../../../../tests/circuit/surface_code/surface_code_d3.deq"|"surface_code_d3.noisy.deq"|' \
    00_teleportation_library.noisy.deq
sed -i 's|"00_teleportation_library.deq"|"00_teleportation_library.noisy.deq"|' \
    02_teleport_compose_conditional.noisy.deq

# 2) Run the LER simulator.
deq simulate ler 02_teleport_compose_conditional.noisy.deq \
    --program TeleportConditionalMemoryZ \
    --shots 3000000 --errors 200 --batch-size 5000 --seed 42
```

At physical error rate $p = 1 \times 10^{-4}$ this prints:

```text
=== Simulation Results ===
  Shots:          3000000
  Logical errors: 22
  Error rate:     7.333333e-06
```

so the gadget's surviving LER is $\approx 7.3 \times 10^{-6}$ — **a factor of ≈14
below the physical error rate**, comfortably more than one order of magnitude.

### LER vs. physical error rate sweep

Repeating the sweep over five noise rates traces out the standard sub-threshold
scaling for the rotated $d=3$ surface code under SI1000:

| Physical rate $p$ | LER $(\bar Z$-basis memory) | $\mathrm{LER}/p$ |
| ----------------- | --------------------------- | ----------------- |
| $1.0\times 10^{-3}$ | $7.85 \times 10^{-4}$  | $0.79$            |
| $5.0\times 10^{-4}$ | $1.98 \times 10^{-4}$  | $0.40$            |
| $3.0\times 10^{-4}$ | $7.09 \times 10^{-5}$  | $0.24$            |
| $2.0\times 10^{-4}$ | $3.17 \times 10^{-5}$  | $0.16$            |
| $1.0\times 10^{-4}$ | $7.33 \times 10^{-6}$  | $0.07$            |

(3 M shots per row, `--seed 42`, with `--errors 200` early-stop.)

At $p \approx 10^{-3}$ we are close to the SI1000 threshold and the protection is
weak; halving the noise to $5 \times 10^{-4}$ already brings the LER below physical;
by $10^{-4}$ the gap is over an order of magnitude.  The three CONDITIONAL variants
(`@REPROPAGATE`, COMPOSE-level, PROGRAM-level) produce byte-identical merge()-
absorbed matrices, so they share the same LER curve.

---

## Where `CONDITIONAL` isn't enough

`CONDITIONAL` (in either its COMPOSE or PROGRAM form) is the right tool whenever
the byproduct is a *per-port* logical Pauli that the framework can derive from an
existing readout — every measurement-based teleport in this chapter fits that
mould.  Some measurement-based operations exceed even CONDITIONAL's reach: the
byproduct spans more than one output port at once, or the same physical body
admits several distinct logical actions and the framework's per-port flow solver
picks the wrong one.  The [lattice-surgery chapter](lattice-surgery.md) works
through the canonical example — a joint-$\bar Z$ merge that needs `CONDITIONAL`
to pin the "honest joint measurement" reading *and* two hand-written `PROPAGATE`
rows to hand-declare the joint-$\bar X$ preservation the per-port solver misses
— and then shows how to restructure the merge so the resulting fault-tolerance
is genuinely below-threshold at $d = 3$.

---

## When to reach for `CONDITIONAL` vs. `@REPROPAGATE`

The two features overlap for any operation where the correction can be expressed
either as natural Heisenberg flow or as an explicit logical Pauli statement.  When
both work, they produce byte-identical merge()-absorbed matrices, so the choice is
a matter of *which form reads more clearly in source*.

| Property                                                 | `@REPROPAGATE` is the better fit                           | `CONDITIONAL` is the better fit                               |
| -------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------- |
| The correction comes from a transversal gate's Heisenberg propagation | ✓                                                          | works, but the user has to spell it out explicitly             |
| The correction comes from a measurement that has no transversal Heisenberg path in the inlined body | ✗ (the flow solver silently omits the row, giving a wrong `correction_propagation`; adding a `CONDITIONAL` to compensate is itself rejected) | ✓ (synthesizes the identity gadget) |
| You want the COMPOSE body to read like a textbook protocol (Bell prep, measure, correct) | works, but the correction is invisible in source            | ✓ (CONDITIONAL spells out the correction)                      |
| You want the absolute minimum number of source lines     | ✓ (no explicit correction)                                  | one extra line per CONDITIONAL                                  |
| You don't yet know whether the correction is a real classical Pauli or a Heisenberg-flow artifact | ✗ (transpiler decides for you)                              | ✓ (explicit declaration)                                       |

A reliable diagnostic recipe when you are unsure which to use:

1. Write the `COMPOSE` block first, **without** any decorator and without
   `CONDITIONAL`.
2. Run `deq annotate` and read the resulting `PROPAGATE OUT*.L*0` rows: for each
   output logical operator, does the right-hand side (the XOR of input columns
   and measurement bits) match what you intended the operation to do?  The
   annotator always produces *some* rows — the default matrix-composition
   strategy picks one self-consistent logical action out of the many that the
   physical body admits, and it may not be the one you had in mind.
3. If every row matches your intent, the default matrix-composition strategy
   happened to pick the reading you wanted — you are done.
4. If some row differs from your intent, decide which mechanism to reach for:

   * If your intended row *can* be derived from a measurement the inlined body
     already carries through via transversal-Heisenberg flow (the
     transversal-CNOT case), add `@REPROPAGATE` and re-run — the flow analysis
     on the flat inlined body will find the row for you.
   * If your intended row is the result of a *classical* Pauli the circuit
     never applies physically (it lives only in the decoder's frame), spell it
     out with a `CONDITIONAL rec[-k] <pauli> <wire>` statement at the COMPOSE
     level (or the PROGRAM level if you don't want a wrapping COMPOSE).

`@REPROPAGATE` and `CONDITIONAL` are **mutually exclusive within one
COMPOSE**: `deq annotate` rejects a `@REPROPAGATE` COMPOSE that (transitively,
through any sub-COMPOSE or sub-GADGET) contains a `CONDITIONAL` statement,
because the flat-circuit Heisenberg re-derivation cannot reconstruct a frame
flip that lives only in the decoder's classical Pauli record.  If a single
operation genuinely needs *both* a flow-derived correction and a
classical-frame correction, drop `@REPROPAGATE` and express every row via
`CONDITIONAL` — the merge-based composition path handles the combined case
uniformly.

---

## Summary

| Concept                                                | Purpose                                                                                                              |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| Pauli frame correction in measurement-based logical operations | A bit derived from mid-circuit measurements that must be applied to a downstream logical observable           |
| `@REPROPAGATE`                                         | Let the inlined-body Heisenberg propagation derive the correction implicitly                                          |
| `CONDITIONAL rec[-k] <paulis> <wire>` in `COMPOSE`     | Logical-level intent; wraps a sub-gadget that exposes the readout, no physical-qubit names at the call site         |
| `CONDITIONAL rec[-k] <paulis> <wire>` in `PROGRAM`     | Same expression inline in the program body, no wrapping COMPOSE needed                                                |
| What changes in the merged matrices                    | The conditional readout's measurement set is folded into the affected logical row of `correction_propagation` / `physical_correction` |
| What stays the same                                    | The readout itself is preserved — the decoder needs it to apply the frame correction at runtime                       |
| When you reach for `CONDITIONAL`                       | When you want the correction visible in source, or when no transversal-Heisenberg path exists for `@REPROPAGATE` to derive |
| LER at $d = 3$, $p = 10^{-4}$                          | Bell-pair teleport gets $\approx 7 \times 10^{-6}$ — over an order of magnitude below physical                        |

Related chapters:

- [Lattice Surgery: The Joint-$\bar Z$ Measurement](lattice-surgery.md) — the follow-on chapter where a joint-parity merge forces `CONDITIONAL` *and* hand-written `PROPAGATE` rows, and where restructuring the merge into single-SE-round GADGETs is what recovers fault tolerance.
- [`@REPROPAGATE`](compose-repropagate.md) — the flow-based alternative for corrections with a transversal-Heisenberg path.
