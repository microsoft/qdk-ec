# Codes with Redundant Stabilizers

In the [language basics chapter](language-basics.md), we defined the repetition code
with its two independent stabilizer generators $Z_0 Z_1$ and $Z_1 Z_2$. These two
generators are a *minimal* (non-redundant) basis for the stabilizer group. But in practice,
syndrome extraction circuits often measure **more stabilizers than the minimum** — and
declaring these redundant stabilizers in the `CODE` block has a significant impact on the
quality of the auto-derived decoding hypergraph.

This chapter demonstrates the difference with a concrete example: the same physical circuit
produces cleaner error models when redundant stabilizers are declared.

---

## The Setup: 3 Ancillae, but How Many Stabilizers?

Consider a repetition code $[[3,1,1]]$ where the syndrome extraction circuit uses
**3 ancilla qubits** to measure:
- Ancilla 1: $Z_0 Z_1$ (parity of qubits 0 and 1)
- Ancilla 3: $Z_1 Z_2$ (parity of qubits 1 and 2)
- Ancilla 5: $Z_0 Z_2$ (parity of qubits 0 and 2) — **redundant!**

The third measurement is redundant because $Z_0 Z_2 = (Z_0 Z_1)(Z_1 Z_2)$ in the
stabilizer group. But measuring it provides **spatial redundancy**: if any single ancilla
suffers a measurement error, the other two measurements are sufficient to detect and
identify the faulty ancilla.

The question is: should we declare 2 or 3 stabilizers in the `CODE` block?

---

## Version 1: Only Generators (Non-Redundant)

[Non-redundant code definition](../examples/redundant-stabilizers/snippet_code_non_redundant.deq)
<!-- deq-highlight-begin: ../examples/redundant-stabilizers/snippet_code_non_redundant.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/redundant-stabilizers/snippet_code_non_redundant.deq -->

The Idle gadget measures all 3 ancillae, but the `CODE` only declares 2 stabilizers.
This means the transpiler tracks **2 virtual stabilizer measurements** per port boundary.

[Idle gadget (shared by both versions)](../examples/redundant-stabilizers/snippet_idle.deq)
<!-- deq-highlight-begin: ../examples/redundant-stabilizers/snippet_idle.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span><span style="color:#008000">           # data qubit errors</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 5</span><span style="color:#008000">                       # 3 ancillae</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#008000">                # round 1: Z0→a1, Z1→a3, Z2→a5</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span><span style="color:#098658"> 0</span><span style="color:#098658"> 5</span><span style="color:#008000">                # round 2: Z1→a1, Z2→a3, Z0→a5</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#098658">1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 5</span><span style="color:#008000">                 # measurement errors</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/redundant-stabilizers/snippet_idle.deq -->

The annotated output for the Idle gadget reveals the problem:

[Non-redundant Idle annotated](../examples/redundant-stabilizers/snippet_01_non_redundant_idle.deq)
<!-- deq-highlight-begin: ../examples/redundant-stabilizers/snippet_01_non_redundant_idle.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">2</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 2 4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 5</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span><span style="color:#098658"> 0</span><span style="color:#098658"> 5</span></span>
<span class="line"><span style="color:#008000">    # M(0.01) 1 3 5</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 5</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span><span style="color:#267F99"> C3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C2</span><span style="color:#267F99"> C4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span><span style="color:#267F99"> IN0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#267F99"> IN0.S1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M2</span><span style="color:#001080"> M1</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S0</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S1</span><span style="color:#001080"> M1</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 3</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:2, 3:1 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 2</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:2 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 6</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 1:3, 2:1, 3:2 }</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/redundant-stabilizers/snippet_01_non_redundant_idle.deq -->

**Problems:**
1. **C2 is a weight-3 spatial check** involving all 3 physical measurements but no virtual
   stabilizer. The transpiler had to construct this check because the $Z_0 Z_2$ measurement
   has no corresponding virtual stabilizer — it must be expressed as
   $m_0 \oplus m_1 \oplus m_2 = 0$
2. **`ERROR(0.01) C2`** — the measurement error on the $Z_0 Z_2$ ancilla triggers
   **only** the spatial check C2, producing a weight-1 error in the decoding hypergraph.
   While decoders can still handle this (it is detected by the spatial check), the
   asymmetry between ancillae — two have time-like checks and one does not — leads to
   an unstructured error model
3. **Measurement errors trigger 3 checks** — e.g. `ERROR(0.01) C0 C2 C3` and
   `ERROR(0.01) C1 C2 C4` are weight-3 hyperedges that are harder for decoders
   to handle than simple edges

---

## Version 2: All Stabilizers (Redundant)

[Redundant code definition](../examples/redundant-stabilizers/snippet_code_redundant.deq)
<!-- deq-highlight-begin: ../examples/redundant-stabilizers/snippet_code_redundant.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#000000"> [[</span><span style="color:#098658">3</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/redundant-stabilizers/snippet_code_redundant.deq -->

Same physical circuit, but now the transpiler tracks **3 virtual stabilizer measurements**
per port — one for each declared stabilizer.

The annotated Idle gadget:

[Redundant Idle annotated](../examples/redundant-stabilizers/snippet_02_redundant_idle.deq)
<!-- deq-highlight-begin: ../examples/redundant-stabilizers/snippet_02_redundant_idle.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">2</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Idle</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#008000">    # X_ERROR(0.01) 0 2 4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C2</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C1</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C2</span><span style="color:#800000"> OUT0.LX0</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 5</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 2</span><span style="color:#098658"> 1</span><span style="color:#098658"> 4</span><span style="color:#098658"> 3</span><span style="color:#098658"> 0</span><span style="color:#098658"> 5</span></span>
<span class="line"><span style="color:#008000">    # M(0.01) 1 3 5</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 1</span><span style="color:#098658"> 3</span><span style="color:#098658"> 5</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#267F99"> C3</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C1</span><span style="color:#267F99"> C4</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.01</span><span style="color:#000000">) </span><span style="color:#267F99">C2</span><span style="color:#267F99"> C5</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span><span style="color:#267F99"> IN0.S0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M1</span><span style="color:#267F99"> IN0.S1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M2</span><span style="color:#267F99"> IN0.S2</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S0</span><span style="color:#001080"> M0</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S1</span><span style="color:#001080"> M1</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#267F99"> OUT0.S2</span><span style="color:#001080"> M2</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 3</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:3 }</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 3</span></span>
<span class="line"><span style="color:#008000">    #   weight distribution: { 2:3 }</span></span>
<span class="line"><span style="color:#008000">    # errors: 6</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 2:6 }</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/redundant-stabilizers/snippet_02_redundant_idle.deq -->

**Improvements:**
1. **Every check is weight-2** — each ancilla has its own time-like check comparing
   the physical measurement against its input virtual stabilizer:
   `M0 IN0.S0`, `M1 IN0.S1`, `M2 IN0.S2`
2. **Every measurement error triggers exactly 2 checks** — one finished (time-like)
   and one unfinished: `C0 C3`, `C1 C4`, `C2 C5`. No hyperedges!
3. **Data errors trigger exactly 2 finished checks** — symmetric: `C0 C2 OUT0.LX0`,
   `C0 C1 OUT0.LX0`, `C1 C2 OUT0.LX0`

With `@CHECKS("syndrome")`, the transpiler uses the syndrome check plugin which
automatically excludes spatial metachecks. The metacheck
$m_0 \oplus m_1 \oplus m_2 = 0$ would otherwise arise from the redundant
stabilizer relation. It is a valid "check of checks" but creates
weight-3 hyperedges for measurement errors. The `syndrome` plugin removes it,
giving a clean graph decoder-friendly structure where every error is an edge
(weight ≤ 2).

> **Note:** The default `@CHECKS("auto")` plugin keeps metachecks, which is useful
> for single-shot decoding where spatial redundancy provides additional error
> correction capability at the cost of hyperedges. For single-shot decoding with
> metachecks, use the `@CHECKS("syndrome", metachecks=1)` plugin.

---

## Why It Matters: The Number of Virtual Stabilizers

The `STABILIZER` list in the `CODE` block determines **how many virtual stabilizer
measurements** the transpiler tracks per port boundary:

| Configuration                  | Virtual stabilizers per port | Behavior                                                             |
| ------------------------------ | ---------------------------- | -------------------------------------------------------------------- |
| `STABILIZER Z0*Z1 Z1*Z2`       | 2                            | Third ancilla has no virtual counterpart → spatial-only check        |
| `STABILIZER Z0*Z1 Z1*Z2 Z0*Z2` | 3                            | Every ancilla maps to a virtual stabilizer → full time-like coverage |

When there are **fewer virtual stabilizers than physical measurements**, the transpiler
must express the "extra" measurements as GF(2) combinations of the declared ones. This
creates high-weight checks and unstructured error patterns.

When there are **as many virtual stabilizers as physical measurements** (even if some are
redundant), each measurement maps 1:1 to a stabilizer → clean weight-2 time-like checks.

---

## Connection to Single-Shot QEC and the `@CHECKS` Plugin System

This example illustrates the key mechanism behind **single-shot quantum error correction**
and the trade-off controlled by the check plugin:

- **Without redundancy** (2 stabilizers, 3 measurements): the third measurement is
  "invisible" to time-like checks. A measurement error on that ancilla can only be detected
  by the spatial check — but to *correct* it, you need temporal redundancy (multiple
  rounds of syndrome extraction)

- **With redundancy + `@CHECKS("syndrome")`** (3 stabilizers, 3 measurements): every
  ancilla has a 1:1 time-like check. All errors are weight ≤ 2 — a clean graph with no
  hyperedges, ideal for matching-based decoders

- **With redundancy + `@CHECKS("auto")`** (default): the transpiler additionally
  emits a spatial metacheck $m_0 \oplus m_1 \oplus m_2 = 0$. This makes measurement
  errors weight-3 (hyperedges) but provides spatial redundancy that enables single-shot
  decoding — a single round suffices to correct measurement errors without temporal
  comparison

The `@CHECKS` decorator selects a check plugin at the gadget level:

| Setting                             | Metachecks | Error weight | Use case                                  |
| ----------------------------------- | ---------- | ------------ | ----------------------------------------- |
| `@CHECKS("auto")`                   | Included   | ≤ 3          | Single-shot decoding, hypergraph decoders |
| `@CHECKS("syndrome")`               | Excluded   | ≤ 2          | Multi-round decoding, matching decoders   |
| `@CHECKS("syndrome", metachecks=1)` | Included   | ≤ 3          | Single-shot with per-stabilizer checks    |

---

## Summary

| Aspect                       | Non-redundant               | Redundant + `@CHECKS("syndrome")` |
| ---------------------------- | --------------------------- | --------------------------------- |
| `STABILIZER` declarations    | 2 (generators only)         | 3 (including $Z_0 Z_2$)           |
| Virtual stabilizers per port | 2                           | 3                                 |
| Finished checks              | 3 (2 time-like + 1 spatial) | 3 (all weight-2 time-like)        |
| Unfinished checks            | 2                           | 3                                 |
| Time-like coverage           | Partial (ancilla 5 missing) | Complete                          |
| Max error weight             | 3 (measurement error)       | 2 (all errors)                    |
| Hyperedges                   | Yes                         | None                              |

**Rule of thumb:** if your syndrome extraction circuit measures redundant stabilizers,
**declare all of them** in the `CODE` block — even though they're not independent generators.
The transpiler will produce clean weight-2 time-like checks for each measurement. Use
`@CHECKS("syndrome")` if you want a hyperedge-free graph (for matching decoders), or
keep the default `@CHECKS("auto")` if you want spatial metachecks (for single-shot
decoding with hypergraph decoders).
