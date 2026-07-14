# Lattice Surgery: The Joint-$\bar Z$ Measurement

The [`CONDITIONAL` chapter](conditional-correction.md) closed with a warning
that the reader can safely ignore for teleportation-style gadgets: **the same
physical circuit realises many inequivalent logical actions**, and deq's
auto-derived flow silently picks *one* of them, which may not be the one you
intended.  For Bell-pair teleportation the choice is invisible because there
is only one natural reading — the flow solver picks it, `@REPROPAGATE` and
`CONDITIONAL` agree with it, and the user never has to think about the
ambiguity.

Lattice surgery is where the ambiguity stops being an abstraction and starts
biting.  The joint-$\bar Z$ merge $\mathrm{MZZ}$ takes two surface-code patches
and reads out the joint parity $\bar Z_A \bar Z_B$ — but its physical body
(six joint-Pauli MPPs plus a destructive MX seam) is *equally consistent*
with two different logical actions: a pure joint-Z measurement (`MZZ`), or a
joint-Z measure-and-reset (`MRZZ`).

**`MZZ` versus `MRZZ`.**  Both readings share the same physical body:

* `MZZ`: read $\bar Z_A \bar Z_B$; leave the individual $\bar Z_A$, $\bar Z_B$
  frames alone so the post-merge state stays in whichever $\bar Z_A
  \bar Z_B = \pm 1$ branch the measurement projected onto.
* `MRZZ`: read $\bar Z_A \bar Z_B$; then classically flip patch B's $\bar Z$
  frame whenever the readout is $1$ so the post-merge state is always in the
  $+1$ eigenspace — equivalently, patch B's post-merge $\bar Z$ frame is
  always forced to agree with patch A's.

Without user guidance, deq's auto-derived flow silently picks the `MRZZ`
reading.  A hand-written `PROPAGATE OUT1.LX0 FROM IN1.LX0` row overrides
that and pins the honest `MZZ` reading (an equivalent `CONDITIONAL R0
OUT1.LX0` byproduct would do the same job — the two forms are derived to be
equivalent below).  The naming mirrors Stim's single-qubit `MZ` / `MRZ`
distinction — `M*` measures only, `MR*` measures and resets.

The rest of this chapter walks through spotting the `MRZZ` pick in
`deq annotate`'s output, deriving why the flow solver settles on it, and
fixing it via either `PROPAGATE` or `CONDITIONAL`.  Declaring the byproduct
makes MZZ *semantically* correct, but a single-round MZZ is not
fault-tolerant on its own — a further refactor into repeated single-SE-round
GADGETs at the COMPOSE level is what restores the $\mathrm{LER} \propto
p^{(d+1)/2}$ surface-code scaling.

**Prerequisites.**  Read the [`CONDITIONAL` chapter](conditional-correction.md)
first; this chapter assumes familiarity with `@REPROPAGATE`,
`CONDITIONAL rec[-k] <pauli> <wire>`, and the COMPOSE-level Pauli-frame
correction machinery that both feed into.

---

## The MZZ merge in one figure

![Stim `detslice-svg` snapshot of the merged-code stabilizer group before and after the MZZ merge. Left: two independent distance-3 rotated surface-code patches with 16 stabilizers (red = X-type, blue = Z-type); the seam column sits bare between them.  Right: after the six merge measurements the same 21 qubits form one merged code with 20 stabilizers — four new bulk plaquettes span the seam and two new Z 2-body boundary plaquettes complete the checkerboard at the top and bottom, while the previously-independent X 2-body edge stabs (patch A's `X5*X8` and patch B's `X9*X12`) get absorbed into the new bulk plaquettes. Regenerate with [`stabilizer_flow.py`](../examples/lattice-surgery/stabilizer_flow.py).](../examples/lattice-surgery/mzz_stabilizer_flow.png)

Two distance-3 rotated surface-code patches sit horizontally side-by-side with
an intermediate column of three data qubits between them.  The single-shot
merge does three things:

1. Initialize the seam column in $|+\rangle$ (`RX 18 19 20`), pinning the
   seam-X stabilizers.
2. Measure six new merge stabilizers on the seam (four bulk plaquettes plus
   two Z-type boundary 2-bodies) — call the outcomes $M_0 \dots M_5$.
3. Destructively measure the seam column in the X basis (`MX 18 19 20`) to
   split the patches back apart — outcomes $M_6, M_7, M_8$.

The joint $\bar Z_A \bar Z_B$ parity comes out of the Z-type merge
measurements as $M_0 \oplus M_3 \oplus M_4 \oplus M_5$; that XOR is exposed as
the merge's single-bit `READOUT R0`.  The full gadget lives in the
lattice-surgery library:

[`MZZ` and its `ComposeMZZ` wrapper (single-shot merge)](../examples/lattice-surgery/00_lattice_surgery_library.deq)
<!-- deq-highlight-begin: ../examples/lattice-surgery/00_lattice_surgery_library.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#008000"># Shared library for the lattice-surgery chapter.</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000"># The physical mechanics of the joint-Z merge (geometry, stabilizer</span></span>
<span class="line"><span style="color:#008000"># derivation, byproduct semantics) are documented in</span></span>
<span class="line"><span style="color:#008000"># ``documents/tutorial/chapters/lattice-surgery.md`` and in the</span></span>
<span class="line"><span style="color:#008000"># annotated fixture at ``tests/circuit/surface_code/lattice_surgery_d3.deq``.</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#000000"> [[</span><span style="color:#098658">9</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">3</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z3</span><span style="color:#000000">*</span><span style="color:#0000FF">Z6</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X3</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z3</span><span style="color:#000000">*</span><span style="color:#0000FF">Z4</span><span style="color:#0000FF"> X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#000000">*</span><span style="color:#0000FF">X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X5</span><span style="color:#0000FF"> X3</span><span style="color:#000000">*</span><span style="color:#0000FF">X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X6</span><span style="color:#000000">*</span><span style="color:#0000FF">X7</span><span style="color:#0000FF"> Z4</span><span style="color:#000000">*</span><span style="color:#0000FF">Z5</span><span style="color:#000000">*</span><span style="color:#0000FF">Z7</span><span style="color:#000000">*</span><span style="color:#0000FF">Z8</span><span style="color:#0000FF"> X5</span><span style="color:#000000">*</span><span style="color:#0000FF">X8</span><span style="color:#0000FF"> Z6</span><span style="color:#000000">*</span><span style="color:#0000FF">Z7</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    RZ</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span></span>
<span class="line"><span style="color:#795E26">    I</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span><span style="color:#008000">  # so that data-qubit error can be injected</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # Single round of syndrome extraction to project into the code space.</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X3</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z3</span><span style="color:#000000">*</span><span style="color:#0000FF">Z4</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#000000">*</span><span style="color:#0000FF">X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X5</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X3</span><span style="color:#000000">*</span><span style="color:#0000FF">X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X6</span><span style="color:#000000">*</span><span style="color:#0000FF">X7</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z4</span><span style="color:#000000">*</span><span style="color:#0000FF">Z5</span><span style="color:#000000">*</span><span style="color:#0000FF">Z7</span><span style="color:#000000">*</span><span style="color:#0000FF">Z8</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X5</span><span style="color:#000000">*</span><span style="color:#0000FF">X8</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z6</span><span style="color:#000000">*</span><span style="color:#0000FF">Z7</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span></span>
<span class="line"><span style="color:#795E26">    I</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span><span style="color:#008000">  # so that data-qubit error can be injected</span></span>
<span class="line"><span style="color:#795E26">    MZ</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-9]</span><span style="color:#001080"> rec[-6]</span><span style="color:#001080"> rec[-3]</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MZZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">    I</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#008000">  # so that data-qubit error can be injected</span></span>
<span class="line"><span style="color:#795E26">    RX</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z2</span><span style="color:#000000">*</span><span style="color:#0000FF">Z5</span><span style="color:#000000">*</span><span style="color:#0000FF">Z18</span><span style="color:#000000">*</span><span style="color:#0000FF">Z19</span><span style="color:#008000">       # M0</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X5</span><span style="color:#000000">*</span><span style="color:#0000FF">X8</span><span style="color:#000000">*</span><span style="color:#0000FF">X19</span><span style="color:#000000">*</span><span style="color:#0000FF">X20</span><span style="color:#008000">       # M1</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X9</span><span style="color:#000000">*</span><span style="color:#0000FF">X12</span><span style="color:#000000">*</span><span style="color:#0000FF">X18</span><span style="color:#000000">*</span><span style="color:#0000FF">X19</span><span style="color:#008000">      # M2</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z12</span><span style="color:#000000">*</span><span style="color:#0000FF">Z15</span><span style="color:#000000">*</span><span style="color:#0000FF">Z19</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span><span style="color:#008000">     # M3</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z9</span><span style="color:#000000">*</span><span style="color:#0000FF">Z18</span><span style="color:#008000">              # M4</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z8</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span><span style="color:#008000">              # M5</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span><span style="color:#008000">             # M6 M7 M8</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> M0</span><span style="color:#001080"> M3</span><span style="color:#001080"> M4</span><span style="color:#001080"> M5</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT1.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN1.LX0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> ComposeMZZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    MZZ</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> ComposeMZZMemoryZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    ComposeMZZ</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-3]</span><span style="color:#098658"> 0</span><span style="color:#008000">   # joint LZ_A*LZ_B parity = +1</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 0</span><span style="color:#008000">   # MeasureZ patch A</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span><span style="color:#008000">   # MeasureZ patch B</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/lattice-surgery/00_lattice_surgery_library.deq -->

The body ends with the *declarative* statement this chapter is about:

[`MZZ` body — READOUT, OUTPUT ports, and the declarative statement](../examples/lattice-surgery/00_lattice_surgery_library.deq#L52-L59)
<!-- deq-highlight-begin: ../examples/lattice-surgery/00_lattice_surgery_library.deq#L52-L59 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> M0</span><span style="color:#001080"> M3</span><span style="color:#001080"> M4</span><span style="color:#001080"> M5</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT1.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN1.LX0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/lattice-surgery/00_lattice_surgery_library.deq#L52-L59 -->

That single hand-written `PROPAGATE` row tells deq *which logical action*
this merge is supposed to represent.  Delete it and the compiler still
accepts the gadget — it just installs the wrong (`MRZZ`, measure-and-reset)
logical map instead of the honest (`MZZ`, measure-only) one.

The rest of this chapter explains why that line is there.

---

## The ambiguity: `MRZZ` versus `MZZ`

### Spotting the problem in `deq annotate`'s naive output

Delete the hand-written `PROPAGATE` row from `MZZ` and let
`deq annotate` derive everything from the physical body alone.  A minimal
self-contained copy of that stripped body lives at
[`01_mzz_before_conditional.deq`](../examples/lattice-surgery/01_mzz_before_conditional.deq),
and running the annotator on it produces
[`01_mzz_before_conditional.annotated.deq`](../examples/lattice-surgery/01_mzz_before_conditional.annotated.deq).
Two pieces of that output matter for what follows.

**Piece 1** — the `READOUT` line, which carries a `#` comment listing the
input-frame bits that (XOR'd with anything already on the line) give the
full flip set of the raw merge readout `R0 = M0 ⊕ M3 ⊕ M4 ⊕ M5` relative
to the input observables (this is what `readout_propagation` computed for
R0):

[the annotator's READOUT line for the un-fixed `MZZ` gadget](../examples/lattice-surgery/01_mzz_before_conditional.annotated.deq#L27)
<!-- deq-highlight-begin: ../examples/lattice-surgery/01_mzz_before_conditional.annotated.deq#L27 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> M0</span><span style="color:#001080"> M3</span><span style="color:#001080"> M4</span><span style="color:#001080"> M5</span><span style="color:#008000">  # IN0.LX0 IN1.LX0 IN0.DS0 IN0.DS2 IN0.DS5 IN0.DS7</span></span></code></pre>
<!-- deq-highlight-end: ../examples/lattice-surgery/01_mzz_before_conditional.annotated.deq#L27 -->

That comment reflects a division of labour.  The user's `READOUT M0
M3 M4 M5` line declares which physical measurement bits make up
`R0`'s raw value:

$$R_0^{\text{raw}} \;=\; M_0 \oplus M_3 \oplus M_4 \oplus M_5.$$

The transpiler analyzes what those physical measurements actually
measure on the two-patch pre-merge state and reports which input
observables' values would flip `R0`'s raw value if they were flipped
in the pre-merge frame — that's the `#` comment's XOR list, and it's
stored on the compiled binary as `readout_propagation`.  At runtime,
the framework combines the two to produce the value of `R0` in the
input frame:

$$R_0 \;=\; R_0^{\text{raw}} \;\oplus\; \bigl(\text{IN0.LX0} \oplus
\text{IN1.LX0} \oplus \text{IN0.DS0} \oplus \text{IN0.DS2} \oplus
\text{IN0.DS5} \oplus \text{IN0.DS7}\bigr).$$

(In the noiseless case this equals the true joint parity $\bar Z_A
\bar Z_B$, evaluated on the pre-merge input observables plus their
stabilizer-syndrome bits.  The runtime uses exactly this combined
expression whenever it substitutes `R0` into a downstream frame
formula.)

Two kinds of input bits appear in the comment's XOR list, each with a
distinct physical meaning.

**The two logical bits (`IN0.LX0`, `IN1.LX0`).**  `R0` measures the
joint Z observable $\bar Z_A \bar Z_B$, and applying an X on either
patch's logical qubit anti-commutes with that patch's Z observable —
so of course either patch's X flips R0.  In the framework's PROPAGATE
convention `LX<i>` names the row that tracks the Z-observable value
(an X error is what flips a Z outcome), so both input LX rows appear:
one X on either patch is enough to flip R0.

**The four destabilizer bits (`IN0.DS0`, `IN0.DS2`, `IN0.DS5`,
`IN0.DS7`).**  These appear because `R0`'s physical operator on patch A
is *not* the code's natural $\bar Z_A$ representative.  Cancelling the
seam-qubit factors from $M_0 \oplus M_3 \oplus M_4 \oplus M_5$ leaves

$$
Z_2 Z_5 Z_{18} Z_{19} \cdot Z_{12} Z_{15} Z_{19} Z_{20} \cdot Z_9 Z_{18} \cdot Z_8 Z_{20}
\;=\; \underbrace{Z_2 Z_5 Z_8}_{\text{patch A, right column}}
\cdot \underbrace{Z_9 Z_{12} Z_{15}}_{\text{patch B, left column}}
$$

On patch B this is already the code's natural $\bar Z$ representative
— the code declares `LOGICAL ... Z0*Z3*Z6`, which on patch B's qubit
range 9–17 is $Z_9 Z_{12} Z_{15}$ (the left column) — so no shift is
needed and no `IN1.DS*` bits appear.  On patch A the code's natural
representative is $Z_0 Z_3 Z_6$ (the left column), but $R_0$ measured
$Z_2 Z_5 Z_8$ (the right column) instead.  To translate R0's flip
condition into the natural input frame, the transpiler shifts the
right column back to the left via stabilizers:

$$
Z_2 Z_5 Z_8 \;=\; \underbrace{Z_0 Z_3 Z_6}_{\text{LZ}_A}
\;\oplus\; \underbrace{Z_1 Z_2}_{S_0}
\;\oplus\; \underbrace{Z_0 Z_1 Z_3 Z_4}_{S_2}
\;\oplus\; \underbrace{Z_4 Z_5 Z_7 Z_8}_{S_5}
\;\oplus\; \underbrace{Z_6 Z_7}_{S_7}.
$$

Every qubit index appears an even number of times on the right except
$\{2, 5, 8\}$, which appear once each — verifying the shift.  The
four stabilizers ($S_0, S_2, S_5, S_7$) needed for the shift show up
as the four destabilizer references `IN0.DS0`, `IN0.DS2`, `IN0.DS5`,
`IN0.DS7` in the comment's XOR list.  (In the framework's PROPAGATE
algebra, an input stabilizer's measurement-outcome bit is XOR'd in via
the destabilizer column `IN<p>.DS<s>` of `correction_propagation`.)

**Piece 2** — the four auto-derived `PROPAGATE` rows at the tail of the
file:

[the auto-derived `PROPAGATE` rows of the un-fixed `MZZ` gadget](../examples/lattice-surgery/01_mzz_before_conditional.annotated.deq#L48-L52)
<!-- deq-highlight-begin: ../examples/lattice-surgery/01_mzz_before_conditional.annotated.deq#L48-L52 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LZ0</span><span style="color:#800000"> IN1.LZ0</span><span style="color:#001080"> M6</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT1.LZ0</span><span style="color:#0000FF"> FROM</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT1.LX0</span><span style="color:#0000FF"> FROM</span><span style="color:#800000"> IN0.LX0</span><span style="color:#267F99"> IN0.DS0</span><span style="color:#267F99"> IN0.DS2</span><span style="color:#267F99"> IN0.DS5</span><span style="color:#267F99"> IN0.DS7</span><span style="color:#001080"> M0</span><span style="color:#001080"> M3</span><span style="color:#001080"> M4</span><span style="color:#001080"> M5</span></span>
<span class="line"></span></code></pre>
<!-- deq-highlight-end: ../examples/lattice-surgery/01_mzz_before_conditional.annotated.deq#L48-L52 -->

Stare at those four rows for a moment.  Three of them look sensible:

* `OUT0.LX0 FROM IN0.LX0` — patch A's pre-merge Z frame propagates to
  OUT0's post-merge Z frame untouched.
* `OUT0.LZ0 FROM IN0.LZ0 IN1.LZ0 M6` — the auto-derived anchor for the
  joint logical-X invariant $\bar X_A \bar X_B$ that the merge preserves
  (with the seam-MX outcome $M_6$ absorbing the measurement-induced
  sign).  The flow solver notices that neither individual $\bar X$
  survives, but their product does, and it charges the surviving joint
  observable to the `OUT0.LZ0` anchor — mathematically arbitrary, but
  consistent.
* `OUT1.LZ0 FROM` — the mirror empty row that goes with the
  `OUT0.LZ0` joint-$\bar X$ anchor above (charging it here instead
  would be an equivalent choice — they differ by the projected
  $\bar Z_A \bar Z_B$ stabilizer).

But then `OUT1.LX0`, by symmetry with `OUT0.LX0`, ought to read
`FROM IN1.LX0` — patch B's pre-merge Z frame propagates to OUT1's
post-merge Z frame untouched.  Instead the annotator produced:

```text
PROPAGATE OUT1.LX0 FROM IN0.LX0 IN0.DS0 IN0.DS2 IN0.DS5 IN0.DS7 M0 M3 M4 M5
```

**`IN1.LX0` is nowhere in the file.**  Patch B's own Z tracker has
vanished from the annotated propagation; patch A's tracker plus a pile
of merge bits and patch-A destabilizers has taken its place.  Where did
patch B's Z frame go?

### Why deq's Heisenberg picks patch A's frame

The natural-Heisenberg row for `OUT1.LX0` is the flow solver's answer
to *"expressed as an XOR of input observables, destabilizers, and body
measurements, what is a propagated version of patch B's post-merge Z
string $Z_9 Z_{12} Z_{15}$?"*.  Two things pin the specific answer down.

First, **the merge projects the two-patch state onto the joint-Z
eigenspace, so patch A's $\bar Z_A$ and patch B's $\bar Z_B$ are no
longer independent observables of the merged system** — they're equal
modulo the four Z-type merge measurements plus a handful of patch-A
destabilizers, which is exactly the identity from Piece 1 above.  In
deq's frame-column convention `LX<i>` names the row that tracks the
$\bar Z$-observable value (an X error is what flips a Z outcome).  The solver therefore
has *GF(2) freedom* in which pre-merge $\bar Z$ frame to charge the
row against: `IN0.LX0 ⊕ (merge bits) ⊕ (destabilizer bits)` and
`IN1.LX0` produce the same physical observable on the output.

Second, `_compute_pc_logical_via_flows` builds a single GF(2) linear
system whose columns are ordered `[flow_generators, port_0's
observables + destabilizers, port_1's observables + destabilizers,
...]` and solves it via reduced row-echelon form.  RREF picks pivots
strictly left-to-right, so patch A's `IN0.LX0` column — which precedes
patch B's `IN1.LX0` — becomes a pivot; by the time RREF reaches
`IN1.LX0` it is already GF(2)-dependent on the columns to its left
(exactly the merge identity above) and gets classified as a free
column, which the solver fixes to zero.  The result is the
`IN0.LX0 + destabilizer bits + measurement bits` row above, and a
different port ordering would have picked the mirror.

**Semantic reading**: deq's naive interpretation is *"patch B's
post-merge Z frame equals patch A's pre-merge Z frame ⊕ the joint-parity
readout"* — patch B has been silently rewritten to agree with patch A
up to R0.  That's the `MRZZ` measure-and-reset behaviour from the
chapter intro.  It's a self-consistent logical map, but it's the wrong
one for the joint-measurement gadget we're trying to build; an honest
`MZZ` should leave both individual Z frames alone and expose the joint
parity as a *separate* readout.

### Two equivalent fixes

The gadget author has two clean ways to restore `OUT1.LX0 = IN1.LX0`.

**Fix A: `CONDITIONAL R0 OUT1.LX0`.**  Leave the naive `PROPAGATE` row
alone; add an R0 XOR on top via a `CONDITIONAL` statement.  At runtime
the framework evaluates

$$\text{residual}[\text{OUT1.LX0}]
\;=\; \underbrace{cp \cdot \text{inputs} \;\oplus\; pc \cdot \text{raw}}_{\text{the naive PROPAGATE row}}
\;\oplus\; \underbrace{lc \cdot \text{readouts}}_{= \;R_0}.$$

**Fix B: `PROPAGATE OUT1.LX0 FROM IN1.LX0`.**  Overwrite the residual
directly.  Since `PROPAGATE` is authoritative, this replaces the naive
row wholesale — same shape as the auto-derived `OUT0.LX0 FROM IN0.LX0`,
just mirrored to patch B.  No `CONDITIONAL`, no R0 arithmetic to
reason about.

Neither fix changes any physical instruction; both change only the
*logical interpretation* the framework installs.

### Why the two fixes are equivalent

We can derive that they produce identical runtime residuals without
running a single shot, using the R0-identity from Piece 1.  Start with
the naive row:

$$\text{naive row for OUT1.LX0}
\;=\; \text{IN0.LX0} \oplus \text{IN0.DS0} \oplus \text{IN0.DS2}
\oplus \text{IN0.DS5} \oplus \text{IN0.DS7} \oplus M_0 \oplus M_3 \oplus M_4 \oplus M_5.$$

Fix A instructs the runtime to XOR `R0`'s value on top.  Substituting
`R0`'s two-part form ($R_0 = R_0^{\text{raw}} \oplus \text{(input-frame
contribution)}$ from Piece 1):

$$
\begin{aligned}
\text{OUT1.LX0} &\;=\; \text{naive row} \;\oplus\; R_0 \\
&\;=\; \bigl(\text{IN0.LX0} \oplus \text{IN0.DS0} \oplus \dots
\oplus \text{IN0.DS7} \oplus M_0 \oplus M_3 \oplus M_4 \oplus M_5\bigr) \\
&\phantom{\;=\;} \oplus\; \underbrace{\bigl(M_0 \oplus M_3 \oplus M_4 \oplus M_5\bigr)}_{R_0^{\text{raw}}} \\
&\phantom{\;=\;} \oplus\; \underbrace{\bigl(\text{IN0.LX0} \oplus \text{IN1.LX0}
\oplus \text{IN0.DS0} \oplus \dots \oplus \text{IN0.DS7}\bigr)}_{\text{readout\_propagation}[R_0] \,\cdot\, \text{inputs}} \\
&\;=\; \text{IN1.LX0}.
\end{aligned}
$$

Every term of the naive row is XOR-cancelled by its counterpart in
`R0`'s two parts, except the single `IN1.LX0` bit that R0's
input-frame contribution carries but the naive row doesn't — that
survives and becomes the final residual.  Fix B installs `IN1.LX0`
directly.  Same value, at the level the runtime evaluates.

**Compiled-binary side effect.**  The two fixes are not byte-identical
in the compiled matrices: Fix A leaves `cp[OUT1.LX0]` and `pc[OUT1.LX0]`
as the naive row and sets `logical_correction[OUT1.LX0, R0] = 1`; Fix B
pushes `cp[OUT1.LX0]` to `IN1.LX0` and leaves `logical_correction`
empty.  The final residual is the same either way — the runtime
evaluates `residual ^= lc · readouts` unconditionally, so Fix A's
`R0` contribution enters at runtime while Fix B pre-computes the same
final residual in `cp` — but the matrices carrying it are different.

### Which one should you write?

Both are valid.  For most cases we recommend **Fix B (direct
`PROPAGATE` rewrite)**:

* The mirror-symmetric shape (`OUT0.LX0 FROM IN0.LX0` /
  `OUT1.LX0 FROM IN1.LX0`) is self-explanatory: patch A's Z frame
  passes to OUT0, patch B's to OUT1.
* No cross-reference to R0's own definition is needed to read the row.
* The correction lives entirely in the static `cp` matrix — no
  runtime `lc · readouts` evaluation is needed for this row.

### Empirical validation

Both fixes pass the same anti-correlation sanity check, and both
un-fixed variants fail it.  With `LogicalX = VIRTUAL LX0` applied to
patch A, the corrected-frame state is
$|\Psi^+\rangle = (|1_L 0_L\rangle + |0_L 1_L\rangle)/\sqrt 2$; the
honest joint measurement predicts anti-correlated individual outcomes
$(joint, A, B) \in \{(1, 0, 1), (1, 1, 0)\}$.
[`PROGRAM BellPairWithLogicalXJointZZ`](../../../tests/circuit/surface_code/lattice_surgery_d3.deq#L271-L281) produces
exactly that pattern under either fix, and produces the
*correlated* pattern $\{(1, 0, 0), (1, 1, 1)\}$ when both are absent
(the joint readout is right but the individual outcomes agree instead
of disagreeing, exposing the silent patch-B rewrite).

---

## Error suppression: making the MZZ fault-tolerant

Getting the byproducts right makes `MZZ` *semantically* the correct joint-$\bar
Z$ measurement.  It does not yet make it a good *fault-tolerant* joint
measurement — that is a separate question about the underlying physical
circuit.

### Single-round MZZ is inherently non-fault-tolerant

`deq inject si1000` layers depolarizing/measurement noise onto the physical
gates in `00_lattice_surgery_library.deq`; `deq simulate ler` then runs the
compiled `ComposeMZZMemoryZ` under a black-box relay-BP decoder.  Repeat
one command per noise rate:

```sh
# Regenerate the noisy library for a given p (both *.noisy.deq are gitignored).
deq inject si1000 00_lattice_surgery_library.deq --p 1e-4 \
    --out 00_lattice_surgery_library.noisy.deq

deq simulate ler 00_lattice_surgery_library.noisy.deq \
    --program ComposeMZZMemoryZ \
    --shots 30000000 --errors 1000 --batch-size 5000 --seed 42
```

Sweeping $p \in \{1{\times}10^{-3},\, 5{\times}10^{-4},\, 3{\times}10^{-4},\,
2{\times}10^{-4},\, 1{\times}10^{-4}\}$ produces:

| Physical rate $p$   | LER `ComposeMZZMemoryZ` |
| ------------------- | ----------------------- |
| $1.0\times 10^{-3}$ | $7.44 \times 10^{-3}$   |
| $5.0\times 10^{-4}$ | $3.60 \times 10^{-3}$   |
| $3.0\times 10^{-4}$ | $2.07 \times 10^{-3}$   |
| $2.0\times 10^{-4}$ | $1.32 \times 10^{-3}$   |
| $1.0\times 10^{-4}$ | $6.57 \times 10^{-4}$   |

`ComposeMZZMemoryZ` stays at $\mathrm{LER} \approx 7 p$ across the
whole range — the log-log slope is $1$, not the $(d+1)/2 = 2$ expected of a
fault-tolerant $d = 3$ protocol.  This is not a threshold-crossing problem
that lowering $p$ (or raising $d$) would fix: a single-shot merge cannot
produce round-to-round comparison syndromes at all, so every measurement
error on a merge MPP feeds straight into the logical readout regardless of
code distance.  `CONDITIONAL` and the hand-written `PROPAGATE` rows make
the gadget *semantically* correct, but the *structure* of the physical
circuit is what determines whether it is fault-tolerant.

### Recovering fault tolerance with repeated merge rounds

The single-round joint merge can be turned into a multi-round one by
repeating the six MPP measurements before the destructive split.  Each
repeated measurement gives the decoder a *time-edge* syndrome: round $k$'s
outcome XORed against round $k+1$'s outcome equals zero in the absence of
measurement error, so the decoder can localize and correct a faulty MPP
rather than letting it slip straight into the logical readout.

Repeating the merge measurements requires the same GADGET / COMPOSE
factoring that turns a single SE round into a fault-tolerant memory: one
GADGET per merged-code SE round, and a COMPOSE-level `REPEAT` around them.
See [Composing Gadgets with COMPOSE](compose-gadgets.md) for the mechanics.
Concretely we factor the merge into three GADGETs operating on a dedicated
`MergedSurface [[21, 1]]` code:

* `MergeBegin` initializes the seam in $|+\rangle$, measures the six new
  merge stabilizers once (their XOR is the joint-parity readout `R0`), and
  lifts the two input `SurfaceCode` patches into the merged code.
* `MergedSE` performs a single SE round on the merged code, measuring all
  20 stabilizers.  Repeating it gives the decoder round-to-round time
  edges on the joint-Z stabilizer.
* `MergeEnd` is just the destructive `MX 18 19 20` of the seam column
  that splits the merged code back into two `SurfaceCode` patches; it does
  not re-measure any stabilizer (that job belongs to `MergedSE`).

The example file Mako-parametrizes the SE-round count `r` ($r \geq 0$)
via a COMPOSE-level `REPEAT ${r} { MergedSE }`.  Because `MergeBegin`'s
measurements are consumed by the readout, the earliest time edge on the
joint stabilizer is between `MergedSE` rounds — recovering the
$\mathrm{LER} \propto p^2$ scaling requires $r \geq 3$:

[`MergedSurface` / `MergeBegin` / `MergedSE` / `MergeEnd` / `ComposeMZZR` (Mako-parametric)](../examples/lattice-surgery/02_ls_merge_multi_round.deq)
<!-- deq-highlight-begin: ../examples/lattice-surgery/02_ls_merge_multi_round.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#0000FF">&#x3C;%</span></span>
<span class="line"><span style="color:#000000FF">r </span><span style="color:#000000">=</span><span style="color:#267F99"> int</span><span style="color:#000000FF">(context.get(</span><span style="color:#A31515">'r'</span><span style="color:#000000FF">, </span><span style="color:#098658">3</span><span style="color:#000000FF">))</span></span>
<span class="line"><span style="color:#AF00DB">assert</span><span style="color:#000000FF"> r </span><span style="color:#000000">>=</span><span style="color:#098658"> 0</span><span style="color:#000000FF">, </span><span style="color:#A31515">"r must be >= 0 (r is the number of MergedSE rounds between MergeBegin and MergeEnd)"</span></span>
<span class="line"><span style="color:#000000FF">inner_rounds </span><span style="color:#000000">=</span><span style="color:#000000FF"> r</span></span>
<span class="line"><span style="color:#0000FF">%></span></span>
<span class="line"><span style="color:#008000"># Multi-round joint-Z lattice surgery: MergeBegin, MergedSE, MergeEnd</span></span>
<span class="line"><span style="color:#008000"># on a MergedSurface [[21,1]] code, with COMPOSE-level REPEAT driving</span></span>
<span class="line"><span style="color:#008000"># the round count.</span></span>
<span class="line"><span style="color:#008000">#</span></span>
<span class="line"><span style="color:#008000"># MergeBegin measures the six merge stabilizers once (this defines the</span></span>
<span class="line"><span style="color:#008000"># joint-parity readout R0) and merges the two input SurfaceCode patches</span></span>
<span class="line"><span style="color:#008000"># into the merged code.  MergedSE performs one SE round on the merged</span></span>
<span class="line"><span style="color:#008000"># code; repeating it gives the decoder temporally local edges that</span></span>
<span class="line"><span style="color:#008000"># catch measurement errors on the joint stabilizer between the merge</span></span>
<span class="line"><span style="color:#008000"># and the split.  MergeEnd is just the destructive MX of the seam</span></span>
<span class="line"><span style="color:#008000"># column that splits the merged code back into two SurfaceCode patches;</span></span>
<span class="line"><span style="color:#008000"># it does not re-measure any stabilizer (that job belongs to</span></span>
<span class="line"><span style="color:#008000"># MergedSE).  See the "Recovering fault tolerance with repeated merge</span></span>
<span class="line"><span style="color:#008000"># rounds" section of</span></span>
<span class="line"><span style="color:#008000"># ``documents/tutorial/chapters/lattice-surgery.md`` for the</span></span>
<span class="line"><span style="color:#008000"># rationale.</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">IMPORT</span><span style="color:#A31515"> "00_lattice_surgery_library.deq"</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> MergedSurface</span><span style="color:#000000"> [[</span><span style="color:#098658">21</span><span style="color:#000000">, </span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X3</span><span style="color:#000000">*</span><span style="color:#0000FF">X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X5</span><span style="color:#000000">*</span><span style="color:#0000FF">X19</span><span style="color:#000000">*</span><span style="color:#0000FF">X12</span><span style="color:#000000">*</span><span style="color:#0000FF">X13</span><span style="color:#000000">*</span><span style="color:#0000FF">X14</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z3</span><span style="color:#000000">*</span><span style="color:#0000FF">Z6</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span></span>
<span class="line"><span style="color:#008000">        # Patch A (right-edge X 2-body X5*X8 absorbed into new bulk X5*X8*X19*X20).</span></span>
<span class="line"><span style="color:#0000FF">        Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span><span style="color:#0000FF">           X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X3</span><span style="color:#0000FF">           Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z3</span><span style="color:#000000">*</span><span style="color:#0000FF">Z4</span><span style="color:#0000FF">    X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#000000">*</span><span style="color:#0000FF">X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X5</span></span>
<span class="line"><span style="color:#0000FF">        X3</span><span style="color:#000000">*</span><span style="color:#0000FF">X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X6</span><span style="color:#000000">*</span><span style="color:#0000FF">X7</span><span style="color:#0000FF">     Z4</span><span style="color:#000000">*</span><span style="color:#0000FF">Z5</span><span style="color:#000000">*</span><span style="color:#0000FF">Z7</span><span style="color:#000000">*</span><span style="color:#0000FF">Z8</span><span style="color:#0000FF">     Z6</span><span style="color:#000000">*</span><span style="color:#0000FF">Z7</span></span>
<span class="line"><span style="color:#008000">        # Patch B (left-edge X 2-body X9*X12 absorbed into new bulk X9*X12*X18*X19).</span></span>
<span class="line"><span style="color:#0000FF">        Z10</span><span style="color:#000000">*</span><span style="color:#0000FF">Z11</span><span style="color:#0000FF">         Z9</span><span style="color:#000000">*</span><span style="color:#0000FF">Z10</span><span style="color:#000000">*</span><span style="color:#0000FF">Z12</span><span style="color:#000000">*</span><span style="color:#0000FF">Z13</span><span style="color:#0000FF">  X10</span><span style="color:#000000">*</span><span style="color:#0000FF">X11</span><span style="color:#000000">*</span><span style="color:#0000FF">X13</span><span style="color:#000000">*</span><span style="color:#0000FF">X14</span><span style="color:#0000FF"> X12</span><span style="color:#000000">*</span><span style="color:#0000FF">X13</span><span style="color:#000000">*</span><span style="color:#0000FF">X15</span><span style="color:#000000">*</span><span style="color:#0000FF">X16</span></span>
<span class="line"><span style="color:#0000FF">        Z13</span><span style="color:#000000">*</span><span style="color:#0000FF">Z14</span><span style="color:#000000">*</span><span style="color:#0000FF">Z16</span><span style="color:#000000">*</span><span style="color:#0000FF">Z17</span><span style="color:#0000FF"> X14</span><span style="color:#000000">*</span><span style="color:#0000FF">X17</span><span style="color:#0000FF">         Z15</span><span style="color:#000000">*</span><span style="color:#0000FF">Z16</span></span>
<span class="line"><span style="color:#008000">        # Four new bulk plaquettes spanning the seam.</span></span>
<span class="line"><span style="color:#0000FF">        Z2</span><span style="color:#000000">*</span><span style="color:#0000FF">Z5</span><span style="color:#000000">*</span><span style="color:#0000FF">Z18</span><span style="color:#000000">*</span><span style="color:#0000FF">Z19</span><span style="color:#0000FF">   X5</span><span style="color:#000000">*</span><span style="color:#0000FF">X8</span><span style="color:#000000">*</span><span style="color:#0000FF">X19</span><span style="color:#000000">*</span><span style="color:#0000FF">X20</span><span style="color:#0000FF">   X9</span><span style="color:#000000">*</span><span style="color:#0000FF">X12</span><span style="color:#000000">*</span><span style="color:#0000FF">X18</span><span style="color:#000000">*</span><span style="color:#0000FF">X19</span><span style="color:#0000FF">  Z12</span><span style="color:#000000">*</span><span style="color:#0000FF">Z15</span><span style="color:#000000">*</span><span style="color:#0000FF">Z19</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span></span>
<span class="line"><span style="color:#008000">        # Two new Z 2-body boundary plaquettes at top and bottom of seam.</span></span>
<span class="line"><span style="color:#0000FF">        Z9</span><span style="color:#000000">*</span><span style="color:#0000FF">Z18</span><span style="color:#0000FF">          Z8</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MergeBegin</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">    I</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#008000">  # so that data-qubit error can be injected</span></span>
<span class="line"><span style="color:#795E26">    RX</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z2</span><span style="color:#000000">*</span><span style="color:#0000FF">Z5</span><span style="color:#000000">*</span><span style="color:#0000FF">Z18</span><span style="color:#000000">*</span><span style="color:#0000FF">Z19</span><span style="color:#008000">       # M0</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X5</span><span style="color:#000000">*</span><span style="color:#0000FF">X8</span><span style="color:#000000">*</span><span style="color:#0000FF">X19</span><span style="color:#000000">*</span><span style="color:#0000FF">X20</span><span style="color:#008000">       # M1</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X9</span><span style="color:#000000">*</span><span style="color:#0000FF">X12</span><span style="color:#000000">*</span><span style="color:#0000FF">X18</span><span style="color:#000000">*</span><span style="color:#0000FF">X19</span><span style="color:#008000">      # M2</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z12</span><span style="color:#000000">*</span><span style="color:#0000FF">Z15</span><span style="color:#000000">*</span><span style="color:#0000FF">Z19</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span><span style="color:#008000">     # M3</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z9</span><span style="color:#000000">*</span><span style="color:#0000FF">Z18</span><span style="color:#008000">              # M4</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z8</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span><span style="color:#008000">              # M5</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> M0</span><span style="color:#001080"> M3</span><span style="color:#001080"> M4</span><span style="color:#001080"> M5</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> MergedSurface</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MergedSE</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> MergedSurface</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">    I</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span><span style="color:#008000">  # so that data-qubit error can be injected</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z2</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X0</span><span style="color:#000000">*</span><span style="color:#0000FF">X3</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z0</span><span style="color:#000000">*</span><span style="color:#0000FF">Z1</span><span style="color:#000000">*</span><span style="color:#0000FF">Z3</span><span style="color:#000000">*</span><span style="color:#0000FF">Z4</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X1</span><span style="color:#000000">*</span><span style="color:#0000FF">X2</span><span style="color:#000000">*</span><span style="color:#0000FF">X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X5</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X3</span><span style="color:#000000">*</span><span style="color:#0000FF">X4</span><span style="color:#000000">*</span><span style="color:#0000FF">X6</span><span style="color:#000000">*</span><span style="color:#0000FF">X7</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z4</span><span style="color:#000000">*</span><span style="color:#0000FF">Z5</span><span style="color:#000000">*</span><span style="color:#0000FF">Z7</span><span style="color:#000000">*</span><span style="color:#0000FF">Z8</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z6</span><span style="color:#000000">*</span><span style="color:#0000FF">Z7</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z10</span><span style="color:#000000">*</span><span style="color:#0000FF">Z11</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z9</span><span style="color:#000000">*</span><span style="color:#0000FF">Z10</span><span style="color:#000000">*</span><span style="color:#0000FF">Z12</span><span style="color:#000000">*</span><span style="color:#0000FF">Z13</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X10</span><span style="color:#000000">*</span><span style="color:#0000FF">X11</span><span style="color:#000000">*</span><span style="color:#0000FF">X13</span><span style="color:#000000">*</span><span style="color:#0000FF">X14</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X12</span><span style="color:#000000">*</span><span style="color:#0000FF">X13</span><span style="color:#000000">*</span><span style="color:#0000FF">X15</span><span style="color:#000000">*</span><span style="color:#0000FF">X16</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z13</span><span style="color:#000000">*</span><span style="color:#0000FF">Z14</span><span style="color:#000000">*</span><span style="color:#0000FF">Z16</span><span style="color:#000000">*</span><span style="color:#0000FF">Z17</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X14</span><span style="color:#000000">*</span><span style="color:#0000FF">X17</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z15</span><span style="color:#000000">*</span><span style="color:#0000FF">Z16</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z2</span><span style="color:#000000">*</span><span style="color:#0000FF">Z5</span><span style="color:#000000">*</span><span style="color:#0000FF">Z18</span><span style="color:#000000">*</span><span style="color:#0000FF">Z19</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X5</span><span style="color:#000000">*</span><span style="color:#0000FF">X8</span><span style="color:#000000">*</span><span style="color:#0000FF">X19</span><span style="color:#000000">*</span><span style="color:#0000FF">X20</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> X9</span><span style="color:#000000">*</span><span style="color:#0000FF">X12</span><span style="color:#000000">*</span><span style="color:#0000FF">X18</span><span style="color:#000000">*</span><span style="color:#0000FF">X19</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z12</span><span style="color:#000000">*</span><span style="color:#0000FF">Z15</span><span style="color:#000000">*</span><span style="color:#0000FF">Z19</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z9</span><span style="color:#000000">*</span><span style="color:#0000FF">Z18</span></span>
<span class="line"><span style="color:#795E26">    MPP</span><span style="color:#0000FF"> Z8</span><span style="color:#000000">*</span><span style="color:#0000FF">Z20</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> MergedSurface</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MergeEnd</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> MergedSurface</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span></span>
<span class="line"></span>
<span class="line"><span style="color:#795E26">    I</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span><span style="color:#008000">  # so that data-qubit error can be injected</span></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#098658"> 18</span><span style="color:#098658"> 19</span><span style="color:#098658"> 20</span><span style="color:#008000">     # M0 M1 M2</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span><span style="color:#098658"> 5</span><span style="color:#098658"> 6</span><span style="color:#098658"> 7</span><span style="color:#098658"> 8</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 9</span><span style="color:#098658"> 10</span><span style="color:#098658"> 11</span><span style="color:#098658"> 12</span><span style="color:#098658"> 13</span><span style="color:#098658"> 14</span><span style="color:#098658"> 15</span><span style="color:#098658"> 16</span><span style="color:#098658"> 17</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> ComposeMZZR</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    MergeBegin</span><span style="color:#0000FF"> IN</span><span style="color:#000000">(</span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#000000">) </span><span style="color:#0000FF">OUT</span><span style="color:#000000">(</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">%</span><span style="color:#AF00DB"> if</span><span style="color:#000000FF"> inner_rounds </span><span style="color:#000000">></span><span style="color:#098658"> 0</span><span style="color:#000000FF">:</span></span>
<span class="line"><span style="color:#AF00DB">    REPEAT</span><span style="color:#098658"> ${inner_rounds}</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">        MergedSE</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">    }</span></span>
<span class="line"><span style="color:#AF00DB">%</span><span style="color:#000000FF"> endif</span></span>
<span class="line"><span style="color:#795E26">    MergeEnd</span><span style="color:#0000FF"> IN</span><span style="color:#000000">(</span><span style="color:#098658">0</span><span style="color:#000000">) </span><span style="color:#0000FF">OUT</span><span style="color:#000000">(</span><span style="color:#098658">0</span><span style="color:#098658"> 1</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#008000">    # MRZZ → MZZ correction: flip patch B's logical-Z frame whenever</span></span>
<span class="line"><span style="color:#008000">    # the joint-parity readout R0 (from MergeBegin, at rec[-1] here)</span></span>
<span class="line"><span style="color:#008000">    # is 1.  Same role as the CONDITIONAL R0 OUT1.LX0 byproduct</span></span>
<span class="line"><span style="color:#008000">    # inside the single-round MZZ.</span></span>
<span class="line"><span style="color:#0000FF">    CONDITIONAL</span><span style="color:#001080"> rec[-1]</span><span style="color:#267F99"> X0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> SurfaceCode</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> ComposeMZZRMemoryZ</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    PrepareZ</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    ComposeMZZR</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MeasureZ</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-3]</span><span style="color:#098658"> 0</span><span style="color:#008000">   # joint LZ_A*LZ_B parity = +1</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-2]</span><span style="color:#098658"> 0</span><span style="color:#008000">   # MeasureZ patch A</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span><span style="color:#008000">   # MeasureZ patch B</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/lattice-surgery/02_ls_merge_multi_round.deq -->

Because `02_ls_merge_multi_round.deq` `IMPORT`s
`00_lattice_surgery_library.deq`, both files must be noise-injected at the
same $p$ and the noisy multi-round file's `IMPORT` rewired to the noisy
library:

```sh
deq inject si1000 00_lattice_surgery_library.deq --p 1e-4 \
    --out 00_lattice_surgery_library.noisy.deq
deq inject si1000 02_ls_merge_multi_round.deq --p 1e-4 --mako r=3 \
    --out 02_ls_merge_multi_round.r3.noisy.deq
sed -i 's|"00_lattice_surgery_library.deq"|"00_lattice_surgery_library.noisy.deq"|' \
    02_ls_merge_multi_round.r3.noisy.deq

deq simulate ler 02_ls_merge_multi_round.r3.noisy.deq \
    --program ComposeMZZRMemoryZ \
    --shots 30000000 --errors 1000 --batch-size 5000 --seed 42
```

Sweeping the round count $r$ against the single-round baseline
($r = 1$, from `ComposeMZZ` in `00_lattice_surgery_library.deq`) at five
noise rates:

| Physical rate $p$   | LER ($r = 1$, single-round) | LER ($r = 3$)         | LER ($r = 5$)         |
| ------------------- | --------------------------- | --------------------- | --------------------- |
| $1.0\times 10^{-3}$ | $7.44 \times 10^{-3}$       | $4.59 \times 10^{-4}$ | $4.62 \times 10^{-4}$ |
| $5.0\times 10^{-4}$ | $3.60 \times 10^{-3}$       | $1.13 \times 10^{-4}$ | $1.11 \times 10^{-4}$ |
| $3.0\times 10^{-4}$ | $2.07 \times 10^{-3}$       | $4.47 \times 10^{-5}$ | $4.17 \times 10^{-5}$ |
| $2.0\times 10^{-4}$ | $1.32 \times 10^{-3}$       | $1.75 \times 10^{-5}$ | $1.90 \times 10^{-5}$ |
| $1.0\times 10^{-4}$ | $6.57 \times 10^{-4}$       | $4.74 \times 10^{-6}$ | $4.44 \times 10^{-6}$ |

(Target of 1000 logical errors per row with `--errors 1000` and
`--seed 42`; per-row shot counts range from $\sim 2 \times 10^5$ at the
highest noise rate up to $\sim 2 \times 10^8$ at the lowest.)

The $r = 1$ column reproduces the non-FT result: $\mathrm{LER} \approx 7 p$
across the whole range.  The $r = 3$ column instead scales as $\mathrm{LER}
\propto p^2$ — the classic $d = 3$ surface-code suppression restored.  At
$p = 10^{-4}$ the $r = 3$ merge reaches $\approx 5 \times 10^{-6}$, **more
than an order of magnitude below physical** — the same regime the
single-patch FT memory in `surface_code_d3_noisy.deq` achieves, and the
strongest signature that the merge is now genuinely fault-tolerant.

---

## Summary

| Concept                                                | Purpose                                                                                                              |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- |
| Physical/logical ambiguity                             | A single physical body (six MPPs + MX seam) is consistent with two inequivalent logical actions (`MZZ` vs `MRZZ`); deq's auto-derivation silently picks the `MRZZ` reading unless the user overrides it |
| `PROPAGATE OUT1.LX0 FROM IN1.LX0`                      | Pins the branch-dependent frame flip, selecting the honest `MZZ` reading (individual $\bar Z$ frames left alone).  Equivalent to `CONDITIONAL R0 OUT1.LX0`, chosen here because it matches the compiled form directly |
| Auto-derived joint $\bar X_A \bar X_B$ preservation     | The flow solver finds the joint-XX invariant on its own and anchors it on `OUT0.LZ0` (with the mirror empty `OUT1.LZ0` row) — no user-written `PROPAGATE` needed |
| Empirical calibration for the fixed-port choice        | Product-state discriminators (`ProductZZ_VirtualXA`, `ProductZZ_VirtualXB`) with deterministic outcomes falsify the wrong port |
| Single-round MZZ                                       | Structurally correct after the byproduct, but $\mathrm{LER} \approx 7 p$ across all noise rates — not fault-tolerant |
| `MergeBegin` / `MergedSE` / `MergeEnd` refactor        | Repeated merge measurements give the decoder temporally local edges, restoring $\mathrm{LER} \propto p^2$ at $d = 3$ (see [Composing Gadgets with COMPOSE](compose-gadgets.md) for the REPEAT mechanics) |
| LER at $d = 3$, $p = 10^{-4}$                          | $r = 1$: $\approx 7 \times 10^{-4}$ (above physical); $r = 3$: $\approx 5 \times 10^{-6}$ (more than an order of magnitude below physical) |

Related chapters:

- [Conditional Pauli Corrections: the `CONDITIONAL` Statement](conditional-correction.md) — the CONDITIONAL mechanic itself; this chapter builds on it.
- [Logical operations with multiple inputs and outputs](multi-port-gadgets.md) — the general multi-port framework that MZZ instantiates.
