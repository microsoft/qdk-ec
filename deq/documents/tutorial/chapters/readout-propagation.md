# READOUT Propagation: Logical Dependency vs. Physical Flow

When you inspect an `annotate`-emitted `.deq` file for a chained composition,
you'll sometimes see `READOUT` lines carrying more than just `M<i>` tokens:

[Annotated `ExerciseReadoutConditions`: `READOUT` line with logical and destabilizer input-frame tokens](../../../tests/circuit/repetition_code/exercise_readout_conditions.annotated.deq#L303)
<!-- deq-highlight-begin: ../../../tests/circuit/repetition_code/exercise_readout_conditions.annotated.deq#L303 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> M2</span><span style="color:#001080"> M5</span><span style="color:#800000"> IN0.LX0</span><span style="color:#267F99"> IN0.DS0</span><span style="color:#267F99"> IN0.DS1</span><span style="color:#008000">  # IN1.LX0 IN1.DS0 IN1.DS1</span></span></code></pre>
<!-- deq-highlight-end: ../../../tests/circuit/repetition_code/exercise_readout_conditions.annotated.deq#L303 -->

The bare `M2 M5` looks like the sort of thing you'd hand-write for a
measurement-basis readout — but why the extras `IN0.LX0 IN0.DS0 IN0.DS1`?
And what does the trailing `# IN1.LX0 IN1.DS0 IN1.DS1` comment mean?

This chapter tells the story of what a `READOUT` statement really declares,
why the transpiler sometimes needs explicit input-frame tokens on that line,
and how to read them when you see them.  A single fixture built on the
[[3,1,1]] repetition code —
`tests/circuit/repetition_code/exercise_readout_conditions.deq` — is enough
to see every mechanism at work.

---

## The two ingredients of a compiled readout

At the binary level, every readout $k$ has three companion pieces of data:

* `measurement_indices` — the list of internal physical measurements whose XOR
  forms the raw readout bit;
* `readout_propagation` (`rp`) — one row per readout, encoding which **input
  frame bits** also flip that readout when set on the input;
* an implicit affine `FLIP` bit (last column of `rp`).

The runtime computes each shot as

$$\text{readouts}[k] \;=\; \bigoplus_{i \in \text{measurement\_indices}[k]} \text{raw}_i
    \;\oplus\; \bigoplus_{c \in \text{rp row}[k]} \text{input}_c
    \;\oplus\; \text{decoded}_k .$$

The `measurement_indices` come directly from the `M<i>` tokens on the source
`READOUT` line — that's straightforward.  The interesting piece is the `rp`
row: which columns of the input frame end up in it, and how do they get
there?

## The input frame: logical observables and stabilizer generators

Each input port contributes a fixed set of columns to the frame:

$$[\underbrace{LX_0, LZ_0, \ldots, LX_{k-1}, LZ_{k-1}}_{\text{logical observables}},
\ \underbrace{S_0, S_1, \ldots, S_{g-1}}_{\text{stabilizer generators}}]$$

concatenated across input ports.  Two flavors of column can appear in an `rp`
row:

* A **logical observable column** (`IN<p>.LZ<i>` / `IN<p>.LX<i>`) means the
  readout flips when the corresponding logical Pauli is applied on the input
  patch.
* A **stabilizer generator column** (`IN<p>.DS<s>`) means the readout flips
  when the corresponding stabilizer's *destabilizer* Pauli is applied on the
  input — i.e., when the input frame has drifted off code space in that
  stabilizer's direction.

For a valid codestate arriving at the gadget's input, every stabilizer-generator
bit is $0$: that's what "in code space" means at the frame level, and the
destabilizer contributions to $\text{rp} \cdot \text{input}$ vanish exactly.
The destabilizer entries earn their keep in compositions: when a preceding
sub-gadget's residual (its `cp · input + pc · raw + lc · readouts`) carries
a non-zero stabilizer-generator bit forward, that bit rides in as the current
gadget's `input`.  A destabilizer entry on the current gadget's `rp` then
correctly toggles the readout by exactly the frame-drift amount that would
otherwise leak into the observed parity — keeping sub-gadget composition
arithmetic closed at the frame level.

## Two sources of `rp` columns

The transpiler builds each `rp` row by XORing two independently derived sets of
columns:

1. **Explicit tokens** on the source `READOUT` line contribute their columns
   directly.  Three families of token are accepted:
   `IN<p>.LX<i>` / `IN<p>.LZ<i>` for logical corrections, and
   `IN<p>.DS<s>` for destabilizers.
2. **Walker-implicit tokens** — the transpiler runs a Heisenberg walker
   (`compute_implicit_readout_propagation`) that pushes each input frame
   column's Pauli representative *forward through the gadget body* and records
   which measurements it anti-commutes with.  If the walked Pauli anti-commutes
   with an odd number of the readout's `measurement_indices`, that column is
   added.  The walker walks all input frame columns — both logical corrections
   and destabilizers.

The final `rp` row is `walker_cols XOR explicit_cols`.  This XOR is the key
mechanism, and it exists precisely so that in most cases user get the correct input frame contributions but in certain complicated cases, user can still override the bits.

## The common case: walker suffices

For most gadgets — anything whose physical body doesn't erase input observables
before the readout's measurements — the walker output *is* the `rp` row, and
no explicit tokens are needed on the `READOUT` line.  Consider
`MeasureZAlternative` from our fixture:

[`MeasureZAlternative` gadget](../../../tests/circuit/repetition_code/exercise_readout_conditions.deq#L18-L22)
<!-- deq-highlight-begin: ../../../tests/circuit/repetition_code/exercise_readout_conditions.deq#L18-L22 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> MeasureZAlternative</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span><span style="color:#008000"> # use a different representative</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../../../tests/circuit/repetition_code/exercise_readout_conditions.deq#L18-L22 -->

The [[3,1,1]] repetition code declares its logical `bar Z` representative as
`Z_0` (see `LOGICAL X0*X1*X2 Z0` in `repetition_code_d3.deq`).  This gadget
instead reads `rec[-1] = M2` — the measurement of qubit 2 — which samples the
operator $Z_2$.  These are the *same* logical operator up to a product of
stabilizers: $Z_2 = Z_0 \cdot S_0 \cdot S_1$ using $S_0 = Z_0 Z_1$ and
$S_1 = Z_1 Z_2$.

When the physical operator you measure differs from the declared
representative by a product of stabilizers, the compiled `rp` records that
difference by picking up the corresponding destabilizer columns.  The
annotator shows it explicitly:

[Annotated `MeasureZAlternative`: `READOUT` line with destabilizer entries in the compiled `rp`](../../../tests/circuit/repetition_code/exercise_readout_conditions.annotated.deq#L280)
<!-- deq-highlight-begin: ../../../tests/circuit/repetition_code/exercise_readout_conditions.annotated.deq#L280 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span><span style="color:#008000">  # IN0.LX0 IN0.DS0 IN0.DS1</span></span></code></pre>
<!-- deq-highlight-end: ../../../tests/circuit/repetition_code/exercise_readout_conditions.annotated.deq#L280 -->

The trailing `# IN0.LX0 IN0.DS0 IN0.DS1` comment shows the compiled `rp` row
in human-readable form: three columns — the logical `IN0.LX0` (flip the
readout when the input has a logical X applied, since that flips the $Z_0$
eigenvalue) and the two destabilizer columns picked up by the representative
shift.  No explicit tokens on the source line: the walker on this flat body
correctly reproduces all three columns just by tracing each input-frame
Pauli's forward flow through the `M 0 1 2` gates.  Compiled `rp` = walker
output, diff is empty.

## The compose case: `CONDITIONAL` hides the flow

Now put two `MeasureZAlternative` calls together with a `CONDITIONAL` in
between:

[`ExerciseReadoutConditions` compose](../../../tests/circuit/repetition_code/exercise_readout_conditions.deq#L24-L30)
<!-- deq-highlight-begin: ../../../tests/circuit/repetition_code/exercise_readout_conditions.deq#L24-L30 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">COMPOSE</span><span style="color:#795E26"> ExerciseReadoutConditions</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> RepetitionCode</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    MeasureZAlternative</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    CONDITIONAL</span><span style="color:#001080"> rec[-1]</span><span style="color:#267F99"> X0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    MeasureZAlternative</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../../../tests/circuit/repetition_code/exercise_readout_conditions.deq#L24-L30 -->

The two patches are physically independent: `MeasureZAlternative 0` runs on
qubits 0–2, `MeasureZAlternative 1` runs on qubits 3–5, and no joint
measurement or entangling gate crosses the patches.  The only link between
them is a **classical feed-forward**: the `CONDITIONAL rec[-1] X0 1` reads
the first readout's classical value and, if it is 1, applies an X gate to
patch 1's qubit 0 before the second readout runs.

At compose-flatten time, each sub-gadget's `rp` matrix is folded into the
composed body's `rp`.  For the *second* `MeasureZAlternative` the flattener
sees two contributions to its rp row:

1. The leaf's own row on its input patch: `{IN1.LX0, IN1.DS0, IN1.DS1}`.
2. The classical feed-forward means the second readout's value ends up
   depending on whatever the first readout depended on: step 9 of the
   canonical compose-flatten pass
   (see [`canonical.merge`](../design/design-transpiler-jit.md#canonical-merge))
   folds the `CONDITIONAL` into the running `cp/pc/lc` formula.  Effectively,
   the first readout's rp row `{IN0.LX0, IN0.DS0, IN0.DS1}` is XOR'd into
   the second readout's row too, and the first readout's `M2` measurement
   joins the second readout's `measurement_indices`.

The composed binary `rp` for the second readout is therefore
`{IN0.LX0, IN0.DS0, IN0.DS1, IN1.LX0, IN1.DS0, IN1.DS1}` — all six columns —
with `measurement_indices = {M2, M5}`.

But when the annotator inlines this compose into a flat body, the walker sees
only what the physical M gates literally sample.  Patch 0's data qubits are
measured out by `M 0 1 2` before the walker reaches the second readout's
`M 3 4 5`.  Walking `IN0.LX0`, `IN0.DS0`, `IN0.DS1` forward past `M 0 1 2`
gives an empty Pauli — the walker records nothing for those columns on the
second readout.  For patch 1 the walker still sees
`{IN1.LX0, IN1.DS0, IN1.DS1}` correctly.

So walker and binary disagree: walker sees three patch-1 columns, binary
insists on all six.

## Reconciling the disagreement: what "input dependency" means

The disagreement is real and it has a clean interpretation:

* The **binary `rp`** is a *logical* statement about the composed operation.
  It is built by folding sub-gadget `rp` matrices together at compose-flatten
  time: the `CONDITIONAL`'s classical feed-forward contributes the first
  readout's dependencies to the second readout's row.
* The **walker's `rp`** is a *physical* statement about the flat body.  It
  follows each input's Pauli representative through actual `R`, `CX`, `M`,
  and `MPP` operations, tracking exactly which measurement outcomes carry
  that Pauli's eigenvalue.  When physical measurements erase the
  representative before subsequent measurements, the walker records no
  dependency — because at the physical measurement level, those later
  outcomes truly *are* independent of the erased input.

Both views are correct at their level.  What makes the compose consistent is
not the physical persistence of patch 0's Paulis but the combination of the
first M-measurement's outcome + the `CONDITIONAL`'s frame update + the runtime
formula `residual = cp · input + pc · raw + lc · readouts`.  The `rp` matrix
is one input to that runtime formula and it must encode the logical
dependency for the frame math to work out.

## Explicit tokens: how the annotator bridges the two views

Given the physical/logical mismatch, the annotator's job is to emit a source
`READOUT` line whose *re-parsed* `rp` row matches the binary.  Because the
transpiler XORs walker and explicit contributions:

$$\text{explicit tokens} \;=\; \text{walker\_cols} \;\oplus\; \text{binary\_cols}$$

$$\Rightarrow\; \text{rp on re-parse} \;=\; \text{walker\_cols} \oplus \text{explicit} \;=\; \text{binary\_cols}$$

For `ExerciseReadoutConditions`'s second readout,
`walker_cols = {IN1.LX0, IN1.DS0, IN1.DS1}` and `binary_cols` is the full
six-column set, so the annotator emits `IN0.LX0 IN0.DS0 IN0.DS1` as explicit
tokens alongside `M2 M5`:

[Annotated `ExerciseReadoutConditions` readouts](../../../tests/circuit/repetition_code/exercise_readout_conditions.annotated.deq#L302-L303)
<!-- deq-highlight-begin: ../../../tests/circuit/repetition_code/exercise_readout_conditions.annotated.deq#L302-L303 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> M2</span><span style="color:#008000">  # IN0.LX0 IN0.DS0 IN0.DS1</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> M2</span><span style="color:#001080"> M5</span><span style="color:#800000"> IN0.LX0</span><span style="color:#267F99"> IN0.DS0</span><span style="color:#267F99"> IN0.DS1</span><span style="color:#008000">  # IN1.LX0 IN1.DS0 IN1.DS1</span></span></code></pre>
<!-- deq-highlight-end: ../../../tests/circuit/repetition_code/exercise_readout_conditions.annotated.deq#L302-L303 -->

Read the second line as three groups:

1. `M2 M5` — physical measurement refs (`measurement_indices`).
2. `IN0.LX0 IN0.DS0 IN0.DS1` — explicit input-frame tokens patching the
   walker's blind spot on patch 0.
3. `# IN1.LX0 IN1.DS0 IN1.DS1` — a trailing comment showing the *remaining*
   rp bits, i.e. the ones the walker still handles physically on patch 1.

Semantically, the compiled `rp` row for this readout is the XOR of everything
on the line (explicit tokens) with everything in the comment (walker-implicit
tokens).  On re-transpile the walker still sees only the patch-1 bits, the
explicit tokens XOR the patch-0 bits back in, and the compiled `rp`
reconstructs byte-for-byte.  Without this fix-up,
`test_annotate_exercise_readout_conditions_destab_readout` in
[tests/circuit/test_annotate.py](../../../tests/circuit/test_annotate.py)
would fail its byte-identity assertion.

The mechanism is uniform across all three token families: the DEQ grammar
accepts `IN<p>.LX<i>` / `IN<p>.LZ<i>` / `IN<p>.DS<s>` on `READOUT` lines
precisely because the walker/binary XOR-patch identity applies to any input
frame column, not just logical ones.  The same fix-up pattern shows up at
much larger scale in surface-code lattice surgery — the `MZZ` merge's
joint-Z parity operator differs from each patch's declared $\bar{Z}$
representative by several patch stabilizers, so its `rp` row picks up
`IN<p>.DS<s>` entries alongside the two logical columns, and any compose
that carries `MZZ`'s dependencies past the walker's physical horizon (via
qubit reuse, reset, or a `CONDITIONAL`) needs explicit destabilizer tokens
in exactly the same shape as the [[3,1,1]] example above.

## When should you write explicit input tokens by hand?

For most hand-written leaf `GADGET` blocks: never. The walker sees whatever
your circuit does with each input observable, and that's what your `rp`
should be.  Just list the `rec[-k]` measurement refs on the `READOUT` line.

You need explicit tokens whenever the walker's *physical* view of your gadget
body diverges from the *logical* dependency you want the readout to have. In
practice this shows up when:

* You are hand-writing a gadget body that mimics a compiled compose (e.g.
  copying the annotator's output as a starting point).  Any qubit reuse,
  reset, or `CONDITIONAL` absorption that erases an input observable before
  a subsequent measurement leaves the walker blind to that observable's
  logical role.
* You are declaring a readout that *should* logically track an input
  observable even though the physical circuit doesn't explicitly measure it
  (e.g. because a subsequent correction cancels the erasure).

In both cases, the rule is the same: put the missing input-frame label
directly on the `READOUT` line.  Use `IN<p>.LX<i>` / `IN<p>.LZ<i>` for a logical correction
and `IN<p>.DS<s>` for a destabilizer.  The
transpiler XORs them with the walker's output the same way in both cases.

## Signal in `annotate` output: extra tokens on `READOUT` lines

Conversely, when you *read* an annotated file and notice extra tokens on a
`READOUT` line beyond the `M<i>` measurement refs, that's a signal that the
compose's compiled `rp` carries a dependency the walker's physical view has
lost — usually via qubit reuse, reset, or a `CONDITIONAL` absorption.  The
trailing `# ...` comment always shows the walker's remaining view; the extra
tokens on the line are the annotator's minimum-diff patch to make the row
survive round-trip.  The compiled `rp` row is the XOR of the two.
