# Pauli-envelope loss decoding

The Pauli-envelope method was introduced by Pengyu Liu, Shi Jie Samuel Tan,
Eric Huang, Umut A. Acar, Hengyun Zhou, and Chen Zhao in
[*Achieving Optimal-Distance Atom-Loss Correction via Pauli
Envelope*](https://arxiv.org/abs/2603.04156) (arXiv:2603.04156). This chapter
describes deq's implementation of their method in a systematic manner.

The [QDK loss-simulation chapter](qdk-loss-simulation.md) explains how
`LOSS_ERROR(p)` is sampled and how a lost measurement reaches the runtime as a
bit in `Outcomes.loss_mask`. This chapter starts at the other end of that wire:
how deq compiles the possible effects of a physical loss, how a herald selects
possible loss sites at runtime, and what a decoder receives.

The implementation has two deliberately separate parts:

1. A **qubit-platform loss model** says what physical operations do when one of their
   operands is absent.
2. A **decoder strategy** decides how to use the resulting Pauli envelope after
   a loss is observed.

Keeping those choices separate lets the same offline transpiler support neutral atoms,
trapped ions, custom hardware models, ordinary black-box decoders, and decoders
that reason about loss explicitly.

## Before you start

This chapter assumes familiarity with [deq gadgets and circuit
syntax](language-basics.md) and with the QDK loss-sampling path described in
[Loss-aware simulation with the QDK backend](qdk-loss-simulation.md). All shell
commands below assume the current directory is the repository's `deq/`
directory. The QDK simulation commands additionally require the QDK Python
package used in the preceding chapter.

## The central idea: store generators, not every Pauli pattern

Loss is not itself a Pauli error. Nevertheless, its effect on checks, readouts,
and output observables can be bounded by a set of Pauli errors. deq represents
that set as a linear span, ignoring global Pauli phase.

For one qubit, the two generators $X$ and $Z$ represent all four possibilities:
$I$, $X$, $Z$, and $Y=XZ$ up to phase. Storing those two generators is enough.
More generally,

$$
\mathcal{E} = \operatorname{span}_{\mathrm{GF}(2)}
    \{g_1, g_2, \ldots, g_r\}.
$$

Each GF(2) coefficient says whether a generator participates in the product.
With $r$ independent generators, this compact basis represents up to $2^r$
patterns. A more specific physical response can use a smaller basis. For
example, the trapped-ion model represents a residual $S^\dagger$ on a surviving
qubit by the envelope $\{I,Z\}$, which needs only the generator $Z$.

Each generator is propagated through the remaining Clifford circuit and
projected onto the gadget's decoder-visible **footprint**:

- finished and unfinished checks,
- readouts,
- output logical and stabilizer-frame residuals.

Generators with the same footprint share one error row. The result is compact
and sampling-free: offline transpilation never enumerates all products in the
envelope. More importantly, the online decoding doesn't need to analyze
Clifford circuit at all.

## Offline transpilation: build a gadget-local loss-event DAG

For each nonzero `LOSS_ERROR(p) q`, the loss analyzer creates a candidate source
event. It then makes one forward pass over the flattened gadget body. At each
gate occurrence, it passes one `LossGate` to the selected platform model while
retaining the state of every active loss event. This is a gadget-wide traversal
implemented as gate-by-gate dispatch, not a separate suffix walk for every
candidate event. During that pass it records:

- **source generators**, active only if the loss starts at this event;
- **continuation generators**, active when a loss that started here or earlier
  reaches this point;
- **herald measurements**, whose QDK result can be `Loss`;
- **successor events**, later candidate first-loss locations that share the same
  remaining lifetime;
- **output physical qubits**, where an unresolved loss leaves the gadget.

Gate boundaries are positions in Stim's decomposed primitive-operation stream,
not source-line numbers. Gates handled directly by a loss model are listed in
its `native_gates`; here “native” means native to that model's dispatch surface,
not necessarily native to the hardware. Other gates are decomposed first. This
gives custom models control at the source-gate level without requiring every
model to reimplement Stim's Clifford decomposition.

Each Pauli generator is then passed through the same fault-propagation machinery
used for ordinary `X_ERROR`, `Z_ERROR`, and depolarizing mechanisms. Its decoder
footprint becomes an `ERROR(0.0)` row when that footprint is nonempty.
Probability zero is intentional: the edge is impossible as an ordinary
unheralded error, but its stable index must exist so a loss observation can
activate it later. A generator with no decoder-visible effect produces no row.

Run `deq annotate` to inspect this representation. Generated annotations use the
following vocabulary:

| Syntax | Meaning |
| --- | --- |
| `LOSS(p) ... # Lk` | Candidate source-loss node `Lk` with source probability `p` |
| `SEj` | Error row `Ej` is a source generator for this node |
| `CEj` | Error row `Ej` is active when an already-lost qubit reaches this node |
| `Lm` | Forward child/successor loss node `Lm` |
| `OUTi.Lj` | The unresolved loss leaves on physical qubit `j` of output port `i` |
| `Mi` | Local measurement `Mi` directly heralds this node |
| `LOSS(INi.Lj)` | Continuation template for loss entering physical qubit `j` of input port `i` |
| `Ck` inside `ERROR` | The error flips check `k` |
| `Rk` inside `ERROR` | The error flips readout `k` |

In generated annotations, the original `LOSS_ERROR` is marked
`@SIMULATE_ONLY`: QDK still needs that instruction to sample physical loss, but
the decode view now uses the explicit `LOSS(...)` and `ERROR(0.0)` metadata.
This split prevents the annotated source from inferring the same model twice.

Loss coordinates and Pauli-frame labels look similar but occur in different
grammar contexts. `OUT0.L0` inside `LOSS(...)` means physical qubit 0 of output
port 0. `OUT0.LX0` or `OUT0.LZ0` inside `ERROR` and `PROPAGATE` refers to the X
or Z frame component of logical qubit 0 on that port.

The annotated form is not merely diagnostic text. It is explicit loss metadata
that transpiles back to the same JIT library; the tutorial generator verifies
that byte equivalence for every example below.

> **Reading annotations:** focus first on `LOSS(...)` and `ERROR(0.0)`.
> `CHECK`, `PROPAGATE`, and the statistics footer explain the surrounding
> compiled gadget, but the footer is comments only and does not affect
> compilation.

## Example 1: one qubit, one gate, one herald

Start with a loss, a Hadamard, and a measurement:

[Single-qubit loss source](../examples/pauli-envelope-loss/01_single_qubit.deq)
<!-- deq-highlight-begin: ../examples/pauli-envelope-loss/01_single_qubit.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> SingleQubitLoss</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#795E26">    H</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/pauli-envelope-loss/01_single_qubit.deq -->

Generate the explicit form:

```sh
deq annotate \
  documents/tutorial/examples/pauli-envelope-loss/01_single_qubit.deq \
    --loss-model neutral-atom \
  --out documents/tutorial/examples/pauli-envelope-loss/01_single_qubit.annotated.deq
```

[Generated single-qubit loss metadata](../examples/pauli-envelope-loss/01_single_qubit.annotated.deq)
<!-- deq-highlight-begin: ../examples/pauli-envelope-loss/01_single_qubit.annotated.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#795E26">@GTYPE</span><span style="color:#000000">(</span><span style="color:#098658">1</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#795E26">@CHECKS</span><span style="color:#000000">(</span><span style="color:#A31515">"manual"</span><span style="color:#000000">, </span><span style="color:#001080">verify</span><span style="color:#000000">=</span><span style="color:#098658">0</span><span style="color:#000000">)</span></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> SingleQubitLoss</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    @SIMULATE_ONLY</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#0000FF">    LOSS</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#267F99">SE0</span><span style="color:#267F99"> CE0</span><span style="color:#001080"> M0</span><span style="color:#008000">  # L0</span></span>
<span class="line"><span style="color:#795E26">    H</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.0</span><span style="color:#000000">) </span><span style="color:#001080">R0</span><span style="color:#008000">  # E0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # errors: 1</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 0:1 }</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/pauli-envelope-loss/01_single_qubit.annotated.deq -->

The key lines are:

```text
LOSS(0.1) SE0 CE0 M0  # L0
ERROR(0.0) R0          # E0
```

`L0` can start with probability `0.1`, and `M0` is its loss-resolving
measurement. The source envelope and the continuation envelope after `H` both
project onto the same visible effect, flipping readout `R0`; the pre-measurement
continuation insertion has that footprint too. They therefore reuse error row
`E0`. This is safe footprint deduplication: errors with identical check,
readout, and output-residual effects are statically indistinguishable to the
decoder.

## Example 2: the platform model changes the envelope

The next circuit loses qubit 0 before a `CZ`. It measures the lost operand in Z
and the surviving operand in X, making a residual Z on the survivor visible in
readout `R1`.

[Platform-dependent CZ example](../examples/pauli-envelope-loss/02_platform_cz.deq)
<!-- deq-highlight-begin: ../examples/pauli-envelope-loss/02_platform_cz.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PlatformCz</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#795E26">    CZ</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-2]</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/pauli-envelope-loss/02_platform_cz.deq -->

The built-in models differ at this gate boundary:

| Model | Controlled-gate behavior |
| --- | --- |
| `neutral-atom` | `CX`, `CY`, and `CZ`: `SKIP` |
| `trapped-ion` | `CZ`: residual $S^\dagger$ envelope; source-level `CX`/`CY` rejected |

Annotate the same circuit under both models:

```sh
deq annotate \
  documents/tutorial/examples/pauli-envelope-loss/02_platform_cz.deq \
    --loss-model neutral-atom \
  --out documents/tutorial/examples/pauli-envelope-loss/02_platform_cz.neutral_atom.annotated.deq

deq annotate \
  documents/tutorial/examples/pauli-envelope-loss/02_platform_cz.deq \
    --loss-model trapped-ion \
  --out documents/tutorial/examples/pauli-envelope-loss/02_platform_cz.trapped_ion.annotated.deq
```

[Neutral-atom annotation excerpt](../examples/pauli-envelope-loss/02_platform_cz.neutral_atom.annotated.deq#L3-L17)
<!-- deq-highlight-begin: ../examples/pauli-envelope-loss/02_platform_cz.neutral_atom.annotated.deq#L3-L17 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PlatformCz</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    @SIMULATE_ONLY</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#0000FF">    LOSS</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#267F99">SE0</span><span style="color:#267F99"> CE1</span><span style="color:#001080"> M0</span><span style="color:#008000">  # L0</span></span>
<span class="line"><span style="color:#795E26">    CZ</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-2]</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.0</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#001080"> R0</span><span style="color:#001080"> R1</span><span style="color:#008000">  # E0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.0</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#001080"> R0</span><span style="color:#008000">  # E1</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span></code></pre>
<!-- deq-highlight-end: ../examples/pauli-envelope-loss/02_platform_cz.neutral_atom.annotated.deq#L3-L17 -->

Under the neutral-atom model, a CZ with a missing operand is skipped. The loss
remains on qubit 0, so `M0` is the only loss herald. No partner-only error row is
needed. Both built-in models are effective gate-level approximations; a
hardware-backed model should be derived from the device's actual decomposition,
pulse ordering, and loss-detection timing.

[Trapped-ion annotation excerpt](../examples/pauli-envelope-loss/02_platform_cz.trapped_ion.annotated.deq#L3-L18)
<!-- deq-highlight-begin: ../examples/pauli-envelope-loss/02_platform_cz.trapped_ion.annotated.deq#L3-L18 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PlatformCz</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    @SIMULATE_ONLY</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#0000FF">    LOSS</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#267F99">SE0</span><span style="color:#267F99"> CE1</span><span style="color:#267F99"> CE2</span><span style="color:#001080"> M0</span><span style="color:#008000">  # L0</span></span>
<span class="line"><span style="color:#795E26">    CZ</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-2]</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.0</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#001080"> R0</span><span style="color:#001080"> R1</span><span style="color:#008000">  # E0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.0</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#001080"> R0</span><span style="color:#008000">  # E1</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.0</span><span style="color:#000000">) </span><span style="color:#001080">R1</span><span style="color:#008000">  # E2</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span></code></pre>
<!-- deq-highlight-end: ../examples/pauli-envelope-loss/02_platform_cz.trapped_ion.annotated.deq#L3-L18 -->

The trapped-ion model implements one explicit compiled-CZ approximation: if the
two-body operation is lost but local phase corrections still execute, a
surviving operand can retain $S^\dagger$. Its Pauli envelope is $\{I,Z\}$, so the
annotation adds:

```text
ERROR(0.0) R1  # E2
```

and lists `CE2` on `L0`. This model is intentionally CZ-specific. Source-level
CX and CY are rejected because their residual local rotations depend on a
different hardware compilation. See the [simulation chapter](qdk-loss-simulation.md)
for the physical scope and references behind both presets.

### Plug in a custom physical model

`--loss-model` also accepts a Python file that defines `create_loss_model()`.
The example plugin changes CZ from `SKIP` to `PROPAGATE`, meaning a loss on one
operand branches onto every operand of the gate:

```python
from deq.transpiler.loss import GateLossPolicy, QdkLossConfig
from deq.transpiler.loss.model_neutral_atom import NeutralAtomLossModel


class PropagatingCzLossModel(NeutralAtomLossModel):
    config = QdkLossConfig(
        gate_policies=(
            ("cx", GateLossPolicy.SKIP),
            ("cy", GateLossPolicy.SKIP),
            ("cz", GateLossPolicy.PROPAGATE),
            ("swap", GateLossPolicy.APPLY_ANYWAY),
        )
    )


def create_loss_model() -> PropagatingCzLossModel:
    return PropagatingCzLossModel()
```

The runnable file is
[`custom_loss_model.py`](../examples/pauli-envelope-loss/custom_loss_model.py).
Use it exactly like a built-in model:

```sh
deq annotate \
  documents/tutorial/examples/pauli-envelope-loss/02_platform_cz.deq \
  --loss-model documents/tutorial/examples/pauli-envelope-loss/custom_loss_model.py \
  --out documents/tutorial/examples/pauli-envelope-loss/02_platform_cz.custom.annotated.deq
```

[Custom propagation-model annotation excerpt](../examples/pauli-envelope-loss/02_platform_cz.custom.annotated.deq#L3-L18)
<!-- deq-highlight-begin: ../examples/pauli-envelope-loss/02_platform_cz.custom.annotated.deq#L3-L18 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PlatformCz</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    @SIMULATE_ONLY</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#0000FF">    LOSS</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#267F99">SE0</span><span style="color:#267F99"> CE1</span><span style="color:#267F99"> CE2</span><span style="color:#001080"> M0</span><span style="color:#001080"> M1</span><span style="color:#008000">  # L0</span></span>
<span class="line"><span style="color:#795E26">    CZ</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#098658"> 1</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-2]</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#0000FF">    CHECK</span><span style="color:#001080"> M0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.0</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#001080"> R0</span><span style="color:#001080"> R1</span><span style="color:#008000">  # E0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.0</span><span style="color:#000000">) </span><span style="color:#267F99">C0</span><span style="color:#001080"> R0</span><span style="color:#008000">  # E1</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.0</span><span style="color:#000000">) </span><span style="color:#001080">R1</span><span style="color:#008000">  # E2</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span></code></pre>
<!-- deq-highlight-end: ../examples/pauli-envelope-loss/02_platform_cz.custom.annotated.deq#L3-L18 -->

The custom annotation contains the same partner envelope edge as the trapped-ion
case, but it also lists `M1`: the physical loss branch itself now reaches qubit
1, so either measurement can herald the event.

Subclassing a built-in model is the shortest path when only policies change. A
fully custom model can implement the public `LossModel` protocol directly. The
model receives `LossGate` records and a constrained `LossAnalysisState` with
operations for adding generator insertions, branching or swapping active losses,
recording heralds, and terminating a loss at reset. All per-gadget mutable state
lives in `LossAnalysisState`; model methods must remain stateless. Use
`native_gates` for source gates the model wants to receive without Stim
decomposition. The complete required surface is:

```python
class LossModel:
  config: QdkLossConfig
  native_gates: frozenset[str]

  def handle_loss_source(
    self, event_id: int, state: LossAnalysisState
  ) -> None: ...

  def handle_gate(self, gate: LossGate, state: LossAnalysisState) -> None: ...
```

The plugin file's zero-argument `create_loss_model()` returns that object.

A custom model should describe the actual device compilation and loss-detection
timing. `PROPAGATE` is useful here because it visibly exercises branching; it is
not a claim about neutral-atom CZ physics. Its `config` is also passed to the QDK
sampler by default. Keep those policies consistent with `handle_gate`, or pass an
explicit `--simulation-loss-model` override when deliberately comparing decoder
and simulator assumptions.

## Example 3: one loss lifetime across gadget boundaries

Loss does not stop at a gadget boundary. The following program loses a qubit in
`Start`, exports it through an output port, imports it into `Finish`, and only
then measures it:

[Cross-gadget loss example](../examples/pauli-envelope-loss/03_cross_gadget.deq)
<!-- deq-highlight-begin: ../examples/pauli-envelope-loss/03_cross_gadget.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> Qubit</span><span style="color:#000000"> [[</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#0000FF"> Z0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Start</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#795E26">    H</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Qubit</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Finish</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Qubit</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    H</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Run</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    Start</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Finish</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/pauli-envelope-loss/03_cross_gadget.deq -->

```sh
deq annotate \
  documents/tutorial/examples/pauli-envelope-loss/03_cross_gadget.deq \
    --loss-model neutral-atom \
  --out documents/tutorial/examples/pauli-envelope-loss/03_cross_gadget.annotated.deq
```

[Generated `Start` metadata](../examples/pauli-envelope-loss/03_cross_gadget.annotated.deq#L8-L26)
<!-- deq-highlight-begin: ../examples/pauli-envelope-loss/03_cross_gadget.annotated.deq#L8-L26 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Start</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    @SIMULATE_ONLY</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#0000FF">    LOSS</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#267F99">SE0</span><span style="color:#267F99"> SE1</span><span style="color:#267F99"> CE0</span><span style="color:#267F99"> CE1</span><span style="color:#800000"> OUT0.L0</span><span style="color:#008000">  # L0</span></span>
<span class="line"><span style="color:#795E26">    H</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Qubit</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LZ0</span><span style="color:#0000FF"> FROM</span></span>
<span class="line"><span style="color:#0000FF">    PROPAGATE</span><span style="color:#800000"> OUT0.LX0</span><span style="color:#0000FF"> FROM</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.0</span><span style="color:#000000">) </span><span style="color:#800000">OUT0.LZ0</span><span style="color:#008000">  # E0</span></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.0</span><span style="color:#000000">) </span><span style="color:#800000">OUT0.LX0</span><span style="color:#008000">  # E1</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # errors: 2</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 0:2 }</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/pauli-envelope-loss/03_cross_gadget.annotated.deq#L8-L26 -->

[Generated `Finish` metadata](../examples/pauli-envelope-loss/03_cross_gadget.annotated.deq#L30-L45)
<!-- deq-highlight-begin: ../examples/pauli-envelope-loss/03_cross_gadget.annotated.deq#L30-L45 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Finish</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Qubit</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    H</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span><span style="color:#008000">  # IN0.LZ0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.0</span><span style="color:#000000">) </span><span style="color:#001080">R0</span><span style="color:#008000">  # E0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    LOSS</span><span style="color:#000000">(</span><span style="color:#800000">IN0.L0</span><span style="color:#000000">) </span><span style="color:#267F99">CE0</span><span style="color:#001080"> M0</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # --- statistics ---</span></span>
<span class="line"><span style="color:#008000">    # finished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # unfinished checks: 0</span></span>
<span class="line"><span style="color:#008000">    # errors: 1</span></span>
<span class="line"><span style="color:#008000">    #   check-weight distribution: { 0:1 }</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/pauli-envelope-loss/03_cross_gadget.annotated.deq#L30-L45 -->

The [complete annotation](../examples/pauli-envelope-loss/03_cross_gadget.annotated.deq)
also shows the code, propagation matrices, and statistics.

The relevant pair is:

```text
# Start
LOSS(0.1) ... OUT0.L0  # L0

# Finish
LOSS(IN0.L0) CE0 M0
```

`OUT0.L0` is a flat physical output-port position, not a logical observable.
At runtime, the program connector maps that position to `Finish`'s input slot,
where `LOSS(IN0.L0)` contributes its continuation generator and terminal herald.
A gadget with no local `LOSS_ERROR` therefore still has an input-loss template:
it must describe what happens if a loss entered from an upstream gadget.

The runtime performs the same linking for dynamically instantiated gadgets. A
window coordinator processes only a subset of nearby gadget instances at once.
If a connection leaves that decoding region, its downstream herald is unknown,
not a measured non-loss; the link ends at the boundary and can be assembled in
a later overlapping region that includes the downstream gadget.

## Example 4: the paper's four-CX data-loss chain

The data-loss example from *Achieving Optimal-Distance Atom-Loss Correction via
Pauli Envelope* places five candidate loss sources along one data-qubit
lifetime, separated by four CX gates:

```text
time ->  L0 --CX-- L1 --CX-- L2 --CX-- L3 --CX-- L4 --M0
```

Only one site can be the true first loss. The chain records the five alternatives
and their shared suffix without treating them as five independent atoms.

[Four-CX paper example](../examples/pauli-envelope-loss/04_four_cx.deq)
<!-- deq-highlight-begin: ../examples/pauli-envelope-loss/04_four_cx.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> Reg</span><span style="color:#000000"> [[</span><span style="color:#098658">5</span><span style="color:#000000">,</span><span style="color:#098658">5</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X0</span><span style="color:#0000FF"> Z0</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X1</span><span style="color:#0000FF"> Z1</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X2</span><span style="color:#0000FF"> Z2</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X3</span><span style="color:#0000FF"> Z3</span></span>
<span class="line"><span style="color:#0000FF">    LOGICAL</span><span style="color:#0000FF"> X4</span><span style="color:#0000FF"> Z4</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> FourCx</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Reg</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 4</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Reg</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/pauli-envelope-loss/04_four_cx.deq -->

Generate its explicit metadata with:

```sh
deq annotate \
  documents/tutorial/examples/pauli-envelope-loss/04_four_cx.deq \
    --loss-model neutral-atom \
  --out documents/tutorial/examples/pauli-envelope-loss/04_four_cx.annotated.deq
```

[Generated `L0` through `L4` chain](../examples/pauli-envelope-loss/04_four_cx.annotated.deq#L12-L32)
<!-- deq-highlight-begin: ../examples/pauli-envelope-loss/04_four_cx.annotated.deq#L12-L32 -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> FourCx</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Reg</span><span style="color:#098658"> 0</span><span style="color:#098658"> 1</span><span style="color:#098658"> 2</span><span style="color:#098658"> 3</span><span style="color:#098658"> 4</span></span>
<span class="line"><span style="color:#795E26">    @SIMULATE_ONLY</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#0000FF">    LOSS</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#267F99">SE0</span><span style="color:#267F99"> SE1</span><span style="color:#267F99"> CE0</span><span style="color:#267F99"> CE2</span><span style="color:#267F99"> L1</span><span style="color:#008000">  # L0</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 1</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    @SIMULATE_ONLY</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#0000FF">    LOSS</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#267F99">SE0</span><span style="color:#267F99"> SE2</span><span style="color:#267F99"> L2</span><span style="color:#008000">  # L1</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 2</span></span>
<span class="line"><span style="color:#795E26">    @SIMULATE_ONLY</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#0000FF">    LOSS</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#267F99">SE2</span><span style="color:#267F99"> SE3</span><span style="color:#267F99"> L3</span><span style="color:#008000">  # L2</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 0</span><span style="color:#098658"> 3</span></span>
<span class="line"><span style="color:#795E26">    @SIMULATE_ONLY</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#0000FF">    LOSS</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#267F99">SE2</span><span style="color:#267F99"> SE4</span><span style="color:#267F99"> CE4</span><span style="color:#267F99"> CE5</span><span style="color:#267F99"> L4</span><span style="color:#008000">  # L3</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#098658"> 4</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    @SIMULATE_ONLY</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#098658">0</span></span>
<span class="line"><span style="color:#0000FF">    LOSS</span><span style="color:#000000">(</span><span style="color:#098658">0.1</span><span style="color:#000000">) </span><span style="color:#267F99">SE4</span><span style="color:#267F99"> SE5</span><span style="color:#267F99"> CE4</span><span style="color:#267F99"> CE5</span><span style="color:#001080"> M0</span><span style="color:#008000">  # L4</span></span></code></pre>
<!-- deq-highlight-end: ../examples/pauli-envelope-loss/04_four_cx.annotated.deq#L12-L32 -->

The successor structure is:

```text
L0 -> L1 -> L2 -> L3 -> L4 --M0
```

Only `L4` directly lists `M0`; the runtime folds the herald evidence backward
through the child links, so a loss at `M0` keeps every upstream candidate whose
suffix reaches it.

The source/continuation distinction preserves the physics without expanding all
patterns:

- `SEj` applies only if that node is the true first loss.
- `CEj` applies whenever a loss that started at that node or an ancestor reaches
  the insertion.
- `Lk` says the same atom could instead have first disappeared at the next
  candidate site.

For this five-qubit circuit, the offline transpiler projects the generator basis
into 15 distinct output-residual footprints. That is still a generator basis,
not an enumeration of its exponentially many products. The structured backend
also retains the child chain and direct herald identities, allowing a loss-aware
decoder to distinguish causally conflicting starts from independent branches.

## Runtime loss compilation: link instances and fold heralds

The offline-transpiled `GadgetType.loss_model` is gadget-local and static. It
cannot know which gadget instances will be connected or which instances fall
inside a decoding window. At decode time, the runtime loss compiler glues those
local models together and adds three dynamic facts:

1. which gadget instances are connected in the current decoding region;
2. which measurement positions were reported as loss;
3. which generator error rows became hyperedges in that region.

It instantiates fresh and input-loss nodes, links `child_losses` inside a gadget,
links `child_output_qubits` to downstream `input_losses`, and folds herald
evidence through the resulting DAG. A node is retained when at least one herald
in its forward reach was observed as loss and none was observed as non-loss.
For a window coordinator, a child outside the current decoding region is not
linked into this graph, so its herald contributes neither support nor
contradiction in that region.

The retained sites are then projected onto hyperedge indices. At this point the
representation is strategy-neutral; the coordinator chooses one of three
`loss_strategy` values:

| Strategy | Decoder-facing behavior |
| --- | --- |
| `ignore` | Drop structured loss information after constructing the syndrome |
| `reweight` | Convert each observed envelope into shot-scoped edge probabilities; send an ordinary decoding problem |
| `handoff` | Keep the base hypergraph unchanged and send structured `LossInfo` alongside it |

`reweight` is the default. Before any strategy in this table runs,
`loss_random_imputation` chooses the ordinary outcome bit used to construct the
syndrome at each lost measurement. That step applies under `ignore`, `reweight`,
and `handoff`; the strategy independently decides what happens to the structured
loss observation afterward.

## Backend option 1: ordinary decoder with edge reweights

A loss-only generator enters the hypergraph with prior probability zero. Under
`reweight`, the coordinator accumulates the probability that each retained site
activates each source or continuation edge. With the default `local` scale, it
combines that activation probability with the edge's current prior and lowers
the resulting likelihood weight by a configurable fraction:

$$
w(p) = -\ln\!\left(\frac{p}{1-p}\right), \qquad
w_{\mathrm{target}} = f\times \max\!\left(w(p_e \oplus p_a), 0\right),
$$

where $p_e$ is the live edge prior, $p_a$ is its accumulated loss activation
probability, $p \oplus q = p + q - 2pq$ is the runtime's exclusive-probability
composition, and $f=0.5$ by default. The fraction is applied once per edge after
all activating sites are accumulated, preventing one edge shared by many
candidate sites from becoming accidentally free.

Configure it with:

```json
{
  "loss_strategy": "reweight",
  "loss_config": {
    "weight_fraction": 0.5,
    "scale": "local"
  },
  "loss_random_imputation_seed": 123
}
```

The available scales are `local`, `global_mean`, and `neighbourhood_mean`.
`global_mean` instead uses the mean weight of all nonzero-prior edges in the
live graph. `neighbourhood_mean` uses the mean over nonzero-prior edges sharing
at least one syndrome vertex with the activated edge, falling back to `local`
when there are no such neighbors. Once an edge is activated, these two mean
scales do not otherwise depend on its activation probability; both assign the
edge probability corresponding to $f$ times the selected mean weight.

What the core decoder receives depends on the coordinator's
`decoder_reweighting` transport setting and the decoder's advertised
capabilities:

- A decoder advertising `reweights` can receive
  `LoadedDecodingProblem.reweights`, a list of `(edge, probability)` assignments
  for the current shot.
- Otherwise the coordinator materializes the same probabilities into a one-shot
  `DecodingHypergraph` before calling an ordinary decoder.

Either way, the core decoder does **not** receive `LossInfo` under this strategy.
It solves its usual weighted graph or hypergraph problem.

## Backend option 2: structured loss handoff

`handoff` is for a decoder that wants to enforce loss structure itself. The
runtime does not turn the loss observation into edge weights; it sends the
hypergraph plus `LossInfo`. Unrelated runtime probability modifiers remain
independent, so a decoder advertising both optional features may receive
`reweights` and `loss` in the same request. Each site contains:

| Field | Index domain | Meaning |
| --- | --- | --- |
| `source_edges` | `DecodingHypergraph.hyperedges` | Generators usable only if loss starts at this site |
| `continuation_edges` | `DecodingHypergraph.hyperedges` | Generators usable when loss starts here or at an ancestor and reaches this site |
| `children` | `LossInfo.sites` | Forward links along the same loss lifetime |
| `probability` | scalar | Declared source probability, or zero for an input continuation site |
| `heralds` | window-local herald IDs | Direct observed loss measurements; equal IDs refer to the same measurement across sites |

This is not a second decoding hypergraph embedded inside the first. It is a
small site DAG whose edge fields reference the ordinary hypergraph. A Python
decoder opts in explicitly:

```python
class Decoder:
    @staticmethod
    def supported_features() -> list[str]:
        return ["loss"]

    def decode(self, syndrome, loss=None):
        for site in loss.sites if loss is not None else ():
            print(
                site.source_edges,
                site.continuation_edges,
                site.children,
                site.probability,
                site.heralds,
            )
        # Return selected hyperedge indices.
        return []
```

The embedded `@mle_loss_decoder` is a complete reference implementation. It
uses a mixed-integer program to cover every observed herald, reject source pairs
whose forward loss lifetimes overlap, gate envelope edges by the selected
sources, and satisfy detector parity. Branch siblings with disjoint forward
lifetimes may both start. Because structured loss refers to distinct edge and
herald identities, handoff preserves those identities instead of merging
same-syndrome hyperedges.

## Run both backend paths

The following fixture uses a one-qubit stabilizer code so its loss generators
become real detector hyperedges. `LOSS_ERROR(1)` is deliberate: every shot
contains a loss, making the request examples deterministic. It is a contract
probe, not a physical benchmark.

[Deterministic backend-contract fixture](../examples/pauli-envelope-loss/05_backend_contract.deq)
<!-- deq-highlight-begin: ../examples/pauli-envelope-loss/05_backend_contract.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">CODE</span><span style="color:#267F99"> FixedZero</span><span style="color:#000000"> [[</span><span style="color:#098658">1</span><span style="color:#000000">,</span><span style="color:#098658">0</span><span style="color:#000000">,</span><span style="color:#098658">1</span><span style="color:#000000">]] {</span></span>
<span class="line"><span style="color:#0000FF">    STABILIZER</span><span style="color:#0000FF"> Z0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Start</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(</span><span style="color:#098658">1</span><span style="color:#000000">) </span><span style="color:#098658">0</span><span style="color:#008000">  # Deterministic loss for backend-contract inspection.</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> FixedZero</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Finish</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> FixedZero</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    READOUT</span><span style="color:#001080"> rec[-1]</span></span>
<span class="line"><span style="color:#000000">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">PROGRAM</span><span style="color:#795E26"> Run</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    Start</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#795E26">    Finish</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#0000FF">    ASSERT_EQ</span><span style="color:#001080"> rec[-1]</span><span style="color:#098658"> 0</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/pauli-envelope-loss/05_backend_contract.deq -->

The accompanying
[`inspect_loss_decoder.py`](../examples/pauli-envelope-loss/inspect_loss_decoder.py)
advertises both optional Python-decoder features and prints the first nonempty
payload it receives. `deq simulate ler` captures runtime output, so the commands
use `--debug-dir` and then read `runtime_output.txt`. They require the
`deq_runtime` Python bindings in addition to QDK; build them as described in the
[Python decoder chapter](python-decoder.md) if `black-box-python` is unavailable.

### Inspect reweight transport

```sh
rm -rf /tmp/deq-loss-reweight-doc
deq simulate ler \
    documents/tutorial/examples/pauli-envelope-loss/05_backend_contract.deq \
    --program Run \
    --simulator qdk \
    --loss-model neutral-atom \
    --decoder black-box-python \
    --decoder-config '{
      "file":"documents/tutorial/examples/pauli-envelope-loss/inspect_loss_decoder.py",
      "parallel":1
    }' \
    --coordinator-config '{
      "loss_strategy":"reweight",
      "loss_config":{"weight_fraction":0.5,"scale":"local"},
      "loss_random_imputation_seed":123
    }' \
    --shots 4 --errors 4 --batch-size 4 --jobs 1 \
    --debug-dir /tmp/deq-loss-reweight-doc

grep REWEIGHT_REQUEST /tmp/deq-loss-reweight-doc/runtime_output.txt
# REWEIGHT_REQUEST [{"edge": 0, "probability": 0.5}]
```

The Python decoder receives a shot-scoped assignment for edge 0 and no
`LossInfo`. If `decoder_reweighting` is configured to materialize updates, the
coordinator instead sends an equivalent one-shot hypergraph with that
probability already installed.

### Inspect structured handoff

```sh
rm -rf /tmp/deq-loss-handoff-doc
deq simulate ler \
    documents/tutorial/examples/pauli-envelope-loss/05_backend_contract.deq \
    --program Run \
    --simulator qdk \
    --loss-model neutral-atom \
    --decoder black-box-python \
    --decoder-config '{
      "file":"documents/tutorial/examples/pauli-envelope-loss/inspect_loss_decoder.py",
      "parallel":1
    }' \
    --coordinator-config '{
      "loss_strategy":"handoff",
      "loss_random_imputation_seed":123
    }' \
    --shots 4 --errors 4 --batch-size 4 --jobs 1 \
    --debug-dir /tmp/deq-loss-handoff-doc

grep LOSS_REQUEST /tmp/deq-loss-handoff-doc/runtime_output.txt
# Abbreviated for readability; the actual command emits one JSON line.
# LOSS_REQUEST [
#   {"source_edges":[0], "children":[1], "probability":1.0, "heralds":[], ...},
#   {"continuation_edges":[1], "children":[], "probability":0.0, "heralds":[0], ...}
# ]
```

The source site points at edge 0 and its child points at continuation edge 1.
The inspector deliberately returns no correction, so its reported logical error
rate is not meaningful. To run an actual structured decoder, replace the
decoder configuration with the embedded reference implementation:

```json
{"file":"@mle_loss_decoder"}
```

Both commands use the same compiled platform model and deterministic loss
condition. In general, separate QDK runs do **not** replay identical random
shots: the current upstream QDK sampler ignores its simulator seed. The
strategy comparison here concerns request shape, not paired statistical output.

## Regenerate every example

The checked-in annotations are generated, not edited by hand:

```sh
python documents/tutorial/examples/pauli-envelope-loss/gen_pauli_envelope_loss.py
```

Run this after changing an example, the offline loss transpiler, or a built-in
model. Each call to `deq annotate` retranspiles its output and checks byte
equality with the source; the generator also asserts the `LOSS`/`ERROR`
fragments and row counts explained by this chapter. From `documents/`, `make
tutorial` runs this generator along with all other tutorial generators, then
uses Shiki and deq's TextMate grammar to refresh the syntax-highlighted blocks
in this chapter.

## Summary

The complete path is:

```text
LOSS_ERROR + platform model
  -> offline gadget-local loss-event DAG
    -> probability-zero decoder error rows + static LossModel
  -> runtime loss compiler: instance linking + herald filtering
    -> ordinary edge reweights OR structured LossInfo
    -> core decoder
```

The offline transpiler owns gadget-local physical propagation and Pauli
projection. The runtime loss compiler owns gadget-instance connectivity,
cross-gadget linking, and shot evidence. The decoder owns how strictly to use
the resulting envelope. That separation is what makes the same machinery usable
across platforms and decoder families without baking one hardware model or one
loss-decoding algorithm into deq.
