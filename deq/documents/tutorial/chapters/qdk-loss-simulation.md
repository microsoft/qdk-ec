# Loss-aware simulation with the QDK backend

Neutral-atom and trapped-ion platforms suffer **qubit loss**: a physical
qubit may leave the trap, leak out of the computational subspace, or
otherwise become unavailable to the rest of the circuit.  Loss is **not**
a Pauli error.  A lost qubit can't be acted on by ordinary gates and
can't yield a clean measurement, but exactly *how* the surrounding
circuit degrades depends on the platform — there is no single
standard loss model:

- **Neutral atoms (CZ-native).**  A `CZ` involving a lost atom has no
  effect on its partner; the gate effectively becomes the identity.
- **Trapped ions (MS/XX-native).**  Trapped-ion processors can implement
  [Mølmer–Sørensen $XX$ interactions](https://doi.org/10.1103/PhysRevLett.82.1971)
  and compile logical gates such as `CNOT` and controlled phase from an $XX$
  interaction plus local rotations ($S^{\dagger}$).
- Other platforms (Rydberg blockade variants, leakage to higher
  levels, atom-array transport, …) come with their own variants.

deq packages these gate-by-gate rules as platform loss models, selected with
``--loss-model``:

| Model | Scope | Explicit gate policies |
| --- | --- | --- |
| ``neutral-atom`` | native CZ and its compiled controlled-Pauli aliases | ``CX/CY/CZ → SKIP``; ``SWAP → APPLY_ANYWAY`` |
| ``trapped-ion`` | one explicit compiled-CZ residual-phase approximation | ``CZ → RESIDUAL_S_DAGGER``; ``SWAP → APPLY_ANYWAY`` |
| ``none`` | opt out of loss entirely | none — loss is not modelled |

``none`` is not a physical model. It compiles the circuit as if loss could not
happen: no gadget gets loss metadata, explicit ``LOSS`` statements are ignored,
and ``LOSS_ERROR`` is dropped from the exported Stim circuit so the simulator
never samples loss the decoder cannot explain. Use it when a circuit declares
``LOSS_ERROR`` for another backend, or when its gates fall outside every
built-in platform model's supported scope and you want to study the Pauli noise
alone.

The neutral-atom row does not claim that all three controlled gates are native.
Neutral-atom processors natively realize CZ and obtain CNOT/CX using local
target rotations around CZ.
QDK's neutral-atom compiler likewise lowers ``CX`` to ``H-CZ-H`` and ``CY`` to
``S†-H-CZ-H-S``. If an atom is absent and the native CZ is skipped, the local
wrappers cancel on a surviving target or act only on the absent atom. The
effective operation is therefore SKIP for all three source-level aliases. This
equivalence assumes the loss is already active at the source-gate boundary;
pulse-resolved loss between the local wrappers requires a more detailed model.

The trapped-ion row is deliberately narrower. The native MS entangler is an
$XX(\chi)$ interaction, not CX. Logical CX, CY, and CZ use different local
rotations around that interaction, so losing the interaction does not leave the
same residual operation for all three. The built-in preset specifies one CZ
compilation and rejects source-level CX/CY. Circuits using them must first expose
a supported CZ-plus-local-gates decomposition or supply a custom loss model.

The selector also accepts a Python file.  The file must define a zero-argument
``create_loss_model()`` function returning an object that implements
``LossModel``: a ``QdkLossConfig`` in ``config``, a ``native_gates`` set, and
the stateless ``handle_loss_source()`` and ``handle_gate()`` methods. Subclassing
a built-in model is sufficient when only its configuration changes:

```python
from deq.transpiler.loss import GateLossPolicy, QdkLossConfig
from deq.transpiler.loss.model_neutral_atom import NeutralAtomLossModel


class UserLossModel(NeutralAtomLossModel):
  config = QdkLossConfig(
    gate_policies=(
      ("cx", GateLossPolicy.PROPAGATE),
      ("cy", GateLossPolicy.SKIP),
      ("cz", GateLossPolicy.SKIP),
      ("swap", GateLossPolicy.APPLY_ANYWAY),
    )
  )


def create_loss_model():
  return UserLossModel()
```

The same selector syntax works for built-ins and files, for example
``--loss-model neutral-atom`` and ``--loss-model ./user_loss.py``.

The same canonical configuration is stored as nested ``loss_strategy`` metadata
in the compiled ``.deq.jit`` and ``.deq.bin`` artifacts.  By default,
``deq simulate ler --simulator qdk`` also passes that configuration to QDK, so
decoder metadata and physical sampling stay synchronized.  To compare different
assumptions deliberately, pass a JSON object with ``--simulation-loss-model``.
This replaces the decoder-derived QDK configuration; it does not merge with it,
and it leaves the compiled decoder metadata unchanged:

```sh
deq simulate ler circuit.deq --program Run --simulator qdk \
  --loss-model trapped-ion \
  --simulation-loss-model '{"cz":"SKIP","swap":"APPLY_ANYWAY"}'
```

Use ``--simulation-loss-model '{}'`` to leave every QDK loss policy at its own
default while retaining the selected decoder loss model.

The stored configuration remains structured rather than becoming a JSON string:

```json
{
  "loss_strategy": {
    "cx": "SKIP",
    "cy": "SKIP",
    "cz": "SKIP",
    "swap": "APPLY_ANYWAY"
  }
}
```

**Scope of the trapped-ion preset.**  This is an effective gate-level model, not
a claim that a physical MS pulse intrinsically applies $S^{\dagger}$ when an ion
is absent.  Experiments describe native $XX$ interactions and compile `CNOT` and
controlled-phase gates from $XX$ plus separate single-qubit rotations; see
[Debnath et al., *Nature* 536, 63-66 (2016)](https://doi.org/10.1038/nature18648)
and the explicit, interaction-sign-dependent decompositions in
[Maslov, *New J. Phys.* 19, 023035 (2017)](https://doi.org/10.1088/1367-2630/aa5e47).
Those references support the composite-gate picture, but neither reports a
universal residual operation caused by ion loss.

The motivation for the preset is the following possible controlled-phase
implementation:

$$
CZ = e^{-i\pi/4}
  e^{+i\pi Z_1/4}
  e^{+i\pi Z_2/4}
  e^{-i\pi Z_1Z_2/4}.
$$

Up to global phase, each local factor $e^{+i\pi Z/4}$ is $S^{\dagger}$.  If a
specific implementation loses the two-body interaction while its local phase
corrections still execute, the survivor does acquire $S^{\dagger}$.  Reversing
the interaction or compilation convention can change the residual rotation.
QDK's
[`RESIDUAL_S_DAGGER` policy](https://github.com/microsoft/qdk/pull/3302)
implements exactly that abstract behavior: skip the requested multi-qubit gate
and apply $S^{\dagger}$ to each surviving operand.

The built-in model consequently applies this policy only to `CZ` and rejects
source-level `CX` and `CY`. A hardware-backed model should instead be derived
from the device's actual gate decomposition, pulse ordering, interaction sign,
and loss detection timing. On the decoding side, deq represents the chosen
$S^{\dagger}$ response by its Pauli envelope $\{I,Z\}$; no native-MS circuit
gate or sampler rewrite is implied.

This chapter focuses on **sampling and transport**: how QDK produces a loss
result, how deq carries it as a `loss_mask`, and how random imputation chooses
the bit used to construct the syndrome. Imputation is not the loss decoder. It
is an orthogonal syndrome policy that applies whether loss information is
ignored, converted into edge reweights, or handed to a loss-aware decoder.

deq now compiles each selected platform model into a Pauli-envelope generator
DAG and supports both ordinary-decoder reweighting and structured loss handoff.
The complete compiler and backend model, including small runnable examples and
the paper's four-CX chain, is covered in
[Pauli-envelope loss decoding](pauli-envelope-loss-decoding.md).

Stim doesn't model loss as a first-class outcome.  The
[QDK](https://github.com/microsoft/qdk) stabilizer simulator does, via
its experimental `qdk.stim` module and a **Stim extension** that adds a
`LOSS_ERROR(p)` instruction.  This chapter walks through that extension
end-to-end through deq's `--simulator python` plug-point.

This chapter walks through:

1. **An example simulation**: a repetition-code memory experiment
   over `3·d` rounds of syndrome extraction, comparing a baseline that
   does nothing about loss against a loss-aware variant that
   replenishes data qubits each cycle.  The loss-aware variant beats
   the baseline by **3+ orders of magnitude** even at modest loss
   rates.
2. **What's in the `.deq`**: how `LOSS_ERROR(p)` shows up in a gadget
   body and what the one-line "replenish" addition does.
3. **How loss flows through deq today**: from the QDK output, through
   the Rust runtime's `PythonSampler` (which forwards the loss
   positions as a `loss_mask` bitvector alongside the placeholder
   outcomes), into the controller, and finally to the coordinator —
   which applies its configurable **loss-random-imputation** policy
   before computing the syndrome the decoder sees.
4. **Where loss info lives today** and pointers to follow-up chapters.

---

## An example simulation: loss kills, replenish saves

The example lives in [loss-simulation/repetition_code.deq](../examples/loss-simulation/repetition_code.deq)
— a Mako-templated repetition-code memory experiment with a single
boolean knob, `replenish`:

- **Baseline (`replenish=False`)**: at the start of each cycle we sprinkle
  `LOSS_ERROR(p_loss)` on every data and ancilla qubit, then run a
  standard Z-stabilizer syndrome extraction.  Lost data qubits **stay
  lost** — every subsequent gate on them is the identity and every
  subsequent measurement returns `Loss` (imputed by the coordinator to
  a fair coin-flip).  After `3·d` rounds, accumulated loss decimates
  the syndrome.
- **Loss-aware (`replenish=True`)**: at cycle end, teleport each
  data qubit `q` onto a fresh buddy `f`.  Textbook teleportation is
  `R f; CX q → f; MX q; CZ rec[-1] f`. The example omits the final
  correction because it reads out only in Z, where that phase correction
  cannot change the result. In this case,
  loss becomes a one-cycle random bit-flip the decoder attributes
  to `X_ERROR`, and code distance still achieves sub-threshold
  scaling.

[loss_ler_sweep.py](../examples/loss-simulation/loss_ler_sweep.py)
sweeps both variants for each `(d, p_loss)` point, transpiles via
`deq transpile`, then drives `python -m deq.runtime server`
with `--simulator python` (the QDK adapter) and `--decoder
black-box-relay-bp` (a real decoder).  It captures `Logical errors: K/N`
from each run, accumulates, and plots:

```sh
python documents/tutorial/examples/loss-simulation/loss_ler_sweep.py \
    --distances 3 5 7 \
    --loss-rates 0.01 0.02 0.05 0.1 0.2 0.3 \
    --target-errors 20 \
    --max-shots 1000000 \
    --workers 4
```

By default the per-instruction Pauli noise rate is set to
`p_Pauli = p_loss / 10` so the decoder always sees a non-trivial
hypergraph — without any Pauli noise the hyperedge probabilities
collapse to zero and the decoder can't pick a meaningful correction
when loss does show up.  Use `--p <value>` to decouple them.

![Logical error rate vs per-cycle loss probability](../examples/loss-simulation/loss_ler_sweep.png)

Two observations:

- **Baseline is not fault-tolerant** — there is no threshold.  The
  per-data-qubit loss probability over `3d` cycles grows as
  `3d · p_loss`, so raising the code distance also raises the loss
  exposure and buys no exponential suppression.
- **Loss-aware is fault-tolerant.**  The teleportation step swaps
  every data qubit onto a fresh buddy each cycle, so per-qubit loss
  exposure is bounded by a single cycle's `p_loss` no matter how many
  rounds we run.  Below threshold the replenish LER drops roughly an
  order of magnitude per two units of code distance (at `p_loss =
  0.01`, `d=3 → 5 → 7` LER is `7.5e-3 → 7.0e-4 → 7.1e-5`) — the
  exponential suppression in `(d+1)/2` that a fault-tolerant scheme should
  give.

### Caveats

The omitted conditional `Z` correction is **safe only for a Z-basis
memory experiment**.  For an arbitrary logical state (X-basis prep,
mid-circuit logical rotations, anything where the Pauli frame
matters), the missing classical-feedforward `CZ rec[-1] q` would
leave a real phase error. QDK and deq now accept record-controlled Pauli
gates. If the controlling measurement is loss, QDK skips the gate and deq's
loss analysis adds the corresponding event-conditioned Pauli generator to the
target. An arbitrary-state protocol should therefore include the correction;
this Z-basis-only example omits it solely because it is observationally inert.

---

## What's in the `.deq`

The full source is [repetition_code.deq](../examples/loss-simulation/repetition_code.deq)
— a few dozen lines of Mako-templated deq.  The interesting piece is
the `Syndrome` gadget; the `% if replenish:` block is the only
difference between the two variants.

[Syndrome gadget (Mako source)](../examples/loss-simulation/snippet_syndrome.deq)
<!-- deq-highlight-begin: ../examples/loss-simulation/snippet_syndrome.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> Syndrome</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#0000FF">    INPUT</span><span style="color:#267F99"> Rep</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(q) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> q </span><span style="color:#AF00DB">in</span><span style="color:#000000FF"> data)</span><span style="color:#0000FF">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # Loss event + per-cycle Pauli noise on data qubits.</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(${p_loss}) </span><span style="color:#0000FF">${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(q) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> q </span><span style="color:#AF00DB">in</span><span style="color:#000000FF"> data)</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(${p}) </span><span style="color:#0000FF">${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(q) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> q </span><span style="color:#AF00DB">in</span><span style="color:#000000FF"> data)</span><span style="color:#0000FF">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # Standard Z-stabilizer syndrome extraction.</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(q) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> q </span><span style="color:#AF00DB">in</span><span style="color:#000000FF"> anc)</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#000000FF">data[i]</span><span style="color:#0000FF">}</span><span style="color:#0000FF"> {</span><span style="color:#000000FF">anc[i]</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d </span><span style="color:#000000">-</span><span style="color:#098658"> 1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#0000FF">f</span><span style="color:#A31515">"</span><span style="color:#0000FF">{</span><span style="color:#000000FF">data[i</span><span style="color:#000000">+</span><span style="color:#098658">1</span><span style="color:#000000FF">]</span><span style="color:#0000FF">}</span><span style="color:#0000FF"> {</span><span style="color:#000000FF">anc[i]</span><span style="color:#0000FF">}</span><span style="color:#A31515">"</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> i </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> range</span><span style="color:#000000FF">(d </span><span style="color:#000000">-</span><span style="color:#098658"> 1</span><span style="color:#000000FF">))</span><span style="color:#0000FF">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#008000">    # Loss on syndrome ancillas + measurement bit-flip noise.</span></span>
<span class="line"><span style="color:#795E26">    LOSS_ERROR</span><span style="color:#000000">(${p_loss}) </span><span style="color:#0000FF">${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(q) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> q </span><span style="color:#AF00DB">in</span><span style="color:#000000FF"> anc)</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(${p}) </span><span style="color:#0000FF">${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(q) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> q </span><span style="color:#AF00DB">in</span><span style="color:#000000FF"> anc)</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    M</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(q) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> q </span><span style="color:#AF00DB">in</span><span style="color:#000000FF"> anc)</span><span style="color:#0000FF">}</span></span>
<span class="line"></span>
<span class="line"><span style="color:#AF00DB">%</span><span style="color:#AF00DB"> if</span><span style="color:#000000FF"> replenish:</span></span>
<span class="line"><span style="color:#008000">    # ── Teleportation replenish: data[i] ─→ fresh[i] (slot rename) ──</span></span>
<span class="line"><span style="color:#008000">    # One single-qubit teleportation per data qubit per cycle.</span></span>
<span class="line"><span style="color:#008000">    # The X-basis measurement (``MX``) clears any accumulated loss on</span></span>
<span class="line"><span style="color:#008000">    # the original data qubit while ``CX q → f`` transfers its</span></span>
<span class="line"><span style="color:#008000">    # Z-eigenstate to the buddy.  The data state now lives on</span></span>
<span class="line"><span style="color:#008000">    # ``fresh``, so we just declare the OUTPUT port on the</span></span>
<span class="line"><span style="color:#008000">    # ``fresh`` slots — the deq compiler wires those physicals into</span></span>
<span class="line"><span style="color:#008000">    # the next ``Syndrome``'s INPUT with no extra gates.  The would-be</span></span>
<span class="line"><span style="color:#008000">    # conditional Z corrections are omitted: see the header comment</span></span>
<span class="line"><span style="color:#008000">    # for why this is safe in a Z-basis memory experiment.</span></span>
<span class="line"><span style="color:#AF00DB">%</span><span style="color:#AF00DB"> for</span><span style="color:#000000FF"> q, f </span><span style="color:#AF00DB">in</span><span style="color:#795E26"> zip</span><span style="color:#000000FF">(data, fresh):</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#0000FF"> ${</span><span style="color:#000000FF">f</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    CX</span><span style="color:#0000FF"> ${</span><span style="color:#000000FF">q</span><span style="color:#0000FF">}</span><span style="color:#0000FF"> ${</span><span style="color:#000000FF">f</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    MX</span><span style="color:#0000FF"> ${</span><span style="color:#000000FF">q</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#008000">    # CZ rec[-1] ${f}  # omitted, see header comment</span></span>
<span class="line"><span style="color:#AF00DB">%</span><span style="color:#000000FF"> endfor</span></span>
<span class="line"><span style="color:#AF00DB">%</span><span style="color:#000000FF"> endif</span></span>
<span class="line"></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Rep</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(q) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> q </span><span style="color:#AF00DB">in</span><span style="color:#000000FF"> (fresh </span><span style="color:#AF00DB">if</span><span style="color:#000000FF"> replenish </span><span style="color:#AF00DB">else</span><span style="color:#000000FF"> data))</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/loss-simulation/snippet_syndrome.deq -->

`LOSS_ERROR(p_loss)` is just an instruction in the gadget body.
deq's transpiler treats it as a **passthrough noise instruction**:
emitted verbatim into the generated `.stim` (with the usual
local→physical qubit relabel), contributing nothing to the detector
graph or to the measurement count.  Upstream Stim doesn't recognise
`LOSS_ERROR`, so the resulting `.stim` is gated to `--simulator
python`; `qdk.stim` is what actually simulates the loss.

### Why `PrepareOne` initializes to physical `|1>`

[PrepareOne gadget (Mako source)](../examples/loss-simulation/snippet_prepareone.deq)
<!-- deq-highlight-begin: ../examples/loss-simulation/snippet_prepareone.deq -->
<pre class="shiki light-plus" style="background-color:#FFFFFF;color:#000000" tabindex="0"><code><span class="line"><span style="color:#AF00DB">GADGET</span><span style="color:#795E26"> PrepareOne</span><span style="color:#000000"> {</span></span>
<span class="line"><span style="color:#795E26">    R</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(q) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> q </span><span style="color:#AF00DB">in</span><span style="color:#000000FF"> data)</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#008000">    # logical X gate to prepare |1> state for testing</span></span>
<span class="line"><span style="color:#795E26">    X</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(q) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> q </span><span style="color:#AF00DB">in</span><span style="color:#000000FF"> data)</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#795E26">    X_ERROR</span><span style="color:#000000">(${p}) </span><span style="color:#0000FF">${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(q) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> q </span><span style="color:#AF00DB">in</span><span style="color:#000000FF"> data)</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#0000FF">    OUTPUT</span><span style="color:#267F99"> Rep</span><span style="color:#0000FF"> ${</span><span style="color:#A31515">" "</span><span style="color:#000000FF">.join(</span><span style="color:#267F99">str</span><span style="color:#000000FF">(q) </span><span style="color:#AF00DB">for</span><span style="color:#000000FF"> q </span><span style="color:#AF00DB">in</span><span style="color:#000000FF"> data)</span><span style="color:#0000FF">}</span></span>
<span class="line"><span style="color:#0000FF">    VIRTUAL</span><span style="color:#800000"> LX0</span><span style="color:#008000">  # added so that this is indeed outputing the logical |1> state</span></span>
<span class="line"><span style="color:#000000">}</span></span></code></pre>
<!-- deq-highlight-end: ../examples/loss-simulation/snippet_prepareone.deq -->

Look back at `PrepareOne`: it applies `R` then `X`, so every data
qubit starts in physical `|1>` rather than the more natural `|0>`.
This is deliberate — we don't want the benchmark to secretly favor
loss.  In the replenish step, `R f` prepares the buddy in `|0>` and
`CX q → f` copies `q`'s Z-eigenvalue onto it.  When `q` is lost the
CX is identity, so `f` stays in `|0>`.  With a `|1>` logical state
that's a deterministic bit-flip; with a `|0>` logical state it would
have been a free pass — loss self-healing on every qubit.  Preparing
`|1>` puts every loss event at its worst case and is why the
replenish curve saturates above 50% LER at high `p_loss`.

---

## How the pipeline works

The stim file that `deq transpile` emits — including the
`LOSS_ERROR(p_loss)` passthrough — are handed to the
[QDK](https://github.com/microsoft/qdk) stabilizer simulator through
deq's `--simulator python` plug-point.  Four short hops:

1. **deq's runtime sets `--simulator python`** and `sampler: "@qdk_sampler"`,
   which the runtime resolves to a **compile-time-embedded** copy of
   [qdk_sampler.py](../../../deq_runtime/src/simulator/qdk_sampler.py) via a
   small registry inside
   [python_sampler.rs](../../../deq_runtime/src/simulator/python_sampler.rs). You can
   still point `sampler` at your own `*.py` adapter when you want to.
2. **`qdk_sampler.py`** calls `qdk.stim.compile(src, None)` once to turn
   the Stim source (with `LOSS_ERROR`) into QIR, then batches
   `qdk.simulation.run_qir(shots=N)` (default `batch_size=256`) to
   amortize the ~0.5 ms-per-call Python overhead.  Each shot is a
   length-N string of `'0'`, `'1'`, or `'-'`; `'-'` marks a measurement
   whose qubit was lost.
3. **The Rust `PythonSampler`** packs each shot into an
   [`ErrorSet`](../../../deq_runtime/src/simulator/python_sampler.rs)
   with a `placeholder=0` bit and a `loss_mask` bit set at every `'-'`
   position.
4. **The coordinator** receives `Outcomes { outcomes, loss_mask }`
   and, by default (`loss_random_imputation=true`), replaces every
   `outcomes` bit whose `loss_mask` bit is set with a uniformly random
   bit drawn from a seeded RNG, then computes
   the syndrome the decoder consumes.

That's the whole pipeline — there's nothing QDK-specific about it
beyond the choice of `qdk_sampler.py` as the adapter.  The `@name`
sentinel only recognises names registered in the
[builtin_samplers module in python_sampler.rs](../../../deq_runtime/src/simulator/python_sampler.rs);
any other value of `sampler` is opened as a filesystem path, so a Python
class implementing
```python
class Sampler:
    def __init__(self, circuit_text: str, config: dict) -> None: ...
    def sample(self) -> str: ...        # length-N string of '0', '1', '-'
```
plugs in the same way; the same `loss_mask` plumbing carries through.

Three QDK-specific caveats are worth knowing if you're writing your
own circuits or adapters:

- The `qdk.stim` module is marked **experimental**; its API may shift.
- The `seed` parameter is currently **ignored** by upstream — successive
  calls produce different shots even with the same seed.  deq still
  passes it through so the contract is right when upstream wires it up.
- QDK's Stim parser does **not** yet accept the compact `M(p) <q>`
  noisy-measurement syntax or `MPP`. Use `X_ERROR(p) <q>; M <q>` for
  noisy measurement. Record-controlled Paulis such as `CX rec[-1] <q>`
  are supported; a loss-valued control skips the Pauli.

---

## Where the loss information lives today

| Layer                                          | Carries loss info? | Form                                      |
| ---------------------------------------------- | ------------------ | ----------------------------------------- |
| `qdk.stim.run` output                          | yes                | `Result.Loss` enum value                  |
| `qdk_sampler.py` shot string                   | yes                | `'-'` character                           |
| `ErrorSet.loss_mask` (Rust)                    | yes                | `Option<BitVector>`                       |
| `ShotSample.loss_mask` (proto)                 | yes                | `optional BitVector` proto field          |
| gRPC `Outcomes.loss_mask` (controller→coord)   | yes                | `optional BitVector` proto field          |
| Coordinator imputation policy                  | yes (consumed)     | `loss_random_imputation` (default `true`) |
| Black-box decoder request                      | strategy-dependent | edge `reweights`, structured `LossInfo`, or both |

By default, the coordinator applies **loss-random-imputation**: every bit
of `outcomes` whose `loss_mask` bit is set is replaced with a uniformly
random bit before the syndrome is computed. Pass
`--coordinator-config '{"loss_random_imputation": false}'` to disable
imputation; the decoder then sees a syndrome built from placeholder `0`
bits.

Imputation is independent of `loss_strategy`. The default `reweight` strategy
turns observed losses into shot-scoped edge probabilities. `handoff` sends
structured `LossInfo` to a decoder that advertises loss support, while `ignore`
drops the loss observation after syndrome construction. Runtime probability
updates from `Outcomes.modifiers` remain independent: under `handoff`, a request
may contain both edge reweights and structured loss.

The coordinator's `decoder_reweighting` setting controls transport only:
`auto` uses loaded reweights when advertised and otherwise materializes an
equivalent one-shot graph, `enabled` requires loaded support, and `disabled`
always materializes. Because `enabled` explicitly selects the loaded interface,
it also requires `persistent_decoder: true`. The policy never changes the
configured loss strategy.

To make the imputation reproducible across runs, also pass
`"loss_random_imputation_seed": <int>`.  When omitted, the RNG is seeded
from OS RNG.
