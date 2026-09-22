# Non-Clifford Gates: Simulation and Conservative Checks

A gadget can contain physical non-Clifford rotations directly. The same body
describes the physical experiment and supplies a conservative model for automatic
check discovery. It no longer needs an empty resource-state gadget solely to
hide the non-Clifford instruction from the compiler.

This is language and analysis support, not an automatic construction of a
fault-tolerant logical gate. Encoding, magic-state preparation, feedforward,
and the correctness of the protocol remain the circuit author's responsibility.

## Gate Syntax

| Instruction | Physical operation |
| --- | --- |
| `R_X(a) q...` | $\exp(-i\pi a X/2)$ on each target |
| `R_Y(a) q...` | $\exp(-i\pi a Y/2)$ on each target |
| `R_Z(a) q...` | $\exp(-i\pi a Z/2)$ on each target |
| `T` | Z rotation with `a=0.25`, up to global phase |
| `T_DAG` | Z rotation with `a=-0.25`, up to global phase |
| `TPP`, `TPP_DAG` | $\exp(\mp i\pi P/8)$ about each Pauli product $P$ |
| `R_XX(a)`, `R_YY(a)`, `R_ZZ(a)` | $\exp(-i\pi a P/2)$ on consecutive pairs |
| `R_PAULI(a)` | $\exp(-i\pi a P/2)$ about each Pauli product $P$ |
| `CH` | Controlled-Hadamard on consecutive control/target pairs |
| `CCX`, `CCZ` | Toffoli or controlled-controlled-Z on consecutive triples |
| `U3(t,p,l)`, `U(t,p,l)` | $R_Z(p)R_Y(t)R_Z(l)$, up to global phase, on each target |

Bare angles are **multiples of pi**, not radians or full turns. A `rad` suffix
accepts radians, such as `R_X(0.5rad)` or `U3(0.5rad,0.25,-1rad)`; the parser
normalizes these to half-turns for annotation and export. Angles must be finite.
Axis and product rotations require one angle, `U3`/`U` require three, and the
named phase/controlled gates take no arguments. Qubit-target gates accept one
or more complete groups of non-inverted indices, including inside `REPEAT`.
Each pair or triple must contain distinct qubits.
`R_X`, `R_Y`, and `R_Z` are deliberately different from Stim's `RX`, `RY`,
and `RZ`, which remain **resets**, not rotations.

Pauli-product targets use `*`, for example `TPP X0*Y1` or
`R_PAULI(0.125) Z0*Z1 X2*X3`. Whitespace separates independent products; the
latter instruction rotates about `Z0*Z1` and then about `X2*X3`.
An inverted factor negates the product. Repeated factors are multiplied in
order; non-Hermitian products and products reducing to identity are rejected,
matching the supported QDK simulation path.

The exported `.stim` keeps the physical rotations and their angles, with normal
qubit relabeling. DEQ uses QDK's gate spellings directly: write `R_X(0.25)` or
`R_Y(0.25)` for X- or Y-axis pi/4 rotations, and use a negative angle for their
inverses. This is **extended Stim**, not a circuit accepted by upstream Stim.
No random-Pauli approximation is written into the physical circuit.

## What the Decoder Sees

Before analysis, a shared decoder-circuit builder replaces each rotation about
axis $P$ with Clifford operations: prepare a private auxiliary qubit in $|+\rangle$,
apply controlled-$P$ to the target, then reset the auxiliary. Check discovery,
logical-flow analysis, and fault propagation all consume this same lowered
representation. It is equivalent to conditionally applying $P$ using an
independent, unobserved random bit.

For a Pauli-product rotation, all factors share the same auxiliary control:
`R_ZZ(a) 0 1` dephases by the joint operator $Z_0Z_1$, not independently by
$Z_0$ and $Z_1$. In particular, it preserves an $X_0X_1$ check. Separate products
on the same instruction use independent controls.

One auxiliary qubit beyond the gadget's physical-qubit range is reused. Resetting
it for every rotation application makes successive uses independent.
It never becomes a port qubit or a sampled measurement: `M0`, `rec[-1]`, READOUTs,
and the sampler's output retain their original indices. The original source and
physical simulation circuit keep the actual rotations; source positions are
mapped to the lowered circuit for error analysis and annotation.

The decoder channel is

$$\mathcal D_P(\rho) = \tfrac12(\rho + P\rho P).$$

A check must hold for **both** branches at every rotation. This preserves
commuting checks while dropping correlations that depend on coherent phase.
The 50% branch is an analysis device, **not** a physical error rate or a new
decoder hyperedge. Ordinary declared noise still contributes error mechanisms.

For `U3`/`U`, the decoder first checks whether the unitary is an X, Y, or Z
rotation up to global phase and dephases only that axis. For example,
`U3(0.25,-0.5,0.5)` is equivalent to `R_X(0.25)`: placed between `RX 0` and
`MX 0`, it retains `CHECK M0`. Similarly, `U3(0.25,0,0)` preserves Y checks
and `U3(0,0.125,0.375)` preserves Z checks. Axis recognition tests modular
relations on the stored half-turn arguments, including periodic equivalents,
with an absolute tolerance of **`1e-12` half-turns** and no relative tolerance.
This accepts floating-point roundoff such as the X-axis relation in
`U3(1,0.1,1.1)`. Near-axis gates within that tolerance are treated as axis
rotations only for decoder analysis; the physical export retains their original
angles. Checks retained through this tolerance are approximate, not exact
guarantees for the unrounded physical circuit. If no Pauli axis matches within
the tolerance, the decoder falls back to independent Z-Y-Z dephasing (rightmost
first), which removes all single-qubit Pauli information.

The chosen axis is still dephased even for zero or Clifford-valued angles;
identity-valued `U3` gates are conservatively Z-dephased. This is axis recognition
within one instruction, not cancellation of separate rotations. `CH` continues
to use two Y rotations around a CNOT, and `CCZ` uses seven commuting Z-product
rotations from its phase polynomial. `CCX` adds Hadamards on the target around
`CCZ`. These are **analysis-only** decompositions, so control-dependent
correlations can still be lost. The physical export keeps the original
composite gate for QDK to simulate exactly.

For example, compare these bodies:

[Baseline and one-T source gadgets](../examples/non-clifford/rotations.deq#L1-L10)
<!-- deq-highlight-begin: ../examples/non-clifford/rotations.deq#L1-L10 -->
```deq
GADGET Baseline {
    RX 0
    MX 0
}

GADGET OneT {
    RX 0
    T 0
    MX 0
}
```
<!-- deq-highlight-end: ../examples/non-clifford/rotations.deq#L1-L10 -->

`Baseline` has `CHECK M0`: its noiseless X measurement is always zero.
`OneT` has no check on `M0`. Physically, its probability of measuring one is
$\sin^2(\pi/8) \simeq 0.1464466$. It is not 50%; the physical sampler applies T,
while the decoder only asks whether a parity is guaranteed. In contrast,
`RZ 0; T 0; MZ 0` retains its zero-parity check because Z commutes with T.

## Four T Gates: An Intentionally Missing Check

[Four-T source gadget](../examples/non-clifford/rotations.deq#L12-L18)
<!-- deq-highlight-begin: ../examples/non-clifford/rotations.deq#L12-L18 -->
```deq
GADGET FourTAuto {
    RX 0
    REPEAT 4 {
        T 0
    }
    MX 0
}
```
<!-- deq-highlight-end: ../examples/non-clifford/rotations.deq#L12-L18 -->

Physically, $T^4=Z$, so the measurement is always **one**. Nevertheless, DEQ
intentionally does **not** infer `CHECK M0 FLIP`. Four independent random Z
branches still allow either outcome. DEQ does not combine rotations, cancel
T with T_DAG, or special-case Clifford-valued angles such as `R_Z(0)` or
`R_Z(1)`. Use an ordinary Clifford gate when you want its stronger automatic
analysis; keep the rotations when they describe the experiment you are testing.

### Inspect, Prove, Then Override

The runnable [example](../examples/non-clifford/rotations.deq) contains all three
cases and a manually checked four-T experiment. From the repository root, enter
the example directory and annotate the circuit:

```sh
cd deq/documents/tutorial/examples/non-clifford
deq annotate rotations.deq
```

Inspect the CHECK, ERROR, READOUT, and PROPAGATE statements in the generated
`rotations.annotated.deq` beside the source file. Baseline
has one finished check, while OneT and FourTAuto have none. Annotation preserves
the T gates and emits `@CHECKS("manual", verify=0)` to freeze its derived checks.
Annotation's round-trip verification checks compiler equivalence, not the
physical validity of a newly supplied parity.

After independently proving the missing relation, add `CHECK M0 FLIP` and
`@CHECKS("manual", verify=0)`, as in the example's `FourTManual` gadget.
Its generated annotation shows the physical noise and the decoder error row:

[Manually checked four-T annotation](../examples/non-clifford/rotations.annotated.deq#L45-L65)
<!-- deq-highlight-begin: ../examples/non-clifford/rotations.annotated.deq#L45-L65 -->
```deq
@CHECKS("manual", verify=0)
GADGET FourTManual {
    RX 0
    T 0
    T 0
    T 0
    T 0
    @SIMULATE_ONLY
    Z_ERROR(0.02) 0
    ERROR(0.02) C0 R0  # E0
    MX 0
    READOUT M0
    CHECK M0 FLIP

    # --- statistics ---
    # finished checks: 1
    #   weight distribution: { 1:1 }
    # unfinished checks: 0
    # errors: 1
    #   check-weight distribution: { 1:1 }
}
```
<!-- deq-highlight-end: ../examples/non-clifford/rotations.annotated.deq#L45-L65 -->

`FLIP` declares expected odd parity. Without it, the correct noiseless outcome
would trigger the check. The final Z error flips the measurement, triggers the
check, and flips the readout, so Tesseract can correct it in this example.
The annotator keeps `Z_ERROR` under `@SIMULATE_ONLY` and records its decoder
effect as `ERROR(0.02) C0 R0`: the same error flips check 0 and readout 0.
Leaving verification enabled rejects this check because it is outside the
conservative check space. `verify=0` is a deliberate assertion by the author,
not a way to discover or prove a check. The manual plugin uses **only** the
checks you provide.

## Joint Pauli Rotations

The following example retains its automatically discovered `CHECK M0`, because
$X_0X_1$ commutes with the rotation generator $Z_0Z_1$:

[Joint-rotation source gadget](../examples/non-clifford/rotations.deq#L37-L43)
<!-- deq-highlight-begin: ../examples/non-clifford/rotations.deq#L37-L43 -->
```deq
GADGET JointRotation {
    RX 0 1
    R_ZZ(0.125) 0 1
    MPP X0*X1
    READOUT M0
}

```
<!-- deq-highlight-end: ../examples/non-clifford/rotations.deq#L37-L43 -->

Replacing the joint rotation with `TPP Z0 Z1` would instead apply two independent
single-qubit rotations; neither the physical operation nor the decoder channel
is the same. This distinction is why Pauli products must be lowered as products.

## Sampling with QDK

[QDK 1.32](https://github.com/microsoft/qdk/releases/tag/v1.32.0) adds both
non-Clifford gates to its Stim frontend and stabilizer branching to its
`"clifford"` simulator. DEQ requires QDK 1.32 and uses this backend through
`--simulator qdk`; no additional simulator or Git-sourced package is needed.
For an existing Python environment, install the supported QDK version with:

```sh
python -m pip install --upgrade 'qdk>=1.32,<1.33'
```

The adapter compiles the physical circuit to QIR once and samples it in batches.
QDK preserves the coherent rotations, including their cancellation, rather than
using the decoder's random-Pauli approximation. Its bare rotation arguments are
half-turns, matching DEQ: `R_Z(0.25)` is T up to global phase.

QDK also supports measurement-record controls, inverted measurements, Pauli
noise, loss, and `SELECT`/`REQUIRE` preselection with these rotations. Run the
example from the same non-Clifford example directory:

```sh
deq simulate ler rotations.deq \
  --program FourTExperiment --simulator qdk \
    --decoder black-box-tesseract --shots 100 --errors 100 \
    --batch-size 100 --jobs 1 --seed 42
```

This example should complete with zero logical errors and zero failed shots.
The same command with `--program JointRotationExperiment` runs the joint-parity
example, also with a deterministic corrected readout.
It is a small correctness demonstration, not an FTQC threshold measurement.
The low-level runtime configuration is `--simulator python` with
`"sampler": "@qdk_sampler"`. Its `py_config` supports `seed`, `skip_shots`,
`num_measurements`, `batch_size`, `type`, and `loss_config`. The default `type`
is `"clifford"`: stabilizer branching scales well with many qubits but its cost
can grow exponentially with non-Clifford content. For small circuits dominated
by non-Clifford gates, `"type": "cpu"` selects full-state simulation.
Replaying the same seed and batch configuration reproduces the same shots;
changing the batch size can change the sequence. Unlike the native Stim backends
`static`, `jit-static`, and `preselect`, QDK can sample the rotations directly.

## Fire & Ice: Teleported X Rotations

Section IX of [Reichardt, Aasen, and Chao, *Fire and ice*](https://arxiv.org/html/2605.15344v1#S9)
proposes preparing a logical Bell pair between the `[[4,2,2]]` Iceberg code and
the `[[20,2,6]]` Fire & Ice code, rotating the small-code half with physical
two-qubit gates, and teleporting into the large code. The paper leaves the
detailed circuit and its error-rate analysis to future work.
The [rotation fixture](../../../tests/circuit/fixtures/fire_ice_rotations.deq)
implements one such construction, importing the existing verified preparation
and Steane error-correction circuits from the
[memory fixture](../../../tests/circuit/fixtures/fire_ice.deq).

`PrepareVerifiedRotationBell` prepares two cross-code logical Bell pairs using
a noisy CNOT network. It filters the resource by measuring both codes'
stabilizers and all four logical Bell stabilizers. Each measurement uses a cat
state whose bit-flip parity checks are verified before coupling it transversally
to the resource. All two-qubit gates and measurements in this verification are
noisy. The factory has no data input, so retries cannot disturb live data.

On Iceberg's first logical qubit, physical `X0*X1` is logical X. Consequently
`R_XX(a) 0 1` implements $R_{X,L}(a)=\exp(-i\pi a X_L/2)$ without disturbing
the other logical qubit. `RotateXAndMeasureIceberg` applies this rotation and
`DEPOLARIZE2(p)`, then destructively measures the small code in Z. Its even
four-qubit parity is preselected; the two logical Z outcomes give Pauli-X
corrections on the large half. `PrepareXRotationResource` produces
$R_{X,L}(a)|0\rangle_L\otimes|0\rangle_L$.

`TeleportedXRotation` couples that resource as the control of a transversal
CNOT onto the data, then `MeasureXRotationResource` measures the resource in X. For its decoded first
logical outcome $r$, the data undergoes $R_{X,L}((-1)^r a)$. Correcting branch
$r=1$ requires $R_{X,L}(2a)$:

- At `a=0.25`, this is a logical $S_X=HSH$ correction. DEQ does not
  implement this conditional Clifford correction through its Pauli-only
  `CONDITIONAL` statement.
- At `a=0.5`, the correction is just logical X, up to global phase. This is the
  runnable surrogate used below. These X-axis T/S rotations are Hadamard
  conjugates of the usual Z-axis T/S gates.

The independent Mako parameters `x_rotation_angle=0.5` and
`decoder_x_rotation_angle=0.25` select the `@SIMULATE_ONLY` and `@DECODE_ONLY`
instructions respectively. The decoder therefore retains the conservative $T_X$
model even though the physical circuit implements $S_X$. `TwoTeleportedSx` applies
the injection twice, each followed by `CONDITIONAL rec[-1] X0 0` and both
Steane error-correction halves. Two $S_X$ gates give X, so it asserts that the
first final logical Z readout is **1** and the untouched second readout is **0**.
Changing the physical angle to `0.25` alone does **not** turn this benchmark
into a corrected $T_X$-gate program; its branch correction and assertions would
also need to change.

### Reproduce the Evaluation

Run from the `deq/` directory in the `feature-144` environment:

```sh
deq simulate ler tests/circuit/fixtures/fire_ice_rotations.deq \
  --program TwoTeleportedSx --simulator qdk --decoder black-box-tesseract \
  --mako loss_fraction=0 --mako p=0.001 \
  --shots 60000 --errors 50 --batch-size 100 --seed 200144

deq simulate ler tests/circuit/fixtures/fire_ice_rotations.deq \
  --program TwoTeleportedSx --simulator qdk --decoder black-box-tesseract \
  --mako loss_fraction=0 --mako p=0.0005 \
  --shots 100000 --errors 50 --batch-size 100 --seed 300144
```

LER statistics are printed directly. `--save DIR` optionally retains compiled
artifacts, and `--simulator-trace-output PATH` optionally records per-shot
decoding results for offline analysis; neither is needed to measure LER.

The experiment explicitly requires `loss_fraction=0`: no physical loss policy
for `R_XX` is assumed. Two-qubit gates have `DEPOLARIZE2(p)` and measurements
have an anticommuting Pauli error with probability `p`; preparations and other
single-qubit Clifford gates are ideal. No post-decoding shot rejection or
nondefault Tesseract search settings are used. QDK retries failed ancilla
preparations internally; their overhead is not included in the LER denominator.

Completed QDK 1.32 / Tesseract runs on 2026-09-21, checked against the saved
per-shot traces:

| Physical `p` | Shots | Logical errors | Failed shots | Two-gate LER (approx. standard error) | LER / `p` |
| --- | ---: | ---: | ---: | --- | ---: |
| `0.001` | 18,800 | 50 | 0 | `0.002660 +/- 0.000376` | 2.66 |
| `0.0005` | 38,100 | 50 | 0 | `0.001312 +/- 0.000186` | 2.62 |

These results are consistent with linear-in-`p` errors from unprotected
injection and are expected due to the non-fault-tolerant implementation.

Regression tests cover the zero-noise `1,0` result, the distinct decoder and
simulation angles, and the true quarter-turn's $\sin^2(\pi/8)$ logical Z
probability on the small code using both QDK simulators. A full encoded
quarter-turn resource probe hit QDK 1.32's stabilizer-branch limit, even with
small batches; it is not qualified by these $S_X$-surrogate measurements.

## Limits and Troubleshooting

- **A check disappeared:** inspect the annotated circuit and identify a
  noncommuting rotation. This can be intended conservatism, not a parser bug.
  Prove the relation before adding a manual check with verification disabled.
- **An output stabilizer cannot be checked:** dephasing can destroy a required
  unfinished check. A manual model needs valid checks for every output
  stabilizer, not just a new finished check.
- **Logical propagation is incomplete:** only surviving Pauli flows constrain
  automatic propagation. A general non-Clifford gate does not map every Pauli
  frame to a Pauli frame. Do not interpret an unconstrained/empty PROPAGATE row
  as an exact logical action. Inspect and supply protocol-specific PROPAGATE,
  CONDITIONAL, and ERROR rules where representable, or split at a resource-state
  or feedforward boundary. This feature does not implement arbitrary adaptive
  non-Pauli frame tracking.
- **Noise crosses a rotation:** error propagation uses the conservative decoder
  channel, not exact coherent error evolution. Surviving checks are conservative;
  arbitrary logical-error probabilities and user-added checks are not thereby
  certified. In particular, validate error rows for manual checks that rely on
  cancellations. The example deliberately puts noise *after* the four T gates.
- **Automatic noise injection:** existing one- and two-qubit noise rules apply
  to the corresponding rotation, Euler, and controlled-H gates. SI1000 and the
  other generic injectors do not invent a three-qubit or Pauli-product noise
  model: add explicit noise instructions for `CCX`, `CCZ`, `TPP`, `TPP_DAG`, and
  `R_PAULI`, or first supply a physical one-/two-qubit decomposition.
- **Loss reaches a multi-qubit non-Clifford gate:** its physical loss behavior
  is not specified by the conservative decoder channel. Automatic loss analysis
  rejects such a gate unless the selected custom loss model declares it native.
  Supply an appropriate custom model or explicit `LOSS` metadata. Single-qubit
  rotations leave loss locations unchanged; gates on operands that cannot be
  lost do not require an extra policy. QDK's ability to sample a loss trajectory
  alone does not certify the decoder's physical loss model.
- **Stim says "unknown gate":** the output is extended Stim. Use the QDK 1.32
  adapter; do not feed it to stock Stim's sampler or detector-error-model builder.
- **Other Stim instructions are unsupported:** QDK-Stim remains experimental,
  not a complete replacement for stock Stim. In 1.32, `MPAD` is ignored,
  heralded-noise and sweep targets are unsupported, and Pauli products reducing
  to identity are unsupported. `DETECTOR` and `OBSERVABLE_INCLUDE` are ignored
  by QDK; DEQ performs decoding separately using the returned measurements.
  See the [QDK-Stim reference notebook](https://github.com/microsoft/qdk/blob/v1.32.0/samples/notebooks/qdk_stim.ipynb).
- **An import fails from the repository root:** run from `deq/` to avoid the
  outer source directory shadowing the editable package. Development runtime
  protobufs can be generated with `python deq/proto/compile.py` from there.

The automated QDK tests cover the physical probabilities and dagger signs, the
intentionally absent four-T check, explicit verification bypass, all axes,
measurement indexing, annotation/export, seeded replay, loss, preselection,
and the runtime path. Physical sampling is checked on both the branching and
CPU backends for every added QDK gate, including alias export, mixed angle units,
and a 65-qubit branching-only example. Small exact-unitary matrix tests verify
that each flow retained by composite/product lowering is physically valid.