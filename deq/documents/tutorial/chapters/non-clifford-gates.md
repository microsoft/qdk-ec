# Non-Clifford Gates and Adaptive Decoding

A non-Clifford gate raises two questions for an error-correction system. Which
measurement relations can the decoder still use? And if a gate needs a correction
chosen from a decoded logical readout, can decoding finish before the rest of the
logical circuit is known?

DEQ lets a decoder run alongside a non-Clifford simulator in an interactive
feedback loop: execute a gadget, stream its measurements into the decoder, and
request an error-corrected logical readout to choose the next instruction. The
remaining logical circuit need not be known. You declare the gadgets and their
noise in `.deq`; DEQ builds their decoding models without hand-written decoder
graphs or gadget-specific decoding code. The only thing
user needs to supply is the adaptive control
flow, with a simple interface to the decoding system.

The non-Clifford simulator used here limits the size of the encoded experiments
we can run, so the interactive demonstration uses a trivial `[[1,1,1]]` code.
It demonstrates the feedback loop, not error suppression. This restriction is
not built into the decoding system: a quantum-computer controller or a more
efficient emulator can execute the gadgets and deliver measurements through the
same DEQ interfaces.

We will start with one T gate, then build an adaptive T-injection loop with
logical feed-forward corrections. The final example considers
encoded rotations and the limits of what these demonstrations establish.

## What Changes When We Add a T Gate?

Prepare a qubit in $|+\rangle$ and measure X. Without noise, the result is always
zero, representing the $+1$ eigenvalue. Now insert a T gate:

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

Running `deq annotate rotations.deq` shows which checks DEQ found:

[Annotated Baseline and OneT gadgets](../examples/non-clifford/rotations.annotated.deq#L1-L26)
<!-- deq-highlight-begin: ../examples/non-clifford/rotations.annotated.deq#L1-L26 -->
```deq
@GTYPE(1)
@CHECKS("manual", verify=0)
GADGET Baseline {
    RX 0
    MX 0
    CHECK M0

    # --- statistics ---
    # finished checks: 1
    #   weight distribution: { 1:1 }
    # unfinished checks: 0
    # errors: 0
}

@GTYPE(2)
@CHECKS("manual", verify=0)
GADGET OneT {
    RX 0
    T 0
    MX 0

    # --- statistics ---
    # finished checks: 0
    # unfinished checks: 0
    # errors: 0
}
```
<!-- deq-highlight-end: ../examples/non-clifford/rotations.annotated.deq#L1-L26 -->

In `Baseline`, DEQ discovers `CHECK M0`: a noiseless measurement must be zero.
In `OneT`, the probability of one is

$$\Pr(M_X=1)=\sin^2(\pi/8)\simeq0.1464.$$

That randomness is part of the intended operation, not evidence of an error.
DEQ therefore does not turn this measurement into a check. By contrast,
preparing and measuring in the Z basis preserves a check:

[Z-basis T source gadget](../examples/non-clifford/rotations.deq#L49-L53)
<!-- deq-highlight-begin: ../examples/non-clifford/rotations.deq#L49-L53 -->
```deq
GADGET OneTInZBasis {
    RZ 0
    T 0
    MZ 0
}
```
<!-- deq-highlight-end: ../examples/non-clifford/rotations.deq#L49-L53 -->

`RZ 0` prepares $|0\rangle$, which T leaves unchanged, so `MZ 0` always returns
zero without noise. The generated annotation retains `CHECK M0`:

[Annotated Z-basis T gadget](../examples/non-clifford/rotations.annotated.deq#L93-L106)
<!-- deq-highlight-begin: ../examples/non-clifford/rotations.annotated.deq#L93-L106 -->
```deq
@GTYPE(6)
@CHECKS("manual", verify=0)
GADGET OneTInZBasis {
    RZ 0
    T 0
    MZ 0
    CHECK M0

    # --- statistics ---
    # finished checks: 1
    #   weight distribution: { 1:1 }
    # unfinished checks: 0
    # errors: 0
}
```
<!-- deq-highlight-end: ../examples/non-clifford/rotations.annotated.deq#L93-L106 -->

DEQ also accepts rotations such as `R_Z(0.25)`, which is T up to global phase.
Angles are measured in **half-turns**: `0.25` means $\pi/4$. Use a `rad` suffix
for radians. Note the underscore: `R_X` is a rotation, while `RX` prepares
$|+\rangle$.

## Which Checks Can DEQ Trust?

For automatic analysis, DEQ replaces a rotation about a Pauli operator $P$ with
an unobserved choice between doing nothing and applying $P$:

$$\mathcal D_P(\rho)=\tfrac12(\rho+P\rho P).$$

A retained check must hold for both choices. This keeps relations that commute
with the rotation, but can lose relations that depend on its precise angle or
on cancellation between gates. We call this model *conservative*: it may miss a
useful check rather than treat an intended random parity check as a check.

The physical simulator does **not** make this replacement. It still runs the
coherent rotation, so `OneT` has the probability above, not 50%. The factor
$1/2$ belongs to the analysis model; it is not an added physical error rate.
Declared noise is modeled separately.

### Why Can Four T Gates Lose a Check?

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

Since $T^4=Z$, the noiseless X measurement is always **one**. But DEQ analyzes
the four rotations independently, so it does not recover this cancellation.
`FourTAuto` has no inferred check. The same limitation applies to `T; T_DAG`
and to rotations written with zero or Clifford-valued angles.

When you can prove a relation that automatic analysis misses, you can declare
it yourself. The example's `FourTManual` uses `CHECK M0 FLIP` for the expected
odd parity and `@CHECKS("manual", verify=0)` to accept that declaration. A Z
error placed **after** the four T gates flips both the check and the readout,
allowing the decoder to correct this particular error.

Disabling verification transfers responsibility to the author. It does not
prove the check or the associated error model. In particular, errors occurring
between rotations need separate analysis. Use ordinary Clifford instructions
when they describe your experiment and you want their stronger automatic checks.

### What About Joint Rotations?

A joint rotation preserves joint checks that its single-qubit factors would
not preserve separately. Here $X_0X_1$ commutes with $Z_0Z_1$, so DEQ retains
`CHECK M0`:

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

The analysis uses one choice for the whole product: identity or $Z_0Z_1$.
It does not choose Z errors independently on the two qubits. In DEQ syntax,
`*` joins factors of one product; for example, `TPP Z0*Z1` is a joint rotation,
whereas `TPP Z0 Z1` specifies two separate rotations.

## Running the Physical Experiment

With DEQ and its QDK dependency installed, run these commands from the
repository root:

```sh
cd deq/documents/tutorial/examples/non-clifford
deq annotate rotations.deq
deq simulate ler rotations.deq \
  --program FourTExperiment --simulator qdk \
  --decoder black-box-tesseract --shots 100 --errors 100 \
  --batch-size 100 --jobs 1 --seed 42
```

Annotation makes the decoder model visible beside the source: `Baseline` has a
check, while `OneT` and `FourTAuto` do not. Annotation verifies that its output
compiles equivalently to the source; it does not independently prove manual
checks.

`FourTExperiment` uses the manually supplied check and should report zero
logical errors and zero failed shots. Replacing its name with
`JointRotationExperiment` runs the joint-parity example. These are correctness
checks for small circuits, not demonstrations of fault tolerance.

Use `--simulator qdk` because the physical export contains **extended Stim**
instructions that stock Stim cannot sample. QDK runs the actual rotations and
their coherent cancellations. Its default stabilizer-branching backend can
handle many qubits, but non-Clifford operations can make it expensive. The
interactive example below instead uses QDK's state-vector simulator on two
qubits.

## Can the Next Gate Depend on a Decoded Result?

T injection can require a conditional Clifford correction, not just a Pauli
correction. Here the decoded logical readout determines whether to inject an
S gate, whose own readout may require a final Pauli correction.

For an axis $P$, define $R_P(\theta)=\exp(-i\theta P/2)$. The injection circuit
applies either $R_P(\pi/4)$ or $R_P(-\pi/4)$, according to its readout. In the
negative branch, a further $R_P(\pi/2)$ gives the desired rotation. That
S-injection can also return the negative branch, which needs a final Pauli $P$.

This is where streaming decoding matters. **The controller can wait for a
decoded result before it even creates the next gadget.** It loads the gadget
types in advance, but does not have to supply a complete logical `PROGRAM`.

### Declare the Gadgets

All gadget definitions live in
[interactive_t.deq](../examples/non-clifford/interactive_t.deq). The file
contains data preparation and measurement, T and S resources and their inverses,
Clifford couplings, resource measurements, and virtual Pauli corrections.

Each injection has three separate stages. For a Z-axis T injection, first
prepare the magic state on resource qubit `1`, without touching data qubit `0`:

[Prepare the T resource](../examples/non-clifford/interactive_t.deq#L43-L48)
<!-- deq-highlight-begin: ../examples/non-clifford/interactive_t.deq#L43-L48 -->
```deq
GADGET PrepareTZ {
    RX 1
    R_Z(0.25) 1
    DEPOLARIZE1(${p}) 1
    OUTPUT Trivial 1
}
```
<!-- deq-highlight-end: ../examples/non-clifford/interactive_t.deq#L43-L48 -->

Next, couple the data and resource with a Clifford gadget. Both output ports
remain available:

[Couple data and resource](../examples/non-clifford/interactive_t.deq#L103-L109)
<!-- deq-highlight-begin: ../examples/non-clifford/interactive_t.deq#L103-L109 -->
```deq
GADGET CoupleZ {
    INPUT Trivial 0
    INPUT Trivial 1
    CX 0 1
    OUTPUT Trivial 0
    OUTPUT Trivial 1
}
```
<!-- deq-highlight-end: ../examples/non-clifford/interactive_t.deq#L103-L109 -->

Finally, measure the resource and request its decoded logical readout:

[Read the resource](../examples/non-clifford/interactive_t.deq#L119-L123)
<!-- deq-highlight-begin: ../examples/non-clifford/interactive_t.deq#L119-L123 -->
```deq
GADGET ReadZ {
    INPUT Trivial 1
    MZ 1
    READOUT M0
}
```
<!-- deq-highlight-end: ../examples/non-clifford/interactive_t.deq#L119-L123 -->

This separation keeps the non-Clifford preparation out of the data's propagation
analysis. DEQ infers both X and Z frame propagation through the Clifford
coupling and the frame's effect on the measurement.
For X-axis injection, the resource preparation adds a Hadamard,
the CNOT direction is reversed, and the resource is measured in X.

### Execute the Adaptive Sequence

The driver in [interactive_t.py](../examples/non-clifford/interactive_t.py)
performs the three stages and then chooses the logical feed-forward correction:

```python
async def inject(self, gate: str, axis: str) -> int:
    await self.step(f"Prepare{gate}{axis}")
    await self.step(f"Couple{axis}")
    outcome, = await self.step(f"Read{axis}")
    return outcome

async def inject_t(self, axis: str, *, inverse: bool = False):
    suffix = "Inv" if inverse else ""
    self.t_branches += 1
    outcome = await self.inject(f"T{suffix}", axis)
    if outcome:
        self.s_branches += 1
        correction_outcome = await self.inject(f"S{suffix}", axis)
        if correction_outcome:
            await self.step(f"Correct{axis}")
```

Each `step()` creates one gadget, applies its physical operations to QDK's
stateful simulator, and sends its measurement outcomes to DEQ. It returns the
decoded readout. Only then does Python decide whether to prepare and inject an
S resource or apply a virtual Pauli correction. The remaining circuit is not
merely hidden from the decoder: that branch has not yet been chosen.

The example uses the check-free `[[1,1,1]]` code, two reusable physical qubits,
Tesseract, and a JIT/window runtime with `buffer_radius=0, lookahead_radius=0`.
With both radii zero, each gadget's decoded result can be obtained immediately
after executing it and submitting its outcomes, without executing future gadgets.
Each injection prepares a fresh resource state. The final Pauli correction is
virtual: `CorrectZ` declares `VIRTUAL LZ0`, and `CorrectX` declares `VIRTUAL LX0`.
DEQ updates the logical Pauli frame; no physical X or Z gate is sent to QDK.

The frame also matters to the adaptive decision. For example, a pending X on
the data propagates through `CoupleZ` to the resource. DEQ therefore flips the
resource's logical Z readout relative to its raw measurement. Python uses that
**decoded logical readout** to decide whether to inject S. This accounts for
both frame components even when X- and Z-axis injections are mixed.

### How Do We Check the Result?

Run from the example directory:

```sh
python interactive_t.py --axis both --shots 2048 --noise-shots 10000 \
  --noise 0.001 --seed 144
```

The script measures X, Y, and Z on separate ensembles after one to four
injections. This checks the rotation, not just whether the program finishes.
For Z-axis injection starting from $|+\rangle$, the expected Bloch vector is
$(\cos(N\pi/4),\sin(N\pi/4),0)$. For X-axis injection starting from $|0\rangle$,
it is $(0,-\sin(N\pi/4),\cos(N\pi/4))$.

Four T gates provide a simple deterministic test: Z-axis injections followed
by X measurement must return one; X-axis injections followed by Z measurement
must also return one. Measuring X after an X rotation would not test its action.
The tutorial generator checks these predictions with virtual corrections. It
also runs X- and Z-axis injections followed by their inverses, starting with
each of the I, X, Y, and Z virtual frames. Across 384 noiseless trials, the
decoded X, Y, and Z measurements recover the initial logical state. The `Inv`
resource gadgets support these inverse-injection checks.
Earlier reference runs with physical Pauli corrections and 2,048 shots per
component also agreed within sampling uncertainty. These tests cover selected
input states, not every possible input state.

The noisy runs add `DEPOLARIZE1(p)` to each T or S resource; all other operations
are ideal. In those physical-correction reference runs at $p=0.001$, four
injections gave 41 errors in 10,000 shots for Z
rotations and 38 in 10,000 for X rotations, with no failed decodes. Their logical
error rates were $0.0041\pm0.00064$ and $0.0038\pm0.00062$ (approximate standard
errors). These results demonstrate working feedback under noise, **not error
suppression**: this example still has no stabilizer checks.

### What Must Be Ready Before Decoding?

Streaming does not mean decoding without enough information. The coordinator
still waits for the outcomes and error models required by its window. For a
nonzero window radius, the immediate execute-then-await pattern generally no
longer works. In an encoded protocol, the controller must schedule enough
syndrome-extraction cycles after the injection and submit their outcomes to
provide the required future context. Only then can the decoder proceed and
return the injection's decoded readout for the logical feed-forward decision.

At an eligible output boundary, the window coordinator can use a *terminal
error model*: a model that does not require the next gadget to be known. This
is what lets the zero-radius example finish each `step()` before choosing its
successor. Interior gadgets still require full error models. The monolithic
coordinator instead waits for the whole connected component and its full models.

Early decoding can discard useful future information. In particular, the
current policy does not commit an error hypothesis whose checks extend outside
the window. A terminal result is therefore not generally equivalent to one
obtained with the eventual full model. Choose the window size for the encoded
protocol; the check-free example avoids this tradeoff.

See [Driving the Runtime from Python](python-runtime.md) for the API and
readiness contract.

## Fire & Ice: Teleported X Rotations

What changes when the resource and data are encoded? Section IX of
[Reichardt, Aasen, and Chao, *Fire and ice*](https://arxiv.org/html/2605.15344v1#S9)
suggests rotating one half of a Bell pair in the small `[[4,2,2]]` Iceberg code,
then teleporting into the larger `[[20,2,6]]` Fire & Ice code. On Iceberg's first
logical qubit, physical $X_0X_1$ acts as logical X, making `R_XX` a natural way
to prepare the rotated resource.

The [rotation fixture](../../../tests/circuit/fixtures/fire_ice_rotations.deq)
implements a candidate construction with resource verification and Steane error
correction. Resource preparation has no live-data input, so rejected attempts
can be retried without disturbing the data. The paper leaves the detailed
circuit and error-rate analysis open.

The same sign ambiguity appears as in the two-qubit example. A logical $T_X$
injection needs a conditional $S_X=HSH$ correction. DEQ's `CONDITIONAL` syntax
handles Pauli corrections, not an arbitrary conditional Clifford gate. An
adaptive controller would need to schedule that S correction explicitly.

For this reason, the runnable `TwoTeleportedSx` benchmark uses physical $S_X$
injections, whose correction is Pauli X, while retaining a conservative $T_X$
decoder model. The parameters are `x_rotation_angle=0.5` and
`decoder_x_rotation_angle=0.25`. Two corrected $S_X$ gates give X, so the final
logical Z readouts must be **1, 0**. Changing only the physical angle to `0.25`
does not produce a corrected T-gate protocol.

For example, run from `deq/`:

```sh
deq simulate ler tests/circuit/fixtures/fire_ice_rotations.deq \
  --program TwoTeleportedSx --simulator qdk --decoder black-box-tesseract \
  --mako loss_fraction=0 --mako p=0.001 \
  --shots 60000 --errors 50 --batch-size 100 --seed 200144
```

Here two-qubit gates have `DEPOLARIZE2(p)` noise, measurements have a Pauli
error with probability $p$, and other preparations and single-qubit Clifford
gates are ideal. `loss_fraction=0` excludes physical loss; a noisy `R_XX` with
loss needs a separately specified loss model. Factory retries are not counted
in the logical-error-rate denominator. No post-decoding rejection is used.

Recorded QDK 1.32 / Tesseract runs on 2026-09-21 gave:

| Physical `p` | Shots | Logical errors | Failed shots | Two-gate LER (approx. standard error) | LER / `p` |
| --- | ---: | ---: | ---: | --- | ---: |
| `0.001` | 18,800 | 50 | 0 | `0.002660 +/- 0.000376` | 2.66 |
| `0.0005` | 38,100 | 50 | 0 | `0.001312 +/- 0.000186` | 2.62 |

The nearly constant LER/$p$ ratio is consistent with errors linear in $p$ from
unprotected injection. These two points do not demonstrate fault-tolerant error
suppression. They also do not validate the true $T_X$ protocol: a full encoded
quarter-turn resource probe hit QDK 1.32's stabilizer-branch limit.

## Gate Reference

The examples use only a few gates. DEQ also accepts these QDK spellings:

| Instruction | Operation or targets |
| --- | --- |
| `R_X(a)`, `R_Y(a)`, `R_Z(a)` | $\exp(-i\pi a P/2)$ on each qubit |
| `T`, `T_DAG` | Z rotations with `a=0.25` and `a=-0.25`, up to global phase |
| `R_XX(a)`, `R_YY(a)`, `R_ZZ(a)` | Joint rotations on consecutive qubit pairs |
| `TPP`, `TPP_DAG` | $\exp(\mp i\pi P/8)$ on each Pauli product |
| `R_PAULI(a)` | $\exp(-i\pi a P/2)$ on each Pauli product |
| `CH`, `CCX`, `CCZ` | Controlled H, Toffoli, and controlled-controlled Z |
| `U3(t,p,l)`, `U(t,p,l)` | Single-qubit Z-Y-Z Euler rotations |

Bare angles are half-turns; `R_X(0.5rad)` uses radians instead. Pair and triple
gates require complete groups of distinct qubits. Pauli products use `*`, as in
`R_PAULI(0.125) X0*Y1`; products must be Hermitian and nonidentity.

For `U3`/`U`, analysis recognizes Pauli-axis rotations within an absolute
tolerance of `1e-12` half-turns. Checks depending on that tolerance are approximate
for the unrounded physical gate. Other Euler and controlled gates use
conservative decompositions that can discard additional checks. Physical export
always retains the authored operation.

## What This Support Does Not Guarantee

Writing a non-Clifford instruction does not construct a fault-tolerant logical
gate. The author still needs to justify the encoding, resource preparation,
corrections, and noise model. Three boundaries are especially important:

- **Pauli-frame tracking is not arbitrary frame tracking.** A non-Clifford gate
  need not map a Pauli correction to another Pauli. Missing `PROPAGATE` relations
  are not proofs of identity. Split the protocol at injection or feedback
  boundaries and specify the required corrections.
- **Physical noise needs a physical model.** Generic noise injectors do not
  invent noise for three-qubit or general Pauli-product gates. Supply explicit
  noise or a suitable decomposition. Loss at multi-qubit non-Clifford gates
  requires an appropriate custom loss model or explicit `LOSS` metadata.
- **Successful simulation is not a fault-tolerance proof.** Check-free examples
  establish gate action and control flow. Encoded error suppression needs an
  error-rate study of the full protocol, including its corrections.

The central distinction is simple: simulate the physical gate, understand which
checks the decoder retains, and use decoded readouts to drive the next operation.
The logical circuit can grow as the experiment runs; the information required
for each decoding decision still has to be available.
