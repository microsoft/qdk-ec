# Forced-gap post-selection

A decoder returns a correction, but some measurement records admit competing
corrections with different logical outcomes. Post-selection rejects the most
ambiguous shots. Its benefit is a lower logical error rate (LER) among retained
shots, paid for by discarding data, not an improvement to every hard correction.

This chapter compares a cheap correction-count score with forced-gap scores for
Fire & Ice and a distance-five surface code. The central lesson is that a
confidence score must describe the **logical outputs whose correctness matters**.
Taking the largest uncertainty over every intermediate readout can answer a
different question and give a worse selection rule.
Window scheduling is a second, separate choice: the recommended Fire & Ice
configuration combines **sliding windows with final-output-only selection**.
Two variants below isolate all-readout ranking and full temporal parallelism.

## The score and the observable

Let $H\hat e=s$ describe the hard decoder's chosen correction. For logical target
$a_j$, a second decode searches for a correction $e_j$ with the same syndrome and the
opposite target value:

$$
H e_j=s,\qquad a_je_j=1\oplus a_j\hat e.
$$

Using the shot's effective edge priors, define

$$
W(e)=\sum_{i:e_i=1}\log\frac{1-p_i}{p_i},\qquad
\Delta_j=W(e_j)-W(\hat e),\qquad q_j=\frac{1}{1+\exp(\Delta_j)}.
$$

Large $q_j$ means a competitive opposite-target explanation was found. A score
above one half means the alternative returned by the heuristic search is cheaper
than the hard correction. This compares two explanations; it does not sum all
errors in each logical class and is **not a calibrated posterior probability**.
The gap query does not replace the hard correction.

The count baseline is the maximum number of committed correction edges in any
one gadget, $C_{\max}=\max_g C_g$. It is not the sum over rounds or the hidden
physical fault count. Stars and upward triangles in the circuit figures retain
shots with $C_{\max}\leq2$ and $C_{\max}\leq3$. Those integer thresholds need not
occur at the same acceptance rate as a gap threshold.

Each configuration is sampled once. Both selectors reuse its recorded hard
readouts and logical-error labels. Ranking uses scores only, never the known
logical-error labels. Curves average uniform random tie breaking; fixed-budget
tables use a fixed, label-independent random tie order. Intervals are pointwise
95%; zero errors imply an upper bound, not zero risk. Decoder/scoring failures
are recorded separately, not silently dropped or called successful shots.

The circuit figures use 800,000 shots in each of three studies. The capacity
comparison is a separate matched-shot experiment under two gap-search profiles.
Both workflows retain trace checksums, frozen source identities, and disjoint
sampler seed ranges within each configuration. Matching profiles deliberately
reuse the same shot schedule, not additional independent samples.

### Window scoring

Monolithic decoding uses the whole connected problem. A window commits a subset
of corrections while retaining surrounding evidence in a buffer. We score the
logical readouts and **output boundary of the commit region**, not the outer
boundary of the entire window. The latter can contain many more logical
observables and includes outputs whose corrections are not yet being committed.

The target location and the errors allowed to change are separate choices.
For a commit-boundary target, the alternative may change errors in its **causal
history inside the window**, including previously committed gadgets. Future and
unrelated errors stay fixed at the hard decoder's baseline.
Input-port ancestry and remote conditional-correction dependencies define that
causal region. Fixed errors' detector contributions are subtracted to form the
restricted problem; this is bounded history, not an unbounded monolithic solve.

#### Why freeze the future buffer?

A short error chain can straddle the logical-observable boundary in time.
An error immediately before that boundary can flip the target, while the same
error immediately after it does not. Their syndrome explanations can differ by
only a small number of measurement faults, sometimes a single measurement fault
near an open boundary, as in a surface-code memory. Allowing both sides to vary
can therefore produce an extremely cheap opposite-target explanation through
**temporal degeneracy**, even when the eventual logical output is well protected.
That ambiguity belongs to an intermediate boundary, not necessarily to the final
logical task. Freezing the future prevents these short cross-boundary alternatives
from dominating the current score.

#### Why not vary only the commit region?

Restricting alternatives to the commit region excludes too much. A shortest
logical error chain or cycle need not fit within one gadget: equally likely
representatives can span several gadgets. A larger buffer supplies more evidence,
but cannot restore alternatives that the scoring constraint forbids from changing.
The commit-only variant we tested scaled poorly even at large buffer radii.
Allowing bounded causal history lets the score consider those multi-gadget
alternatives without reopening the future.

#### What about chains crossing this boundary?

As decoding advances, today's boundary lies in the historical buffer of a later
window. A short chain crossing today's target can then be considered as part of
that later target's causal history. Uncertainty contributions propagated to the
selected logical outputs are aggregated by a maximum, so later windows can expose
ambiguity excluded from the current query. This is the rationale for delaying
such comparisons, not a guarantee that every chain is found: the buffer must
cover it, the target must contribute to the selected outputs, and the heuristic
search must find the competing explanation.

The scoring graph is captured at hard-decode time with the same shot-specific
probability modifiers and loss transform, before hard-decoder edge merging.
Historical edges retain their original effective priors and selected corrections;
later loss evidence does not retrospectively reweight them. Restoring historical
edges reverses their subtracted syndrome contribution. Checks outside the retained
history are projected out; available checks remain constraints. These are
counterfactual queries only: they never rewrite committed hard corrections.
History is cleared between shots. Eager and lazy scoring use the same immutable
problems, differing only in when requested scores are computed.

### Scheduling is a separate choice

The runtime defaults to `window_parallelism: "sliding"`: a window waits for
non-free-hop causal history to commit before competing for reservation. This
preserves spatial parallelism between independent branches, while advancing
dependent windows in time. `window_parallelism: "fully_parallel"` removes this wait and
permits temporal parallelism too; overlapping windows still cannot decode
concurrently.

`"serial"` orders eligible window leaders by gadget ID. At positive buffer radius,
free-hop gadgets are absorbed by adjacent windows rather than leading their own.
With zero lookahead this policy is useful for
repeatable window construction. This policy sacrifices spatial concurrency too.
These policies operate within a shot; independent shots can still run in parallel.

The main studies use sliding windows and zero lookahead. The full-parallelism
variant changes only this scheduling policy, not Tesseract's search parameters.

### Hard decoding and gap search

DEQ supports a separate `--gap-decoder` and `--gap-decoder-config`, independently
of the hard decoder selected by `--decoder` and `--decoder-config`. The hard
decoder chooses the correction that determines the logical result. The gap
decoder searches for the forced, opposite-logical-outcome alternatives used
only for scoring; it does not replace that hard correction.

This separation is especially useful for search-based decoders such as Tesseract.
At low physical error rates, a hard-decoding solution is usually low weight.
Forcing the opposite logical outcome typically requires a substantially
higher-weight solution, making the gap search much harder. Reusing the hard
decoder's configuration can exhaust the search budget or time out even when
hard decoding is fast. A different decoder type, or the same type with a
different configuration, lets the two searches make different speed/accuracy
tradeoffs and can alleviate this problem. It does not guarantee that every
forced search will succeed; failures remain explicit rather than becoming
zero-risk scores.

The studies use Tesseract for both searches, with these settings:

| Search | `det_beam` | `pqlimit` | `det_penalty` | `beam_climbing` |
| --- | ---: | ---: | ---: | --- |
| Hard correction, all studies | 5 | 200000 | 0 | false |
| gap-config1 | 5 | 200000 | 0 | false |
| gap-config2 | 2 | 2000 | 30 | false |

`gap-config1` is the higher-accuracy profile and uses the same search parameters
as the hard decoder. `gap-config2` is the cheaper, lower-accuracy profile used
for the circuit studies. It trades search accuracy for feasibility without changing
hard corrections. The capacity figure below compares both profiles directly:
only the search for forced alternatives changes. A successful search need not
find a minimum-weight alternative, so valid scores can still rank shots poorly.
The larger, zero-penalty search is a more expensive accuracy setting, not an
optimality guarantee. Even an exact minimum-weight gap is not a posterior summed
over logical classes.

The runtime cost becomes much more severe at circuit level. On one fixed
ten-round `SteaneZMemory` shot with identical hard settings, `gap-config2`
completed in 17.7 seconds, while `gap-config1` did not finish
within a 120-second limit. The latter is a censored runtime, not a successful
decode or a measured completion time. This greater-than-6.7-fold cost illustrates
why the more expensive profile is not practical for the circuit study's
throughput budget. It is a bounded feasibility probe, not a universal speed ratio
or proof that the profile is infeasible for every circuit.

On search failure, Tesseract may retry bounded beam/order combinations within the
configured limits; a successful primary result is not replaced by a fallback.
Each attempt has its own queue limit, so retries increase worst-case work without
guaranteeing an optimum. DEQ reports the failing forced target, graph size, and
reachability. An unsuccessful search is **unavailable evidence**, never a zero-risk
score. Changing this search policy requires a new scientific run.

Setup, command lines, checkpointing, and cluster execution are described in the
[reproduction guide](../examples/post-selection/RUNNING.md).

## Fire & Ice: code-capacity noise

The public Fire & Ice $[[20,2,6]]$ code encodes two logical qubits. The fixture
is derived from the public code and preparation diagrams in *Fire and ice:
Partially fault-tolerant quantum computing with selective state filtering*,
by Ben W. Reichardt, David Aasen, and Rui Chao. These examples do not claim to
reproduce the authors' full circuit-level benchmark.

`CodeCapacityZMemory` is **Z memory with X-error noise**: the memory basis is
defined by the prepared and measured logical observables, not by the noise type.
It starts with ideal MPP preparation, applies one data-noise
layer, and measures every data qubit individually in Z. Both gadgets are written
explicitly. There are no reference blocks, auxiliary loss flags, or noisy
preparation/readout gates. QDK samples the circuit; DEQ infers its loss model and
performs native monolithic decoding and forced-gap scoring.

Two models use $p=0.02$: X errors only, and 30% X errors plus 70% independent
data-qubit loss. For loss fraction $f$, the noise layer is exactly
`X_ERROR((1-f)*p)` followed by `LOSS_ERROR(f*p)` on data qubits 0 through 19.
The first model uses $f=0$ and the second $f=0.7$. A missing atom is reported by
its own final measurement. DEQ uses that native loss mask, its inferred loss
envelope, and the same local weight-fraction-0.5 reweighting used elsewhere.

Individual Z outcomes supply the nine Z-stabilizer parities and two logical-Z
parities. X errors commute with the X stabilizers, so measuring that sector would
not add X-error syndrome information. No later quantum gate acts on the lost data;
phase uncertainty is irrelevant to the stated logical-Z failure label. This is
why individual Z readout is appropriate here, whereas it would discard useful
information in a joint X/Y/Z benchmark. A many-qubit MPP loss result alone also
does not identify each lost constituent qubit. There is no synthetic correlated
Pauli-plus-flag channel in this experiment.

A shot fails logically if either final logical Z assertion fails. This tests
the prepared logical state, not every possible residual logical Pauli. Scores
and correction counts come from the same native shot trace. Both searches use
the Tesseract settings above; the scores are approximate, not exact posteriors.

The four curves pair `gap-config1` (solid) and `gap-config2`
(dashed) for each noise model (color). Each profile uses the same sampled shots,
hard correction, and logical-error label; correction-count markers are shared
and drawn only once per noise model. The comparison isolates the benefit and
cost of searching more thoroughly for competing logical explanations. It is
not a comparison between hard decoders.

The completed run contains **100,000,000 matched shots per noise model**, scored
with both profiles: 200 million noise shots and 400 million decoder evaluations.
All 40,000 batch traces passed validation, with zero execution, decoding, or
scoring failures and no reused refill seeds within a case. Both profiles have
identical hard readouts, correction statistics, and logical-failure labels on
every matched shot.

| Noise model | Raw logical errors | Raw LER | gap-config1: LER at 1% rejection | gap-config2: LER at 1% rejection |
| --- | ---: | ---: | ---: | ---: |
| X errors only | 529,009 | 0.00529009 | $3.60\times10^{-4}$ | $4.39\times10^{-4}$ |
| 30% X errors + 70% native loss | 24,281 | 0.00024281 | $2.65\times10^{-7}$ | $6.06\times10^{-7}$ |

For X-only noise, the more thorough search reduces retained LER by about 18%
at 1% rejection. For mixed noise, at 0.5% rejection the corresponding LERs are
$2.47\times10^{-6}$ versus $1.01\times10^{-5}$. At 1% rejection, 99 million shots
are retained and the mixed-model tails contain about 26 versus 60 expected
errors after averaging uniform ties. The larger sample supports the observed
advantage more clearly, although tail estimates still have sampling uncertainty.
These are tie-averaged error rates, not fractional physical errors or exact risk
estimates; more accurate gap search does not guarantee a uniformly better ranking.
With 10,000-shot batches, the mean recorded time per shot, including process
overhead, was approximately 6.0--6.2 times higher for `gap-config1`.

This X-only/native-loss experiment replaces the earlier depolarizing and
synthetically flagged-erasure study. Their error rates are not directly
comparable: both the channel and the available syndrome information changed.

![Native DEQ Fire & Ice code-capacity comparison](../examples/post-selection/figures/fire_ice_capacity_post_selection.png)

## Surface-code circuit control

The second study uses the repository's rotated $[[25,1,5]]$ surface code, with
**ten syndrome-extraction rounds total**: the preparation gadget includes the
first round, followed by nine syndrome gadgets and one final logical-Z readout.
It uses the repository's `inject_si1000` channel: depolarization after gates,
reset faults and measurement flips at the same physical rate, with no added idle
noise or loss. These channel definitions matter more than the noise-model name.

The physical rate $p=0.00345$ was fixed from excluded calibration samples near a
monolithic LER of $2\times10^{-3}$. Calibration and evaluation use disjoint seeds.
The final comparison samples 100,000 new shots per configuration with the same
Tesseract hard/gap settings as Fire & Ice. Unlike teleporting Steane extraction,
these syndrome gadgets emit no logical readouts: the only logical readout is the
final asserted Z output. Thus all-readout and final-output gap ranking coincide.

The completed run had no execution or scoring failures. Monolithic decoding
produced 212 logical errors in 100,000 shots, LER $2.12\times10^{-3}$ with a
95% interval of $[1.84,2.43]\times10^{-3}$. At 3% and 5% rejection, 97,000 and
95,000 shots are retained:

| Configuration | Raw errors | Errors at 3% rejection | Errors at 5% rejection |
| --- | ---: | ---: | ---: |
| Monolithic | 212 | 32 | 19 |
| Radius 0 | 16192 | 15106 | 14326 |
| Radius 1 | 730 | 267 | 147 |
| Radius 2 | 359 | 73 | 48 |
| Radius 3 | 246 | 40 | 26 |
| Radius 4 | 218 | 41 | 28 |
| Radius 5 | 213 | 40 | 24 |
| Radius 6 | 213 | 39 | 23 |

Radius zero lacks the time context needed for noisy syndrome extraction. Increasing
the buffer improves both raw decoding and post-selection markedly; radii 4--6
approach monolithic raw performance in these data. Their sparse selected-error
counts do not prove identical performance. Larger windows also require additional
decoding and gap queries: improved accuracy is not by itself evidence of a
latency advantage.

![Distance-five surface-code post-selection](../examples/post-selection/figures/surface_code_post_selection.png)

## Fire & Ice: all readouts are not the objective

The circuit study uses `SteaneZMemory`: ten Z-only teleporting Steane EC rounds,
not ten Z+X cycles. Verified ancilla preparation occurs before interaction with
the data. Its `PRESELECT` retries are distinct from rejecting a completed shot
using decoder scores.

At each two-qubit gate, total noise is $p=0.007$: 30% Pauli and 70% correlated
loss, with equal first-qubit, second-qubit, and both-qubit loss branches.
Measurements have Pauli faults at $0.3p$ and introduce no new loss. QDK's neutral-
atom SKIP sampler handles existing loss; the decoder uses local loss-envelope
reweighting with weight fraction 0.5. This is an approximate decoding model,
not an exact joint posterior or a reproduction of the paper's entire protocol.

Each EC round measures the old data block in X and emits **two logical readouts**
for teleportation-frame updates. Ten rounds therefore contribute 20 intermediate
X readouts. `MeasureZNoisy` adds the final two Z readouts. Only those last two
appear in the memory experiment's success assertions:

```deq
PROGRAM SteaneZMemory {
    PrepareZeroVerified 0
    REPEAT 10 { ec_round_z 0 }
    MeasureZNoisy 0
    ASSERT_EQ rec[-2] 0
    ASSERT_EQ rec[-1] 0
}
```

The initial ranking was $S_{\mathrm{all}}=\max_{j=1}^{22}q_j$. It asks about
uncertainty in *any* intermediate or final logical readout. That is a poor match
to this Z-memory LER: intermediate X-readout uncertainty can concern phase-frame
information that does not flip the final asserted Z result. In these data it
creates a common score floor, hiding distinctions in the final-output scores.
Once selection reaches the large tied population, further rejection is mostly
random and the LER curve plateaus.

![Readout variant: sliding windows, ranking all 22 logical readouts](../examples/post-selection/figures/fire_ice_sliding_all_readouts.png)

All eight configurations completed 100,000 attempts with no execution/scoring
failures. Radius 6 improves substantially on smaller windows, but the original
metric still leaves radius 1 worse than radius 0 after 3% rejection. Distance six
alone does not make a radius-six decoder identical to monolithic decoding.

## Fire & Ice: full temporal parallelism

The second variant uses full temporal parallelism (`window_parallelism: "fully_parallel"`)
but keeps final-output-only ranking. It uses the same frozen runtime, sampler,
seeds, noise, and hard/gap search settings as the sliding study. Future windows
may commit before earlier ones, changing which corrections are fixed when
neighboring windows decode.

Both policies completed the fixed budget of **100,000 shots per configuration**,
800,000 per policy, with **zero execution/scoring failures**. The comparison
uses the same shot schedule and label-independent tie order for each policy,
without selecting on outcomes:

| Configuration | Raw errors: sliding | Raw errors: fully parallel | Final-output errors at 3%: sliding | Final-output errors at 3%: fully parallel |
| --- | ---: | ---: | ---: | ---: |
| Monolithic | 253 | 253 | 56 | 56 |
| Radius 0 | 2023 | 2023 | 263 | 263 |
| Radius 1 | 462 | 920 | 73 | 235 |
| Radius 2 | 438 | 904 | 67 | 195 |
| Radius 3 | 389 | 904 | 63 | 162 |
| Radius 4 | 349 | 674 | 58 | 128 |
| Radius 5 | 330 | 533 | 58 | 133 |
| Radius 6 | 301 | 426 | 62 | 96 |

Full parallelism increases the **zero-rejection logical error rate** for radii
1--6, before post-selection is applied. Final-output selection still helps, but
does not erase the scheduling disadvantage. The monolithic and radius-zero
controls match shot-for-shot, including scores and hard readouts. These observations
support sliding as the default for this benchmark; they do not establish a
universal ordering theorem or measure a temporal-parallelism speedup.

![Scheduling variant: full parallelism, final-output-only selection](../examples/post-selection/figures/fire_ice_fully_parallel_final_readouts.png)

This figure includes the complete 100,000-shot budget for every configuration.

## Select for the final logical outputs

For the stated benchmark, use $S_{\mathrm{final}}=\max(q_{21},q_{22})$: the
uncertainty of the two outputs that define logical failure. These scores include
the implemented history and frame propagation; they are not confidence estimates
from the final measurement gadget in isolation.

**Nothing is resampled or re-decoded for this figure.** It uses the same noisy
shots, hard corrections, and logical-error labels. Only the ranking statistic
changes. Correction-count markers are also unchanged. In particular, the
postprocessor never consults the error label to decide which shots to keep.

| Configuration | Raw errors | All 22: errors at 3% rejection | Final two: errors at 3% | Final two: errors at 5% |
| --- | ---: | ---: | ---: | ---: |
| Monolithic | 253 | 72 | 56 | 36 |
| Radius 0 | 2023 | 264 | 263 | 215 |
| Radius 1 | 462 | 383 | 73 | 45 |
| Radius 2 | 438 | 317 | 67 | 49 |
| Radius 3 | 389 | 248 | 63 | 39 |
| Radius 4 | 349 | 203 | 58 | 32 |
| Radius 5 | 330 | 117 | 58 | 26 |
| Radius 6 | 301 | 108 | 62 | 33 |

The 3% and 5% columns retain 97,000 and 95,000 shots. Radius 6 and monolithic
have close final-output-selected rates here; this is not a formal equivalence
proof, and small differences among nonzero radii need uncertainty estimates.
Under the all-readout rule radius six retains more observed errors than monolithic
decoding. The three figures keep the effects of target selection and window
scheduling separate: this final figure uses sliding windows and the benchmark's
asserted outputs, the recommended combination.

For a Z-memory benchmark, final asserted outputs are the appropriate selection
objective. For preserving an arbitrary logical quantum state, phase and bit-flip
information can both matter: the final-Z-only curve does **not** establish full
logical-channel protection. The transferable lesson is to choose observables from
the task definition first, and only then evaluate a confidence score.

![Recommended: sliding windows, post-selection on final asserted logical outputs](../examples/post-selection/figures/fire_ice_sliding_final_readouts.png)
