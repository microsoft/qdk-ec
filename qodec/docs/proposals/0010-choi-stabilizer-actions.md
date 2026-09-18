# 0010 - Choi-stabilizer instruction actions

**Status:** proposed

## Context

Measuring Z and resetting the measured qubit makes two promises: the reported
bit records the input Z value, and the output qubit is in the +1 Z eigenspace.
This proposal writes both promises as relations in one action. Proposed syntax:

```yaml
stabilizer:
  Z_0: R_0
  I: Z_0
```

`Z_0:R_0` connects the input operator to outcome zero. `I:Z_0` constrains the
output without requiring an input sign. A relation `P:Q` can also describe a
Clifford transfer from input P to output Q.

These relations describe stabilizers of a **Choi operator**, a representation
of a quantum channel. The proposed `stabilizer` action replaces `clifford`,
`stabilize`, and `observe`; `R_i` represents the sign of outcome bit i.

The current actions can already describe measurement and reset. The reason to
consider one relation form is to give them a common meaning for analysis:
which input information survives, which output signs are fixed, and which are
reported as bits. That matters when checking a gadget against a partial
specification without demanding an arbitrary recovery procedure. Putting the
rules in the action contract would keep each consumer from choosing its own
interpretation. That gain must outweigh the less familiar notation; the
completion rules still need proof.

## Benefits

- Use the same algebra to compose and check gates, measurements, and resets.
- Compare different recovery implementations against the same required output,
  without making one recovery procedure part of the specification.

## Drawbacks

- Familiar actions become relations with less obvious omission rules.
- Existing instructions and their consumers need migration.
- Generic rotations still need a separate action.
- The completion rule is not yet established for every allowed declaration.

## Proposal

This is proposed syntax and semantics. The general completion rule remains
subject to [Open Questions](#open-questions).

### Mathematical model

For a channel $\mathcal E:\mathcal H_{\mathrm{in}}\to
\mathcal H_{\mathrm{out}}$, use the normalized Choi operator

$$
J(\mathcal E)
=
(\operatorname{id}_{\mathrm{in}}\otimes\mathcal E)
\left(|\Phi\rangle\!\langle\Phi|\right),
$$

where $|\Phi\rangle=d_{\mathrm{in}}^{-1/2}\sum_j |j\rangle\otimes|j\rangle$.
Trace preservation is

$$
\operatorname{Tr}_{\mathrm{out}} J(\mathcal E)
=
I_{\mathrm{in}}/d_{\mathrm{in}}.
$$

Tensor factors below are ordered as input reference, quantum output, then
classical readout registers $C_0,C_1,\ldots$. Formulas include identity factors
when needed to keep that ordering explicit.

A mixed stabilizer Choi operator is described by an abelian signed Pauli group
that excludes $-I$. If its independent generators are $g_1,\ldots,g_r$ on $N$
qubits, its normalized projector is

$$
J=2^{-N}\prod_{j=1}^r (I+g_j).
$$

Every $J$ in this proposal has trace one. Here $N$ counts the input-reference,
quantum-output, and binary classical factors.

The relation `P:Q` stands for $P^T\otimes Q$. The transpose is a convention,
not a physical operation, because

$$
(M\otimes I)|\Phi\rangle=(I\otimes M^T)|\Phi\rangle.
$$

For example, if a Clifford $U$ maps $P$ to $Q=UPU^\dagger$, then its Choi state
$|J_U\rangle=(I\otimes U)|\Phi\rangle$ satisfies

$$
(P^T\otimes Q)|J_U\rangle=|J_U\rangle.
$$

$X^T=X$, $Z^T=Z$, and $Y^T=-Y$. Thus identity has $-Y\otimes Y$
correlation, but the author writes `Y_0: Y_0`; interpretation supplies the transpose.

Generators impose $gJ=J$. A physical channel must be completely positive and
trace preserving (CPTP). The [completion rules](#completion-and-elision)
distinguish a single channel from constraints admitting several channels.

### Relation grammar

`stabilizer` is a sparse map of authored relations, not a total operator map.
Each key occurs once. A list value supplies several constraints for that key;
it is not a probability distribution. Quantum indices address the flat logical
space on the corresponding side of the step. `R_i` indexes the instruction's
observe outcomes, not circuit readouts or flags.

Each side is a signed product of:

- quantum Pauli atoms (`X_0`, `Y_1`, `Z_0 Z_2`);
- readout characters (`R_0`, `R_1`, `R_0 R_1`);
- the identity `I`.

To interpret a signed relation `L:R`, split each side into its quantum Pauli and
readout character. If these are $L_q,L_c,R_q,R_c$ with overall sign $s$, the
declared Choi generator is

$$
g(L:R)
=
s\,L_q^T\otimes R_q\otimes L_cR_c,
$$

in the global factor order defined above. Classical atoms occupy their fixed
$C_i$ slots regardless of which YAML side contains them. Thus `P:R_i` gives
$P^T\otimes I\otimes Z_{C_i}$, while `R_i:P` gives
$I\otimes P\otimes Z_{C_i}$.

`R_i` is the $Z$ character of the conceptual classical register holding readout
bit $i$:

$$
R_i=|0\rangle\!\langle0|-|1\rangle\!\langle1|.
$$

There are no `X` or `Y` readout atoms: readouts are classical, not coherent
quantum operands. Products encode parity:

$$
R_i^2=I,\qquad [R_i,R_j]=0,
$$

and $R_0R_1$ has eigenvalue $(-1)^{r_0\oplus r_1}$. The first occurrence of the
next `R_i` must be on the right; it creates one immutable classical register.
Later occurrences on either side reference that same register.

A right-hand value may be a list when several generators share one left side:

```yaml
# Two independent output constraints.
stabilizer:
  I: [Z_0, Z_1]
```

This declares both $I\otimes Z_0$ and $I\otimes Z_1$. A list is conjunction,
never a choice.

### Core examples

#### Identity and code switching

Under the proposed completion contract, an empty relation set carries matching
flat logical positions unchanged. The example uses `action: []` for that case;
acceptance must explicitly settle this meaning because the current model can
also preserve empty actions as undeclared drafts. Changing the block type changes
the encoding boundary, not the logical channel:

```yaml
mnemonic: switch
in: [surface]
out: [color]
action: []
```

For $n$ carried logical qubits, completion supplies the identity-channel Choi
group

$$
\mathcal S_{\mathrm{id}}
=
\left\langle X_i\otimes X_i,\ Z_i\otimes Z_i\right\rangle_{i=0}^{n-1}.
$$

The different block types change how the input and output tensor factors are
encoded, not these logical correlations.

#### Clifford gates

```yaml
# Hadamard
- stabilizer:
    X_0: Z_0
    Z_0: X_0
```

Its Choi state is the pure stabilizer state with group

$$
\mathcal S_H
=
\left\langle X_0\otimes Z_0,\ Z_0\otimes X_0\right\rangle.
$$

```yaml
# CNOT; unchanged generator relations are elided.
- stabilizer:
    X_0: X_0 X_1
    Z_1: Z_0 Z_1
```

After completing the unchanged transfers, its Choi group is

$$
\mathcal S_{\mathrm{CX}}
=
\left\langle
X_0\otimes (X_0X_1),\
Z_0\otimes Z_0,\
X_1\otimes X_1,\
Z_1\otimes (Z_0Z_1)
\right\rangle.
$$

A Pauli unitary also fits. For example, an $X$ gate preserves $X$ and negates
$Z$, so only the nontrivial signed transfer is needed:

```yaml
- stabilizer:
    Z_0: -Z_0
```

The completed Choi group is

$$
\mathcal S_X
=
\left\langle X_0\otimes X_0,\ -Z_0\otimes Z_0\right\rangle.
$$

A dedicated `pauli:` spelling may remain as ergonomic sugar for predicated Pauli
frame corrections.

#### Preparation and stabilization

```yaml
# Prepare or stabilize Z_0 = +1.
- stabilizer:
    I: Z_0
```

This declares the Choi generator $I\otimes Z_0$. For a fresh one-qubit output,
the resulting state is

$$
\rho=\frac{I+Z_0}{2}=|0\rangle\!\langle0|.
$$

```yaml
# Prepare |00>.
- stabilizer:
    I: [Z_0, Z_1]
```

The output state has generators $Z_0$ and $Z_1$:

$$
\rho
=
\frac{1}{4}(I+Z_0)(I+Z_1)
=
|00\rangle\!\langle00|.
$$

```yaml
# Stabilize a joint parity, not each qubit individually.
- stabilizer:
    I: X_0 X_1
```

This declares only the output generator $X_0X_1$. Its canonical maximally mixed
completion is

$$
\rho=\frac{1}{4}(I+X_0X_1),
$$

though the action contract also admits other trace-preserving completions with
the same $XX=+1$ guarantee.

```yaml
# Bell-state constraints.
- stabilizer:
    I: [X_0 X_1, Z_0 Z_1]
```

The two independent output generators define the Bell state

$$
\rho
=
\frac{1}{4}(I+X_0X_1)(I+Z_0Z_1)
=
|\Phi^+\rangle\!\langle\Phi^+|.
$$

#### Measurement

A readout atom on the right introduces a classical output:

```yaml
# Measure Z_0 into the first readout.
- stabilizer:
    Z_0: R_0
```

This is a measurement *declaration*, not a full generator list. In general
`P:R_i` adds the input/readout generator

$$
P^T\otimes I\otimes Z_{C_i}
$$

and, on carried positions, the centralizer transport

$$
Q^T\otimes Q\otimes I_{C_i}
$$

for a generating basis of $P$'s centralizer on that space; anticommuting transfers
are absent. Where $P$'s support survives, `P:P` is already included, so `P:[P,R_i]`
is redundant.

If flat logical qubit 0 is present in `out`, completion gives the nondestructive
Lüders instrument. Writing $R_0=Z_{C_0}$, its Choi state is

$$
J
=
\frac{1}{8}
(I+Z_0\otimes Z_0\otimes I_{C_0})
(I+Z_0\otimes I\otimes Z_{C_0}).
$$

If qubit 0 is absent from `out`, the quantum output is traced out and only
$Z_0\otimes Z_{C_0}$ remains: the measurement is destructive, with

$$
J_{\mathrm{destructive}}
=
\frac{1}{4}(I+Z_0\otimes Z_{C_0}).
$$

The same action relation therefore works with the instruction boundary rather
than restating lifecycle in the action.

Several commuting observables can be measured in one step:

```yaml
- stabilizer:
    Z_0 Z_1: R_0
    X_0 X_1: R_1
```

For a nondestructive Bell measurement, with $P_Z=Z_0Z_1$ and $P_X=X_0X_1$,
the completed Choi group is

$$
\mathcal S_{\mathrm{Bell\ meas.}}
=
\left\langle
P_Z\otimes P_Z\otimes I_{C_0}\otimes I_{C_1},\
P_Z\otimes I\otimes Z_{C_0}\otimes I_{C_1},\
P_X\otimes P_X\otimes I_{C_0}\otimes I_{C_1},\
P_X\otimes I\otimes I_{C_0}\otimes Z_{C_1}
\right\rangle.
$$

A single `stabilizer` step requires all completed Choi generators to commute.
Sequential noncommuting measurements use separate steps:

```yaml
- stabilizer: {Z_0: R_0}
- stabilizer: {X_0: R_1}
```

After composing the two nondestructive instruments, a convenient final Choi
generating set is

$$
\left\langle
Z_0\otimes I\otimes Z_{C_0}\otimes I_{C_1},\
I\otimes X_0\otimes I_{C_0}\otimes Z_{C_1}
\right\rangle.
$$

The first outcome remembers the input $Z$ value; the final quantum output is in
the $X$ eigenspace reported by the second outcome.

Repeated measurements also use separate steps when their chronology matters:

```yaml
- stabilizer: {Z_0: R_0}
- stabilizer: {Z_0: R_1}
```

The completed group contains

$$
Z_0\otimes Z_0\otimes I_{C_0}\otimes I_{C_1},\qquad
Z_0\otimes I\otimes Z_{C_0}\otimes I_{C_1},\qquad
I\otimes Z_0\otimes I_{C_0}\otimes Z_{C_1}.
$$

Their product is $I\otimes I\otimes Z_{C_0}\otimes Z_{C_1}=R_0R_1$, proving
that the two noiseless outcomes agree unless an intervening operation disturbs
$Z_0$.

#### Measurement and reset

```yaml
in: [qubit]
out: [qubit]
action:
  - stabilizer:
      Z_0: R_0
      I: Z_0
```

The Choi state is generated by $Z_0\otimes I\otimes Z_{C_0}$ and
$I\otimes Z_0\otimes I_{C_0}$:

$$
J
=
\frac{1}{8}
(I+Z_0\otimes I\otimes Z_{C_0})
(I+I\otimes Z_0\otimes I_{C_0}).
$$

`Z_0:R_0` reports the input eigenvalue; `I:Z_0` fixes the output to $+1$. The
action exposes the outcome without prescribing a physical reset.

#### Classical feed-forward

A prior readout can participate in a later relation:

```yaml
# Prepare Z_3 = (-1)^r1.
- stabilizer:
    R_1: Z_3
```

This declares $Z_3\otimes Z_{C_1}$, correlating the prepared output eigenvalue with
readout 1.

```yaml
# Prepare from the parity r0 XOR r1.
- stabilizer:
    R_0 R_1: Z_3
```

This declares $Z_3\otimes Z_{C_0}\otimes Z_{C_1}$, so the output sign is
$(-1)^{r_0\oplus r_1}$.

```yaml
# Invert the convention.
- stabilizer:
    R_1: -Z_3
```

This declares $-Z_3\otimes Z_{C_1}$, reversing which classical value prepares the
$+1$ output eigenspace.

A classically controlled Pauli can also be represented as a signed transport. An
$X_3$ conditioned on `R_1` preserves $X_3$ and changes the sign of $Z_3$:

```yaml
- stabilizer:
    Z_3: Z_3 R_1
```

After completing the unchanged $X$ transfer, the Choi group contains

$$
X_3\otimes X_3\otimes I_{C_1},\qquad
Z_3\otimes Z_3\otimes Z_{C_1},
$$

which is the classically controlled $X_3$ channel.

The unchanged `X_3:X_3` is elided. This covers XOR-conditioned Pauli frame
updates; general branch-dependent Cliffords and rotations still need `if`/`unless`.

### Readout order

Readouts remain positional and chronological. Traversing action steps in order,
the first new right-side atom must be `R_0`, then `R_1`, with no gaps. Numeric
indices order several outcomes introduced by one commuting step; YAML map order
has no meaning. A left-side `R_i` must refer to an earlier outcome. Reusing an
existing `R_i` on the right adds a correlation, not another bit. The public record
is therefore `R_0,...,R_(n-1)`, followed by top-level flags as today.

### Completion and elision

Completion fills omitted relations. The candidate adds pass-through transport
that commutes with the declarations and preserves trace. The commuting operators
form their **centralizer**. Existence and uniqueness of a largest permitted
transport subgroup remain open; the examples specify the intended results.

#### Ingredients

An **input-only** Pauli is identity on all quantum outputs and classical factors.
A nontrivial input-only stabilizer would constrain the input, violating trace
preservation. The completed group must contain none.

Completion uses two ingredients:

- the **declared generators** $D$: the atoms
  $g(L{:}R)=s\,L_q^{\mathsf T}\otimes R_q\otimes L_cR_c$ of the step's relations; and
- the **pass-through transports**: the group generated by $X_i\otimes X_i$ and
  $Z_i\otimes Z_i$ over the flat indices $i$ carried from `in` to `out` — the
  identity channel on the carried space. Its elements include multi-qubit
  transports such as $X_0X_1\otimes X_0X_1$, and none is input-only (each acts on
  an output factor).

#### The completion rule

An interpreting consumer first checks the declarations:

- reject the step if they do not all commute, or if they close to $-I$ (a sign
  contradiction) — for a Clifford tableau this pairwise commuting check is exactly
  the symplectic condition on the rows; and
- reject it if they already contain a nontrivial input-only generator (a declared
  postselection).

The candidate completed group $\mathcal S$ contains the declarations together with the
**trace-preserving centralizer transport**: every pass-through $Q\!:\!Q$ whose $Q$
commutes with all declared generators, *except* any transport that — combined with
the declared generators — would yield a generator acting on the input alone, which
trace preservation forbids. Equivalently, $\mathcal S$ is $\langle D\rangle$
extended by the largest subgroup of that centralizer holding no input-only element.

Two constraints explain the intended result:

- **Not a per-qubit basis.** The centralizer is a subgroup, so measuring
  $Z_0Z_1$ keeps the joint transport `X_0 X_1:X_0 X_1` even though neither
  `X_0:X_0` nor `X_1:X_1` survives on its own.
- **The exclusion only bites for input-free constraints (`I:P`).** For `I:Z_0`,
  `Z_0:Z_0` commutes yet forms the input-only `Z_0:I`, so it is dropped and the
  input on that qubit is discarded. For `I:Z_0 Z_1`, `Z_0:Z_0` and `Z_1:Z_1`
  each commute but together with the declared `Z_0 Z_1` give an input-only
  product, so **both** go, leaving only the joint parity constrained.

Completion must be independent of map order and generating basis.

#### Complete actions versus contracts

$\mathcal S$, when it satisfies the stated group and trace conditions, defines a Choi operator
$J_{\mathcal S}=2^{-N}\prod_j (I+g_j)$. Whether the action *is* that operator or
merely constrains it turns on maximality:

- **Complete.** If no further generator can join $\mathcal S$ without creating an
  input-only element, $\mathcal S$ is maximal: every CPTP Choi operator it
  stabilizes equals $J_{\mathcal S}$, so the action denotes exactly
  $J_{\mathcal S}$. A full Clifford tableau is complete, and so is a reset such as
  `I: Z_0` — `X_0:X_0` anticommutes and `Z_0:Z_0` would act on the input alone, so
  no transport survives and the unique channel is "trace the input, output
  $|0\rangle$".
- **Contract.** Otherwise $\mathcal S$ leaves output/classical freedom that no
  input transport can fill, and the action denotes *every* CPTP Choi operator
  stabilized by $\mathcal S$ — it does not choose a recovery. `I: X_0 X_1` and
  `I: Z_0 Z_1` are contracts: their declared group is not maximal (the Bell
  group, among others, extends `I: X_0 X_1`). Validation proves the set nonempty
  by exhibiting $J_{\mathcal S}$ and checking its input marginal (see
  [Validation](#validation)), never a dense positivity test.

#### Required completion cases

- Untouched carried positions retain both X and Z identity transport.
- Sparse Clifford rows retain compatible omitted identity rows. The CNOT
  example supplies `Z_0:Z_0` and `X_1:X_1`; a complete tableau fixes one unitary.
- Measurement retains the commuting transport, including joint operators such
  as `X_0 X_1:X_0 X_1` for a `Z_0 Z_1` measurement. Omitting the quantum output
  makes it destructive. `P:[P,R_i]` is redundant with `P:R_i` if accepted.
- Output constraints discard input transport that would force an input-only
  stabilizer. Joint constraints may remain contracts rather than specify a
  recovery. Compatible transport can still be declared explicitly.

Each case must yield a commuting group without $-I$ or a nontrivial input-only
element. Product relations that leave freedom denote contracts, not arbitrary
choices made by the loader.

### Interaction with `in` and `out`

`in` and `out` define independent, ordered flat logical spaces. They type the
relation sides and determine which quantum factors exist:

- quantum atoms on the left address the input flat space of the step;
- quantum atoms on the right address its output flat space;
- every `R_i` atom references one immutable instruction-local classical register;
- newly introduced `R_i` atoms are classical outputs.

At the instruction boundary:

- a flat index on both sides starts with implicit identity transport;
- an input-only index is traced out;
- an output-only index has no inherited state or implicit $|0\rangle$ preparation;
- equal flat indices hosted by different block types express a pure code switch;
- explicit support lists partition and reorder each boundary but do not add
  channel relations.

Any promised state of a fresh output must therefore appear as `I:P` generators.
An unconstrained fresh output is valid only when the instruction intentionally
makes no state guarantee.

For a variadic block group, the caller fixes the flat size. Definition-time
validation checks symbolic shape; call-site validation checks concrete bounds and
commutation.

### Gadget integration

The contiguous produced range `R_0,...,R_(n-1)` defines the instruction's semantic
readouts. Gadget entry $i$ realizes `R_i` with its existing parity equation over
`circuit.readouts[...]` and boundary signs.

Compose steps in order, carrying quantum outputs into the next step. Retain
classical outcomes for later feed-forward and final output. A consumer checks
gadget conformance by:

1. compose the lower-level circuit's stabilizer instrument;
2. apply the input and output encoding maps;
3. substitute each gadget parity equation for the corresponding `R_i`;
4. trace out hidden lower-level outcome registers;
5. prove equality to a complete action or refinement of a contract;
6. verify checks and flags under the existing ideal-zero rules.

Equality means equal completed Choi operators. Refinement means the gadget Choi
operator is CPTP and satisfies every declared or inferred generator, possibly
adding compatible guarantees.

Flags stay ideal-zero side channels, following the `R_i` entries in the gadget
record and unavailable to the action as predicates; this proposal does not fold
them into the grammar.

### Rotations and non-stabilizer actions

A generic Pauli rotation has no commuting Pauli-generator presentation, so keep
`rotate`:

```yaml
- rotate: {pauli: Z_0, angle: theta}
```

For $U_\theta=e^{-i\theta Z_0/2}$, the Choi state is the pure state

$$
|J_{U_\theta}\rangle
=
(I\otimes U_\theta)|\Phi\rangle.
$$

For Clifford angles this state can be reduced to `stabilizer`; generic $\theta$
requires the explicit rotation.

The ordered action list remains necessary for mixed stabilizer/non-stabilizer
sequences and sequential noncommuting measurements.

### Validation

Loading preserves declarations without completing them. Consumers and audit
establish channel validity.

Consumers and audit check:

- quantum indices against the flat space on the appropriate relation side;
- pairwise commutation and consistency of the declared signed group;
- contiguous production and valid use of instruction-local `R_i` outcomes;
- existence of a CPTP completion with the required input marginal;
- conformance of a gadget to the complete action or partial contract.

Report invalid readout order or unsatisfiable constraints rather than return
partial results or choose a nonconforming recovery.

## Alternatives

### Keep `observe` separate

Merging only `clifford` and `stabilize` avoids `R_i` completion but leaves
measurement and feed-forward in a second algebra. It is the fallback if the
unified completion rules prove too subtle.

### Serialize PTM entries

A Pauli transfer matrix (PTM) makes loss of an operator component explicit
(`X\mapsto0` under $Z$ dephasing) and composes by
matrix multiplication, but authoring it requires coefficients and expanded sums.
Use PTMs as a derived equality form, not the source format.

### Expose classical qubits

An instrument Choi state has a qubit for every outcome. Exposing those qubits
would imply coherent readout $X/Y$ operations; `R_i` exposes only the classical
$Z$ character qodec uses.

### Require every Choi generator explicitly

A complete generator list makes omission unambiguous but expands identity and
Clifford actions. Typed completion keeps sparse authoring and makes each inference
rule explicit.

## Discussion

### Names and public surface

One key (`stabilizer`) and one atom family (`R_i`) replace three action keys.
`stabilizer` names the group constraint; `stabilize` already means forcing an
eigenspace. `R_i` names an outcome sign; `M_i` or `C_i` are alternatives, but
must still distinguish instruction outcomes from circuit and gadget readouts.
Binding APIs and retention of `pauli` are open.

### Migration

This is a breaking on-disk change and requires a `schema_version` bump. Common
actions migrate mechanically:

```text
clifford: {P: Q, ...}  -> stabilizer: {P: Q, ...}
stabilize: P           -> stabilizer: {I: P}
stabilize: [P, Q]      -> stabilizer: {I: [P, Q]}
observe: P             -> stabilizer: {P: R_next}
pauli: P               -> stabilizer signed transports, or retained sugar
rotate: ...            -> unchanged
```

Assign each old `observe` entry the next `R_i`; gadget positions remain stable.
Preserve action-step boundaries. Split a noncommuting `stabilize` list into
sequential steps because it cannot be one Choi stabilizer group.

Update Rust, schemas, bindings, and examples together. Outcome production
analysis replaces counting `observe` entries; tests need values as well as
signatures and complete export sets. Consumer work is defined in
[Gadget integration](#gadget-integration). Gadget parity syntax need not change.

## Open Questions

- Does the proposed completion exist uniquely for every supported declaration?
  Test basis and map-order independence, especially partial output constraints,
  before treating the completion cases as consequences of one general rule.
- Should an empty action mean identity, an undeclared draft, or require an
  explicit identity relation? Resolve this before claiming mechanical migration.
- Is `R_i` the right spelling, or should the classical atom be `M_i`, `C_i`, or a
  bracketed form? Decide here against the existing Pauli notation; a separate
  grammar change should be justified by this action language's requirements.
- Should the redundant expanded spelling `P:[P,R_i]` be accepted and normalized
  to canonical `P:R_i`, or rejected to keep one measurement spelling?
- Does constructing a CPTP stabilizer-projector witness accept every useful
  contract without choosing a recovery?
- Should `pauli` remain ergonomic sugar, especially with `if`/`unless`?
- How should named external `bit` parameters enter the classical character
  grammar alongside positional `R_i` atoms?
- Can ideal-zero flags eventually become classical Choi atoms, or is their
  decoder-blind role fundamentally separate?
- Does one step denote only simultaneous constraints, forcing noncommuting
  sequences into separate steps? This proposal says yes; examples should confirm
  the authoring is acceptable.
