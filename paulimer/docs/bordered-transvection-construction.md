# Checked bordered transvection construction

The minimal decomposer first runs its existing congruence search.
If that search rejects the residue core, the decomposer tries the algorithm below.
The new path replaces neither the initial search nor its alternating restriction guard.
It constructs a complete candidate, then checks that candidate before returning it.

This repository does not prove that the construction succeeds for every input.
The exhaustive residue fix search remains as a fallback for that reason.
If construction or verification fails, the decomposer uses that search.
The retained search still panics if it exhausts its candidates.
The open question is whether $\mathrm{Res}(F)$ always contains a fix vector of the same residue rank.

## Coordinates and algorithm

All arithmetic uses bits, with addition as XOR.
The action matrix $F$ uses row vectors, with X coordinates before Z coordinates.
The matrix $\Omega$ exchanges those two coordinate halves.
The existing code computes

$$
\widehat F=\Omega(I+F), \qquad V=R\widehat F, \qquad E=VR^{\mathsf T}.
$$

Here $V$ contains the nonzero reduced rows of $\widehat F$.
The matrix $R$ records their row operations.
The residue rank $r$ is the number of rows in $V$.
The construction works in those $r$ coordinates with the form $\beta(x,y)=xEy^{\mathsf T}$.
Its coefficient rows form a matrix $Q$.
The Pauli vectors are the rows of $QV$, in the same order, without a transpose.

The recursion starts with the standard coordinate basis.
For a current basis matrix $B$, it computes the restriction $BEB^{\mathsf T}$.
It chooses the first basis vector with diagonal entry one.
If all diagonal entries are zero, it chooses the first pair with unequal off-diagonal entries and adds those two vectors.
These choices use ascending basis indices.

For a chosen vector $p$, the code reuses the right-orthogonal complement helper.
The helper eliminates one basis vector with nonzero coupling to $p$.
It returns a basis of the vectors $u$ that satisfy $\beta(p,u)=0$.
The recursion processes that basis, then prepends $p$ with border bit zero.

If neither choice exists, the restriction is symmetric with zero diagonal.
The code uses the simplex procedure below instead of enumerating the span.
It assigns border bit one to each simplex point.
The list of border bits is $\rho$.

### Simplex procedure

The empty basis produces one zero vector.
For a nonempty basis, choose its first vector $a$.
Choose the first later vector $b$ with $\beta(a,b)=1$.
If no such vector exists, return construction failure.

For each remaining basis vector $u$, compute

$$
u+\beta(b,u)a+\beta(a,u)b.
$$

Recurse on these projected vectors.
Add $a$ to each returned point.
Prepend $b$ and $a+b$ to that list.
This defines the candidate procedure, not a guarantee that every input passes the checks below.

## Runtime acceptance

The code checks the coefficient dimensions, the border length, and the rank of $Q$.
It requires $r+1$ rows, $r$ columns, and rank $r$.
It also checks

$$
QE Q^{\mathsf T}+\rho\rho^{\mathsf T}
\text{ is unit lower triangular}, \qquad
\rho^{\mathsf T}Q=0, \qquad
\rho^{\mathsf T}\rho=1.
$$

Unit lower triangular means zero entries above the diagonal and one entries on it.
These are runtime conditions, not assumptions about the recursion.
Failure returns no candidate.

After the lift through $V$, the code requires exactly $r+1$ nonzero vectors of length $2n$.
It multiplies their transvection matrices in order, starting from the identity.
It accepts the candidate only when that product equals the complete input action matrix.
The Pauli conversion uses the existing positive Hermitian representative.
As with the other decomposers, the result does not reproduce Pauli-image signs or global phase.

The successful path returns the full factorization.
It does not extract a fix vector and repeat the congruence search on an updated core.
The fallback retains the original candidate order, search, and recursive reconstruction.
The [minimality note](transvection-minimality-correction.md) describes that search and its limits.

## Test evidence

The existing shortest-path oracles test the public minimal decomposer.
They compare the factor count, replay the factors, and check tableau validity.
An explicit run on 2026-10-04 passed all three finite domains:

| Qubits | Actions tested | Test |
| --- | ---: | --- |
| 1 | 6 | `minimal_matches_brute_force_oracle_on_every_one_and_two_qubit_action` |
| 2 | 720 | `minimal_matches_brute_force_oracle_on_every_one_and_two_qubit_action` |
| 3 | 1,451,520 | `minimal_matches_brute_force_oracle_on_every_three_qubit_action` |

The three-qubit oracle is ignored by default.
Those public oracles do not exercise the retained search.
Separate direct tests enumerate the same domains and call `find_fix_vector` on every rank-plus-one case.
They check residue membership, preserved rank, the updated core, and the replayed action.

| Qubits | Actions enumerated | Direct search calls |
| --- | ---: | ---: |
| 1 | 6 | 0 |
| 2 | 720 | 225 |
| 3 | 1,451,520 | 150,255 |

The three-qubit direct test is also ignored by default.
Run the following commands from the repository root to include it:

```bash
cargo test --profile ci-test -p paulimer --lib bordered_
cargo test --profile ci-test -p paulimer --lib retained_search_covers -- --include-ignored
cargo test --profile ci-test -p paulimer --test transvection_test -- --include-ignored
```

`bordered_seeded_larger_actions_replay` tries 32 fixed seeds at each of 4, 6, 8, 12, 16, and 32 qubits.
It tests 192 general Clifford inputs directly through the construction.
It tests another 192 conjugated SWAP layers through the minimal decomposer.
Every sampled family replays correctly.
Every sampled SWAP layer uses the construction without any span enumeration.

The general-input test checks construction and replay, not minimality.
A family for a triangularizable core can contain a zero vector.
The test retains that vector, but the public acceptance check rejects it.
The conjugated SWAP tests also check the nonzero factors and the $r+1$ count.
These 384 inputs provide sampled evidence, not exhaustive coverage at those sizes.

The focused tests also cover the zero-dimensional simplex and a nonsymmetric form with zero diagonal.
They reject malformed border data and malformed factor lists.
They force missing and incorrect candidates through the retained fallback.
The private validator reports each failed condition.
Some conditions overlap, so the tests compare the complete diagnostic result.
The count, parity, and rank fixtures also fail another condition.
Separate fixtures isolate the shape, relation-sum, and unit-lower conditions.
The public nonsymmetric test requires zero fallback calls on a non-involutory class-A input.
The following results cover the original construction tests.
Each test failed under its mutation and passed after restoration:

| Test | Mutation | Failure message |
| --- | --- | --- |
| `bordered_family_handles_nonsymmetric_zero_diagonal` | Reverse the pair-pivot condition | `a nonsymmetric zero diagonal requires a pair pivot` |
| `bordered_swap_layers_do_not_search` | Discard every constructed candidate | `the bordered SWAP path must not enumerate a span`, with 7 visits instead of 0 |
| `bordered_verification_rejects_invalid_candidates` | Accept the count without action equality | `invalid candidate passed verification: reversed factors` |
| `bordered_fallback_recovers_from_candidate_failure` | Return without the retained search | `a failed candidate must use the retained search` |
| `bordered_seeded_larger_actions_replay` | Reverse the constructed factor order | `bordered factors must reproduce the input action` |

The review regressions below each failed under the listed mutation and passed after restoration.
The count, parity, and rank rows test diagnostic checks because their rejection conditions overlap.
They do not claim an independent change in Boolean acceptance.

The GIL test passed 40 of 40 repetitions pinned to CPU 6.
It uses three class-A blocks followed by a `Z_4 X_6` transvection on seven qubits.
One native call takes a median of 0.221 seconds across five pinned release runs after one warmup.
Removing `py.detach` still fails with `the search held the GIL`.

| Test | Mutation | Failure message |
| --- | --- | --- |
| `bordered_core_requires_square_shape` | Remove the square-core check | `a nonsquare core must be rejected without a panic: Any { .. }` |
| `bordered_family_requires_vector_count` | Remove the vector-count check | `validation must report the wrong vector count` |
| `bordered_family_requires_relation_length` | Remove the border-length check | `validation must report the wrong relation length` |
| `bordered_family_requires_vector_length` | Remove the coefficient-length check | `validation must report the wrong vector length` |
| `bordered_family_requires_odd_relation` | Remove the odd-parity check | `validation must report an even relation` |
| `bordered_family_requires_unit_lower_form` | Remove the unit-lower check | `validation must report a non-unit-lower form` |
| `bordered_family_requires_zero_relation_sum` | Remove the relation-sum check | `validation must report a nonzero relation sum` |
| `bordered_family_requires_full_rank` | Remove the full-rank check | `validation must report a missing rank` |
| `bordered_verification_rejects_identity_padding` | Allow zero factors | `identity padding must not pass verification` |
| `bordered_verification_requires_rank_plus_one_factors` | Remove the factor-count check | `verification must reject the wrong factor count` |
| `bordered_public_nonsymmetric_path_does_not_fall_back` | Transpose the core at the public call site | `the public nonsymmetric path must use the construction` |
| `retained_search_covers_one_and_two_qubit_actions` | Return zero instead of the accepted fix | `the retained search must return a nonzero fix` |
| `retained_search_covers_three_qubit_actions` | Return zero instead of the accepted fix | `the retained search must return a nonzero fix` |
| `test_minimal_decomposition_releases_the_gil` | Remove `py.detach` | `the search held the GIL` |

## Suite costs

These measurements compare `20a44da` before the five review fixes with `6421570` after them.
Both revisions already contain the construction.
Revision `b4d8a1f` replaces the four-block GIL input with the sub-second input.
That change does not alter Rust code.

The Rust command is `cargo test -p paulimer --all-features`, run from the repository root.
It uses the default unoptimized test profile, not the gate's `ci-test` profile.
The Python command is `python -m pytest -q tests`, run from `paulimer/bindings/python`.
The extension uses a release build.
All measurements use the same private environment and the default affinity of CPUs 0 through 15.

Each suite has one warmup and three measured runs.
The table reports whole-command wall time, including command startup and test discovery.
Rust builds are cached, and extension builds occur before timing starts.
Every measured run passes.

| Suite | Revision | Median, seconds | Range, seconds |
| --- | --- | ---: | ---: |
| Rust | `20a44da` | 32.235 | 31.196 to 33.014 |
| Rust | `6421570` | 31.224 | 30.935 to 46.213 |
| Python | `20a44da` | 4.581 | 4.531 to 4.661 |
| Python | `6421570` | 11.308 | 10.927 to 11.499 |
| Python | `b4d8a1f` | 4.764 | 4.641 to 4.819 |

The Rust ranges overlap, so these samples do not establish a speedup.
The sub-second GIL input removes most of the added Python suite cost.
Its suite median is 0.182 seconds above the baseline median.

The direct-search command is `cargo test -p paulimer --all-features --lib retained_search_covers`.
It uses the same unoptimized profile and cached build.
The normal row reports a median after one warmup.
The ignored row reports one completed run.

| Direct-search mode | Measured runs | Wall time, seconds |
| --- | ---: | ---: |
| Normal | 3 | 0.434 |
| With `-- --include-ignored` | 1 | 613.920 |

The normal command costs less than half a second in this measurement.
The expensive three-qubit part already uses `#[ignore]`.
The normal test run does not execute that search.

## Measured costs

The measurements use the release Python extension on Linux x86_64.
The host has an AMD EPYC 7763 processor and uses Rust 1.99.0 and Python 3.13.15.
Both builds and all inputs use the same environment.
The before build disables the construction and uses the original retained search.
The after build enables the construction and all acceptance checks.
Neither build contains timing instrumentation for the tables below.

Each process runs on CPU 6 with `taskset -c 6`.
Before each batch, that CPU was at least 98 percent idle during a one-second sample.
No builds or other local validation runs overlap these measurements.
Each elapsed time covers one `to_transvections_minimal()` call with `time.perf_counter_ns()`.
Input construction and replay occur outside the timed call.
Each completed case has one untimed warmup.
The tables report medians and sample counts for each build.
These measurements are observations, not runtime bounds.

A SWAP layer exchanges each adjacent pair of qubits.

| SWAP qubits | Factors before and after | Runs per build | Before median, seconds | After median, seconds |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 3 | 31 | 0.0000364 | 0.0000497 |
| 4 | 5 | 31 | 0.0000598 | 0.0000763 |
| 8 | 9 | 31 | 0.0001447 | 0.0001666 |
| 32 | 33 | 5 | 0.511360 | 0.003588 |
| 34 | 35 | 5 | 1.108985 | 0.004342 |
| 36 | 37 | 5 | 2.454685 | 0.005234 |
| 40 | 41 | 5 | 11.920274 | 0.007573 |

The small SWAP layers are slower with the construction.
Their medians increase by about 37, 28, and 15 percent at 2, 4, and 8 qubits.
The construction adds coefficient work, witness checks, and exact replay before accepting its candidate.
Those costs exceed the small retained-search costs for these inputs.
The larger SWAP layers avoid two expensive updated-core searches.

A separate instrumented release run measures those search stages at 40 qubits.
It uses five calls after one warmup, pinned to CPU 6.
Every call accepts the first fix candidate.
The initial congruence search has a median of 0.002995 seconds.
The retained fix search has a median of 6.133 seconds.
Recursive reconstruction has a median of 6.112 seconds.
The last two stages each run congruence triangularization on the updated core.
The construction avoids both stages without changing the search implementation.

Each class-A block applies $X_0$, $X_1$, $X_0X_1$, and $Z_0$ transvections in that order.
Several blocks use disjoint qubit pairs.
Two through four blocks complete in the initial congruence search, so they never use the construction.
Their timing differences do not measure construction costs.

| Class-A blocks | Factors before and after | Runs per build | Before median, seconds | After median, seconds |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 31 | 0.0000392 | 0.0000653 |
| 2 | 6 | 31 | 0.0001863 | 0.0001849 |
| 3 | 9 | 31 | 0.011932 | 0.012157 |
| 4 | 12 | 5 | 6.523068 | 6.848681 |

One class-A block uses the construction and is slower, not faster.
Its median increases by about 67 percent.
It incurs the same extra construction and acceptance costs as the small SWAP inputs.
The earlier claimed improvement does not reproduce.

Five blocks do not finish within 120 seconds in either build.
All three trials per build stop at that limit.
Peak memory for this case cannot be measured reproducibly from these runs.
The search does not finish, and the figure depends on when the measurement stops.
This note therefore reports no five-block memory figure.
The unchanged initial search still limits class-A inputs.
The construction does not give the complete minimal decomposer a polynomial runtime bound.
