//! Exact unsigned transvection minimization with a final signed Pauli correction.
//!
//! The rotation count minimizes the binary symplectic action, not a signed gate count.
//! The final Pauli restores all tableau signs. The input does not record global phase.
//!
//! ## Minimum rotation count
//!
//! The residue rank `r = rank(I + F)` is a lower bound on rotations for the binary action `F`.
//! The unsigned minimum is `r` when the residue core is congruence-triangularizable, and `r + 1` otherwise.
//! A nonzero alternating core requires the extra rotation.
//!
//! A non-hyperbolic action does not always attain `r` over GF(2).
//! Among the 720 elements of `Sp(4;2)`, 210 non-hyperbolic elements require `r + 1` rotations.
//! See `docs/transvection-minimality-correction.md` for the criterion, evidence, and limits of the residue fix search.

use std::collections::HashSet;

use binar::matrix::AlignedBitMatrix;
use binar::vec::AlignedBitVec;
use binar::{Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IndexSet};

use super::transvection::{assign_positive_hermitian_phase, symplectic_basis};
use super::{Clifford, CliffordMutable, CliffordUnitary, PreimageViews};
use crate::{Pauli, SparsePauli};

/// Ordered quarter-turns and a final Pauli that reproduce a signed Clifford action.
///
/// The input tableau does not record global phase, so this result does not reconstruct it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransvectionDecomposition {
    /// Positive Hermitian generators, applied in list order with
    /// [`CliffordMutable::left_mul_pauli_exp`].
    pub transvections: Vec<SparsePauli>,
    /// Positive Hermitian Pauli applied last with [`CliffordMutable::left_mul_pauli`],
    /// not `left_mul_pauli_exp`. Identity means no correction is needed.
    pub pauli_correction: SparsePauli,
}

/// Decomposes `clifford` into a minimum unsigned rotation list and a final Pauli correction.
///
/// Apply the returned `transvections` in list order to identity with
/// [`CliffordMutable::left_mul_pauli_exp`]. Apply `pauli_correction` last with
/// [`CliffordMutable::left_mul_pauli`]. The result equals the input signed tableau.
/// Global phase is not reconstructed because the input tableau does not record it.
///
/// The number of rotations `k` is the unsigned minimum: `k = r` when the residue core is
/// congruence-triangularizable and `k = r + 1` otherwise, where `r = 2n − dim Fix(clifford)` is the
/// dimension of the residue space (see [`super::clifford_fixed_space`] for `Fix`). This corrects the
/// minimality criterion of [arXiv:2102.11380](https://arxiv.org/abs/2102.11380).
/// See the module's minimum rotation count section and `docs/transvection-minimality-correction.md` for the correction.
/// Contrast with [`super::clifford_to_transvections`], which returns an O(n)-factor
/// decomposition that can exceed the minimum by Θ(n) on structured inputs.
/// This minimum excludes the final Pauli. It is not a minimum signed gate count.
///
/// Every factor is returned with xyz phase exponent `0` (the positive Hermitian representative).
/// Its xz phase exponent is its number of Y factors modulo 4. The sign of a transvection does not
/// affect its symplectic action. Changing a factor's sign requires a new final correction.
///
/// # Panics
///
/// The input must be a valid Clifford, as reported by `is_valid`. An invalid tableau, such as the
/// one produced by `CliffordUnitary::zero`, can make the residue-fix step fail and panic.
/// Candidate exhaustion reaches `unreachable!`. A separate Lean proof establishes residue-fix
/// existence in every dimension, but does not verify the compiled Rust implementation.
/// The public oracles cover complete decompositions through three qubits, not the retained search.
/// Separate direct tests exercise the search on every rank-plus-one action in those domains.
/// The three-qubit direct test is ignored by default.
/// See `docs/transvection-minimality-correction.md` for the proof links and implementation limits.
///
/// # Running time and memory
///
/// This exact minimal search can be exponential in the residue rank, in both running time and
/// memoization space. It can be impractical on structured high-rank inputs such as Callan
/// exceptional sums. The search rejects alternating restrictions without enumerating their spans,
/// which avoids the initial exponential scan for a SWAP layer.
/// It memoizes failed subspaces and generates candidates lazily, but it gives no polynomial bound.
/// When the core is not triangularizable, a bordered construction supplies a candidate factorization.
/// The code checks its dimensions, nonzero factors, factor count, and exact unsigned (symplectic) action before acceptance.
/// Construction or verification failure uses the retained exhaustive residue search.
/// See `docs/bordered-transvection-construction.md` for the algorithm and its measured coverage.
/// See `docs/transvection-minimality-correction.md` for measured limits.
/// Use [`super::clifford_to_transvections`] when a linear factor count is sufficient and strict minimality
/// is unnecessary.
///
/// # Examples
///
/// ```
/// use paulimer::CliffordUnitary;
/// use paulimer::clifford::{clifford_to_transvections_minimal, Clifford, CliffordMutable};
/// use paulimer::pauli::Pauli;
///
/// let mut clifford = CliffordUnitary::identity(2);
/// clifford.left_mul_hadamard(0);
/// clifford.left_mul_root_z(0);
///
/// let decomposition = clifford_to_transvections_minimal(&clifford);
/// assert!(decomposition.transvections.iter().all(|pauli| pauli.is_order_two()));
///
/// let mut rebuilt = CliffordUnitary::identity(2);
/// for pauli in &decomposition.transvections {
///     rebuilt.left_mul_pauli_exp(pauli);
/// }
/// rebuilt.left_mul_pauli(&decomposition.pauli_correction);
/// assert_eq!(rebuilt, clifford);
/// ```
#[must_use]
pub fn clifford_to_transvections_minimal(clifford: &CliffordUnitary) -> TransvectionDecomposition {
    let qubit_count = clifford.num_qubits();
    let action = action_matrix(clifford);
    let vectors = minimal_decomposition(&action, qubit_count);
    let transvections = vectors
        .iter()
        .map(|vector| vector_to_pauli(vector, qubit_count))
        .collect::<Vec<_>>();
    let pauli_correction = final_pauli_correction(clifford, &transvections);
    TransvectionDecomposition {
        transvections,
        pauli_correction,
    }
}

/// For target = P * rebuilt, target† X target = rebuilt† (P X P) rebuilt.
/// Thus X-preimage sign differences give P's Z bits, and Z differences give its X bits.
/// Equal binary preimages have equal Y weights, so their xz-phase differences are signs.
fn final_pauli_correction(clifford: &CliffordUnitary, transvections: &[SparsePauli]) -> SparsePauli {
    let qubit_count = clifford.num_qubits();
    let mut rebuilt = CliffordUnitary::identity(qubit_count);
    for transvection in transvections {
        rebuilt.left_mul_pauli_exp(transvection);
    }
    debug_assert_eq!(
        clifford.symplectic_matrix(),
        rebuilt.symplectic_matrix(),
        "the binary preimages must agree before correcting their signs"
    );
    let x_bits: IndexSet = (0..qubit_count)
        .filter(|&qubit| {
            clifford.preimage_z_view(qubit).xz_phase_exponent() != rebuilt.preimage_z_view(qubit).xz_phase_exponent()
        })
        .collect();
    let z_bits: IndexSet = (0..qubit_count)
        .filter(|&qubit| {
            clifford.preimage_x_view(qubit).xz_phase_exponent() != rebuilt.preimage_x_view(qubit).xz_phase_exponent()
        })
        .collect();
    let mut correction = SparsePauli::from_bits(x_bits, z_bits, 0);
    assign_positive_hermitian_phase(&mut correction);
    correction
}

/// The `2n × 2n` symplectic action matrix of `clifford`, in the "image" convention: row `k` is the
/// symplectic vector of the image of the `k`-th standard basis Pauli (`X₀, …, X_{n−1}, Z₀, …,
/// Z_{n−1}`), with `x`-bits in columns `[0, n)` and `z`-bits in columns `[n, 2n)`.
fn action_matrix(clifford: &CliffordUnitary) -> AlignedBitMatrix {
    let qubit_count = clifford.num_qubits();
    let dimension = 2 * qubit_count;
    let mut matrix = AlignedBitMatrix::zeros(dimension, dimension);
    for (row, basis) in symplectic_basis(qubit_count).enumerate() {
        let image = clifford.image(&basis);
        for qubit in 0..qubit_count {
            if image.x_bits().index(qubit) {
                matrix.set((row, qubit), true);
            }
            if image.z_bits().index(qubit) {
                matrix.set((row, qubit_count + qubit), true);
            }
        }
    }
    matrix
}

/// The symplectic transvection matrix `T_v` (row `k` = `e_k + ⟨e_k, v⟩·v`), whose row-vector action
/// `x ↦ x·T_v` equals `x + ⟨x, v⟩·v`. The solver applies `T_v` with [`right_multiply_transvection`];
/// this dense, bit-by-bit form is the independent reference for its tests.
#[cfg(test)]
fn transvection_matrix(vector: &AlignedBitVec, qubit_count: usize) -> AlignedBitMatrix {
    let dimension = 2 * qubit_count;
    let mut matrix = AlignedBitMatrix::identity(dimension);
    for row in 0..dimension {
        let coupling = if row < qubit_count {
            vector.index(qubit_count + row)
        } else {
            vector.index(row - qubit_count)
        };
        if coupling {
            for column in vector.support() {
                matrix.negate((row, column));
            }
        }
    }
    matrix
}

/// The vector `ṽ` with `x·ṽᵀ = ⟨x, v⟩` for every `x`: the `x` and `z` halves of `v` swapped.
fn symplectic_dual(vector: &AlignedBitVec, qubit_count: usize) -> AlignedBitVec {
    let mut dual = AlignedBitVec::zeros(2 * qubit_count);
    for bit in vector.support() {
        dual.assign_index(
            if bit < qubit_count {
                bit + qubit_count
            } else {
                bit - qubit_count
            },
            true,
        );
    }
    dual
}

/// The residue matrix `F̂ = Ω·(I + F)`, where `Ω` swaps the `x` and `z` halves of the rows. Its row
/// space is the residue space `Res(F)`.
fn residue_matrix(action: &AlignedBitMatrix, qubit_count: usize) -> AlignedBitMatrix {
    let dimension = 2 * qubit_count;
    let mut residue = AlignedBitMatrix::zeros(dimension, dimension);
    for row in 0..dimension {
        let swapped = if row < qubit_count {
            row + qubit_count
        } else {
            row - qubit_count
        };
        for column in 0..dimension {
            let mut bit = action.get((swapped, column));
            if swapped == column {
                bit ^= true;
            }
            if bit {
                residue.set((row, column), true);
            }
        }
    }
    residue
}

/// Row-reduces `matrix` to reduced echelon form while tracking the transform.
///
/// Returns `(basis, transform)` where `basis` holds the `r` nonzero echelon rows (a basis of the row
/// space) and `transform` is `r × rows` with `basis = transform · matrix`. Pivoting is over the
/// columns of `matrix` only.
fn row_reduce_with_transform(matrix: &AlignedBitMatrix) -> (AlignedBitMatrix, AlignedBitMatrix) {
    let rows = matrix.row_count();
    let columns = matrix.column_count();
    let mut augmented = AlignedBitMatrix::zeros(rows, columns + rows);
    for row in 0..rows {
        for column in 0..columns {
            if matrix.get((row, column)) {
                augmented.set((row, column), true);
            }
        }
        augmented.set((row, columns + row), true);
    }
    let mut pivot_row = 0;
    for column in 0..columns {
        let Some(selected) = (pivot_row..rows).find(|&row| augmented.get((row, column))) else {
            continue;
        };
        augmented.swap_rows(pivot_row, selected);
        for row in 0..rows {
            if row != pivot_row && augmented.get((row, column)) {
                augmented.add_into_row(row, pivot_row);
            }
        }
        pivot_row += 1;
    }
    let rank = pivot_row;
    let mut basis = AlignedBitMatrix::zeros(rank, columns);
    let mut transform = AlignedBitMatrix::zeros(rank, rows);
    for row in 0..rank {
        for column in 0..columns {
            if augmented.get((row, column)) {
                basis.set((row, column), true);
            }
        }
        for column in 0..rows {
            if augmented.get((row, columns + column)) {
                transform.set((row, column), true);
            }
        }
    }
    (basis, transform)
}

/// Copies row `index` of `matrix`.
fn matrix_row(matrix: &AlignedBitMatrix, index: usize) -> AlignedBitVec {
    AlignedBitVec::from_view(&matrix.row(index))
}

/// The sum `left + right` of two vectors of the same dimension.
fn xor_vectors(left: &AlignedBitVec, right: &AlignedBitVec) -> AlignedBitVec {
    let mut sum = left.clone();
    sum.bitxor_assign(right);
    sum
}

/// The unit vectors `e_0, …, e_{dimension−1}`.
fn standard_basis(dimension: usize) -> Vec<AlignedBitVec> {
    (0..dimension)
        .map(|index| {
            let mut unit = AlignedBitVec::zeros(dimension);
            unit.assign_index(index, true);
            unit
        })
        .collect()
}

/// Whether `vector` is stored for `dimension` bits and has no bit set outside them.
fn lies_in(vector: &AlignedBitVec, dimension: usize) -> bool {
    vector.len() == AlignedBitVec::zeros(dimension).len() && vector.max_support().is_none_or(|bit| bit < dimension)
}

/// The row vector `x·E`, so that the bilinear form `x·E·yᵀ` is `left_form(core, x).dot(y)`.
fn left_form(core: &AlignedBitMatrix, vector: &AlignedBitVec) -> AlignedBitVec {
    core.right_multiply(&vector.as_view())
}

/// Overwrites `form` with `x·E`, reusing its storage. `form` must be stored for `E`'s columns.
fn left_form_into(core: &AlignedBitMatrix, vector: &AlignedBitVec, form: &mut AlignedBitVec) {
    form.clear();
    for row in vector.support() {
        form.bitxor_assign(&core.row(row));
    }
}

/// Whether `x·E·yᵀ` is alternating on `span(basis)`: zero on the diagonal and symmetric.
/// Builds each basis vector's form only when it is reached, so a failure stops further work;
/// the first form uses the reusable `form` buffer and needs no allocation.
fn is_alternating(core: &AlignedBitMatrix, basis: &[AlignedBitVec], form: &mut AlignedBitVec) -> bool {
    let Some(first) = basis.first() else {
        return true;
    };
    left_form_into(core, first, form);
    if form.dot(first) {
        return false;
    }
    let mut forms: Vec<AlignedBitVec> = Vec::with_capacity(basis.len());
    forms.push(form.clone());
    for (index, vector) in basis.iter().enumerate().skip(1) {
        let vector_form = left_form(core, vector);
        if vector_form.dot(vector) || (0..index).any(|other| vector_form.dot(&basis[other]) != forms[other].dot(vector))
        {
            return false;
        }
        forms.push(vector_form);
    }
    true
}

/// Stacks `vectors`, each of dimension `columns`, as the rows of a matrix.
fn vectors_to_matrix(vectors: &[AlignedBitVec], columns: usize) -> AlignedBitMatrix {
    AlignedBitMatrix::from_row_iter(vectors.iter().map(AlignedBitVec::as_view), columns)
}

/// Attempts to triangularize the `r × r` matrix `core` by congruence.
///
/// On success returns `Ok(q)` with `q ∈ GL(r, 2)` such that `q·core·qᵀ` is lower triangular; the
/// rows of `q` are an ordered basis in which each vector is right-orthogonal (under the form
/// `x·core·yᵀ`) to all later ones and non-isotropic (`x·core·xᵀ = 1`). Since `core` is invertible,
/// a lower-triangular `q·core·qᵀ` automatically has an all-ones diagonal.
///
/// A triangularization exists exactly when the associated symplectic map is a product of `r`
/// transvections. It is found by a backtracking search over the choice of each successive basis
/// vector: after picking a non-isotropic `pick`, the search recurses into its right-orthogonal
/// complement. A greedy (first-choice) search can dead-end even when a triangularization exists, so
/// the choices are explored exhaustively, with subspaces proven unsolvable memoized to prune the
/// search. On failure returns `Err(())`.
fn congruence_triangularize(core: &AlignedBitMatrix) -> Result<AlignedBitMatrix, ()> {
    let dimension = core.row_count();
    if dimension == 0 {
        return Ok(AlignedBitMatrix::zeros(0, 0));
    }
    let standard = standard_basis(dimension);
    let key = subspace_key(&standard, dimension);
    let mut unsolvable: HashSet<SubspaceKey> = HashSet::new();
    triangularize_subspace(core, &standard, &key, dimension, &mut unsolvable)
        .map(|picks| vectors_to_matrix(&picks, dimension))
        .ok_or(())
}

/// Backtracking core of [`congruence_triangularize`]: finds an ordered basis of `span(basis)` in
/// which each vector is non-isotropic and right-orthogonal to all later ones, or `None` if none
/// exists. Subspaces proven to have no such basis are recorded in `unsolvable` (keyed by their
/// canonical row-reduced form) so that they are never re-explored. `key` is
/// `subspace_key(basis, dimension)`, which the caller has already computed.
fn triangularize_subspace(
    core: &AlignedBitMatrix,
    basis: &[AlignedBitVec],
    key: &SubspaceKey,
    dimension: usize,
    unsolvable: &mut HashSet<SubspaceKey>,
) -> Option<Vec<AlignedBitVec>> {
    if basis.is_empty() {
        return Some(Vec::new());
    }
    if unsolvable.contains(key) {
        return None;
    }
    let mut form = AlignedBitVec::zeros(dimension);
    if is_alternating(core, basis, &mut form) {
        unsolvable.insert(key.clone());
        return None;
    }
    let mut explored: HashSet<SubspaceKey> = HashSet::new();
    let mut picks = span_vectors(basis);
    while let Some(pick) = picks.advance() {
        left_form_into(core, pick, &mut form);
        if !form.dot(pick) {
            continue;
        }
        let Some(complement) = right_orthogonal_complement(&form, basis) else {
            continue;
        };
        let complement_key = subspace_key(&complement, dimension);
        if explored.contains(&complement_key) {
            continue;
        }
        if let Some(mut rest) = triangularize_subspace(core, &complement, &complement_key, dimension, unsolvable) {
            let mut found = Vec::with_capacity(rest.len() + 1);
            found.push(pick.clone());
            found.append(&mut rest);
            return Some(found);
        }
        explored.insert(complement_key);
    }
    unsolvable.insert(key.clone());
    None
}

/// The reduced row-echelon basis that identifies a subspace in the search memo.
type SubspaceKey = Vec<AlignedBitVec>;

/// Lazily generates all `2ᵈ − 1` nonzero vectors in the span of a `d`-vector basis.
struct SpanVectors<'a> {
    basis: &'a [AlignedBitVec],
    coefficients: AlignedBitVec,
    current: AlignedBitVec,
    exhausted: bool,
}

impl<'a> SpanVectors<'a> {
    fn new(basis: &'a [AlignedBitVec]) -> Self {
        Self {
            basis,
            coefficients: AlignedBitVec::zeros(basis.len()),
            current: AlignedBitVec::zeros(basis.first().map_or(0, AlignedBitVec::len)),
            exhausted: basis.is_empty(),
        }
    }
}

impl SpanVectors<'_> {
    /// Steps to the next vector and lends it, so that callers clone only the vectors they keep.
    /// The coefficients count up in binary, so the order is ascending in binary coordinates.
    fn advance(&mut self) -> Option<&AlignedBitVec> {
        if self.exhausted {
            return None;
        }
        for (index, member) in self.basis.iter().enumerate() {
            self.coefficients.negate_index(index);
            self.current.bitxor_assign(member);
            if self.coefficients.index(index) {
                #[cfg(test)]
                tests::SPAN_VECTOR_VISITS.with(|visits| visits.set(visits.get() + 1));
                return Some(&self.current);
            }
        }
        self.exhausted = true;
        None
    }
}

impl Iterator for SpanVectors<'_> {
    type Item = AlignedBitVec;

    fn next(&mut self) -> Option<Self::Item> {
        self.advance().cloned()
    }
}

fn span_vectors(basis: &[AlignedBitVec]) -> SpanVectors<'_> {
    SpanVectors::new(basis)
}

/// A basis of `{y ∈ span(basis) : pick·core·yᵀ = 0}`, one dimension smaller than `basis`, or `None`
/// if `pick` is right-orthogonal to the whole span (which cannot happen for a non-isotropic `pick`).
/// The caller passes `form = pick·core`, computed with [`left_form`].
fn right_orthogonal_complement(form: &AlignedBitVec, basis: &[AlignedBitVec]) -> Option<Vec<AlignedBitVec>> {
    let pivot = basis.iter().position(|vector| form.dot(vector))?;
    let complement = basis
        .iter()
        .enumerate()
        .filter(|&(index, _)| index != pivot)
        .map(|(_, vector)| {
            if form.dot(vector) {
                xor_vectors(vector, &basis[pivot])
            } else {
                vector.clone()
            }
        })
        .collect();
    Some(complement)
}

/// A canonical key for the subspace spanned by `basis`: its reduced row-echelon basis, so that any
/// two bases of the same subspace produce the same key.
fn subspace_key(basis: &[AlignedBitVec], dimension: usize) -> SubspaceKey {
    let mut rows = basis.to_vec();
    let mut pivot = 0;
    for column in 0..dimension {
        if pivot == rows.len() {
            break;
        }
        let Some(selected) = (pivot..rows.len()).find(|&row| rows[row].index(column)) else {
            continue;
        };
        rows.swap(pivot, selected);
        let (before, rest) = rows.split_at_mut(pivot);
        let (reference, after) = rest.split_first_mut().expect("the pivot row exists");
        for vector in before.iter_mut().chain(after) {
            if vector.index(column) {
                vector.bitxor_assign(&*reference);
            }
        }
        pivot += 1;
    }
    rows.truncate(pivot);
    rows
}

/// The residue core `E` and its residue-space basis `V` for the action matrix `action`.
///
/// Returns `(basis, rank, core)` where `basis` (`rank × 2n`) spans `Res(F)` and `core = V·Rᵀ` with
/// `V = R·F̂` (`rank × rank`) is the matrix whose congruence-triangularizability governs minimality.
fn residue_core(action: &AlignedBitMatrix, qubit_count: usize) -> (AlignedBitMatrix, usize, AlignedBitMatrix) {
    let residue = residue_matrix(action, qubit_count);
    let (basis, transform) = row_reduce_with_transform(&residue);
    let rank = basis.row_count();
    let core = basis.dot(&transform.transposed());
    (basis, rank, core)
}

/// The minimal ordered transvection vectors for the symplectic action matrix `action`.
fn minimal_decomposition(action: &AlignedBitMatrix, qubit_count: usize) -> Vec<AlignedBitVec> {
    let (basis, rank, core) = residue_core(action, qubit_count);
    if rank == 0 {
        return Vec::new();
    }
    let Ok(transform) = congruence_triangularize(&core) else {
        // A bordered candidate can exceed the minimum, so the rank-length search must run first.
        let candidate = bordered_decomposition(&core, &basis);
        return verified_decomposition_or_search(action, qubit_count, &basis, candidate);
    };
    let defining = transform.dot(&basis);
    (0..rank).map(|row| matrix_row(&defining, row)).collect()
}

struct BorderedFamily {
    vectors: Vec<AlignedBitVec>,
    relation: Vec<bool>,
}

#[derive(Debug, PartialEq, Eq)]
enum BorderedFamilyIssue {
    CoreShape,
    VectorCount,
    RelationLength,
    VectorDimension,
    EvenRelation,
    NonUnitLower,
    NonzeroRelation,
    MissingRank,
}

impl BorderedFamily {
    fn is_valid(&self, core: &AlignedBitMatrix) -> bool {
        self.validate(core).is_ok()
    }

    fn validate(&self, core: &AlignedBitMatrix) -> Result<(), Vec<BorderedFamilyIssue>> {
        let dimension = core.row_count();
        if core.column_count() != dimension {
            return Err(vec![BorderedFamilyIssue::CoreShape]);
        }
        if self.relation.len() != self.vectors.len() {
            return Err(vec![BorderedFamilyIssue::RelationLength]);
        }
        if self.vectors.iter().any(|vector| !lies_in(vector, dimension)) {
            return Err(vec![BorderedFamilyIssue::VectorDimension]);
        }
        let mut issues = Vec::new();
        if self.vectors.len() != dimension + 1 {
            issues.push(BorderedFamilyIssue::VectorCount);
        }
        if self.relation.iter().filter(|&&bit| bit).count() % 2 != 1 {
            issues.push(BorderedFamilyIssue::EvenRelation);
        }
        let coefficients = vectors_to_matrix(&self.vectors, dimension);
        let gram = coefficients.dot(core).dot(&coefficients.transposed());
        let unit_lower = (0..self.vectors.len()).all(|row| {
            (row..self.vectors.len()).all(|column| {
                (gram.get((row, column)) ^ (self.relation[row] & self.relation[column])) == (row == column)
            })
        });
        if !unit_lower {
            issues.push(BorderedFamilyIssue::NonUnitLower);
        }
        let mut relation_sum = AlignedBitVec::zeros(dimension);
        for (vector, _) in self
            .vectors
            .iter()
            .zip(&self.relation)
            .filter(|(_, selected)| **selected)
        {
            relation_sum.bitxor_assign(vector);
        }
        if !relation_sum.is_zero() {
            issues.push(BorderedFamilyIssue::NonzeroRelation);
        }
        if row_reduce_with_transform(&coefficients).0.row_count() != dimension {
            issues.push(BorderedFamilyIssue::MissingRank);
        }
        if issues.is_empty() { Ok(()) } else { Err(issues) }
    }
}

/// Peels a non-isotropic vector, or constructs a simplex on an alternating restriction.
fn bordered_family(core: &AlignedBitMatrix, basis: &[AlignedBitVec]) -> Option<BorderedFamily> {
    let dimension = core.row_count();
    let coordinates = vectors_to_matrix(basis, dimension);
    let restricted = coordinates.dot(core).dot(&coordinates.transposed());
    let pick = (0..basis.len())
        .find(|&index| restricted.get((index, index)))
        .map(|index| basis[index].clone())
        .or_else(|| {
            (0..basis.len()).find_map(|first| {
                (first + 1..basis.len())
                    .find(|&second| restricted.get((first, second)) != restricted.get((second, first)))
                    .map(|second| xor_vectors(&basis[first], &basis[second]))
            })
        });
    if let Some(pick) = pick {
        let complement = right_orthogonal_complement(&left_form(core, &pick), basis)?;
        let mut family = bordered_family(core, &complement)?;
        family.vectors.insert(0, pick);
        family.relation.insert(0, false);
        Some(family)
    } else {
        let vectors = symplectic_simplex(core, basis)?;
        let relation = vec![true; vectors.len()];
        Some(BorderedFamily { vectors, relation })
    }
}

/// Extends the simplex on a pair's orthogonal complement by two points.
fn symplectic_simplex(core: &AlignedBitMatrix, basis: &[AlignedBitVec]) -> Option<Vec<AlignedBitVec>> {
    let Some(first) = basis.first() else {
        return Some(vec![AlignedBitVec::zeros(core.row_count())]);
    };
    let first_form = left_form(core, first);
    let partner = (1..basis.len()).find(|&index| first_form.dot(&basis[index]))?;
    let second = &basis[partner];
    let second_form = left_form(core, second);
    let mut complement = Vec::with_capacity(basis.len() - 2);
    for (index, vector) in basis.iter().enumerate().skip(1) {
        if index == partner {
            continue;
        }
        let mut projected = vector.clone();
        if second_form.dot(vector) {
            projected.bitxor_assign(first);
        }
        if first_form.dot(vector) {
            projected.bitxor_assign(second);
        }
        complement.push(projected);
    }
    let mut vectors = vec![second.clone(), xor_vectors(first, second)];
    vectors.extend(symplectic_simplex(core, &complement)?.into_iter().map(|mut vector| {
        vector.bitxor_assign(first);
        vector
    }));
    Some(vectors)
}

fn bordered_decomposition(core: &AlignedBitMatrix, basis: &AlignedBitMatrix) -> Option<Vec<AlignedBitVec>> {
    let rank = core.row_count();
    let family = bordered_family(core, &standard_basis(rank))?;
    if !family.is_valid(core) {
        return None;
    }
    let defining = vectors_to_matrix(&family.vectors, rank).dot(basis);
    Some(
        (0..defining.row_count())
            .map(|row| matrix_row(&defining, row))
            .collect(),
    )
}

/// Replaces `matrix` by `matrix·T_v` with one rank-one update per row: `row += ⟨row, v⟩·v`.
fn right_multiply_transvection(matrix: &mut AlignedBitMatrix, vector: &AlignedBitVec, qubit_count: usize) {
    let dual = symplectic_dual(vector, qubit_count);
    for row in 0..matrix.row_count() {
        if matrix.row(row).dot(&dual) {
            matrix.row_mut(row).bitxor_assign(vector);
        }
    }
}

fn verifies_bordered_decomposition(
    action: &AlignedBitMatrix,
    qubit_count: usize,
    rank: usize,
    vectors: &[AlignedBitVec],
) -> bool {
    let dimension = 2 * qubit_count;
    if vectors.len() != rank + 1
        || vectors
            .iter()
            .any(|vector| !lies_in(vector, dimension) || vector.is_zero())
    {
        return false;
    }
    let mut rebuilt = AlignedBitMatrix::identity(dimension);
    for vector in vectors {
        right_multiply_transvection(&mut rebuilt, vector, qubit_count);
    }
    rebuilt == *action
}

fn verified_decomposition_or_search(
    action: &AlignedBitMatrix,
    qubit_count: usize,
    basis: &AlignedBitMatrix,
    candidate: Option<Vec<AlignedBitVec>>,
) -> Vec<AlignedBitVec> {
    let rank = basis.row_count();
    if let Some(vectors) =
        candidate.filter(|vectors| verifies_bordered_decomposition(action, qubit_count, rank, vectors))
    {
        return vectors;
    }
    #[cfg(test)]
    bordered_tests::FALLBACK_CALLS.set(bordered_tests::FALLBACK_CALLS.get() + 1);
    let fix = find_fix_vector(action, qubit_count, basis, rank);
    let mut updated = action.clone();
    right_multiply_transvection(&mut updated, &fix, qubit_count);
    let mut vectors = minimal_decomposition(&updated, qubit_count);
    vectors.push(fix);
    vectors
}

/// Searches for a residue vector `v` such that `F·T_v` has a congruence-triangularizable residue core of
/// the same rank, so that `F` decomposes into `rank + 1` transvections.
///
/// A separate Lean proof establishes that a nonzero fix exists in every dimension.
/// Candidate exhaustion reaches `unreachable!`; the proof does not verify compiled Rust.
/// Direct tests exercise all 225 two-qubit and 150,255 three-qubit rank-plus-one cases.
/// No one-qubit action needs this step. The three-qubit direct test is ignored by default.
/// The public decomposition oracles do not exercise this retained search.
/// See `docs/transvection-minimality-correction.md` for the proof links and implementation limits.
///
/// Candidates are the nonzero residue vectors in ascending binary-coordinate order with respect to
/// the rows of `basis`. The search is exhaustive over `Res(F)`.
fn find_fix_vector(
    action: &AlignedBitMatrix,
    qubit_count: usize,
    basis: &AlignedBitMatrix,
    rank: usize,
) -> AlignedBitVec {
    let candidate_accepts = |vector: &AlignedBitVec| -> bool {
        if vector.is_zero() {
            return false;
        }
        let mut updated = action.clone();
        right_multiply_transvection(&mut updated, vector, qubit_count);
        let (_, updated_rank, updated_core) = residue_core(&updated, qubit_count);
        updated_rank == rank && congruence_triangularize(&updated_core).is_ok()
    };
    let rows: Vec<AlignedBitVec> = (0..rank).map(|row| matrix_row(basis, row)).collect();
    let mut candidates = span_vectors(&rows);
    while let Some(vector) = candidates.advance() {
        if candidate_accepts(vector) {
            return vector.clone();
        }
    }
    unreachable!("no residue fix vector exists for this non-triangularizable core")
}

/// Converts a `2n`-bit symplectic vector into a positive Hermitian Pauli (`x`-bits in `[0, n)`,
/// `z`-bits in `[n, 2n)`).
fn vector_to_pauli(vector: &AlignedBitVec, qubit_count: usize) -> SparsePauli {
    let x_bits: IndexSet = vector.support().take_while(|&bit| bit < qubit_count).collect();
    let z_bits: IndexSet = vector
        .support()
        .filter(|&bit| bit >= qubit_count)
        .map(|bit| bit - qubit_count)
        .collect();
    let mut pauli = SparsePauli::from_bits(x_bits, z_bits, 0);
    assign_positive_hermitian_phase(&mut pauli);
    pauli
}

#[cfg(test)]
#[path = "transvection_exact_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "transvection_bordered_tests.rs"]
mod bordered_tests;

#[cfg(test)]
#[path = "transvection_search_tests.rs"]
mod retained_search_tests;
