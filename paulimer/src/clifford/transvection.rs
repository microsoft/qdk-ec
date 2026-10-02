//! Decomposition of Clifford unitaries into Clifford transvections (`π/4` Pauli exponents).
//!
//! A *Clifford transvection* is the `π/4` Pauli exponent `exp(iπ/4·P_v)`, whose conjugation action
//! on Pauli operators is the *symplectic transvection*
//!
//! ```text
//! x ↦ x + ⟨x, v⟩ v,
//! ```
//!
//! where `⟨·,·⟩` is the symplectic (commutation) form. This module follows the transvection
//! framework of [arXiv:2102.11380](https://arxiv.org/abs/2102.11380) (Pllaha, Volanto & Tirkkonen,
//! *Decomposition of Clifford Gates*): every Clifford is a product of transvections, and no
//! decomposition is shorter than the residue rank `r = rank(I + F) = 2n − dim Fix(F)`, where `F` is
//! the symplectic action and `Fix(F)` is the space of Pauli operators fixed by conjugation. The
//! exact minimum is `r` or `r + 1`.
//!
//! Two decompositions are provided:
//!
//! * [`clifford_to_transvections`] uses a greedy O'Meara-style reduction. It returns a linear
//!   number of factors and reproduces the symplectic action exactly, but it is not a
//!   minimal-length algorithm.
//! * [`clifford_to_transvections_minimal`] returns the strict minimum (`r` or `r + 1`) through a
//!   congruence triangulation of the residue core.
//!
//! Both decompositions reproduce the symplectic action only. The Pauli-image signs and the global
//! phase are not reproduced.
//!
//! ## The minimum factor count
//!
//! A symplectic action is *hyperbolic* when `⟨v, vF⟩ = 0` for every `v`. That condition is stronger
//! than the residue form having zero diagonal, which only tests the basis vectors. The paper states
//! that every non-hyperbolic action attains `r` factors. That statement is too strong over GF(2).
//! Of the 720 elements of `Sp(4;2)`, 210 non-hyperbolic elements need `r + 1` factors, and so does
//! every non-identity hyperbolic element. For the 90 of those elements that have residue rank 3,
//! every rank-lowering transvection leaves a hyperbolic action of rank 2.
//!
//! The criterion used here is: the minimum is `r` when the residue core is
//! congruence-triangularizable, and `r + 1` otherwise. Hyperbolicity is the special case where the
//! core is alternating.
//!
//! The greedy reduction attains the minimum on every element of `Sp(4;2)`. From three qubits upward
//! it can exceed the minimum, and the excess can grow with the qubit count.

use std::collections::HashSet;

use binar::matrix::{AlignedBitMatrix, kernel_basis_matrix};
use binar::{Bitwise, IndexSet};

use crate::clifford::{Clifford, CliffordMutable, CliffordUnitary};
use crate::pauli::DensePauli;
use crate::{Pauli, PauliBinaryOps, PauliMutable, SparsePauli, anti_commutes_with};

/// Decomposes `clifford` into an ordered product of Clifford transvections.
///
/// Returns a list of Hermitian Pauli operators `[P₁, …, P_k]` such that left-multiplying the
/// identity by the transvections `exp(iπ/4·P₁)`, then `exp(iπ/4·P₂)`, …, then `exp(iπ/4·P_k)`
/// reproduces the symplectic action of `clifford` (its conjugation map on Pauli operators).
/// The Pauli-image signs and the global phase are not reproduced.
///
/// The number of factors is linear in the qubit count and never fewer than the residue rank
/// `r = rank(I + F) = 2n − dim Fix(clifford)`. This is a greedy reduction, not a minimal-length
/// algorithm. See the module documentation for when the true minimum exceeds `r`.
///
/// Every factor is returned with xyz phase exponent `0` (the positive Hermitian representative).
/// Its xz phase exponent is its number of Y factors modulo 4. The sign of a transvection does not
/// affect its symplectic action, so `exp(iπ/4·P)` and `exp(−iπ/4·P)` are interchangeable here.
///
/// # Panics
///
/// The input must be a valid Clifford, as reported by `is_valid`. An invalid tableau, such as the
/// one produced by `CliffordUnitary::zero`, can make the reduction exceed its linear termination
/// bound and panic. For a valid input a panic here reports a defect in the reduction.
///
/// # Examples
///
/// ```
/// use paulimer::CliffordUnitary;
/// use paulimer::clifford::{clifford_to_transvections, Clifford, CliffordMutable};
///
/// let mut clifford = CliffordUnitary::identity(2);
/// clifford.left_mul_hadamard(0);
/// clifford.left_mul_cx(0, 1);
///
/// let transvections = clifford_to_transvections(&clifford);
///
/// let mut rebuilt = CliffordUnitary::identity(2);
/// for pauli in &transvections {
///     rebuilt.left_mul_pauli_exp(pauli);
/// }
/// assert!(rebuilt.is_valid());
/// // The symplectic actions agree (signs and global phase may differ).
/// assert_eq!(rebuilt.symplectic_matrix(), clifford.symplectic_matrix());
/// ```
#[must_use]
pub fn clifford_to_transvections(clifford: &CliffordUnitary) -> Vec<SparsePauli> {
    let qubit_count = clifford.num_qubits();
    let mut working = clifford.clone();
    let mut recorded = Vec::new();
    // Reduce the symplectic action to the identity by left-multiplying transvections `T_{v₁}, …,
    // T_{v_k}`, so that `T_{v_k} ⋯ T_{v₁} · F = I` and hence `F = T_{v₁} ⋯ T_{v_k}`. Replaying the
    // factors in reverse order rebuilds `F` from the identity.
    while let Some(transvection) = next_transvection(&working) {
        working.left_mul_pauli_exp(&transvection);
        recorded.push(transvection);
        assert!(
            recorded.len() <= 4 * qubit_count + 2,
            "transvection reduction exceeded its linear termination bound"
        );
    }
    recorded.reverse();
    recorded
}

/// Returns generators of the Clifford's centralizer: the Pauli operators fixed (up to sign) by
/// conjugation, i.e. the `P` with `clifford · P · clifford† = ±P`.
///
/// This is `Fix(F)`, the kernel of the residue map `P ↦ conj(P) · P`, computed as the left null
/// space of the residue matrix over GF(2). The returned Paulis are independent positive Hermitian
/// generators. The space they span has dimension `dim Fix(F) = 2n − r`, where `r = rank(I + F)` is
/// the residue rank. That rank is a lower bound on the number of factors returned by
/// [`clifford_to_transvections`], not the factor count itself.
///
/// # Examples
///
/// ```
/// use paulimer::{CliffordUnitary, Pauli};
/// use paulimer::clifford::{clifford_centralizer, Clifford, CliffordMutable};
///
/// let mut clifford = CliffordUnitary::identity(1);
/// clifford.left_mul_root_z(0); // S fixes Z, sends X -> Y
///
/// let generators = clifford_centralizer(&clifford);
/// // Every generator is fixed (up to sign) under conjugation.
/// assert!(generators.iter().all(|pauli| {
///     let image = clifford.image(pauli);
///     image.x_bits() == pauli.x_bits() && image.z_bits() == pauli.z_bits()
/// }));
/// ```
#[must_use]
pub fn clifford_centralizer(clifford: &CliffordUnitary) -> Vec<SparsePauli> {
    let qubit_count = clifford.num_qubits();
    let dimension = 2 * qubit_count;
    let mut residue = AlignedBitMatrix::zeros(dimension, dimension);
    for (row, basis) in symplectic_basis(qubit_count).enumerate() {
        let vector = residue_vector(&basis, &clifford.image(&basis));
        for qubit in 0..qubit_count {
            if vector.x_bits().index(qubit) {
                residue.set((row, qubit), true);
            }
            if vector.z_bits().index(qubit) {
                residue.set((row, qubit_count + qubit), true);
            }
        }
    }
    let kernel = kernel_basis_matrix(&residue.transposed());
    (0..kernel.row_count())
        .map(|row| {
            let x_bits: IndexSet = (0..qubit_count).filter(|&qubit| kernel[(row, qubit)]).collect();
            let z_bits: IndexSet = (0..qubit_count)
                .filter(|&qubit| kernel[(row, qubit_count + qubit)])
                .collect();
            let mut generator = SparsePauli::from_bits(x_bits, z_bits, 0);
            assign_positive_hermitian_phase(&mut generator);
            generator
        })
        .collect()
}

/// The `2n` standard basis Pauli operators `X₀, …, X_{n−1}, Z₀, …, Z_{n−1}`.
fn symplectic_basis(qubit_count: usize) -> impl Iterator<Item = SparsePauli> {
    (0..qubit_count)
        .map(move |qubit| SparsePauli::x(qubit, qubit_count))
        .chain((0..qubit_count).map(move |qubit| SparsePauli::z(qubit, qubit_count)))
}

/// The next transvection `T_v` reducing the residue of `working`, or `None` if `working` already
/// acts as the identity on Pauli operators (up to sign).
///
/// Following the O'Meara strategy of [arXiv:2102.11380](https://arxiv.org/abs/2102.11380): find a
/// vector `x` with `⟨x, conj(x)⟩ = 1` (`x` anticommutes with its own image) and set `v = x + conj(x)`
/// — a residue vector — which lowers the residue rank by one. If no such `x` exists but `working`
/// is non-trivial (the hyperbolic case), any nonzero residue vector `v` makes the action
/// non-hyperbolic while preserving the residue space, costing one extra transvection.
fn next_transvection(working: &CliffordUnitary) -> Option<SparsePauli> {
    let qubit_count = working.num_qubits();
    let basis: Vec<SparsePauli> = symplectic_basis(qubit_count).collect();
    let images: Vec<DensePauli> = basis.iter().map(|pauli| working.image(pauli)).collect();

    for (pauli, image) in basis.iter().zip(&images) {
        if anti_commutes_with(pauli, image) {
            return Some(residue_vector(pauli, image));
        }
    }

    let dimension = basis.len();
    for first in 0..dimension {
        for second in (first + 1)..dimension {
            let anticommuting =
                anti_commutes_with(&basis[first], &images[second]) ^ anti_commutes_with(&basis[second], &images[first]);
            if anticommuting {
                let mut sum = basis[first].clone();
                sum.mul_assign_left(&basis[second]);
                let mut image = images[first].clone();
                image.mul_assign_left(&images[second]);
                return Some(residue_vector(&sum, &image));
            }
        }
    }

    basis
        .iter()
        .zip(&images)
        .find(|(pauli, image)| !acts_trivially_on(pauli, image))
        .map(|(pauli, image)| residue_vector(pauli, image))
}

/// The residue vector `v = x + conj(x)` as a positive Hermitian Pauli (its symplectic vector is the
/// product `x · conj(x)`).
fn residue_vector(pauli: &SparsePauli, image: &DensePauli) -> SparsePauli {
    let mut vector: SparsePauli = image.clone().into();
    vector.mul_assign_left(pauli);
    assign_positive_hermitian_phase(&mut vector);
    vector
}

/// Sets the phase so that `pauli` is the positive Hermitian representative of its symplectic
/// vector. The `xyz` phase exponent equals the `xz` exponent minus the `Y` weight, so assigning
/// the `Y` weight as the `xz` exponent leaves the `xyz` exponent at zero.
fn assign_positive_hermitian_phase(pauli: &mut SparsePauli) {
    pauli.assign_phase_exp(u8::try_from(pauli.y_weight() % 4).expect("phase exponent fits in u8"));
}

/// Whether `image` equals `pauli` as a symplectic vector (i.e. conjugation fixes `pauli` up to sign).
fn acts_trivially_on(pauli: &SparsePauli, image: &DensePauli) -> bool {
    let mut difference: SparsePauli = image.clone().into();
    difference.mul_assign_left(pauli);
    difference.x_bits().is_zero() && difference.z_bits().is_zero()
}

/// Decomposes `clifford` into a **minimal** ordered product of Clifford transvections.
///
/// Returns a list of Hermitian Pauli operators `[P₁, …, P_k]` such that left-multiplying the
/// identity by the transvections `exp(iπ/4·P₁)`, then `exp(iπ/4·P₂)`, …, then `exp(iπ/4·P_k)`
/// reproduces the **symplectic action** of `clifford` (its conjugation map on Pauli operators).
/// The Pauli-image signs are *not* reproduced. No tableau-level decomposition reproduces the
/// global phase of the input unitary, because a tableau does not record it.
///
/// The number of factors `k` is the strict minimum: `k = r` when the residue core is
/// congruence-triangularizable and `k = r + 1` otherwise, where `r = 2n − dim Fix(clifford)` is the
/// dimension of the residue space (see [`clifford_centralizer`] for `Fix`). This corrects the
/// minimality criterion of [arXiv:2102.11380](https://arxiv.org/abs/2102.11380) (see the module
/// docs). Contrast with [`clifford_to_transvections`], which returns an O(n)-factor
/// decomposition that can exceed the minimum by Θ(n) on structured inputs.
///
/// Every factor is returned with xyz phase exponent `0` (the positive Hermitian representative).
/// Its xz phase exponent is its number of Y factors modulo 4. The sign of a transvection does not
/// affect its symplectic action, so `exp(iπ/4·P)` and `exp(−iπ/4·P)` are interchangeable here.
///
/// # Panics
///
/// The input must be a valid Clifford, as reported by `is_valid`. An invalid tableau, such as the
/// one produced by `CliffordUnitary::zero`, can make the residue-fix step fail and panic. The same
/// panic reports a residue-fix search that found no vector, which the exhaustive one-, two-, and
/// three-qubit tests rule out at those sizes.
///
/// # Running time and memory
///
/// This exact minimal search can be exponential in the residue rank, in both running time and
/// memoization space. It can be impractical on structured high-rank inputs such as swap layers or
/// Callan exceptional sums. A 20-qubit swap layer takes about one second, but a 10-qubit sum of
/// five Callan class-A blocks runs for minutes and uses hundreds of megabytes. The search memoizes
/// failed subspaces and generates candidates lazily, but the implementation alone gives no
/// polynomial bound. Use [`clifford_to_transvections`] when a linear factor count is sufficient and
/// strict minimality is unnecessary.
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
/// let transvections = clifford_to_transvections_minimal(&clifford);
/// assert!(transvections.iter().all(|pauli| pauli.is_order_two()));
///
/// let mut rebuilt = CliffordUnitary::identity(2);
/// for pauli in &transvections {
///     rebuilt.left_mul_pauli_exp(pauli);
/// }
/// assert!(rebuilt.is_valid());
/// assert_eq!(rebuilt.symplectic_matrix(), clifford.symplectic_matrix());
/// ```
#[must_use]
pub fn clifford_to_transvections_minimal(clifford: &CliffordUnitary) -> Vec<SparsePauli> {
    let qubit_count = clifford.num_qubits();
    let action = action_matrix(clifford);
    let vectors = minimal_decomposition(&action, qubit_count);
    vectors
        .iter()
        .map(|vector| vector_to_pauli(vector, qubit_count))
        .collect()
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
/// `x ↦ x·T_v` equals `x + ⟨x, v⟩·v`.
fn transvection_matrix(vector: &[bool], qubit_count: usize) -> AlignedBitMatrix {
    let dimension = 2 * qubit_count;
    let mut matrix = AlignedBitMatrix::identity(dimension);
    for row in 0..dimension {
        let coupling = if row < qubit_count {
            vector[qubit_count + row]
        } else {
            vector[row - qubit_count]
        };
        if coupling {
            for (column, &bit) in vector.iter().enumerate() {
                if bit {
                    matrix.negate((row, column));
                }
            }
        }
    }
    matrix
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

/// Extracts row `index` of `matrix` as a boolean vector of length `length`.
fn matrix_row(matrix: &AlignedBitMatrix, index: usize, length: usize) -> Vec<bool> {
    (0..length).map(|column| matrix.get((index, column))).collect()
}

/// The bitwise XOR of two equal-length boolean vectors.
fn xor_vectors(left: &[bool], right: &[bool]) -> Vec<bool> {
    left.iter().zip(right).map(|(&a, &b)| a ^ b).collect()
}

/// The value `x·E·yᵀ` of the bilinear form given by the square matrix `core`.
fn bilinear(core: &AlignedBitMatrix, left: &[bool], right: &[bool]) -> bool {
    let dimension = core.row_count();
    (0..dimension).fold(false, |acc, i| {
        let row = (0..dimension).fold(false, |inner, j| inner ^ (core.get((i, j)) & right[j]));
        acc ^ (left[i] & row)
    })
}

/// Packs `vectors` (each of length `columns`) into an `AlignedBitMatrix`.
fn vectors_to_matrix(vectors: &[Vec<bool>], columns: usize) -> AlignedBitMatrix {
    let mut matrix = AlignedBitMatrix::zeros(vectors.len(), columns);
    for (row, vector) in vectors.iter().enumerate() {
        for (column, &bit) in vector.iter().enumerate() {
            if bit {
                matrix.set((row, column), true);
            }
        }
    }
    matrix
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
    let standard: Vec<Vec<bool>> = (0..dimension)
        .map(|index| (0..dimension).map(|column| column == index).collect())
        .collect();
    let mut unsolvable: HashSet<Vec<bool>> = HashSet::new();
    triangularize_subspace(core, &standard, dimension, &mut unsolvable)
        .map(|picks| vectors_to_matrix(&picks, dimension))
        .ok_or(())
}

/// Backtracking core of [`congruence_triangularize`]: finds an ordered basis of `span(basis)` in
/// which each vector is non-isotropic and right-orthogonal to all later ones, or `None` if none
/// exists. Subspaces proven to have no such basis are recorded in `unsolvable` (keyed by their
/// canonical row-reduced form) so that they are never re-explored.
fn triangularize_subspace(
    core: &AlignedBitMatrix,
    basis: &[Vec<bool>],
    dimension: usize,
    unsolvable: &mut HashSet<Vec<bool>>,
) -> Option<Vec<Vec<bool>>> {
    if basis.is_empty() {
        return Some(Vec::new());
    }
    let key = subspace_key(basis, dimension);
    if unsolvable.contains(&key) {
        return None;
    }
    let mut explored: HashSet<Vec<bool>> = HashSet::new();
    for pick in span_vectors(basis) {
        if !bilinear(core, &pick, &pick) {
            continue;
        }
        let Some(complement) = right_orthogonal_complement(core, &pick, basis) else {
            continue;
        };
        let complement_key = subspace_key(&complement, dimension);
        if !explored.insert(complement_key) {
            continue;
        }
        if let Some(mut rest) = triangularize_subspace(core, &complement, dimension, unsolvable) {
            let mut picks = Vec::with_capacity(rest.len() + 1);
            picks.push(pick);
            picks.append(&mut rest);
            return Some(picks);
        }
    }
    unsolvable.insert(key);
    None
}

/// Lazily generates all `2ᵈ − 1` nonzero vectors in the span of a `d`-vector basis.
struct SpanVectors<'a> {
    basis: &'a [Vec<bool>],
    coefficients: Vec<bool>,
    current: Vec<bool>,
    exhausted: bool,
}

impl<'a> SpanVectors<'a> {
    fn new(basis: &'a [Vec<bool>]) -> Self {
        Self {
            basis,
            coefficients: vec![false; basis.len()],
            current: vec![false; basis.first().map_or(0, Vec::len)],
            exhausted: basis.is_empty(),
        }
    }
}

impl Iterator for SpanVectors<'_> {
    type Item = Vec<bool>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        for (index, member) in self.basis.iter().enumerate() {
            self.coefficients[index] ^= true;
            for (slot, &bit) in self.current.iter_mut().zip(member) {
                *slot ^= bit;
            }
            if self.coefficients[index] {
                return Some(self.current.clone());
            }
        }
        self.exhausted = true;
        None
    }
}

fn span_vectors(basis: &[Vec<bool>]) -> SpanVectors<'_> {
    SpanVectors::new(basis)
}

/// A basis of `{y ∈ span(basis) : pick·core·yᵀ = 0}`, one dimension smaller than `basis`, or `None`
/// if `pick` is right-orthogonal to the whole span (which cannot happen for a non-isotropic `pick`).
fn right_orthogonal_complement(core: &AlignedBitMatrix, pick: &[bool], basis: &[Vec<bool>]) -> Option<Vec<Vec<bool>>> {
    let couplings: Vec<bool> = basis.iter().map(|vector| bilinear(core, pick, vector)).collect();
    let pivot = couplings.iter().position(|&bit| bit)?;
    let mut complement = Vec::with_capacity(basis.len() - 1);
    for (index, vector) in basis.iter().enumerate() {
        if index == pivot {
            continue;
        }
        if couplings[index] {
            complement.push(xor_vectors(vector, &basis[pivot]));
        } else {
            complement.push(vector.clone());
        }
    }
    Some(complement)
}

/// A canonical key for the subspace spanned by `basis`: its rows reduced to reduced row-echelon
/// form and flattened, so that any two bases of the same subspace produce the same key.
fn subspace_key(basis: &[Vec<bool>], dimension: usize) -> Vec<bool> {
    let mut rows: Vec<Vec<bool>> = basis.to_vec();
    let mut pivot = 0;
    for column in 0..dimension {
        let Some(selected) = (pivot..rows.len()).find(|&row| rows[row][column]) else {
            continue;
        };
        rows.swap(pivot, selected);
        for row in 0..rows.len() {
            if row != pivot && rows[row][column] {
                let reference = rows[pivot].clone();
                for (slot, bit) in rows[row].iter_mut().zip(&reference) {
                    *slot ^= *bit;
                }
            }
        }
        pivot += 1;
    }
    rows.truncate(pivot);
    rows.into_iter().flatten().collect()
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
fn minimal_decomposition(action: &AlignedBitMatrix, qubit_count: usize) -> Vec<Vec<bool>> {
    let dimension = 2 * qubit_count;
    let (basis, rank, core) = residue_core(action, qubit_count);
    if rank == 0 {
        return Vec::new();
    }
    let Ok(transform) = congruence_triangularize(&core) else {
        let fix = find_fix_vector(action, qubit_count, &basis, rank);
        let updated = action.dot(&transvection_matrix(&fix, qubit_count));
        let mut vectors = minimal_decomposition(&updated, qubit_count);
        vectors.push(fix);
        return vectors;
    };
    let defining = transform.dot(&basis);
    (0..rank).map(|row| matrix_row(&defining, row, dimension)).collect()
}

/// Finds a residue vector `v` such that `F·T_v` has a congruence-triangularizable residue core of
/// the same rank, so that `F` decomposes into `rank + 1` transvections.
///
/// Callan gives the `r`/`r + 1` length bound over GF(2), so a decomposition of length `rank + 1`
/// exists. That such a vector always lies in `Res(F)` is verified exhaustively for one, two, and
/// three qubits by the integration tests. It is not proved here for larger qubit counts.
///
/// Candidates are the nonzero residue vectors in ascending binary-coordinate order. The search is
/// exhaustive over `Res(F)`.
fn find_fix_vector(action: &AlignedBitMatrix, qubit_count: usize, basis: &AlignedBitMatrix, rank: usize) -> Vec<bool> {
    let dimension = 2 * qubit_count;
    let lift = |coordinates: &[bool]| -> Vec<bool> {
        let mut vector = vec![false; dimension];
        for (row, &selected) in coordinates.iter().enumerate() {
            if selected {
                for (column, slot) in vector.iter_mut().enumerate() {
                    *slot ^= basis.get((row, column));
                }
            }
        }
        vector
    };
    let candidate_accepts = |vector: &[bool]| -> bool {
        if vector.iter().all(|&bit| !bit) {
            return false;
        }
        let updated = action.dot(&transvection_matrix(vector, qubit_count));
        let (_, updated_rank, updated_core) = residue_core(&updated, qubit_count);
        updated_rank == rank && congruence_triangularize(&updated_core).is_ok()
    };
    let coordinate_basis: Vec<Vec<bool>> = (0..rank)
        .map(|selected| (0..rank).map(|index| index == selected).collect())
        .collect();
    for coordinates in span_vectors(&coordinate_basis) {
        let vector = lift(&coordinates);
        if candidate_accepts(&vector) {
            return vector;
        }
    }
    panic!("no residue fix vector exists for this non-triangularizable core")
}

/// Converts a `2n`-bit symplectic vector into a positive Hermitian Pauli (`x`-bits in `[0, n)`,
/// `z`-bits in `[n, 2n)`).
fn vector_to_pauli(vector: &[bool], qubit_count: usize) -> SparsePauli {
    let x_bits: IndexSet = (0..qubit_count).filter(|&qubit| vector[qubit]).collect();
    let z_bits: IndexSet = (0..qubit_count).filter(|&qubit| vector[qubit_count + qubit]).collect();
    let mut pauli = SparsePauli::from_bits(x_bits, z_bits, 0);
    assign_positive_hermitian_phase(&mut pauli);
    pauli
}

#[cfg(test)]
mod tests {
    use super::span_vectors;

    #[test]
    fn span_vectors_supports_more_than_u64_bits_lazily() {
        let dimension = 65;
        let basis: Vec<Vec<bool>> = (0..dimension)
            .map(|row| (0..dimension).map(|column| row == column).collect())
            .collect();
        let vectors: Vec<Vec<bool>> = span_vectors(&basis).take(4).collect();

        assert_eq!(vectors.len(), 4);
        assert_eq!(vectors[0].iter().filter(|&&bit| bit).count(), 1);
        assert_eq!(vectors[1].iter().filter(|&&bit| bit).count(), 1);
        assert_eq!(vectors[2].iter().filter(|&&bit| bit).count(), 2);
        assert_eq!(vectors[3].iter().filter(|&&bit| bit).count(), 1);
    }
}
