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
//! the symplectic action and `Fix(F)` is the space of Pauli operators fixed by conjugation.
//!
//! A symplectic action is *hyperbolic* when `⟨v, vF⟩ = 0` for every `v`. That condition is stronger
//! than the residue form having zero diagonal, which only tests the basis vectors. The paper states
//! that every non-hyperbolic action attains `r` factors. That statement is too strong over GF(2).
//! Of the 720 elements of `Sp(4;2)`, 210 non-hyperbolic elements need `r + 1` factors, and so does
//! every non-identity hyperbolic element. For the 90 of those elements that have residue rank 3,
//! every rank-lowering transvection leaves a hyperbolic action of rank 2.
//!
//! The decomposition here uses a greedy O'Meara-style reduction. It returns a linear number of
//! factors and reproduces the symplectic action exactly, but it is not a minimal-length algorithm.
//! It attains the minimum on every element of `Sp(4;2)`. From three qubits upward it can exceed the
//! minimum, and the excess can grow with the qubit count.
//!
//! This decomposition reproduces the symplectic action only. The Pauli-image signs and the global
//! phase are not reproduced.

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
/// Panics if the reduction exceeds its linear termination bound. A valid Clifford cannot trigger
/// this, so a panic here reports a defect in the reduction rather than an invalid input.
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

/// Returns generators of the fixed space `Fix(F)`: Paulis with `clifford · P · clifford† = ±P`.
///
/// This is the projective centralizer of the Clifford in the Pauli group with phase quotiented out.
/// In contrast, [`crate::pauli_group::centralizer_of`] requires exact commutation.
/// For example, Clifford `X` fixes `Z` up to sign because `X Z X = -Z`.
/// Thus `Z` belongs to this fixed space but not to the centralizer of `X`.
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
/// use paulimer::clifford::{clifford_fixed_space, Clifford, CliffordMutable};
///
/// let mut clifford = CliffordUnitary::identity(1);
/// clifford.left_mul_root_z(0); // S fixes Z, sends X -> Y
///
/// let generators = clifford_fixed_space(&clifford);
/// // Every generator is fixed (up to sign) under conjugation.
/// assert!(generators.iter().all(|pauli| {
///     let image = clifford.image(pauli);
///     image.x_bits() == pauli.x_bits() && image.z_bits() == pauli.z_bits()
/// }));
/// ```
#[must_use]
pub fn clifford_fixed_space(clifford: &CliffordUnitary) -> Vec<SparsePauli> {
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
