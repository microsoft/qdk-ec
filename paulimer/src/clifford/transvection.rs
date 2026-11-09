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
//! *Decomposition of Clifford Gates*): every symplectic action is a product of transvections, and no
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
use binar::vec::AlignedBitVec;
use binar::{Bitwise, BitwisePair, BitwisePairMut, IndexSet};

use crate::clifford::{Clifford, CliffordUnitary};
use crate::{Pauli, PauliMutable, SparsePauli};

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
/// The reduction keeps a packed binary residue matrix and updates it with row XORs.
/// It computes Pauli-image bits once and does not track signs during the reduction.
///
/// Every factor is returned with xyz phase exponent `0` (the positive Hermitian representative).
/// Its xz phase exponent is its number of Y factors modulo 4. The sign of a transvection does not
/// affect its symplectic action, so `exp(iπ/4·P)` and `exp(−iπ/4·P)` are interchangeable here.
///
/// # Panics
///
/// The input must be a valid Clifford, as reported by `is_valid`.
/// An invalid tableau such as `CliffordUnitary::zero(1)` can exceed the linear termination bound and panic.
/// For a valid input, a panic reports a defect in the reduction.
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
    let mut residue = symplectic_residue_matrix(clifford);
    let mut recorded = Vec::new();
    // In the row-image convention, left multiplication of the Clifford gives F <- F T_v.
    // Reverse the recorded reductions to reconstruct F.
    while let Some(vector) = next_transvection(&residue, qubit_count) {
        let dual: AlignedBitVec = (qubit_count..2 * qubit_count)
            .chain(0..qubit_count)
            .map(|column| vector.index(column))
            .collect();
        for row in 0..residue.row_count() {
            // The residue is F + I, so include the identity row's pairing.
            if residue.row(row).dot(&dual.as_view()) ^ dual.index(row) {
                residue.row_mut(row).bitxor_assign(&vector.as_view());
            }
        }
        recorded.push(pauli_from_vector(&vector, qubit_count));
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
/// The residue is built directly from the packed tableau without computing signed Pauli images.
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
    let residue = symplectic_residue_matrix(clifford);
    let kernel = kernel_basis_matrix(&residue.transposed());
    (0..kernel.row_count())
        .map(|row| pauli_from_vector(&kernel.row(row), qubit_count))
        .collect()
}

fn symplectic_residue_matrix(clifford: &CliffordUnitary) -> AlignedBitMatrix {
    let qubit_count = clifford.num_qubits();
    let mut preimages = clifford.symplectic_matrix();
    // Image rows are F = Omega G^T Omega, where G contains preimage rows.
    for qubit in 0..qubit_count {
        preimages.swap_rows(qubit, qubit_count + qubit);
    }
    let mut residue = preimages.transposed();
    for qubit in 0..qubit_count {
        residue.swap_rows(qubit, qubit_count + qubit);
    }
    for diagonal in 0..2 * qubit_count {
        residue.negate((diagonal, diagonal));
    }
    residue
}

/// The `2n` standard basis Pauli operators `X₀, …, X_{n−1}, Z₀, …, Z_{n−1}`.
pub(super) fn symplectic_basis(qubit_count: usize) -> impl Iterator<Item = SparsePauli> {
    (0..qubit_count)
        .map(move |qubit| SparsePauli::x(qubit, qubit_count))
        .chain((0..qubit_count).map(move |qubit| SparsePauli::z(qubit, qubit_count)))
}

/// The next transvection `T_v` reducing the residue, or `None` if the Clifford already
/// acts as the identity on Pauli operators (up to sign).
///
/// Following the O'Meara strategy of [arXiv:2102.11380](https://arxiv.org/abs/2102.11380): find a
/// vector `x` with `⟨x, conj(x)⟩ = 1` (`x` anticommutes with its own image) and set `v = x + conj(x)`
/// — a residue vector — which lowers the residue rank by one. If no such `x` exists but the Clifford
/// is non-trivial (the hyperbolic case), any nonzero residue vector `v` makes the action
/// non-hyperbolic while preserving the residue space, costing one extra transvection.
fn next_transvection(residue: &AlignedBitMatrix, qubit_count: usize) -> Option<AlignedBitVec> {
    let dimension = residue.row_count();
    let paired_column = |index: usize| {
        if index < qubit_count {
            index + qubit_count
        } else {
            index - qubit_count
        }
    };
    for row in 0..dimension {
        if residue[(row, paired_column(row))] {
            return Some(AlignedBitVec::from_view(&residue.row(row)));
        }
    }

    for first in 0..dimension {
        for second in (first + 1)..dimension {
            let anticommuting = residue[(second, paired_column(first))] ^ residue[(first, paired_column(second))];
            if anticommuting {
                let mut vector = AlignedBitVec::from_view(&residue.row(first));
                vector.as_view_mut().bitxor_assign(&residue.row(second));
                return Some(vector);
            }
        }
    }

    (0..dimension)
        .find(|&row| !residue.row(row).is_zero())
        .map(|row| AlignedBitVec::from_view(&residue.row(row)))
}

fn pauli_from_vector(vector: &impl Bitwise, qubit_count: usize) -> SparsePauli {
    let x_bits: IndexSet = (0..qubit_count).filter(|&qubit| vector.index(qubit)).collect();
    let z_bits: IndexSet = (0..qubit_count)
        .filter(|&qubit| vector.index(qubit_count + qubit))
        .collect();
    let mut pauli = SparsePauli::from_bits(x_bits, z_bits, 0);
    assign_positive_hermitian_phase(&mut pauli);
    pauli
}

/// Sets the phase so that `pauli` is the positive Hermitian representative of its symplectic
/// vector. The `xyz` phase exponent equals the `xz` exponent minus the `Y` weight, so assigning
/// the `Y` weight as the `xz` exponent leaves the `xyz` exponent at zero.
pub(super) fn assign_positive_hermitian_phase(pauli: &mut SparsePauli) {
    pauli.assign_phase_exp(u8::try_from(pauli.y_weight() % 4).expect("phase exponent fits in u8"));
}

#[cfg(test)]
#[path = "transvection_tests.rs"]
mod tests;
