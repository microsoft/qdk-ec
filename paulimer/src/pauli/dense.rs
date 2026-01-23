use super::{PauliUnitaryProjective, SparsePauli};
use crate::pauli::{generic::PauliUnitary, NeutralElement, Pauli, PauliBinaryOps};
use binar::{vec::AlignedBitVec, Bitwise, BitwiseMut};

/// Dense representation of a Pauli operator using bit vectors.
///
/// A Pauli operator on n qubits is a tensor product of single-qubit Paulis {I, X, Y, Z}
/// with an optional phase factor {±1, ±i}. `DensePauli` stores this using two bit vectors
/// (one for X components, one for Z components) plus a phase exponent.
///
/// # Memory Layout
///
/// - X bits: one bit per qubit indicating X or Y component
/// - Z bits: one bit per qubit indicating Z or Y component  
/// - Phase: u8 exponent where phase = i^exp (0→+1, 1→+i, 2→-1, 3→-i)
///
/// Memory usage: O(n) where n is the number of qubits
///
/// # When to Use
///
/// Use `DensePauli` when:
/// - Operating on most or all qubits in the system
/// - Qubit count is small to moderate (< 1000s)
/// - Need fast iteration over all qubit positions
/// - Performing many multiplications or commutator checks
///
/// For operators acting on few qubits in large systems, prefer [`SparsePauli`].
///
/// # Examples
///
/// ```
/// use paulimer::{DensePauli, Pauli, commutes_with};
///
/// // Parse from string: Y ⊗ I ⊗ Z
/// let pauli: DensePauli = "YIZ".parse().unwrap();
/// assert_eq!(pauli.weight(), 2);  // Acts on 2 qubits
///
/// // Create single-qubit operators
/// let x0 = DensePauli::x(0, 3);  // X on qubit 0 of 3-qubit system
/// let z0 = DensePauli::z(0, 3);  // Z on qubit 0
///
/// // Check commutation (X and Z anticommute)
/// assert!(!commutes_with(&x0, &z0));
/// ```
///
/// # String Format
///
/// Dense notation: one character per qubit in order
/// - `"XII"` → X₀ ⊗ I₁ ⊗ I₂
/// - `"XYZ"` → X₀ ⊗ Y₁ ⊗ Z₂
/// - Phases: prefix with `+`, `-`, `+i`, `-i` (e.g., `"-XYZ"`)
pub type DensePauli = PauliUnitary<AlignedBitVec, u8>;

/// Dense representation of a projective Pauli operator (without phase tracking).
///
/// Like [`DensePauli`] but does not track the ±1, ±i phase factor. Useful when
/// only the Pauli structure matters, not the global phase.
pub type DensePauliProjective = PauliUnitaryProjective<AlignedBitVec>;
use quantum_core::PositionedPauliObservable;
use std::collections::HashMap;

impl From<&[PositionedPauliObservable]> for DensePauli {
    fn from(value: &[PositionedPauliObservable]) -> Self {
        let r: SparsePauli = value.into();
        match super::Pauli::max_support(&r) {
            Some(max_id) => {
                let mut dense = <DensePauli as NeutralElement>::neutral_element_of_size(max_id + 1);
                dense.assign(&r);
                dense
            }
            None => <DensePauli as NeutralElement>::default_size_neutral_element(),
        }
    }
}

impl<const LENGTH: usize> From<[PositionedPauliObservable; LENGTH]> for DensePauli {
    fn from(pauli_observable: [PositionedPauliObservable; LENGTH]) -> Self {
        pauli_observable.as_slice().into()
    }
}

impl<const LENGTH: usize> From<&[PositionedPauliObservable; LENGTH]> for DensePauli {
    fn from(pauli_observable: &[PositionedPauliObservable; LENGTH]) -> Self {
        pauli_observable.as_slice().into()
    }
}

impl From<Vec<PositionedPauliObservable>> for DensePauli {
    fn from(value: Vec<PositionedPauliObservable>) -> Self {
        value.as_slice().into()
    }
}

impl From<&Vec<PositionedPauliObservable>> for DensePauli {
    fn from(value: &Vec<PositionedPauliObservable>) -> Self {
        value.as_slice().into()
    }
}

pub fn dense_from<PauliLike: Pauli>(pauli: &PauliLike, qubit_count: usize) -> DensePauli
where
    DensePauli: PauliBinaryOps<PauliLike>,
{
    let mut result = DensePauli::neutral_element_of_size(qubit_count);
    result.assign(pauli);
    result
}

pub fn condense_from<PauliLike: Pauli<PhaseExponentValue = u8>, Hasher: std::hash::BuildHasher>(
    pauli: &PauliLike,
    mapping: &HashMap<usize, usize, Hasher>,
) -> DensePauli
where
    DensePauli: PauliBinaryOps<PauliLike>,
{
    let length = mapping.len();
    let mut x_bits = AlignedBitVec::zeros(length);
    for index in pauli.x_bits().support() {
        x_bits.assign_index(mapping[&index], true);
    }
    let mut z_bits = AlignedBitVec::zeros(length);
    for index in pauli.z_bits().support() {
        z_bits.assign_index(mapping[&index], true);
    }
    DensePauli::from_bits(x_bits, z_bits, pauli.xz_phase_exponent())
}
