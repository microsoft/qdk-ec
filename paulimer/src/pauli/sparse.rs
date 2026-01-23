use crate::core::{PauliObservable, PositionedPauliObservable};
use binar::{Bitwise, BitwiseMut, IndexSet};
use std::collections::HashMap;

use crate::pauli::generic::{PauliCharacterError, PauliUnitary};

use super::{Pauli, PauliUnitaryProjective};

/// Sparse representation of a Pauli operator using index sets.
///
/// A Pauli operator on n qubits is a tensor product of single-qubit Paulis {I, X, Y, Z}
/// with an optional phase factor {±1, ±i}. `SparsePauli` stores only the non-identity
/// positions using index sets, making it memory-efficient for operators acting on few qubits.
///
/// # Memory Layout
///
/// - X indices: set of qubit positions with X or Y components
/// - Z indices: set of qubit positions with Z or Y components
/// - Phase: u8 exponent where phase = i^exp (0→+1, 1→+i, 2→-1, 3→-i)
///
/// Memory usage: O(k) where k is the weight (number of non-identity terms)
///
/// # When to Use
///
/// Use `SparsePauli` when:
/// - Operating on few qubits in large systems (e.g., 5 qubits out of 1000)
/// - Weight is much smaller than system size
/// - Need memory efficiency for many operators
///
/// For operators acting on most qubits, prefer [`crate::DensePauli`].
///
/// # Examples
///
/// ```
/// use paulimer::{SparsePauli, Pauli};
///
/// // Sparse notation: only specify non-identity positions
/// let pauli: SparsePauli = "X0 Z5 Y100".parse().unwrap();
/// assert_eq!(pauli.weight(), 3);
///
/// // Efficient for large sparse operators
/// let large_sparse: SparsePauli = "Z0 Z1000000".parse().unwrap();
/// // Uses O(2) memory, not O(1000000)
/// ```
///
/// # String Format
///
/// Sparse notation: space-separated positioned Paulis
/// - `"X0"` → X on qubit 0 (all other qubits are identity)
/// - `"X0 Z5"` → X₀ ⊗ I₁ ⊗ ... ⊗ I₄ ⊗ Z₅
/// - `"Y100 X200"` → operators on qubits 100 and 200
/// - Phases: prefix with `+`, `-`, `+i`, `-i` (e.g., `"-X0 Z5"`)
pub type SparsePauli = PauliUnitary<IndexSet, u8>;

/// Sparse representation of a projective Pauli operator (without phase tracking).
///
/// Like [`SparsePauli`] but does not track the ±1, ±i phase factor. Useful when
/// only the Pauli structure matters, not the global phase.
pub type SparsePauliProjective = PauliUnitaryProjective<IndexSet>;

// Note: Can be improved using PauliMutable trait
// Question: Should 'i' be interpreted as I or complex phase ?
impl<Hasher> TryFrom<HashMap<usize, char, Hasher>> for SparsePauli {
    type Error = PauliCharacterError;

    fn try_from(characters: HashMap<usize, char, Hasher>) -> Result<Self, Self::Error> {
        let mut x_bits = IndexSet::new();
        let mut z_bits = IndexSet::new();
        let mut exponent: u8 = 0;
        for (index, character) in characters {
            match character {
                'X' | 'x' => x_bits.assign_index(index, true),
                'Z' | 'z' => z_bits.assign_index(index, true),
                'Y' | 'y' => {
                    exponent = exponent.wrapping_add(1);
                    x_bits.assign_index(index, true);
                    z_bits.assign_index(index, true);
                }
                'I' => {}
                _ => return Err(PauliCharacterError {}),
            }
        }
        Ok(SparsePauli::from_bits(x_bits, z_bits, exponent))
    }
}

// Note: Allow repeated indicies so that conversion never fails and replace with generic that relies on PauliMutable
impl From<&[PositionedPauliObservable]> for SparsePauli {
    fn from(pauli_observable: &[PositionedPauliObservable]) -> Self {
        let mut obs_copy = Vec::from(pauli_observable);
        obs_copy.sort_unstable();
        if obs_copy.len() > 1 {
            for j in 0..obs_copy.len() - 1 {
                assert!(
                    obs_copy[j].qubit_id < obs_copy[j + 1].qubit_id,
                    "Repeated qubit positions"
                );
            }
        }

        let mut x_indices = IndexSet::new();
        let mut z_indices = IndexSet::new();
        let mut phase = 0u8;

        for PositionedPauliObservable { qubit_id, observable } in obs_copy {
            match observable {
                PauliObservable::PlusI => (),
                PauliObservable::MinusI => phase += 2,
                PauliObservable::PlusX => x_indices.assign_index(qubit_id, true),
                PauliObservable::PlusZ => z_indices.assign_index(qubit_id, true),
                PauliObservable::MinusX => {
                    x_indices.assign_index(qubit_id, true);
                    phase += 2;
                }
                PauliObservable::MinusZ => {
                    z_indices.assign_index(qubit_id, true);
                    phase += 2;
                }
                PauliObservable::PlusY => {
                    x_indices.assign_index(qubit_id, true);
                    z_indices.assign_index(qubit_id, true);
                    phase += 1;
                }
                PauliObservable::MinusY => {
                    x_indices.assign_index(qubit_id, true);
                    z_indices.assign_index(qubit_id, true);
                    phase += 3;
                }
            }
        }
        PauliUnitary::from_bits(x_indices, z_indices, phase)
    }
}

impl From<&[PositionedPauliObservable]> for SparsePauliProjective {
    fn from(pauli_observable: &[PositionedPauliObservable]) -> Self {
        let mut obs_copy = Vec::from(pauli_observable);
        obs_copy.sort_unstable();
        if obs_copy.len() > 1 {
            for j in 0..obs_copy.len() - 1 {
                assert!(
                    obs_copy[j].qubit_id < obs_copy[j + 1].qubit_id,
                    "Repeated qubit positions"
                );
            }
        }

        let mut x_indices = IndexSet::new();
        let mut z_indices = IndexSet::new();

        for crate::core::PositionedPauliObservable { qubit_id, observable } in obs_copy {
            match observable {
                crate::core::PauliObservable::PlusI | crate::core::PauliObservable::MinusI => (),
                crate::core::PauliObservable::PlusX | crate::core::PauliObservable::MinusX => {
                    x_indices.assign_index(qubit_id, true);
                }
                crate::core::PauliObservable::PlusZ | crate::core::PauliObservable::MinusZ => {
                    z_indices.assign_index(qubit_id, true);
                }
                crate::core::PauliObservable::PlusY | crate::core::PauliObservable::MinusY => {
                    x_indices.assign_index(qubit_id, true);
                    z_indices.assign_index(qubit_id, true);
                }
            }
        }
        PauliUnitaryProjective::from_bits(x_indices, z_indices)
    }
}

impl<const LENGTH: usize> From<[PositionedPauliObservable; LENGTH]> for SparsePauli {
    fn from(pauli_observable: [PositionedPauliObservable; LENGTH]) -> Self {
        pauli_observable.as_slice().into()
    }
}

impl From<Vec<PositionedPauliObservable>> for SparsePauli {
    fn from(value: Vec<PositionedPauliObservable>) -> Self {
        value.as_slice().into()
    }
}

impl<const LENGTH: usize> From<[PositionedPauliObservable; LENGTH]> for SparsePauliProjective {
    fn from(pauli_observable: [PositionedPauliObservable; LENGTH]) -> Self {
        pauli_observable.as_slice().into()
    }
}

impl From<Vec<PositionedPauliObservable>> for SparsePauliProjective {
    fn from(value: Vec<PositionedPauliObservable>) -> Self {
        value.as_slice().into()
    }
}

pub fn remapped_sparse(pauli: &impl Pauli<PhaseExponentValue = u8>, support: &[usize]) -> SparsePauli {
    let x_bits = pauli.x_bits().support().map(|id| support[id]).collect::<IndexSet>();
    let z_bits = pauli.z_bits().support().map(|id| support[id]).collect::<IndexSet>();
    SparsePauli::from_bits(x_bits, z_bits, pauli.xz_phase_exponent())
}

pub fn as_sparse(pauli: &impl Pauli<PhaseExponentValue = u8>) -> SparsePauli {
    let x_bits = pauli.x_bits().into();
    let z_bits = pauli.z_bits().into();
    SparsePauli::from_bits(x_bits, z_bits, pauli.xz_phase_exponent())
}

pub fn as_sparse_projective(pauli: &impl Pauli) -> SparsePauliProjective {
    let x_bits = pauli.x_bits().into();
    let z_bits = pauli.z_bits().into();
    SparsePauliProjective::from_bits(x_bits, z_bits)
}
