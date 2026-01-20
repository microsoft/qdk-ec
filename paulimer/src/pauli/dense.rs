use super::{PauliUnitaryProjective, SparsePauli};
use crate::pauli::{generic::PauliUnitary, NeutralElement, Pauli, PauliBinaryOps};
use binar::{vec::AlignedBitVec, Bitwise, BitwiseMut};
pub type DensePauli = PauliUnitary<AlignedBitVec, u8>;
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
