use quantum_core::PositionedPauliObservable;

use crate::subscript_digits;
use crate::traits::{BitwiseNeutralElement, NeutralElement};
use binar::{
    self,
    vec::{AlignedBitVec, AlignedBitView},
    IndexSet,
};
use binar::{Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, FromBits};

pub use core::str::FromStr;
use std::collections::btree_map::Entry;
use std::fmt::Debug;
use std::num::ParseIntError;
use std::{collections::BTreeMap, fmt::Display};

use super::sparse::SparsePauliProjective;
use super::{Pauli, PauliBinaryOps, PauliBits, PauliMutable, PauliMutableBits, PauliNeutralElement, SparsePauli};

impl<Bits, Phase> AsRef<PauliUnitaryProjective<Bits>> for PauliUnitary<Bits, Phase>
where
    Bits: PauliBits,
    Phase: PhaseExponent,
{
    fn as_ref(&self) -> &PauliUnitaryProjective<Bits> {
        &self.projective
    }
}

pub trait PhaseExponent {
    fn raw_value(&self) -> u8;

    fn value(&self) -> u8 {
        self.raw_value() % 4
    }

    fn is_even(&self) -> bool {
        self.raw_value() & 1 == 0
    }

    fn is_odd(&self) -> bool {
        self.raw_value() & 1 != 0
    }

    #[must_use]
    fn raw_eq(raw_value1: u8, raw_value2: u8) -> bool {
        raw_value1.wrapping_sub(raw_value2).is_multiple_of(4)
    }

    fn eq(&self, other: &Self) -> bool {
        Self::raw_eq(self.raw_value(), other.raw_value())
    }

    fn is_zero(&self) -> bool {
        self.raw_value().trailing_zeros() >= 2
    }
}

pub trait PhaseExponentMutable: PhaseExponent {
    fn add_assign(&mut self, value: u8);
    fn assign(&mut self, value: u8);
    fn complex_conjugate_in_place(&mut self) {
        self.assign(4u8 - self.raw_value() % 4);
    }
    fn set_random(&mut self, random_number_generator: &mut impl rand::Rng);
}

pub trait PhaseNeutralElement: PhaseExponent + NeutralElement<NeutralElementType: PhaseExponentMutable> {}

impl PhaseExponent for u8 {
    fn raw_value(&self) -> u8 {
        *self
    }
}

impl PhaseExponent for &u8 {
    fn raw_value(&self) -> u8 {
        **self
    }
}

impl PhaseExponent for &mut u8 {
    fn raw_value(&self) -> u8 {
        **self
    }
}

impl PhaseExponentMutable for u8 {
    fn add_assign(&mut self, value: u8) {
        *self = self.wrapping_add(value);
    }

    fn assign(&mut self, value: u8) {
        *self = value;
    }

    fn set_random(&mut self, random_number_generator: &mut impl rand::Rng) {
        *self = random_number_generator.gen::<u8>();
    }
}

impl PhaseExponentMutable for &mut u8 {
    fn add_assign(&mut self, value: u8) {
        **self = self.wrapping_add(value);
    }

    fn assign(&mut self, value: u8) {
        **self = value;
    }

    fn set_random(&mut self, random_number_generator: &mut impl rand::Rng) {
        **self = random_number_generator.gen::<u8>();
    }
}

impl PhaseNeutralElement for u8 {}
impl PhaseNeutralElement for &u8 {}
impl PhaseNeutralElement for &mut u8 {}

// PauliUnitary & PauliUnitaryProjective structs

#[must_use]
#[derive(Clone, Eq)]
pub struct PauliUnitary<Bits: PauliBits, Phase: PhaseExponent> {
    projective: PauliUnitaryProjective<Bits>,
    xz_phase_exp: Phase,
}

impl<Bits: PauliBits + std::hash::Hash, Phase: PhaseExponent> std::hash::Hash for PauliUnitary<Bits, Phase> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.projective.hash(state);
        self.xz_phase_exponent().hash(state);
    }
}

#[must_use]
#[derive(Clone, Eq, Hash)]
pub struct PauliUnitaryProjective<Bits: PauliBits> {
    x_bits: Bits,
    z_bits: Bits,
}

// Pauli

impl<Bits: PauliBits + BitwisePair> Pauli for PauliUnitaryProjective<Bits> {
    type Bits = Bits;
    type PhaseExponentValue = ();

    fn x_bits(&self) -> &Self::Bits {
        &self.x_bits
    }

    fn z_bits(&self) -> &Self::Bits {
        &self.z_bits
    }

    fn is_order_two(&self) -> bool {
        true
    }

    fn is_identity(&self) -> bool {
        self.x_bits.is_zero() && self.z_bits.is_zero()
    }

    fn is_pauli_x(&self, qubit: usize) -> bool {
        self.x_bits.is_unit(qubit) && self.z_bits.is_zero()
    }

    fn is_pauli_z(&self, qubit: usize) -> bool {
        self.x_bits.is_zero() && self.z_bits.is_unit(qubit)
    }

    fn is_pauli_y(&self, qubit: usize) -> bool {
        self.x_bits.is_unit(qubit) && self.z_bits.is_unit(qubit)
    }

    fn equals_to(&self, rhs: &Self) -> bool {
        self == rhs
    }

    fn to_xz_bits(self) -> (Self::Bits, Self::Bits) {
        (self.x_bits, self.z_bits)
    }

    fn xz_phase_exponent(&self) -> Self::PhaseExponentValue {}
}

impl<Bits: PauliBits + BitwisePair, PhExp: PhaseExponent> Pauli for PauliUnitary<Bits, PhExp> {
    type Bits = Bits;
    type PhaseExponentValue = u8;

    fn x_bits(&self) -> &Bits {
        &self.projective.x_bits
    }

    fn z_bits(&self) -> &Bits {
        &self.projective.z_bits
    }

    fn is_order_two(&self) -> bool {
        self.y_parity() ^ self.xz_phase_exp.is_even()
    }

    fn is_identity(&self) -> bool {
        self.projective.x_bits.is_zero() && self.projective.z_bits.is_zero() && self.xz_phase_exp.is_zero()
    }

    fn is_pauli_x(&self, qubit: usize) -> bool {
        self.projective.x_bits.is_unit(qubit) && self.projective.z_bits.is_zero() && self.xz_phase_exp.is_zero()
    }

    fn is_pauli_z(&self, qubit: usize) -> bool {
        self.projective.x_bits.is_zero() && self.projective.z_bits.is_unit(qubit) && self.xz_phase_exp.is_zero()
    }

    fn is_pauli_y(&self, qubit: usize) -> bool {
        self.projective.x_bits.is_unit(qubit) && self.projective.z_bits.is_unit(qubit) && self.xz_phase_exp.value() == 1
    }

    fn equals_to(&self, rhs: &Self) -> bool {
        self == rhs
    }

    fn to_xz_bits(self) -> (Self::Bits, Self::Bits) {
        (self.projective.x_bits, self.projective.z_bits)
    }

    fn xz_phase_exponent(&self) -> Self::PhaseExponentValue {
        self.xz_phase_exp.value()
    }
}

// PauliMutableBits

impl<OtherBits: Bitwise, Bits: BitwisePairMut<OtherBits> + PauliBits> PauliMutableBits<OtherBits>
    for PauliUnitaryProjective<Bits>
{
    type BitsMutable = Bits;

    fn x_bits_mut(&mut self) -> &mut Self::BitsMutable {
        &mut self.x_bits
    }

    fn z_bits_mut(&mut self) -> &mut Self::BitsMutable {
        &mut self.z_bits
    }
}

impl<OtherBits: Bitwise, Bits: BitwisePairMut<OtherBits> + PauliBits, PhExp: PhaseExponentMutable>
    PauliMutableBits<OtherBits> for PauliUnitary<Bits, PhExp>
{
    type BitsMutable = Bits;

    fn x_bits_mut(&mut self) -> &mut Self::BitsMutable {
        &mut self.projective.x_bits
    }

    fn z_bits_mut(&mut self) -> &mut Self::BitsMutable {
        &mut self.projective.z_bits
    }
}

// PauliOps

impl<Bits: PauliBits + BitwiseMut, Exponent: PhaseExponentMutable> PauliMutable for PauliUnitary<Bits, Exponent> {
    fn assign_phase_exp(&mut self, rhs: u8) {
        self.xz_phase_exp.assign(rhs);
    }

    fn add_assign_phase_exp(&mut self, rhs: u8) {
        self.xz_phase_exp.add_assign(rhs);
    }

    fn complex_conjugate(&mut self) {
        self.xz_phase_exp.complex_conjugate_in_place();
    }

    fn invert(&mut self) {
        self.complex_conjugate();
        if self.y_parity() {
            self.negate();
        }
    }

    fn negate(&mut self) {
        self.xz_phase_exp.add_assign(2u8);
    }

    fn assign_phase_from<PauliLike: Pauli<PhaseExponentValue = Self::PhaseExponentValue>>(
        &mut self,
        other: &PauliLike,
    ) {
        self.xz_phase_exp.assign(other.xz_phase_exponent());
    }

    fn mul_assign_phase_from<PauliLike: Pauli<PhaseExponentValue = Self::PhaseExponentValue>>(
        &mut self,
        other: &PauliLike,
    ) {
        self.xz_phase_exp.add_assign(other.xz_phase_exponent());
    }

    fn mul_assign_left_x(&mut self, qubit_id: usize) {
        self.projective.x_bits.negate_index(qubit_id);
    }

    fn mul_assign_right_x(&mut self, qubit_id: usize) {
        self.projective.x_bits.negate_index(qubit_id);
        if self.projective.z_bits().index(qubit_id) {
            self.xz_phase_exponent().add_assign(2);
        }
    }

    fn mul_assign_left_z(&mut self, qubit_id: usize) {
        self.projective.z_bits.negate_index(qubit_id);
        if self.x_bits().index(qubit_id) {
            self.xz_phase_exponent().add_assign(2);
        }
    }

    fn mul_assign_right_z(&mut self, qubit_id: usize) {
        self.projective.z_bits.negate_index(qubit_id);
    }

    fn set_identity(&mut self) {
        self.projective.x_bits.clear_bits();
        self.projective.z_bits.clear_bits();
        self.xz_phase_exp.assign(0);
    }

    fn set_random(&mut self, num_qubits: usize, random_number_generator: &mut impl rand::Rng) {
        self.projective
            .x_bits
            .assign_random(num_qubits, random_number_generator);
        self.projective
            .z_bits
            .assign_random(num_qubits, random_number_generator);
        self.xz_phase_exp.set_random(random_number_generator);
    }

    fn set_random_order_two(&mut self, num_qubits: usize, random_number_generator: &mut impl rand::Rng) {
        self.set_random(num_qubits, random_number_generator);
        if !self.is_order_two() {
            self.xz_phase_exp.add_assign(1u8);
        }
        debug_assert!(self.is_order_two());
    }
}

// PauliBinaryOps

impl<Bits: PauliBits + BitwiseMut> PauliMutable for PauliUnitaryProjective<Bits> {
    #[inline]
    fn assign_phase_exp(&mut self, _rhs: u8) {}

    #[inline]
    fn add_assign_phase_exp(&mut self, _rhs: u8) {}

    #[inline]
    fn complex_conjugate(&mut self) {}

    #[inline]
    fn invert(&mut self) {}

    #[inline]
    fn negate(&mut self) {}

    #[inline]
    fn assign_phase_from<PauliLike: Pauli<PhaseExponentValue = Self::PhaseExponentValue>>(
        &mut self,
        _other: &PauliLike,
    ) {
    }

    #[inline]
    fn mul_assign_phase_from<PauliLike: Pauli<PhaseExponentValue = Self::PhaseExponentValue>>(
        &mut self,
        _other: &PauliLike,
    ) {
    }

    #[inline]
    fn mul_assign_left_x(&mut self, qubit_id: usize) {
        self.x_bits.negate_index(qubit_id);
    }

    #[inline]
    fn mul_assign_right_x(&mut self, qubit_id: usize) {
        self.x_bits.negate_index(qubit_id);
    }

    #[inline]
    fn mul_assign_left_z(&mut self, qubit_id: usize) {
        self.z_bits.negate_index(qubit_id);
    }

    #[inline]
    fn mul_assign_right_z(&mut self, qubit_id: usize) {
        self.z_bits.negate_index(qubit_id);
    }

    #[inline]
    fn set_identity(&mut self) {
        self.x_bits.clear_bits();
        self.z_bits.clear_bits();
    }

    fn set_random(&mut self, num_qubits: usize, random_number_generator: &mut impl rand::Rng) {
        self.x_bits.assign_random(num_qubits, random_number_generator);
        self.z_bits.assign_random(num_qubits, random_number_generator);
    }

    fn set_random_order_two(&mut self, num_qubits: usize, random_number_generator: &mut impl rand::Rng) {
        self.set_random(num_qubits, random_number_generator);
    }
}

#[inline]
pub fn add_assign_bits<T, U>(to: &mut T, from: &U)
where
    T: PauliMutableBits<U::Bits>,
    U: Pauli,
{
    to.x_bits_mut().bitxor_assign(from.x_bits());
    to.z_bits_mut().bitxor_assign(from.z_bits());
}

#[inline]
fn assign_bits<T, U>(to: &mut T, from: &U)
where
    T: PauliMutableBits<U::Bits>,
    U: Pauli,
{
    to.x_bits_mut().assign(from.x_bits());
    to.z_bits_mut().assign(from.z_bits());
}

#[inline]
fn assign_bits_with_offset<T, U>(to: &mut T, from: &U, start_qubit_index: usize, num_qubits: usize)
where
    T: PauliMutableBits<U::Bits>,
    U: Pauli,
{
    to.x_bits_mut()
        .assign_with_offset(from.x_bits(), start_qubit_index, num_qubits);
    to.z_bits_mut()
        .assign_with_offset(from.z_bits(), start_qubit_index, num_qubits);
}

#[inline]
fn bits_eq<T, U>(x: &T, z: &T, b: &U) -> bool
where
    U: Pauli,
    T: PartialEq<U::Bits>,
{
    x == b.x_bits() && z == b.z_bits()
}

impl<Bits, OtherPauli: Pauli<PhaseExponentValue = ()>> PauliBinaryOps<OtherPauli> for PauliUnitaryProjective<Bits>
where
    Bits: BitwisePairMut<OtherPauli::Bits> + PauliBits,
    OtherPauli: Pauli,
{
    #[inline]
    fn mul_assign_right(&mut self, rhs: &OtherPauli) {
        add_assign_bits(self, rhs);
    }

    #[inline]
    fn mul_assign_left(&mut self, lhs: &OtherPauli) {
        add_assign_bits(self, lhs);
    }

    #[inline]
    fn assign(&mut self, rhs: &OtherPauli) {
        assign_bits(self, rhs);
    }

    #[inline]
    fn assign_with_offset(&mut self, rhs: &OtherPauli, start_qubit_index: usize, num_qubits: usize) {
        assign_bits_with_offset(self, rhs, start_qubit_index, num_qubits);
    }
}

// impl<Bits, OtherPauli : Pauli<PhaseExponentValue = ()>> PauliPhaseBinaryOps<OtherPauli> for PauliUnitaryProjective<Bits>
// where
//     Bits: PauliBits,
// {
//     fn assign_phase_from(&mut self, _other: &OtherPauli) {}
//     fn mul_assign_phase_from(&mut self, _other: &OtherPauli) {}
// }

impl<Bits, Exponent, OtherPauli: Pauli<PhaseExponentValue = u8>> PauliBinaryOps<OtherPauli>
    for PauliUnitary<Bits, Exponent>
where
    Bits: BitwisePairMut<OtherPauli::Bits> + BitwisePair<OtherPauli::Bits> + PauliBits + BitwiseMut,
    Exponent: PhaseExponentMutable,
{
    #[inline]
    fn mul_assign_right(&mut self, rhs: &OtherPauli) {
        let cross: u8 = if self.z_bits().dot(rhs.x_bits()) { 2u8 } else { 0u8 };
        add_assign_bits(self, rhs);
        self.add_assign_phase_exp(cross.wrapping_add(rhs.xz_phase_exponent()));
    }

    #[inline]
    fn mul_assign_left(&mut self, lhs: &OtherPauli) {
        let cross: u8 = if self.x_bits().dot(lhs.z_bits()) { 2u8 } else { 0u8 };
        add_assign_bits(self, lhs);
        self.add_assign_phase_exp(cross.wrapping_add(lhs.xz_phase_exponent()));
    }

    #[inline]
    fn assign(&mut self, rhs: &OtherPauli) {
        self.assign_phase_exp(rhs.xz_phase_exponent());
        assign_bits(self, rhs);
    }

    #[inline]
    fn assign_with_offset(&mut self, rhs: &OtherPauli, start_qubit_index: usize, num_qubits: usize) {
        self.assign_phase_exp(rhs.xz_phase_exponent());
        assign_bits_with_offset(self, rhs, start_qubit_index, num_qubits);
    }
}

// impl<Bits, Exponent, OtherPauli : Pauli<PhaseExponentValue = u8> > PauliPhaseBinaryOps<OtherPauli> for PauliUnitary<Bits, Exponent>
// where
//     Bits: PauliBits,
//     Exponent: PhaseExponentMutable,
// {
//     fn assign_phase_from(&mut self, other: &OtherPauli) {
//         self.assign_phase_exp(other.xz_phase_exponent());
//     }

//     fn mul_assign_phase_from(&mut self, other: &OtherPauli) {
//         self.add_assign_phase_exp(other.xz_phase_exponent());
//     }
// }

impl<Bits: PauliBits, Exponent: PhaseExponent> PauliUnitary<Bits, Exponent> {
    pub fn from_bits(x_bits: Bits, z_bits: Bits, phase: Exponent) -> PauliUnitary<Bits, Exponent> {
        PauliUnitary {
            projective: PauliUnitaryProjective::from_bits(x_bits, z_bits),
            xz_phase_exp: phase,
        }
    }

    pub fn from_bits_tuple(bits: (Bits, Bits), phase: Exponent) -> PauliUnitary<Bits, Exponent> {
        PauliUnitary {
            projective: PauliUnitaryProjective::from_bits_tuple(bits),
            xz_phase_exp: phase,
        }
    }
}

impl<Bits: PauliBits> PauliUnitaryProjective<Bits> {
    pub fn from_bits(x_bits: Bits, z_bits: Bits) -> PauliUnitaryProjective<Bits> {
        PauliUnitaryProjective { x_bits, z_bits }
    }

    pub fn from_bits_tuple(xz_bits: (Bits, Bits)) -> PauliUnitaryProjective<Bits> {
        PauliUnitaryProjective {
            x_bits: xz_bits.0,
            z_bits: xz_bits.1,
        }
    }
}

impl<Exponent: PhaseExponent> PauliUnitary<binar::vec::AlignedBitVec, Exponent> {
    pub fn size(&self) -> usize {
        self.projective.x_bits.len()
    }
}

impl PauliUnitaryProjective<binar::vec::AlignedBitVec> {
    #[must_use]
    pub fn size(&self) -> usize {
        self.x_bits.len()
    }
}

// Partial and Full equality

impl<LeftBits, LeftPhase, RightBits, RightPhase> PartialEq<PauliUnitary<RightBits, RightPhase>>
    for PauliUnitary<LeftBits, LeftPhase>
where
    LeftBits: PartialEq<RightBits> + PauliBits,
    RightBits: PauliBits,
    LeftPhase: PhaseExponent,
    RightPhase: PhaseExponent,
{
    #[inline]
    fn eq(&self, other: &PauliUnitary<RightBits, RightPhase>) -> bool {
        (self.xz_phase_exponent() == other.xz_phase_exponent())
            && bits_eq(&self.projective.x_bits, &self.projective.z_bits, other)
    }
}

impl<LeftBits, LeftPhase, RightBits, RightPhase> PartialEq<PauliUnitary<RightBits, RightPhase>>
    for &PauliUnitary<LeftBits, LeftPhase>
where
    LeftBits: PartialEq<RightBits> + PauliBits,
    RightBits: PauliBits,
    LeftPhase: PhaseExponent,
    RightPhase: PhaseExponent,
{
    #[inline]
    fn eq(&self, other: &PauliUnitary<RightBits, RightPhase>) -> bool {
        *self == other
    }
}

impl<LeftBits, LeftPhase, RightBits, RightPhase> PartialEq<&PauliUnitary<RightBits, RightPhase>>
    for PauliUnitary<LeftBits, LeftPhase>
where
    LeftBits: PartialEq<RightBits> + PauliBits,
    RightBits: PauliBits,
    LeftPhase: PhaseExponent,
    RightPhase: PhaseExponent,
{
    #[inline]
    fn eq(&self, other: &&PauliUnitary<RightBits, RightPhase>) -> bool {
        self == *other
    }
}

// impl<LeftBits: PauliBits, RightPauli: Pauli + ProjectivePauli> PartialEq<RightPauli>
//     for PauliUnitaryProjective<LeftBits>
// where
//     LeftBits: Bitwise + PartialEq<RightPauli::Bits>,
// {
//     fn eq(&self, other: &RightPauli) -> bool {
//         bits_eq(&self.x_bits, &self.z_bits, other)
//     }
// }

impl<LeftBits, RightBits> PartialEq<PauliUnitaryProjective<RightBits>> for PauliUnitaryProjective<LeftBits>
where
    LeftBits: PartialEq<RightBits> + PauliBits,
    RightBits: PauliBits,
{
    #[inline]
    fn eq(&self, other: &PauliUnitaryProjective<RightBits>) -> bool {
        bits_eq(&self.x_bits, &self.z_bits, other)
    }
}

impl<LeftBits, RightBits> PartialEq<PauliUnitaryProjective<RightBits>> for &PauliUnitaryProjective<LeftBits>
where
    LeftBits: PartialEq<RightBits> + PauliBits,
    RightBits: PauliBits,
{
    #[inline]
    fn eq(&self, other: &PauliUnitaryProjective<RightBits>) -> bool {
        *self == other
    }
}

impl<LeftBits, RightBits> PartialEq<&PauliUnitaryProjective<RightBits>> for PauliUnitaryProjective<LeftBits>
where
    LeftBits: PartialEq<RightBits> + PauliBits,
    RightBits: PauliBits,
{
    #[inline]
    fn eq(&self, other: &&PauliUnitaryProjective<RightBits>) -> bool {
        self == *other
    }
}

fn string_map(pauli: &impl Pauli) -> (u8, BTreeMap<usize, char>) {
    let mut phase = 0;
    let mut support = BTreeMap::new();
    for index in pauli.x_bits().support() {
        support.insert(index, 'X');
    }
    for index in pauli.z_bits().support() {
        match support.entry(index) {
            Entry::Occupied(mut e) => {
                e.insert('Y');
                phase = (phase + 3) % 4;
            }
            Entry::Vacant(e) => {
                e.insert('Z');
            }
        }
    }
    (phase, support)
}

pub fn pauli_string(
    pauli: &impl Pauli,
    phase: u8,
    add_phase: bool,
    sign_plus: bool,
    dense: bool,
    size: Option<usize>,
) -> String {
    let mut string = String::new();
    let last_index = size.unwrap_or_else(|| {
        if let Some(last_index) = pauli.max_support() {
            last_index + 1
        } else {
            0
        }
    });
    if last_index > 0 {
        if dense {
            // Optimized dense string: iterate directly through bits instead of using BTreeMap
            let mut extra_phase = 0u8;
            string.reserve(last_index + 3);
            for index in 0..last_index {
                let is_x = pauli.x_bits().index(index);
                let is_z = pauli.z_bits().index(index);
                if is_x && is_z {
                    extra_phase = (extra_phase + 3) % 4;
                }
                string.push(match (is_x, is_z) {
                    (false, false) => 'I',
                    (true, false) => 'X',
                    (true, true) => 'Y',
                    (false, true) => 'Z',
                });
            }
            if add_phase {
                string.insert_str(0, &phase_to_string((phase.wrapping_add(extra_phase)) % 4u8, sign_plus));
            }
        } else {
            // Sparse format: still use string_map for sparse representation
            let (extra_phase, id_to_character) = string_map(pauli);
            if add_phase {
                string.push_str(&phase_to_string((phase.wrapping_add(extra_phase)) % 4u8, sign_plus));
            }
            for (index, character) in &id_to_character {
                string.push(*character);
                string.push_str(&subscript_digits(*index));
            }
        }
    } else {
        if add_phase {
            string.push_str(&phase_to_string(phase % 4u8, sign_plus));
        }
        string.push('I');
    }
    string
}

// Display

impl<Bits: PauliBits, Phase: PhaseExponent> Display for PauliUnitary<Bits, Phase> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            // sparse (when using {:#})
            f.pad(&pauli_string(
                self,
                self.xz_phase_exp.value(),
                true,
                f.sign_plus(),
                false,
                None,
            ))
        } else {
            // dense (default for to_string())
            f.pad(&pauli_string(
                self,
                self.xz_phase_exp.value(),
                true,
                f.sign_plus(),
                true,
                None,
            ))
        }
    }
}

impl<Bits: PauliBits, Phase: PhaseExponent> Debug for PauliUnitary<Bits, Phase> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl<Bits: PauliBits> Display for PauliUnitaryProjective<Bits> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            f.pad(&pauli_string(self, 0, false, f.sign_plus(), true, None))
        } else {
            f.pad(&pauli_string(self, 0, false, f.sign_plus(), false, None))
        }
    }
}

impl<Bits: PauliBits> Debug for PauliUnitaryProjective<Bits> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

/// # Panics
///
/// Will panic
#[must_use]
pub fn phase_to_string(phase: u8, with_plus: bool) -> String {
    let s = match phase {
        0 => {
            if with_plus {
                "+"
            } else {
                ""
            }
        }
        1 => {
            if with_plus {
                "+𝑖"
            } else {
                "𝑖"
            }
        }
        2 => "-",
        3 => "-𝑖",
        _ => panic!("Unexpected phase"),
    };
    String::from(s)
}

impl<Bits, Phase> NeutralElement for PauliUnitary<Bits, Phase>
where
    Bits: BitwiseNeutralElement + PauliBits,
    Bits::NeutralElementType: PauliBits,
    Phase: PhaseNeutralElement,
{
    type NeutralElementType = PauliUnitary<Bits::NeutralElementType, Phase::NeutralElementType>;

    fn neutral_element(&self) -> Self::NeutralElementType {
        PauliUnitary::from_bits(
            self.projective.x_bits.neutral_element(),
            self.projective.z_bits.neutral_element(),
            self.xz_phase_exp.neutral_element(),
        )
    }

    fn default_size_neutral_element() -> Self::NeutralElementType {
        PauliUnitary::from_bits(
            <Bits as NeutralElement>::default_size_neutral_element(),
            <Bits as NeutralElement>::default_size_neutral_element(),
            <Phase as NeutralElement>::default_size_neutral_element(),
        )
    }

    fn neutral_element_of_size(size: usize) -> Self::NeutralElementType {
        PauliUnitary::from_bits(
            <Bits as NeutralElement>::neutral_element_of_size(size),
            <Bits as NeutralElement>::neutral_element_of_size(size),
            <Phase as NeutralElement>::default_size_neutral_element(),
        )
    }
}

impl<Bits> NeutralElement for PauliUnitaryProjective<Bits>
where
    Bits: BitwiseNeutralElement + PauliBits,
    Bits::NeutralElementType: PauliBits,
{
    type NeutralElementType = PauliUnitaryProjective<Bits::NeutralElementType>;

    #[inline]
    fn neutral_element(&self) -> Self::NeutralElementType {
        PauliUnitaryProjective::from_bits(self.x_bits.neutral_element(), self.z_bits.neutral_element())
    }

    #[inline]
    fn default_size_neutral_element() -> Self::NeutralElementType {
        PauliUnitaryProjective::from_bits(
            <Bits as NeutralElement>::default_size_neutral_element(),
            <Bits as NeutralElement>::default_size_neutral_element(),
        )
    }

    #[inline]
    fn neutral_element_of_size(size: usize) -> Self::NeutralElementType {
        PauliUnitaryProjective::from_bits(
            <Bits as NeutralElement>::neutral_element_of_size(size),
            <Bits as NeutralElement>::neutral_element_of_size(size),
        )
    }
}

impl<Bits> PauliNeutralElement for PauliUnitaryProjective<Bits>
where
    Bits: BitwiseNeutralElement + PauliBits,
    Bits::NeutralElementType: PauliBits + BitwiseMut,
{
}

impl<Bits, Phase> PauliNeutralElement for PauliUnitary<Bits, Phase>
where
    Bits: BitwiseNeutralElement + PauliBits,
    Phase: PhaseNeutralElement,
    Bits::NeutralElementType: PauliBits + BitwisePair<Bits> + BitwiseMut,
{
}

impl<BitsFrom: PauliBits, Bits: PauliBits> FromBits<PauliUnitaryProjective<BitsFrom>> for PauliUnitaryProjective<Bits>
where
    Self: PauliNeutralElement<NeutralElementType = Self>,
    Bits: FromBits<BitsFrom>,
{
    fn from_bits(other: &PauliUnitaryProjective<BitsFrom>) -> Self {
        let x = Bits::from_bits(other.x_bits());
        let z = Bits::from_bits(other.z_bits());
        PauliUnitaryProjective::<Bits>::from_bits(x, z)
    }
}

impl<
        BitsFrom: PauliBits,
        Bits: PauliBits,
        PhaseFrom: PhaseExponent,
        Phase: PhaseExponentMutable + NeutralElement<NeutralElementType = Phase>,
    > FromBits<PauliUnitary<BitsFrom, PhaseFrom>> for PauliUnitary<Bits, Phase>
where
    Self: PauliNeutralElement<NeutralElementType = Self>,
    Bits: FromBits<BitsFrom>,
    PauliUnitary<Bits, Phase>: Pauli<PhaseExponentValue = u8>,
{
    fn from_bits(other: &PauliUnitary<BitsFrom, PhaseFrom>) -> Self {
        let x = Bits::from_bits(other.x_bits());
        let z = Bits::from_bits(other.z_bits());
        let mut res = PauliUnitary::<Bits, Phase>::from_bits(x, z, Phase::default_size_neutral_element());
        res.add_assign_phase_exp(other.xz_phase_exponent());
        res
    }
}

fn digits_to_int(digits: &[u32]) -> Result<u32, ParseIntError> {
    let mut normal_digits = String::with_capacity(digits.len());
    for digit_value in digits {
        let digit_char = std::char::from_digit(*digit_value, 10).expect("expected a digit");
        normal_digits.push(digit_char);
    }
    normal_digits.parse()
}

fn pauli_from_str<T>(pauli_string: &str) -> Result<T, PauliStringParsingError>
where
    T: PauliMutable + NeutralElement<NeutralElementType = T>,
{
    let no_whitespace = pauli_string.trim();
    let index_chars = "₀₁₂₃₄₅₆₇₈₉0123456789_";
    let phase_prefix_options = ["+i", "i", "-i", "+𝑖", "𝑖", "-𝑖", "+", "-"];

    let (no_whitespace, phase_exp) = parse_phase(no_whitespace, phase_prefix_options);

    if index_chars.contains(no_whitespace.chars().nth(1).unwrap_or(' ')) {
        // Sparse string
        parse_sparse_pauli(no_whitespace, phase_exp)
    } else {
        // Dense string
        let mut res: T = <T as NeutralElement>::neutral_element_of_size(pauli_string.len());
        res.add_assign_phase_exp(phase_exp);
        for (index, character) in no_whitespace.chars().enumerate() {
            match character {
                'X' | 'x' => res.mul_assign_right_x(index),
                'Z' | 'z' => res.mul_assign_right_z(index),
                'Y' | 'y' => res.mul_assign_right_y(index),
                'I' | ' ' => {}
                _ => {
                    return Err(PauliStringParsingError);
                }
            }
        }
        Ok(res)
    }
}

fn parse_sparse_pauli<T>(no_whitespace: &str, phase_exp: u8) -> Result<T, PauliStringParsingError>
where
    T: PauliMutable + NeutralElement<NeutralElementType = T>,
{
    let mut character_and_positions = Vec::new();
    let mut digit_group = Vec::<u32>::new();
    let mut pauli_char: char = 'I';

    for character in no_whitespace.chars() {
        match character {
            'X' | 'x' | 'Z' | 'z' | 'Y' | 'y' => {
                if pauli_char != 'I' {
                    character_and_positions.push((pauli_char, digit_group.clone()));
                    digit_group.clear();
                }
                pauli_char = character;
            }
            '₀'..='₉' => {
                digit_group.push(character as u32 - '₀' as u32);
            }
            '0'..='9' => {
                digit_group.push(character as u32 - '0' as u32);
            }
            '{' | '}' | ' ' | '_' => {}
            _ => {
                return Err(PauliStringParsingError);
            }
        }
    }
    if pauli_char != 'I' {
        character_and_positions.push((pauli_char, digit_group.clone()));
    }

    let mut max_index = 0;
    for (_, digits) in &character_and_positions {
        if let Ok(index) = digits_to_int(digits) {
            if let Ok(index_usize) = usize::try_from(index) {
                max_index = usize::max(max_index, index_usize);
            } else {
                return Err(PauliStringParsingError);
            }
        } else {
            return Err(PauliStringParsingError);
        }
    }
    let mut res: T = <T as NeutralElement>::neutral_element_of_size(no_whitespace.len());
    res.add_assign_phase_exp(phase_exp);
    for (pauli_char, digits) in &character_and_positions {
        if let Ok(index) = digits_to_int(digits) {
            if let Ok(index_usize) = usize::try_from(index) {
                match pauli_char {
                    'X' | 'x' => res.mul_assign_left_x(index_usize),
                    'Z' | 'z' => res.mul_assign_left_z(index_usize),
                    'Y' | 'y' => res.mul_assign_left_y(index_usize),
                    _ => {
                        return Err(PauliStringParsingError);
                    }
                }
            } else {
                return Err(PauliStringParsingError);
            }
        } else {
            return Err(PauliStringParsingError);
        }
    }
    Ok(res)
}

fn parse_phase<'life>(no_whitespace: &'life str, phase_prefix_options: [&'static str; 8]) -> (&'life str, u8) {
    for phase_prefix in phase_prefix_options {
        if no_whitespace.starts_with(phase_prefix) {
            let (phase_string, remainder) = no_whitespace.split_at(phase_prefix.len());
            let phase_exp = match phase_string {
                "-" => 2,
                "+i" | "+𝑖" | "i" | "𝑖" => 1,
                "-i" | "-𝑖" => 3,
                "+" => 0,
                _ => {
                    unreachable!();
                }
            };
            return (remainder, phase_exp);
        }
    }
    (no_whitespace, 0)
}

impl<Bits: PauliBits + BitwiseNeutralElement> FromStr for PauliUnitaryProjective<Bits>
where
    Bits: PauliBits + BitwiseNeutralElement,
    Self: PauliNeutralElement<NeutralElementType = Self>,
{
    type Err = PauliStringParsingError;

    fn from_str(characters: &str) -> Result<Self, Self::Err> {
        pauli_from_str(characters)
    }
}

impl<Bits, Phase> FromStr for PauliUnitary<Bits, Phase>
where
    Bits: BitwiseNeutralElement + PauliBits,
    Phase: PhaseNeutralElement,
    Self: PauliNeutralElement<NeutralElementType = Self>,
{
    type Err = PauliStringParsingError;

    fn from_str(characters: &str) -> Result<Self, Self::Err> {
        pauli_from_str(characters)
    }
}

impl<Bits, Phase> PartialEq<&[PositionedPauliObservable]> for PauliUnitary<Bits, Phase>
where
    Bits: PauliBits + std::cmp::PartialEq<binar::IndexSet>,
    Phase: PhaseNeutralElement,
{
    #[inline]
    fn eq(&self, other: &&[PositionedPauliObservable]) -> bool {
        self == <&[PositionedPauliObservable] as Into<SparsePauli>>::into(other)
    }
}

impl<Bits, Phase, const LENGTH: usize> PartialEq<[PositionedPauliObservable; LENGTH]> for PauliUnitary<Bits, Phase>
where
    Bits: PauliBits + std::cmp::PartialEq<binar::IndexSet>,
    Phase: PhaseNeutralElement,
{
    #[inline]
    fn eq(&self, other: &[PositionedPauliObservable; LENGTH]) -> bool {
        self == <&[PositionedPauliObservable] as Into<SparsePauli>>::into(other)
    }
}

impl<Bits, Phase> PartialEq<Vec<PositionedPauliObservable>> for PauliUnitary<Bits, Phase>
where
    Bits: PauliBits + std::cmp::PartialEq<binar::IndexSet>,
    Phase: PhaseNeutralElement,
{
    #[inline]
    fn eq(&self, other: &Vec<PositionedPauliObservable>) -> bool {
        self == <&[PositionedPauliObservable] as Into<SparsePauli>>::into(other)
    }
}

impl<Bits> PartialEq<&[PositionedPauliObservable]> for PauliUnitaryProjective<Bits>
where
    Bits: PauliBits + std::cmp::PartialEq<binar::IndexSet>,
{
    #[inline]
    fn eq(&self, other: &&[PositionedPauliObservable]) -> bool {
        self == <&[PositionedPauliObservable] as Into<SparsePauliProjective>>::into(other)
    }
}

impl<Bits, const LENGTH: usize> PartialEq<[PositionedPauliObservable; LENGTH]> for PauliUnitaryProjective<Bits>
where
    Bits: PauliBits + std::cmp::PartialEq<binar::IndexSet>,
{
    #[inline]
    fn eq(&self, other: &[PositionedPauliObservable; LENGTH]) -> bool {
        self == <&[PositionedPauliObservable] as Into<SparsePauliProjective>>::into(other)
    }
}

impl<Bits> PartialEq<Vec<PositionedPauliObservable>> for PauliUnitaryProjective<Bits>
where
    Bits: PauliBits + std::cmp::PartialEq<binar::IndexSet>,
{
    #[inline]
    fn eq(&self, other: &Vec<PositionedPauliObservable>) -> bool {
        self == <&[PositionedPauliObservable] as Into<SparsePauliProjective>>::into(other)
    }
}

#[derive(Debug, PartialEq, Eq, Default)]
pub struct PauliCharacterError;

#[derive(Debug, PartialEq, Eq, Default)]
pub struct PauliStringParsingError;

impl<Bits: PauliBits> From<(Bits, Bits)> for PauliUnitaryProjective<Bits> {
    fn from(value: (Bits, Bits)) -> Self {
        PauliUnitaryProjective::<Bits>::from_bits_tuple(value)
    }
}

impl<Bits: PauliBits, Phase: PhaseExponent + NeutralElement<NeutralElementType = Phase>> From<(Bits, Bits)>
    for PauliUnitary<Bits, Phase>
{
    fn from(value: (Bits, Bits)) -> Self {
        PauliUnitary::<Bits, Phase>::from_bits_tuple(value, Phase::default_size_neutral_element())
    }
}

impl<'life> From<PauliUnitaryProjective<AlignedBitView<'life>>> for PauliUnitaryProjective<AlignedBitVec> {
    fn from(value: PauliUnitaryProjective<AlignedBitView<'life>>) -> Self {
        Self::from_bits(value.x_bits.into(), value.z_bits.into())
    }
}

impl<Bits: PauliBits, T: Pauli<PhaseExponentValue = u8, Bits = Bits>> From<T> for PauliUnitaryProjective<AlignedBitVec>
where
    AlignedBitVec: for<'life> From<&'life Bits>,
{
    fn from(value: T) -> Self {
        Self::from_bits(value.x_bits().into(), value.z_bits().into())
    }
}

impl<Bits: PauliBits, T: Pauli<PhaseExponentValue = (), Bits = Bits>> From<T> for PauliUnitary<AlignedBitVec, u8>
where
    AlignedBitVec: for<'life> From<&'life Bits>,
{
    fn from(value: T) -> Self {
        let weight = value.x_bits().and_weight(value.z_bits());
        Self::from_bits(
            value.x_bits().into(),
            value.z_bits().into(),
            (weight % 4).try_into().unwrap(),
        )
    }
}

impl<'life, const WORD_COUNT: usize> From<PauliUnitaryProjective<&'life [u64; WORD_COUNT]>>
    for PauliUnitaryProjective<[u64; WORD_COUNT]>
{
    fn from(value: PauliUnitaryProjective<&'life [u64; WORD_COUNT]>) -> Self {
        Self::from_bits(value.x_bits.to_owned(), value.z_bits.to_owned())
    }
}

impl<'life> From<PauliUnitary<AlignedBitView<'life>, &u8>> for PauliUnitary<AlignedBitVec, u8> {
    fn from(value: PauliUnitary<AlignedBitView<'life>, &u8>) -> Self {
        Self::from_bits(
            value.projective.x_bits.into(),
            value.projective.z_bits.into(),
            *value.xz_phase_exp,
        )
    }
}

pub fn pauli_random<PauliLike: NeutralElement<NeutralElementType = PauliLike> + PauliMutable>(
    num_qubits: usize,
    random_number_generator: &mut impl rand::Rng,
) -> PauliLike {
    let mut res = PauliLike::neutral_element_of_size(num_qubits);
    res.set_random(num_qubits, random_number_generator);
    res
}

/// # Example
/// `pauli_random_order_two(6, &mut thread_rng());`
pub fn pauli_random_order_two<PauliLike: NeutralElement<NeutralElementType = PauliLike> + PauliMutable>(
    num_qubits: usize,
    random_number_generator: &mut impl rand::Rng,
) -> PauliLike {
    let mut res = PauliLike::neutral_element_of_size(num_qubits);
    res.set_random_order_two(num_qubits, random_number_generator);
    res
}

// Conversion from DensePauli to SparsePauli
impl From<PauliUnitary<AlignedBitVec, u8>> for PauliUnitary<IndexSet, u8> {
    fn from(value: PauliUnitary<AlignedBitVec, u8>) -> Self {
        Self::from_bits(
            IndexSet::from(value.x_bits()),
            IndexSet::from(value.z_bits()),
            value.xz_phase_exponent(),
        )
    }
}
