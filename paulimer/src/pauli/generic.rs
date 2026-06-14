use crate::core::PositionedPauliObservable;

use crate::subscript_digits;
use crate::traits::{BitwiseNeutralElement, NeutralElement};
use binar::{
    self, IndexSet,
    vec::{AlignedBitVec, AlignedBitView},
};
use binar::{Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, FromBits};

pub use core::str::FromStr;
use std::collections::btree_map::Entry;
use std::fmt::{self, Debug};
use std::{collections::BTreeMap, fmt::Display};

use super::sparse::SparsePauliProjective;
use super::{Pauli, PauliBinaryOps, PauliBits, PauliMutable, PauliMutableBits, PauliNeutralElement, SparsePauli};

/// Controls the notation used when displaying Pauli and Clifford operators.
///
/// Determines how the imaginary unit, qubit subscripts, arrows (for Clifford
/// maps), inter-term separators, and the minus sign are rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringNotation {
    /// Unicode notation: italic `𝑖`, subscript digits (`₀₁₂…`), and `→`.
    Unicode,
    /// ASCII notation: plain `i`, underscore-prefixed digits (`_0`, `_1`, …), and `": "`.
    Ascii,
    /// TeX notation: `\mathrm{i}`, brace-delimited subscripts (`_{0}`), and `\mapsto`.
    /// Clifford operators are wrapped in an `align` environment.
    Tex,
}

impl StringNotation {
    pub(crate) fn imaginary_unit(self) -> &'static str {
        match self {
            Self::Unicode => "𝑖",
            Self::Ascii => "i",
            Self::Tex => "\\mathrm{i}",
        }
    }

    pub(crate) fn subscript(self, index: usize) -> String {
        match self {
            Self::Unicode => subscript_digits(index),
            Self::Ascii => format!("_{index}"),
            Self::Tex => format!("_{{{index}}}"),
        }
    }

    pub(crate) fn arrow(self) -> &'static str {
        match self {
            Self::Unicode => "→",
            Self::Ascii => ": ",
            Self::Tex => " &\\mapsto ",
        }
    }

    pub(crate) fn separator(self) -> &'static str {
        match self {
            Self::Unicode => "",
            Self::Ascii | Self::Tex => " ",
        }
    }

    pub(crate) fn minus() -> &'static str {
        "-"
    }

    pub(crate) fn mapping_separator(self) -> &'static str {
        match self {
            Self::Tex => " \\\\\n",
            Self::Unicode | Self::Ascii => ", ",
        }
    }

    pub(crate) fn mapping_preamble(self) -> &'static str {
        match self {
            Self::Tex => "\\begin{align}\n",
            Self::Unicode | Self::Ascii => "",
        }
    }

    pub(crate) fn mapping_epilogue(self) -> &'static str {
        match self {
            Self::Tex => "\n\\end{align}",
            Self::Unicode | Self::Ascii => "",
        }
    }
}

/// Controls whether the Pauli string uses sparse (e.g., `X₀Y₁Z₂`) or dense (e.g., `XYZ`) format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringLayout {
    /// Dense layout: emits every qubit position (including identities), e.g., `XYZ`, `XIIZ`.
    Dense,
    /// Sparse layout: emits only non-identity qubit positions with subscripts, e.g., `X₀Z₂`.
    Sparse,
}

fn format_phase_prefix(phase: Option<u8>, extra_phase: u8, sign_plus: bool, notation: StringNotation) -> String {
    let Some(phase) = phase else {
        return String::new();
    };
    let combined = (phase.wrapping_add(extra_phase)) % 4u8;
    match combined {
        0 => {
            if sign_plus {
                "+".to_string()
            } else {
                String::new()
            }
        }
        1 => {
            let mut s = if sign_plus { "+".to_string() } else { String::new() };
            s.push_str(notation.imaginary_unit());
            s
        }
        2 => StringNotation::minus().to_string(),
        3 => {
            let mut s = StringNotation::minus().to_string();
            s.push_str(notation.imaginary_unit());
            s
        }
        _ => panic!("Unexpected phase"),
    }
}

/// Formats a Pauli operator as a string with the given layout, notation, and optional size override.
///
/// This is the primary formatting entry point, used internally by
/// [`PauliUnitary::to_string_with`] and the `Display` implementations.
pub fn format_pauli(
    pauli: &(impl Pauli + ?Sized),
    phase: Option<u8>,
    layout: StringLayout,
    notation: StringNotation,
    size: Option<usize>,
    sign_plus: bool,
) -> String {
    match layout {
        StringLayout::Dense => format_dense_pauli(pauli, phase, notation, size, sign_plus),
        StringLayout::Sparse => format_sparse_pauli(pauli, phase, notation, sign_plus),
    }
}

fn format_dense_pauli(
    pauli: &(impl Pauli + ?Sized),
    phase: Option<u8>,
    notation: StringNotation,
    size: Option<usize>,
    sign_plus: bool,
) -> String {
    let last_index = size.unwrap_or_else(|| pauli.max_support().map_or(0, |i| i + 1));
    if last_index == 0 {
        let prefix = format_phase_prefix(phase, 0, sign_plus, notation);
        return format!("{prefix}I");
    }
    let mut extra_phase = 0u8;
    let mut body = String::with_capacity(last_index);
    for index in 0..last_index {
        let is_x = pauli.x_bits().index(index);
        let is_z = pauli.z_bits().index(index);
        if is_x && is_z {
            extra_phase = extra_phase.wrapping_add(3) % 4;
        }
        body.push(match (is_x, is_z) {
            (false, false) => 'I',
            (true, false) => 'X',
            (true, true) => 'Y',
            (false, true) => 'Z',
        });
    }
    let prefix = format_phase_prefix(phase, extra_phase, sign_plus, notation);
    format!("{prefix}{body}")
}

fn format_sparse_pauli(
    pauli: &(impl Pauli + ?Sized),
    phase: Option<u8>,
    notation: StringNotation,
    sign_plus: bool,
) -> String {
    let (extra_phase, id_to_character) = string_map(pauli);
    if id_to_character.is_empty() {
        let prefix = format_phase_prefix(phase, 0, sign_plus, notation);
        return format!("{prefix}I");
    }
    let prefix = format_phase_prefix(phase, extra_phase, sign_plus, notation);
    let sep = notation.separator();
    let mut result = prefix;
    for (i, (index, character)) in id_to_character.iter().enumerate() {
        if i > 0 && !sep.is_empty() {
            result.push_str(sep);
        }
        result.push(*character);
        result.push_str(&notation.subscript(*index));
    }
    result
}

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
    fn set_random(&mut self, random_number_generator: &mut impl rand::RngExt);
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

    fn set_random(&mut self, random_number_generator: &mut impl rand::RngExt) {
        *self = random_number_generator.random::<u8>();
    }
}

impl PhaseExponentMutable for &mut u8 {
    fn add_assign(&mut self, value: u8) {
        **self = self.wrapping_add(value);
    }

    fn assign(&mut self, value: u8) {
        **self = value;
    }

    fn set_random(&mut self, random_number_generator: &mut impl rand::RngExt) {
        **self = random_number_generator.random::<u8>();
    }
}

impl PhaseNeutralElement for u8 {}
impl PhaseNeutralElement for &u8 {}
impl PhaseNeutralElement for &mut u8 {}

// PauliUnitary & PauliUnitaryProjective structs

/// Generic Pauli unitary operator with phase tracking.
///
/// `PauliUnitary<Bits, Phase>` is the generic type underlying both [`DensePauli`](crate::DensePauli)
/// and [`SparsePauli`]. It represents a Pauli operator P = i^k · X^a ⊗ Z^b
/// where:
/// - `Bits` stores the X and Z components (bit vectors or index sets)
/// - `Phase` stores the phase exponent k ∈ {0, 1, 2, 3} representing {+1, +i, -1, -i}
///
/// # Type Parameters
///
/// - `Bits`: Storage for Pauli components ([`AlignedBitVec`] for dense,
///   [`IndexSet`] for sparse)
/// - `Phase`: Phase exponent type (typically `u8`)
///
/// # Representation
///
/// A Pauli operator on n qubits is uniquely determined by:
/// - X bits: which qubits have X or Y components
/// - Z bits: which qubits have Z or Y components  
/// - Phase: overall factor i^k where k = `phase_exponent` mod 4
///
/// The mapping is:
/// - I at position j: `x[j]=0, z[j]=0`
/// - X at position j: `x[j]=1, z[j]=0`
/// - Y at position j: `x[j]=1, z[j]=1` (with phase contribution)
/// - Z at position j: `x[j]=0, z[j]=1`
///
/// # Examples
///
/// This type is typically used through type aliases:
///
/// ```
/// use paulimer::{DensePauli, SparsePauli};
///
/// // DensePauli = PauliUnitary<AlignedBitVec, u8>
/// let dense: DensePauli = "XYZ".parse().unwrap();
///
/// // SparsePauli = PauliUnitary<IndexSet, u8>
/// let sparse: SparsePauli = "X0 Y1 Z2".parse().unwrap();
/// ```
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

/// Projective Pauli operator without phase tracking.
///
/// `PauliUnitaryProjective<Bits>` represents a Pauli operator modulo global phase.
/// It stores only the X and Z bit patterns, not the ±1 or ±i phase factor.
///
/// This is useful when:
/// - Only the Pauli structure matters, not the global phase
/// - Working with stabilizer generators (phase can be computed separately)
/// - Memory optimization when phase is not needed
///
/// # Type Parameters
///
/// - `Bits`: Storage for Pauli components (bit vectors or index sets)
///
/// # Relationship to `PauliUnitary`
///
/// `PauliUnitaryProjective` is embedded in `PauliUnitary` and can be accessed
/// via the projective field. Many operations work on the projective part alone.
///
/// # Examples
///
/// ```
/// use paulimer::{DensePauli, Pauli};
///
/// // Projective Paulis don't track phase
/// let x: DensePauli = "XII".parse().unwrap();
/// // The underlying projective part is just the X and Z bit patterns
/// assert_eq!(x.weight(), 1);
/// ```
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
    fn xyz_phase_exponent(&self) -> Self::PhaseExponentValue {}
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

    fn xyz_phase_exponent(&self) -> Self::PhaseExponentValue {
        #[allow(clippy::cast_possible_truncation)] // y_weight() % 4 is always 0-3
        let y_weight_mod4 = (self.y_weight() % 4) as u8;
        self.xz_phase_exponent().wrapping_add(3u8.wrapping_mul(y_weight_mod4)) % 4u8
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

    fn set_random(&mut self, num_qubits: usize, random_number_generator: &mut impl rand::RngExt) {
        self.projective
            .x_bits
            .assign_random(num_qubits, random_number_generator);
        self.projective
            .z_bits
            .assign_random(num_qubits, random_number_generator);
        self.xz_phase_exp.set_random(random_number_generator);
    }

    fn set_random_order_two(&mut self, num_qubits: usize, random_number_generator: &mut impl rand::RngExt) {
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

    fn set_random(&mut self, num_qubits: usize, random_number_generator: &mut impl rand::RngExt) {
        self.x_bits.assign_random(num_qubits, random_number_generator);
        self.z_bits.assign_random(num_qubits, random_number_generator);
    }

    fn set_random_order_two(&mut self, num_qubits: usize, random_number_generator: &mut impl rand::RngExt) {
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

impl<Bits: PauliBits, Phase: PhaseExponent> PauliUnitary<Bits, Phase> {
    /// Formats this Pauli operator as a string with the given layout and notation.
    ///
    /// # Examples
    ///
    /// ```
    /// use paulimer::{DensePauli, StringLayout, StringNotation};
    ///
    /// let pauli: DensePauli = "XYZ".parse().unwrap();
    /// assert_eq!(pauli.to_string(), "XYZ");
    /// assert_eq!(pauli.to_string_with(StringLayout::Sparse, StringNotation::Unicode), "X₀Y₁Z₂");
    /// assert_eq!(pauli.to_string_with(StringLayout::Dense, StringNotation::Ascii), "XYZ");
    /// assert_eq!(pauli.to_string_with(StringLayout::Sparse, StringNotation::Ascii), "X_0 Y_1 Z_2");
    /// ```
    #[must_use]
    pub fn to_string_with(&self, layout: StringLayout, notation: StringNotation) -> String {
        format_pauli(self, Some(self.xz_phase_exp.value()), layout, notation, None, false)
    }

    #[must_use]
    pub fn to_string_with_size(&self, layout: StringLayout, notation: StringNotation, size: usize) -> String {
        format_pauli(
            self,
            Some(self.xz_phase_exp.value()),
            layout,
            notation,
            Some(size),
            false,
        )
    }
}

impl PauliUnitaryProjective<binar::vec::AlignedBitVec> {
    #[must_use]
    pub fn size(&self) -> usize {
        self.x_bits.len()
    }
}

impl<Bits: PauliBits> PauliUnitaryProjective<Bits> {
    /// Formats this projective Pauli operator as a string (phase is always omitted).
    #[must_use]
    pub fn to_string_with(&self, layout: StringLayout, notation: StringNotation) -> String {
        format_pauli(self, None, layout, notation, None, false)
    }

    #[must_use]
    pub fn to_string_with_size(&self, layout: StringLayout, notation: StringNotation, size: usize) -> String {
        format_pauli(self, None, layout, notation, Some(size), false)
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

fn string_map(pauli: &(impl Pauli + ?Sized)) -> (u8, BTreeMap<usize, char>) {
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

// Display

impl<Bits: PauliBits, Phase: PhaseExponent> Display for PauliUnitary<Bits, Phase> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layout = if f.alternate() {
            StringLayout::Sparse
        } else {
            StringLayout::Dense
        };
        let s = format_pauli(
            self,
            Some(self.xz_phase_exp.value()),
            layout,
            StringNotation::Unicode,
            None,
            f.sign_plus(),
        );
        f.pad(&s)
    }
}

impl<Bits: PauliBits, Phase: PhaseExponent> Debug for PauliUnitary<Bits, Phase> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as Display>::fmt(self, f)
    }
}

impl<Bits: PauliBits> Display for PauliUnitaryProjective<Bits> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layout = if f.alternate() {
            StringLayout::Sparse
        } else {
            StringLayout::Dense
        };
        let s = format_pauli(self, None, layout, StringNotation::Unicode, None, f.sign_plus());
        f.pad(&s)
    }
}

impl<Bits: PauliBits> Debug for PauliUnitaryProjective<Bits> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <Self as Display>::fmt(self, f)
    }
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

fn pauli_from_str<T>(pauli_string: &str) -> Result<T, PauliStringParsingError>
where
    T: PauliMutable + NeutralElement<NeutralElementType = T>,
{
    let pauli_string_trimmed = pauli_string.trim();

    let (phase_exp, pauli_string_without_phase) = parse_phase(pauli_string_trimmed);

    if "₀₁₂₃₄₅₆₇₈₉0123456789_".contains(pauli_string_without_phase.chars().nth(1).unwrap_or(' ')) {
        // Sparse string
        parse_sparse_pauli(pauli_string_without_phase, phase_exp)
    } else {
        // Dense string
        let mut res: T = <T as NeutralElement>::neutral_element_of_size(pauli_string_without_phase.len());
        res.add_assign_phase_exp(phase_exp);
        for (index, character) in pauli_string_without_phase.chars().enumerate() {
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

fn parse_sparse_pauli<T>(pauli_string_without_phase: &str, phase_exp: u8) -> Result<T, PauliStringParsingError>
where
    T: PauliMutable + NeutralElement<NeutralElementType = T>,
{
    let mut entries = Vec::new();
    let mut pauli_char: Option<char> = None;
    let mut index: usize = 0;
    let mut has_digits = false;

    for character in pauli_string_without_phase.chars() {
        if let Some(digit) = get_digit(character) {
            index = index
                .checked_mul(10)
                .and_then(|i| i.checked_add(digit))
                .ok_or(PauliStringParsingError)?;
            has_digits = true;
        } else {
            match character {
                'X' | 'x' | 'Y' | 'y' | 'Z' | 'z' => {
                    if let Some(prev) = pauli_char {
                        if !has_digits {
                            return Err(PauliStringParsingError);
                        }
                        entries.push((prev, index));
                    }
                    pauli_char = Some(character.to_ascii_uppercase());
                    index = 0;
                    has_digits = false;
                }
                '{' | '}' | ' ' | '_' => {}
                _ => return Err(PauliStringParsingError),
            }
        }
    }
    if let Some(prev) = pauli_char {
        if !has_digits {
            return Err(PauliStringParsingError);
        }
        entries.push((prev, index));
    }

    let max_index = entries.iter().map(|(_, idx)| *idx).max().unwrap_or(0);
    let mut pauli: T = <T as NeutralElement>::neutral_element_of_size(max_index + 1);
    pauli.add_assign_phase_exp(phase_exp);

    for (pauli_char, index) in &entries {
        match pauli_char {
            'X' => pauli.mul_assign_left_x(*index),
            'Y' => pauli.mul_assign_left_y(*index),
            'Z' => pauli.mul_assign_left_z(*index),
            _ => unreachable!(),
        }
    }

    Ok(pauli)
}

fn get_digit(character: char) -> Option<usize> {
    match character {
        '₀'..='₉' => Some((character as u32 - '₀' as u32) as usize),
        '0'..='9' => Some((character as u32 - '0' as u32) as usize),
        _ => None,
    }
}

fn parse_phase(pauli_string_trimmed: &str) -> (u8, &str) {
    for phase_prefix in ["+i", "i", "-i", "+𝑖", "𝑖", "-𝑖", "+", "-"] {
        if pauli_string_trimmed.starts_with(phase_prefix) {
            let (phase_string, remainder) = pauli_string_trimmed.split_at(phase_prefix.len());
            let phase_exp = match phase_string {
                "-" => 2,
                "+i" | "+𝑖" | "i" | "𝑖" => 1,
                "-i" | "-𝑖" => 3,
                "+" => 0,
                _ => {
                    unreachable!();
                }
            };
            return (phase_exp, remainder);
        }
    }
    (0, pauli_string_trimmed)
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
    random_number_generator: &mut impl rand::RngExt,
) -> PauliLike {
    let mut res = PauliLike::neutral_element_of_size(num_qubits);
    res.set_random(num_qubits, random_number_generator);
    res
}

/// # Example
/// `pauli_random_order_two(6, &mut rand::rng());`
pub fn pauli_random_order_two<PauliLike: NeutralElement<NeutralElementType = PauliLike> + PauliMutable>(
    num_qubits: usize,
    random_number_generator: &mut impl rand::RngExt,
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
