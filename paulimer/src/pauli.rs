pub mod generic;
pub use generic::{add_assign_bits, pauli_random, pauli_random_order_two, PauliUnitary, PauliUnitaryProjective};

pub mod operators;
pub use operators::Phase;

mod dense;
pub use dense::{condense_from, dense_from, DensePauli, DensePauliProjective};
use sorted_iter::SortedIterator;

mod sparse;
pub use sparse::{as_sparse, as_sparse_projective, remapped_sparse, SparsePauli, SparsePauliProjective};

mod algorithms;
pub use algorithms::{
    apply_pauli_exponent, apply_root_x, apply_root_y, apply_root_z, are_mutually_commuting,
    are_the_same_group_up_to_phases, complete_to_full_pauli_basis, paulis_qubit_count,
};

use crate::traits::NeutralElement;
use binar::{Bitwise, BitwiseMut, BitwisePair, BitwisePairMut};

/// Marker trait for types that can store Pauli bit patterns.
///
/// This trait is automatically implemented for types that satisfy the required bounds.
/// It represents the storage requirements for X and Z bit patterns in Pauli operators.
pub trait PauliBits: Bitwise + BitwisePair + PartialEq + std::hash::Hash {}
impl<T: Bitwise + BitwisePair + PartialEq + std::hash::Hash> PauliBits for T {}

/// Core trait for Pauli operator types.
///
/// `Pauli` provides the fundamental interface for working with Pauli operators,
/// whether dense, sparse, or with/without phase tracking. All Pauli representations
/// implement this trait.
///
/// # Key Operations
///
/// - **Access**: [`x_bits`](Pauli::x_bits), [`z_bits`](Pauli::z_bits) for component access
/// - **Queries**: [`weight`](Pauli::weight), [`support`](Pauli::support), [`is_identity`](Pauli::is_identity)
/// - **Structure**: [`y_parity`](Pauli::y_parity) for phase calculations
/// - **Constructors**: [`x`](Pauli::x), [`y`](Pauli::y), [`z`](Pauli::z) for single-qubit operators
///
/// # Type Parameters
///
/// - `Bits`: The storage type for X and Z components
/// - `PhaseExponentValue`: Phase representation (`u8` for full Paulis, `()` for projective)
///
/// # Examples
///
/// ```
/// use paulimer::{DensePauli, Pauli};
///
/// let pauli: DensePauli = "XYZ".parse().unwrap();
/// assert_eq!(pauli.weight(), 3);
/// assert_eq!(pauli.qubit_count(), 3);
/// assert!(!pauli.is_identity());
/// 
/// // Test single-qubit operators
/// let x = DensePauli::x(0, 3);
/// assert!(x.is_pauli_x(0));
/// ```
pub trait Pauli: PartialEq {
    type Bits: PauliBits;
    type PhaseExponentValue;

    fn xz_phase_exponent(&self) -> Self::PhaseExponentValue;
    fn x_bits(&self) -> &Self::Bits;
    fn z_bits(&self) -> &Self::Bits;

    #[inline]
    fn support(&self) -> impl SortedIterator<Item = usize> {
        self.x_bits().support().union(self.z_bits().support())
    }

    #[inline]
    fn max_support(&self) -> Option<usize> {
        match (self.x_bits().max_support(), self.z_bits().max_support()) {
            (None, None) => None,
            (None, Some(id)) | (Some(id), None) => Some(id),
            (Some(id1), Some(id2)) => Some(std::cmp::max(id1, id2)),
        }
    }

    #[inline]
    fn min_support(&self) -> Option<usize> {
        match (self.x_bits().min_support(), self.z_bits().min_support()) {
            (None, None) => None,
            (None, Some(id)) | (Some(id), None) => Some(id),
            (Some(id1), Some(id2)) => Some(std::cmp::min(id1, id2)),
        }
    }

    #[inline]
    fn y_parity(&self) -> bool {
        self.x_bits().dot(self.z_bits())
    }

    #[inline]
    fn weight(&self) -> usize {
        self.x_bits().or_weight(self.z_bits())
    }

    fn is_order_two(&self) -> bool;
    fn is_identity(&self) -> bool;
    fn is_pauli_x(&self, qubit: usize) -> bool;
    fn is_pauli_z(&self, qubit: usize) -> bool;
    fn is_pauli_y(&self, qubit: usize) -> bool;
    fn equals_to(&self, rhs: &Self) -> bool;
    fn to_xz_bits(self) -> (Self::Bits, Self::Bits);

    #[inline]
    fn qubit_count(&self) -> usize {
        self.max_support().map_or(0, |max_id| max_id + 1)
    }

    #[must_use]
    fn x(qubit_index: usize, qubit_count: usize) -> Self
    where
        Self: Sized,
        Self: NeutralElement<NeutralElementType = Self>,
        Self: PauliMutable,
    {
        let mut result = Self::neutral_element_of_size(qubit_count);
        result.mul_assign_left_x(qubit_index);
        result
    }

    #[must_use]
    fn y(qubit_index: usize, qubit_count: usize) -> Self
    where
        Self: Sized,
        Self: NeutralElement<NeutralElementType = Self>,
        Self: PauliMutable,
    {
        let mut result = Self::neutral_element_of_size(qubit_count);
        result.mul_assign_left_y(qubit_index);
        result
    }

    #[must_use]
    fn z(qubit_index: usize, qubit_count: usize) -> Self
    where
        Self: Sized,
        Self: NeutralElement<NeutralElementType = Self>,
        Self: PauliMutable,
    {
        let mut result = Self::neutral_element_of_size(qubit_count);
        result.mul_assign_left_z(qubit_index);
        result
    }
}

/// Trait for mutable Pauli operations.
///
/// `PauliMutable` extends [`Pauli`] with operations that modify the operator in place.
/// This includes phase manipulation, single-qubit gate application, and Pauli multiplication.
///
/// # Key Operations
///
/// - **Phase control**: [`assign_phase_exp`](PauliMutable::assign_phase_exp), [`add_assign_phase_exp`](PauliMutable::add_assign_phase_exp)
/// - **Conjugation**: [`complex_conjugate`](PauliMutable::complex_conjugate), [`invert`](PauliMutable::invert)
/// - **Single-qubit gates**: [`mul_assign_left_x`](PauliMutable::mul_assign_left_x), [`mul_assign_left_y`](PauliMutable::mul_assign_left_y), [`mul_assign_left_z`](PauliMutable::mul_assign_left_z)
/// - **Construction**: [`set_identity`](PauliMutable::set_identity), [`set_random`](PauliMutable::set_random)
///
/// # Left vs Right Multiplication
///
/// Methods come in pairs for left and right multiplication:
/// - `mul_assign_left_x(q)`: Multiply by X_q on the left (prepend X_q)
/// - `mul_assign_right_x(q)`: Multiply by X_q on the right (append X_q)
///
/// These differ in phase: left multiplication accounts for commutation with existing operators.
///
/// # Examples
///
/// ```
/// use paulimer::{DensePauli, Pauli, PauliMutable};
///
/// let mut pauli: DensePauli = "XII".parse().unwrap();
/// 
/// // Apply single-qubit gate
/// pauli.mul_assign_left_z(0);  // Now iY on qubit 0
/// assert_eq!(pauli.weight(), 1);
/// 
/// // Phase manipulation
/// pauli.add_assign_phase_exp(3);  // Adjust phase
/// ```
pub trait PauliMutable: Pauli<Bits: BitwiseMut> {
    fn assign_phase_exp(&mut self, rhs: u8);
    fn add_assign_phase_exp(&mut self, rhs: u8);
    fn complex_conjugate(&mut self);
    fn invert(&mut self);
    fn negate(&mut self);
    fn assign_phase_from<PauliLike: Pauli<PhaseExponentValue = Self::PhaseExponentValue>>(&mut self, other: &PauliLike);
    fn mul_assign_phase_from<PauliLike: Pauli<PhaseExponentValue = Self::PhaseExponentValue>>(
        &mut self,
        other: &PauliLike,
    );

    fn mul_assign_left_x(&mut self, qubit_id: usize);
    fn mul_assign_right_x(&mut self, qubit_id: usize);
    fn mul_assign_left_z(&mut self, qubit_id: usize);
    fn mul_assign_right_z(&mut self, qubit_id: usize);
    fn set_identity(&mut self);
    fn set_random(&mut self, num_qubits: usize, random_number_generator: &mut impl rand::Rng);
    fn set_random_order_two(&mut self, num_qubits: usize, random_number_generator: &mut impl rand::Rng);

    fn mul_assign_left_y(&mut self, qubit_id: usize) {
        self.mul_assign_left_z(qubit_id);
        self.mul_assign_left_x(qubit_id);
        self.add_assign_phase_exp(1u8);
    }

    fn mul_assign_right_y(&mut self, qubit_id: usize) {
        self.mul_assign_right_x(qubit_id);
        self.mul_assign_right_z(qubit_id);
        self.add_assign_phase_exp(1u8);
    }
}

pub fn anti_commutes_with<LeftPauli: Pauli, RightPauli: Pauli>(left: &LeftPauli, right: &RightPauli) -> bool
where
    LeftPauli::Bits: BitwisePair<RightPauli::Bits>,
{
    left.x_bits().dot(right.z_bits()) ^ left.z_bits().dot(right.x_bits())
}

pub fn commutes_with<LeftPauli: Pauli, RightPauli: Pauli>(left: &LeftPauli, right: &RightPauli) -> bool
where
    LeftPauli::Bits: BitwisePair<RightPauli::Bits>,
{
    !anti_commutes_with(left, right)
}

/// Low-level trait for mutable access to Pauli bit storage.
///
/// This trait provides direct mutable access to the X and Z bit patterns.
/// Most users should use [`PauliMutable`] or [`PauliBinaryOps`] instead.
pub trait PauliMutableBits<Bits: Bitwise>: PauliMutable {
    type BitsMutable: BitwisePairMut<Bits>;
    fn x_bits_mut(&mut self) -> &mut Self::BitsMutable;
    fn z_bits_mut(&mut self) -> &mut Self::BitsMutable;
}

/// Trait for binary operations between Pauli operators.
///
/// `PauliBinaryOps` provides operations for combining two Pauli operators:
/// multiplication, assignment, and commutator calculations.
///
/// # Type Parameter
///
/// - `Other`: The type of the right-hand operand (defaults to `Self`)
///
/// This allows operations between different Pauli types (e.g., dense and sparse).
///
/// # Key Operations
///
/// - **Multiplication**: Returns product P₁ · P₂
/// - **In-place multiplication**: [`mul_assign_left`](PauliBinaryOps::mul_assign_left), [`mul_assign_right`](PauliBinaryOps::mul_assign_right)
/// - **Assignment**: [`assign`](PauliBinaryOps::assign) copies from another Pauli
/// - **Commutation**: Methods for checking and computing commutators
///
/// # Examples
///
/// ```
/// use paulimer::{DensePauli, PauliBinaryOps, Pauli};
///
/// let x: DensePauli = "XII".parse().unwrap();
/// let z: DensePauli = "ZII".parse().unwrap();
///
/// // In-place multiplication: X · Z = iY
/// let mut product = x.clone();
/// product.mul_assign_right(&z);
/// assert_eq!(product.weight(), 1);
/// ```
pub trait PauliBinaryOps<Other: ?Sized + Pauli = Self>: PauliMutable {
    fn assign(&mut self, rhs: &Other);
    fn assign_with_offset(&mut self, rhs: &Other, start_qubit_index: usize, num_qubits: usize);
    fn mul_assign_right(&mut self, rhs: &Other);
    fn mul_assign_left(&mut self, lhs: &Other);
}

pub trait PauliNeutralElement: Pauli + NeutralElement<NeutralElementType: PauliBinaryOps<Self>> {}
