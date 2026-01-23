use crate::pauli::{Pauli, PauliBinaryOps, PauliMutable};
use crate::UnitaryOp;
use binar::{matrix::AlignedBitMatrix, Bitwise};
use derive_more::{TryFrom, TryInto};

#[derive(Clone, Copy, Debug, PartialEq, Eq, TryInto, TryFrom)]
#[try_from(repr)]
pub enum XOrZ {
    X = Axis::X as isize,
    Z = Axis::Z as isize,
}

/// Core trait for Clifford unitary operations.
///
/// `Clifford` provides the fundamental interface for working with Clifford gates.
/// All Clifford types implement this trait.
///
/// # Key Capabilities
///
/// - **Pauli conjugation**: [`image`](Clifford::image) and [`preimage`](Clifford::preimage) for propagating Paulis
/// - **Composition**: [`multiply_with`](Clifford::multiply_with) for gate sequences
/// - **Construction**: [`identity`](Clifford::identity), [`random`](Clifford::random), [`from_preimages`](Clifford::from_preimages)
/// - **Structure queries**: [`is_identity`](Clifford::is_identity), [`is_css`](Clifford::is_css), [`is_diagonal`](Clifford::is_diagonal)
/// - **Matrix access**: [`symplectic_matrix`](Clifford::symplectic_matrix) for the binary representation
///
/// # Conjugation: Image and Preimage
///
/// The fundamental Clifford operation is conjugation of Pauli operators:
/// - **Image**: `C · P · C†` (forward propagation)
/// - **Preimage**: `C† · P · C` (backward propagation)
///
/// These satisfy: `C.image(&C.preimage(&P)) == P`
///
/// # Type Parameters
///
/// - `PhaseExponentValue`: Phase representation for Paulis (typically `u8`)
/// - `DensePauli`: The dense Pauli type used for results
///
/// # Examples
///
/// ```
/// use paulimer::{CliffordUnitary, Clifford, DensePauli};
///
/// let cnot = CliffordUnitary::identity(2);
/// // After applying CNOT gates via CliffordMutable...
///
/// // Propagate Pauli through Clifford
/// let x0: DensePauli = "XI".parse().unwrap();
/// let image = cnot.image(&x0);
///
/// // Compose Cliffords
/// let h = CliffordUnitary::identity(2);
/// let composed = cnot.multiply_with(&h);
/// ```
pub trait Clifford {
    type PhaseExponentValue;
    type DensePauli: PauliMutable + Pauli<PhaseExponentValue = Self::PhaseExponentValue>;
    fn num_qubits(&self) -> usize;
    fn qubits(&self) -> std::ops::Range<usize> {
        0..self.num_qubits()
    }

    fn is_valid(&self) -> bool;
    fn is_identity(&self) -> bool;

    #[must_use]
    fn identity(num_qubits: usize) -> Self;
    #[must_use]
    fn zero(num_qubits: usize) -> Self;
    #[must_use]
    fn random(num_qubits: usize, random_number_generator: &mut impl rand::Rng) -> Self;

    fn preimage_x(&self, qubit_index: usize) -> Self::DensePauli;
    fn preimage_z(&self, qubit_index: usize) -> Self::DensePauli;
    fn preimage_x_bits(&self, x_bits: &impl Bitwise) -> Self::DensePauli;
    fn preimage_z_bits(&self, z_bits: &impl Bitwise) -> Self::DensePauli;
    fn preimage<PauliLike: Pauli<PhaseExponentValue = Self::PhaseExponentValue>>(
        &self,
        pauli: &PauliLike,
    ) -> Self::DensePauli;

    fn image_x(&self, qubit_index: usize) -> Self::DensePauli;
    fn image_z(&self, qubit_index: usize) -> Self::DensePauli;
    fn image_x_bits(&self, x_bits: &impl Bitwise) -> Self::DensePauli;
    fn image_z_bits(&self, z_bits: &impl Bitwise) -> Self::DensePauli;
    fn image<PauliLike: Pauli<PhaseExponentValue = Self::PhaseExponentValue>>(
        &self,
        pauli: &PauliLike,
    ) -> Self::DensePauli;

    #[must_use]
    fn multiply_with(&self, rhs: &Self) -> Self;
    #[must_use]
    fn tensor(&self, rhs: &Self) -> Self;
    #[must_use]
    fn inverse(&self) -> Self;
    #[must_use]
    fn from_preimages(preimages: &[Self::DensePauli]) -> Self;
    #[must_use]
    fn from_css_preimage_indicators(x_indicators: &AlignedBitMatrix, z_indicators: &AlignedBitMatrix) -> Self;

    fn is_diagonal(&self, axis: XOrZ) -> bool;
    fn is_diagonal_resource_encoder(&self, axis: XOrZ) -> bool;
    fn is_css(&self) -> bool;
    fn unitary_from_diagonal_resource_state(&self, axis: XOrZ) -> Option<Self>
    where
        Self: Sized;

    fn symplectic_matrix(&self) -> AlignedBitMatrix;
}

/// Trait for mutable Clifford operations.
///
/// `CliffordMutable` extends [`Clifford`] with operations that modify the Clifford gate in place
/// by left-multiplying with standard gates. This is the primary way to build Clifford circuits.
///
/// # Gate Application
///
/// All gates are applied via **left multiplication**: `C' = G · C` where G is the gate.
/// This means gates are applied in the order written:
///
/// ```ignore
/// clifford.left_mul_hadamard(0);   // Apply H₀
/// clifford.left_mul_cx(0, 1);      // Then apply CNOT₀₁
/// // Result: C = CNOT₀₁ · H₀
/// ```
///
/// # Available Gates
///
/// - **Single-qubit Paulis**: [`left_mul_x`](CliffordMutable::left_mul_x), [`left_mul_y`](CliffordMutable::left_mul_y), [`left_mul_z`](CliffordMutable::left_mul_z)
/// - **Single-qubit Cliffords**: [`left_mul_hadamard`](CliffordMutable::left_mul_hadamard), [`left_mul_root_z`](CliffordMutable::left_mul_root_z), [`left_mul_root_x`](CliffordMutable::left_mul_root_x)
/// - **Two-qubit gates**: [`left_mul_cx`](CliffordMutable::left_mul_cx), [`left_mul_cz`](CliffordMutable::left_mul_cz), [`left_mul_swap`](CliffordMutable::left_mul_swap)
/// - **Generic**: [`left_mul`](CliffordMutable::left_mul) for any [`UnitaryOp`]
/// - **Advanced**: [`left_mul_pauli`](CliffordMutable::left_mul_pauli), [`left_mul_clifford`](CliffordMutable::left_mul_clifford)
///
/// # Examples
///
/// ```
/// use paulimer::{CliffordUnitary, CliffordMutable, Clifford};
/// use paulimer::{DensePauli, Pauli, UnitaryOp};
///
/// // Build Bell state circuit
/// let mut circuit = CliffordUnitary::identity(2);
/// circuit.left_mul_hadamard(0);
/// circuit.left_mul_cx(0, 1);
///
/// // Verify action on X
/// let x0: DensePauli = "XI".parse().unwrap();
/// let result = circuit.image(&x0);
/// // X₀ → (CNOT · H)(X₀) → CNOT(Z₀) → Z₀ ⊗ Z₁
/// ```
pub trait CliffordMutable {
    type PhaseExponentValue;

    fn left_mul_x(&mut self, qubit_index: usize);
    fn left_mul_y(&mut self, qubit_index: usize);
    fn left_mul_z(&mut self, qubit_index: usize);

    fn left_mul_hadamard(&mut self, qubit_index: usize);
    fn left_mul_root_z(&mut self, qubit_index: usize);
    fn left_mul_root_z_inverse(&mut self, qubit_index: usize);
    fn left_mul_root_x(&mut self, qubit_index: usize);
    fn left_mul_root_x_inverse(&mut self, qubit_index: usize);
    fn left_mul_root_y(&mut self, qubit_index: usize);
    fn left_mul_root_y_inverse(&mut self, qubit_index: usize);

    fn left_mul_cx(&mut self, control_qubit_index: usize, target_qubit_index: usize);
    fn left_mul_cz(&mut self, qubit1_index: usize, qubit2_index: usize);
    fn left_mul_prepare_bell(&mut self, qubit1_index: usize, qubit2_index: usize);
    fn left_mul_swap(&mut self, qubit_index1: usize, qubit_index2: usize);

    fn left_mul(&mut self, unitary_op: UnitaryOp, support: &[usize]);

    fn left_mul_pauli<PauliLike: Pauli>(&mut self, pauli: &PauliLike);
    fn left_mul_pauli_exp<PauliLike: Pauli<PhaseExponentValue = Self::PhaseExponentValue>>(
        &mut self,
        pauli: &PauliLike,
    );
    fn left_mul_controlled_pauli<PauliLike: Pauli<PhaseExponentValue = Self::PhaseExponentValue>>(
        &mut self,
        control: &PauliLike,
        target: &PauliLike,
    );

    fn left_mul_permutation(&mut self, permutation: &[usize], support: &[usize]);
    fn left_mul_clifford<CliffordLike>(&mut self, clifford: &CliffordLike, support: &[usize])
    where
        CliffordLike: Clifford<PhaseExponentValue = Self::PhaseExponentValue>
            + PreimageViews<PhaseExponentValue = Self::PhaseExponentValue>;

    fn resize(&mut self, new_qubit_count: usize);
}

pub trait PreimageViews {
    type PhaseExponentValue;
    type PreImageView<'life>: Pauli<PhaseExponentValue = Self::PhaseExponentValue>
    where
        Self: 'life;
    type ImageViewUpToPhase<'life>: Pauli<PhaseExponentValue = ()>
    where
        Self: 'life;

    fn preimage_x_view(&self, index: usize) -> Self::PreImageView<'_>;
    fn preimage_z_view(&self, index: usize) -> Self::PreImageView<'_>;
    fn x_image_view_up_to_phase(&self, qubit_index: usize) -> Self::ImageViewUpToPhase<'_>;
    fn z_image_view_up_to_phase(&self, qubit_index: usize) -> Self::ImageViewUpToPhase<'_>;
}

pub trait MutablePreImages {
    type PhaseExponentValue;
    type PreImageViewMut<'life>: PauliBinaryOps + Pauli<PhaseExponentValue = Self::PhaseExponentValue>
    where
        Self: 'life;
    fn preimage_x_view_mut(&mut self, qubit_index: usize) -> Self::PreImageViewMut<'_>;
    fn preimage_z_view_mut(&mut self, qubit_index: usize) -> Self::PreImageViewMut<'_>;
    fn preimage_xz_views_mut(&mut self, index: usize) -> (Self::PreImageViewMut<'_>, Self::PreImageViewMut<'_>);
    fn preimage_xz_views_mut_distinct(&mut self, index: (usize, usize)) -> crate::Tuple2x2<Self::PreImageViewMut<'_>>;
}

pub mod generic_algos;

/// Clifford unitary on n qubits.
///
/// A Clifford gate is a unitary that maps Pauli operators to Pauli operators
/// under conjugation. `CliffordUnitary` stores this mapping using a 2n×2n binary matrix
/// (the symplectic or "Choi" representation) plus phase information.
///
/// # Representation
///
/// For each Pauli basis element (X₀, Z₀, X₁, Z₁, ..., Xₙ₋₁, Zₙ₋₁), the data structure
/// stores its image under conjugation by the Clifford. This allows O(n²) conjugation
/// operations instead of the naive O(2ⁿ) by exploiting the Pauli group structure.
///
/// # Key Operations
///
/// - **Conjugation**: [`image`](Clifford::image) and [`preimage`](Clifford::preimage) for
///   propagating Paulis through the circuit
/// - **Composition**: [`multiply_with`](Clifford::multiply_with) for combining gates
/// - **Inversion**: [`inverse`](Clifford::inverse) for reversing circuits
/// - **Construction**: [`from_preimages`](Clifford::from_preimages) from Pauli basis images
///
/// # Common Gates
///
/// Use [`CliffordMutable`] methods for standard gates:
/// - Single-qubit: `left_mul_hadamard`, `left_mul_root_z`
/// - Two-qubit: `left_mul_cx`, `left_mul_cz`
/// - Or use [`left_mul`](CliffordMutable::left_mul) with [`UnitaryOp`]
///
/// # Examples
///
/// ```
/// use paulimer::{CliffordUnitary, Clifford, CliffordMutable};
/// use paulimer::{DensePauli, Pauli, UnitaryOp};
///
/// // Create CNOT gate
/// let mut cnot = CliffordUnitary::identity(2);
/// cnot.left_mul_cx(0, 1);  // Control on qubit 0, target on qubit 1
///
/// // Propagate X₀ through CNOT: X₀ → X₀ ⊗ X₁
/// let x0: DensePauli = "XI".parse().unwrap();
/// let image = cnot.image(&x0);
/// let expected: DensePauli = "XX".parse().unwrap();
/// assert_eq!(image, expected);
///
/// // Compose gates
/// let mut circuit = CliffordUnitary::identity(2);
/// circuit.left_mul(UnitaryOp::Hadamard, &[0]);
/// circuit.left_mul(UnitaryOp::ControlledX, &[0, 1]);
/// // Now circuit represents H₀ followed by CNOT
/// ```
///
/// # Performance
///
/// - Memory: O(n²) for the binary matrix
/// - Conjugation: O(n²) to compute image/preimage of a Pauli
/// - Composition: O(n³) to multiply two Cliffords (Gaussian elimination)
/// - Construction: O(n³) to verify validity
///
/// This representation enables efficient simulation and verification of
/// Clifford circuits, which is essential for quantum error correction.
#[must_use]
#[derive(Eq, Clone)]
pub struct CliffordUnitary {
    projective: CliffordUnitaryModPauli,
    preimage_phase_exponents: Vec<u8>,
}

#[must_use]
#[derive(Eq, Clone)]
pub struct CliffordUnitaryModPauli {
    bits: AlignedBitMatrix,
}

#[must_use]
#[repr(C, align(64))]
#[derive(Eq, PartialEq, Clone, Debug)]
pub struct CliffordModPauliBatch<const WORD_COUNT: usize, const QUBIT_COUNT: usize> {
    pub preimages: [[[u64; WORD_COUNT]; QUBIT_COUNT]; 4],
}

mod clifford_impl;
use crate::core::Axis;
pub use clifford_impl::{
    apply_qubit_clifford_by_axis, group_encoding_clifford_of, prepare_all_plus, prepare_all_zero,
    random_clifford_via_operations_sampling, recover_z_images_phases, split_clifford_encoder,
    split_clifford_encoder_mod_pauli, split_clifford_mod_pauli_with_transforms, split_phased_css,
    split_qubit_cliffords_and_css, split_qubit_tensor_product_encoder,
};

#[derive(Debug, PartialEq, Eq, Default)]
pub struct CliffordStringParsingError;
