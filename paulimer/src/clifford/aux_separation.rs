//! Separation of disentangled auxiliary qubits from a phased encoder.
//!
//! Implements the "Separating auxiliary qubits" primitive of §4.3 of
//! [arXiv:2603.24717](https://arxiv.org/abs/2603.24717) (eq. aux-out-separation). Given a phased
//! encoder `Co` on `n` qubits whose auxiliary qubits are *disentangled* from the output qubits over
//! the whole computational-basis family `{Co|r⟩ : r ∈ {0,1}ⁿ}`, it factors the encoder state as a
//! tensor product of an output encoder `Co₁` and an auxiliary encoder `Co₂`, together with linear
//! basis-relabelling maps `A₁`, `A₂` and the recovered global phase:
//!
//! ```text
//! Co|r⟩ = ζ₈^{φ(r)} (Co₁|A₁r⟩ ⊗ Co₂|A₂r⟩).
//! ```
//!
//! The separation phase `φ(r)` is a degree-at-most-two polynomial in `r` modulo `8` (a constant
//! `l`, a linear part and a quadratic part). This is exactly the `(l, p, s, B)` global-phase
//! bookkeeping that the phased outcome-complete simulation already tracks, so the recovered
//! polynomial folds directly into that data.

use binar::matrix::AlignedBitMatrix;
use binar::vec::AlignedBitVec;
use binar::{BitMatrix, Bitwise, BitwiseMut};

use super::{
    Clifford, CliffordUnitary, PhasedCliffordUnitary, clifford_to_pauli_exponents, group_encoding_clifford_of,
    standard_restriction_with_sign_matrix,
};
use crate::DensePauli;
use crate::pauli::Pauli;

/// The degree-at-most-two `ζ₈` phase polynomial recovered by [`separate_auxiliary_qubits`].
///
/// [`Self::evaluate`] returns the `ζ₈` exponent `φ(r)` (modulo `8`) such that
/// `Co|r⟩ = ζ₈^{φ(r)} (Co₁|A₁r⟩ ⊗ Co₂|A₂r⟩)`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SeparationPhase {
    constant: u8,
    linear: Vec<u8>,
    quadratic: Vec<u8>,
}

impl SeparationPhase {
    /// The constant term `l = φ(0)`: the global `ζ₈` phase relating `Co|0…0⟩` to the canonical phase
    /// of `Co₁|0…0⟩ ⊗ Co₂|0…0⟩`.
    #[must_use]
    pub fn constant(&self) -> u8 {
        self.constant
    }

    /// The number of input bits `n` the polynomial is defined over.
    #[must_use]
    pub fn num_bits(&self) -> usize {
        self.linear.len()
    }

    /// The linear coefficient `c_i` (a `ζ₈` exponent) of input bit `i`.
    #[must_use]
    pub fn linear_coefficient(&self, bit: usize) -> u8 {
        self.linear[bit]
    }

    /// The quadratic coefficient `c_{ij}` (a `ζ₈` exponent) of the pair of input bits `i < j`.
    #[must_use]
    pub fn quadratic_coefficient(&self, first: usize, second: usize) -> u8 {
        let (low, high) = if first <= second {
            (first, second)
        } else {
            (second, first)
        };
        self.quadratic[low * self.linear.len() + high]
    }

    /// Evaluates the polynomial at `bits`, returning the `ζ₈` exponent `φ(bits)` modulo `8`.
    #[must_use]
    pub fn evaluate(&self, bits: &AlignedBitVec) -> u8 {
        let mut total = self.constant % 8;
        let set: Vec<usize> = bits.support().collect();
        for (position, &first) in set.iter().enumerate() {
            total = (total + self.linear[first]) % 8;
            for &second in &set[position + 1..] {
                total = (total + self.quadratic[first * self.linear.len() + second]) % 8;
            }
        }
        total
    }
}

/// The result of separating the auxiliary qubits of a phased encoder; see
/// [`separate_auxiliary_qubits`].
#[derive(Clone, Debug)]
pub struct AuxiliarySeparation {
    output_encoder: CliffordUnitary,
    auxiliary_encoder: CliffordUnitary,
    output_basis_map: AlignedBitMatrix,
    auxiliary_basis_map: AlignedBitMatrix,
    phase: SeparationPhase,
}

impl AuxiliarySeparation {
    /// The output-qubit encoder `Co₁`, acting on the output qubits (in the order supplied to
    /// [`separate_auxiliary_qubits`]).
    pub fn output_encoder(&self) -> &CliffordUnitary {
        &self.output_encoder
    }

    /// The auxiliary-qubit encoder `Co₂`, acting on the auxiliary qubits (the complement of the
    /// output qubits, in increasing order).
    pub fn auxiliary_encoder(&self) -> &CliffordUnitary {
        &self.auxiliary_encoder
    }

    /// The basis-relabelling map `A₁` (with `Co₁`'s qubit count rows and `n` columns): for input
    /// label `r`, the output encoder reproduces `Co₁|A₁r⟩`.
    pub fn output_basis_map(&self) -> &AlignedBitMatrix {
        &self.output_basis_map
    }

    /// The basis-relabelling map `A₂` (with `Co₂`'s qubit count rows and `n` columns): for input
    /// label `r`, the auxiliary encoder reproduces `Co₂|A₂r⟩`.
    pub fn auxiliary_basis_map(&self) -> &AlignedBitMatrix {
        &self.auxiliary_basis_map
    }

    /// The recovered separation phase polynomial `φ`.
    #[must_use]
    pub fn phase(&self) -> &SeparationPhase {
        &self.phase
    }
}

/// The reason an auxiliary-qubit separation could not be computed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuxiliarySeparationError {
    /// An output-qubit index was out of range or repeated.
    InvalidOutputQubits,
    /// The auxiliary qubits are entangled with the output qubits, so the encoder state does not
    /// factor as `Co₁|0…0⟩ ⊗ Co₂|0…0⟩` and cannot be separated.
    AuxiliaryQubitsEntangled,
}

/// Separates the auxiliary qubits of the phased encoder `encoder` into an output factor and an
/// auxiliary factor, recovering the global phase (§4.3 of arXiv:2603.24717, eq. aux-out-separation).
///
/// `output_qubits` lists the qubits of `encoder` that form the output register; the remaining qubits
/// are the auxiliary register. The auxiliary qubits must be disentangled from the output qubits over
/// the whole computational-basis family `{encoder|r⟩}`. On success the returned
/// [`AuxiliarySeparation`] yields encoders `Co₁`, `Co₂`, basis-relabelling maps `A₁`, `A₂` and a
/// degree-at-most-two phase polynomial `φ` with
/// `encoder|r⟩ = ζ₈^{φ(r)} (Co₁|A₁r⟩ ⊗ Co₂|A₂r⟩)` for every `r`.
///
/// `Co₁` and `Co₂` are the signed output/auxiliary marginal state encoders, obtained from the signed
/// marginal stabiliser groups via [`standard_restriction_with_sign_matrix`] and
/// [`group_encoding_clifford_of`]. `A₁`/`A₂` are read off the preimages of the relabelled marginal
/// Pauli images. The phase polynomial is interpolated from `φ` evaluated at `0`, the unit labels
/// `e_i` and the pairs `e_i + e_j`, using the exact `ζ₈` phase of the encoder state at a single
/// support representative (no statevector materialisation).
///
/// # Errors
///
/// Returns [`AuxiliarySeparationError::InvalidOutputQubits`] if `output_qubits` contains a repeated
/// index or an index that is not a qubit of `encoder`, and
/// [`AuxiliarySeparationError::AuxiliaryQubitsEntangled`] if the auxiliary qubits are not
/// disentangled from the output qubits.
pub fn separate_auxiliary_qubits(
    encoder: &PhasedCliffordUnitary,
    output_qubits: &[usize],
) -> Result<AuxiliarySeparation, AuxiliarySeparationError> {
    let num_qubits = encoder.num_qubits();

    let mut is_output = vec![false; num_qubits];
    for &qubit in output_qubits {
        if qubit >= num_qubits || is_output[qubit] {
            return Err(AuxiliarySeparationError::InvalidOutputQubits);
        }
        is_output[qubit] = true;
    }
    let output: Vec<usize> = output_qubits.to_vec();
    let auxiliary: Vec<usize> = (0..num_qubits).filter(|qubit| !is_output[*qubit]).collect();

    let clifford = encoder.clifford();
    let output_encoder = marginal_encoder(clifford, &output)?;
    let auxiliary_encoder = marginal_encoder(clifford, &auxiliary)?;
    let blocks = phased_from_clifford(&output_encoder.tensor(&auxiliary_encoder));

    let basis_map = basis_label_map(clifford, &output_encoder, &auxiliary_encoder, &output, &auxiliary);

    let phase = interpolate_phase(encoder, &blocks, &basis_map, num_qubits)?;

    let (output_basis_map, auxiliary_basis_map) = split_rows(&basis_map, output.len());

    Ok(AuxiliarySeparation {
        output_encoder,
        auxiliary_encoder,
        output_basis_map,
        auxiliary_basis_map,
        phase,
    })
}

/// Builds the signed marginal state encoder of `Co|0…0⟩` restricted to `support`, reindexed to
/// `support.len()` local qubits. Fails if the marginal is mixed (the auxiliary/output split is
/// entangled), detected by the restricted stabiliser group having fewer than `support.len()`
/// generators.
fn marginal_encoder(
    clifford: &CliffordUnitary,
    support: &[usize],
) -> Result<CliffordUnitary, AuxiliarySeparationError> {
    let sign_matrix = BitMatrix::zeros(clifford.num_qubits(), 1);
    let (generators, _) = standard_restriction_with_sign_matrix(clifford, &sign_matrix, support);
    if generators.len() != support.len() {
        return Err(AuxiliarySeparationError::AuxiliaryQubitsEntangled);
    }
    Ok(group_encoding_clifford_of(&generators, support.len()))
}

/// Builds the combined basis-relabelling map `A` (with `num_qubits` rows and columns): column `j`
/// places the X-bits of `preimage_{Co₁}(P_out(e_j))` on the leading `output.len()` rows and of
/// `preimage_{Co₂}(P_aux(e_j))` on the trailing rows, where `P_*(e_j)` is the support-restriction of
/// `Co X_j Co†`.
fn basis_label_map(
    clifford: &CliffordUnitary,
    output_encoder: &CliffordUnitary,
    auxiliary_encoder: &CliffordUnitary,
    output: &[usize],
    auxiliary: &[usize],
) -> AlignedBitMatrix {
    let num_qubits = clifford.num_qubits();
    let mut matrix = AlignedBitMatrix::zeros(num_qubits, num_qubits);
    for column in 0..num_qubits {
        let image = clifford.image_x(column);
        let output_preimage = output_encoder.preimage(&restrict_pauli_to(&image, output, num_qubits));
        let auxiliary_preimage = auxiliary_encoder.preimage(&restrict_pauli_to(&image, auxiliary, num_qubits));
        for bit in output_preimage.x_bits().support() {
            matrix.set((bit, column), true);
        }
        for bit in auxiliary_preimage.x_bits().support() {
            matrix.set((output.len() + bit, column), true);
        }
    }
    matrix
}

/// Restricts the signed `DensePauli` `pauli` (on `num_qubits` qubits) to `support`, reindexing to
/// `support.len()` local qubits and dropping the phase.
fn restrict_pauli_to(pauli: &DensePauli, support: &[usize], num_qubits: usize) -> DensePauli {
    let mut local = vec![usize::MAX; num_qubits];
    for (index, &global) in support.iter().enumerate() {
        local[global] = index;
    }
    let mut x_bits = AlignedBitVec::zeros(support.len());
    let mut z_bits = AlignedBitVec::zeros(support.len());
    for bit in pauli.x_bits().support() {
        if local[bit] != usize::MAX {
            x_bits.assign_index(local[bit], true);
        }
    }
    for bit in pauli.z_bits().support() {
        if local[bit] != usize::MAX {
            z_bits.assign_index(local[bit], true);
        }
    }
    DensePauli::from_bits(x_bits, z_bits, 0)
}

/// Builds the phased Clifford whose tableau equals `clifford`, with the canonical global phase fixed
/// by decomposing the whole Clifford into `π/4` Pauli exponents and replaying them.
fn phased_from_clifford(clifford: &CliffordUnitary) -> PhasedCliffordUnitary {
    let mut phased = PhasedCliffordUnitary::identity(clifford.num_qubits());
    for exponent in clifford_to_pauli_exponents(clifford) {
        phased.left_mul_pauli_exp(&exponent);
    }
    phased
}

/// Materialises the phased state `clifford|bits⟩ = clifford X^{bits}|0…0⟩`.
fn encoded_basis_state(clifford: &PhasedCliffordUnitary, bits: &AlignedBitVec) -> PhasedCliffordUnitary {
    let pauli = DensePauli::from_bits(bits.clone(), AlignedBitVec::zeros(clifford.num_qubits()), 0);
    let image = clifford.clifford().image(&pauli);
    let mut state = clifford.clone();
    state.left_mul_pauli(&image);
    state
}

/// Computes the separation phase `φ(r)` at the single label `r`: the `ζ₈` exponent relating
/// `encoder|r⟩` to `blocks|A r⟩`. Returns `None` if the two states do not agree up to a global phase
/// (the auxiliary qubits are entangled).
fn separation_phase_at(
    encoder: &PhasedCliffordUnitary,
    blocks: &PhasedCliffordUnitary,
    basis_map: &AlignedBitMatrix,
    label: &AlignedBitVec,
) -> Option<u8> {
    let encoder_state = encoded_basis_state(encoder, label);
    let block_labels = basis_map * &label.as_view();
    let block_state = encoded_basis_state(blocks, &block_labels);
    let representative = encoder_state.support_representative();
    let encoder_phase = encoder_state.state_amplitude_phase_exponent(&representative)?;
    let block_phase = block_state.state_amplitude_phase_exponent(&representative)?;
    Some(u8::try_from((i64::from(encoder_phase) - i64::from(block_phase)).rem_euclid(8)).expect("modulo 8"))
}

/// Interpolates the degree-at-most-two separation phase polynomial from its values at `0`, the unit
/// labels `e_i` and the pairs `e_i + e_j`.
fn interpolate_phase(
    encoder: &PhasedCliffordUnitary,
    blocks: &PhasedCliffordUnitary,
    basis_map: &AlignedBitMatrix,
    num_qubits: usize,
) -> Result<SeparationPhase, AuxiliarySeparationError> {
    let phase_at = |label: &AlignedBitVec| {
        separation_phase_at(encoder, blocks, basis_map, label).ok_or(AuxiliarySeparationError::AuxiliaryQubitsEntangled)
    };
    let unit = |index: usize| {
        let mut label = AlignedBitVec::zeros(num_qubits);
        label.assign_index(index, true);
        label
    };

    let constant = phase_at(&AlignedBitVec::zeros(num_qubits))?;
    let mut linear = vec![0u8; num_qubits];
    for (index, value) in linear.iter_mut().enumerate() {
        *value =
            u8::try_from((i64::from(phase_at(&unit(index))?) - i64::from(constant)).rem_euclid(8)).expect("modulo 8");
    }

    let mut quadratic = vec![0u8; num_qubits * num_qubits];
    for first in 0..num_qubits {
        for second in (first + 1)..num_qubits {
            let mut label = unit(first);
            label.assign_index(second, true);
            let value = i64::from(phase_at(&label)?)
                - i64::from(constant)
                - i64::from(linear[first])
                - i64::from(linear[second]);
            quadratic[first * num_qubits + second] = u8::try_from(value.rem_euclid(8)).expect("modulo 8");
        }
    }

    Ok(SeparationPhase {
        constant,
        linear,
        quadratic,
    })
}

/// Splits the `n × n` basis-label matrix into its leading `output_count` rows (`A₁`) and the
/// remaining rows (`A₂`).
fn split_rows(matrix: &AlignedBitMatrix, output_count: usize) -> (AlignedBitMatrix, AlignedBitMatrix) {
    let columns = matrix.column_count();
    let mut output = AlignedBitMatrix::zeros(output_count, columns);
    let mut auxiliary = AlignedBitMatrix::zeros(matrix.row_count() - output_count, columns);
    for row in 0..output_count {
        for column in 0..columns {
            output.set((row, column), matrix.get((row, column)));
        }
    }
    for row in output_count..matrix.row_count() {
        for column in 0..columns {
            auxiliary.set((row - output_count, column), matrix.get((row, column)));
        }
    }
    (output, auxiliary)
}
