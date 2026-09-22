//! Phase-tracking Clifford unitaries.
//!
//! A [`CliffordUnitary`] represents a Clifford operator only up to a global phase: its symplectic
//! matrix together with the signs of the Pauli images fixes the operator on the Pauli group, but
//! not the overall `ζ₈` factor of the unitary. Many stabilizer algorithms do not need this factor,
//! but the *phased* outcome-complete simulation of arXiv:2603.24717 does, because it tracks the
//! exact amplitudes (including global phase) of the simulated state.
//!
//! [`PhasedCliffordUnitary`] augments a [`CliffordUnitary`] with an exact global-phase tracker.
//! The key observation is that everywhere a phase is needed it is the phase of the *encoder state*
//! `C|0…0⟩`: for any bit string `a`, `C|a⟩ = C X^a C† · C|0…0⟩`, and the sign of the Pauli image
//! `C X^a C†` is already tracked by [`CliffordUnitary`]. It is therefore enough to maintain one
//! exactly-known amplitude of `C|0…0⟩`, namely its value at a fixed *reference* basis string.
//!
//! Concretely the tracker stores a basis string `r` with `⟨r|C|0…0⟩ ≠ 0` and the `ζ₈` exponent of
//! that amplitude. Every amplitude of the stabilizer state `C|0…0⟩` has the same magnitude, so the
//! magnitude is recovered from the rank of the stabilizer tableau and only the phase needs to be
//! propagated. Each elementary left-multiplication updates the underlying [`CliffordUnitary`] with
//! the existing tableau code and updates the reference amplitude in `O(n²)` time.

use super::{Clifford, CliffordMutable, CliffordUnitary};
use crate::UnitaryOp;
use crate::pauli::{Pauli, PauliBinaryOps};
use binar::matrix::AlignedBitMatrix;
use binar::vec::AlignedBitVec;
use binar::{BitMatrix, BitVec, Bitwise, BitwiseMut, EchelonForm};

fn normalize_exponent(value: i64) -> u8 {
    u8::try_from(value.rem_euclid(8)).expect("a value reduced modulo 8 is in 0..8")
}

/// Sums two unit `ζ₈` powers `ζ₈^x + ζ₈^y` exactly, returning the `ζ₈` exponent of the result.
///
/// In this simulator every such sum that arises is the amplitude of a stabilizer state, hence a
/// non-negative real multiple of a single `ζ₈` power. Two unit `ζ₈` powers can therefore only be
/// separated by `d = (x − y) mod 8 ∈ {0, 2, 6}` (parallel or orthogonal), or by `d = 4`, in which
/// case they cancel exactly. This makes the sum computable with pure integer arithmetic, with no
/// floating-point amplitudes. Returns `None` when the two terms cancel.
fn add_zeta8_powers(x: i64, y: i64) -> Option<u8> {
    match (x - y).rem_euclid(8) {
        0 => Some(normalize_exponent(x)),
        2 => Some(normalize_exponent(y + 1)),
        6 => Some(normalize_exponent(y + 7)),
        4 => None,
        separation => unreachable!("ζ₈^{x} + ζ₈^{y} is not a ζ₈ power (separation {separation})"),
    }
}

/// Combines the at-most-two unit `ζ₈` powers contributing to a single amplitude, returning the
/// `ζ₈` exponent of their sum, or `None` when the amplitude vanishes. See [`add_zeta8_powers`].
fn combine_zeta8_powers(terms: &[i64]) -> Option<u8> {
    match *terms {
        [] => None,
        [only] => Some(normalize_exponent(only)),
        [x, y] => add_zeta8_powers(x, y),
        _ => unreachable!("an encoder amplitude is a sum of at most two ζ₈ powers"),
    }
}

/// A Clifford unitary that additionally tracks the exact global phase of its encoder state.
///
/// The operator agrees with [`Self::clifford`] on the Pauli group (same symplectic action and same
/// image signs) and in addition fixes the overall `ζ₈` phase, so that the amplitudes of `C|0…0⟩`
/// are determined exactly rather than only up to a global factor.
///
/// # Examples
///
/// ```
/// use paulimer::clifford::PhasedCliffordUnitary;
///
/// let mut phased = PhasedCliffordUnitary::identity(1);
/// phased.left_mul_hadamard(0);
/// // |0⟩ -> (|0⟩ + |1⟩)/√2: the amplitude at |0⟩ is positive real (ζ₈ exponent 0).
/// assert_eq!(phased.state_amplitude_phase_exponent_usize(0), Some(0));
/// ```
#[must_use]
#[derive(Clone)]
pub struct PhasedCliffordUnitary {
    clifford: CliffordUnitary,
    reference_string: AlignedBitVec,
    reference_phase_exponent: u8,
}

impl PhasedCliffordUnitary {
    /// Returns the identity operator on `num_qubits` qubits, with encoder state `|0…0⟩`.
    pub fn identity(num_qubits: usize) -> Self {
        Self {
            clifford: CliffordUnitary::identity(num_qubits),
            reference_string: AlignedBitVec::zeros(num_qubits),
            reference_phase_exponent: 0,
        }
    }

    /// Returns the number of qubits the operator acts on.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.clifford.num_qubits()
    }

    /// Returns the underlying phaseless [`CliffordUnitary`].
    pub fn clifford(&self) -> &CliffordUnitary {
        &self.clifford
    }

    /// Consumes the operator and returns the underlying phaseless [`CliffordUnitary`].
    pub fn into_clifford(self) -> CliffordUnitary {
        self.clifford
    }

    /// Left-multiplies by the global scalar `ζ₈^exponent = e^{i π exponent / 4}`.
    ///
    /// This leaves the underlying [`CliffordUnitary`] (and hence the symplectic action and image
    /// signs) unchanged and only advances the tracked global phase, since multiplying the operator
    /// by a scalar scales every amplitude of the encoder state by the same factor.
    pub fn left_mul_global_phase(&mut self, exponent: u8) {
        self.reference_phase_exponent =
            normalize_exponent(i64::from(self.reference_phase_exponent) + i64::from(exponent));
    }

    /// Returns the `ζ₈` exponent `e` such that `⟨basis|C|0…0⟩ = ζ₈^e · 2^{-k/2}` for some rank `k`,
    /// or `None` when that amplitude vanishes.
    #[must_use]
    pub fn state_amplitude_phase_exponent(&self, basis: &AlignedBitVec) -> Option<u8> {
        let relative = self.relative_phase(basis)?;
        Some(normalize_exponent(i64::from(self.reference_phase_exponent) + relative))
    }

    /// Convenience wrapper around [`Self::state_amplitude_phase_exponent`] taking the basis string
    /// as an integer whose qubit `q` bit is `(value >> q) & 1`.
    #[must_use]
    pub fn state_amplitude_phase_exponent_usize(&self, value: usize) -> Option<u8> {
        let mut basis = AlignedBitVec::zeros(self.num_qubits());
        for qubit in 0..self.num_qubits() {
            basis.assign_index(qubit, (value >> qubit) & 1 == 1);
        }
        self.state_amplitude_phase_exponent(&basis)
    }

    fn x_parts_matrix(&self) -> BitMatrix {
        let num_qubits = self.num_qubits();
        let mut matrix = AlignedBitMatrix::zeros(num_qubits, num_qubits);
        for generator in 0..num_qubits {
            let image = self.clifford.image_z(generator);
            for qubit in image.x_bits().support() {
                matrix.row_mut(generator).assign_index(qubit, true);
            }
        }
        BitMatrix::from_aligned(matrix)
    }

    fn relative_phase(&self, target: &AlignedBitVec) -> Option<i64> {
        let num_qubits = self.num_qubits();
        let difference: BitVec = (0..num_qubits)
            .map(|qubit| target.index(qubit) ^ self.reference_string.index(qubit))
            .collect();
        let echelon = EchelonForm::new(self.x_parts_matrix());
        let combination = echelon.transpose_solve(&difference.as_view())?;
        let mut product = self.clifford.image_z(0);
        let mut started = false;
        for generator in combination.support() {
            let image = self.clifford.image_z(generator);
            if started {
                product.mul_assign_right(&image);
            } else {
                product = image;
                started = true;
            }
        }
        if !started {
            return Some(0);
        }
        let phase_exponent = i64::from(product.xz_phase_exponent());
        let mut sign_parity = false;
        for qubit in product.z_bits().support() {
            if self.reference_string.index(qubit) {
                sign_parity = !sign_parity;
            }
        }
        let relative = (2 * phase_exponent + if sign_parity { 4 } else { 0 }).rem_euclid(8);
        Some(relative)
    }

    fn apply_one_qubit(
        &mut self,
        qubit: usize,
        amplitudes: [[Option<i64>; 2]; 2],
        symplectic: impl FnOnce(&mut CliffordUnitary),
    ) {
        for output_bit in [self.reference_string.index(qubit), !self.reference_string.index(qubit)] {
            let mut candidate = self.reference_string.clone();
            candidate.assign_index(qubit, output_bit);
            let mut terms = [0i64; 2];
            let mut count = 0usize;
            for input_bit in [false, true] {
                let Some(entry) = amplitudes[usize::from(output_bit)][usize::from(input_bit)] else {
                    continue;
                };
                let mut source = candidate.clone();
                source.assign_index(qubit, input_bit);
                if let Some(relative) = self.relative_phase(&source) {
                    terms[count] = entry + relative;
                    count += 1;
                }
            }
            if let Some(increment) = combine_zeta8_powers(&terms[..count]) {
                self.reference_phase_exponent =
                    normalize_exponent(i64::from(self.reference_phase_exponent) + i64::from(increment));
                self.reference_string = candidate;
                symplectic(&mut self.clifford);
                return;
            }
        }
        unreachable!("a unitary maps a nonzero state to a nonzero state");
    }

    fn apply_two_qubit(
        &mut self,
        qubit_a: usize,
        qubit_b: usize,
        inverse: impl Fn(bool, bool) -> (bool, bool, i64),
        symplectic: impl FnOnce(&mut CliffordUnitary),
    ) {
        for output_a in [
            self.reference_string.index(qubit_a),
            !self.reference_string.index(qubit_a),
        ] {
            for output_b in [
                self.reference_string.index(qubit_b),
                !self.reference_string.index(qubit_b),
            ] {
                let mut candidate = self.reference_string.clone();
                candidate.assign_index(qubit_a, output_a);
                candidate.assign_index(qubit_b, output_b);
                let (input_a, input_b, entry) = inverse(output_a, output_b);
                let mut source = candidate.clone();
                source.assign_index(qubit_a, input_a);
                source.assign_index(qubit_b, input_b);
                if let Some(relative) = self.relative_phase(&source) {
                    self.reference_phase_exponent =
                        normalize_exponent(i64::from(self.reference_phase_exponent) + entry + relative);
                    self.reference_string = candidate;
                    symplectic(&mut self.clifford);
                    return;
                }
            }
        }
        unreachable!("a unitary maps a nonzero state to a nonzero state");
    }

    /// Left-multiplies by a Hadamard gate on `qubit`.
    pub fn left_mul_hadamard(&mut self, qubit: usize) {
        self.apply_one_qubit(qubit, [[Some(0), Some(0)], [Some(0), Some(4)]], |clifford| {
            clifford.left_mul_hadamard(qubit);
        });
    }

    /// Left-multiplies by a Pauli `X` gate on `qubit`.
    pub fn left_mul_x(&mut self, qubit: usize) {
        self.apply_one_qubit(qubit, [[None, Some(0)], [Some(0), None]], |clifford| {
            clifford.left_mul_x(qubit);
        });
    }

    /// Left-multiplies by a Pauli `Y` gate on `qubit`.
    pub fn left_mul_y(&mut self, qubit: usize) {
        self.apply_one_qubit(qubit, [[None, Some(6)], [Some(2), None]], |clifford| {
            clifford.left_mul_y(qubit);
        });
    }

    /// Left-multiplies by a Pauli `Z` gate on `qubit`.
    pub fn left_mul_z(&mut self, qubit: usize) {
        self.apply_one_qubit(qubit, [[Some(0), None], [None, Some(4)]], |clifford| {
            clifford.left_mul_z(qubit);
        });
    }

    /// Left-multiplies by `√Z` (the phase gate `S = diag(1, i)`) on `qubit`.
    pub fn left_mul_root_z(&mut self, qubit: usize) {
        self.apply_one_qubit(qubit, [[Some(0), None], [None, Some(2)]], |clifford| {
            clifford.left_mul_root_z(qubit);
        });
    }

    /// Left-multiplies by `√Z†` (`S† = diag(1, -i)`) on `qubit`.
    pub fn left_mul_root_z_inverse(&mut self, qubit: usize) {
        self.apply_one_qubit(qubit, [[Some(0), None], [None, Some(6)]], |clifford| {
            clifford.left_mul_root_z_inverse(qubit);
        });
    }

    /// Left-multiplies by `√X` on `qubit`.
    pub fn left_mul_root_x(&mut self, qubit: usize) {
        self.apply_one_qubit(qubit, [[Some(1), Some(7)], [Some(7), Some(1)]], |clifford| {
            clifford.left_mul_root_x(qubit);
        });
    }

    /// Left-multiplies by `√X†` on `qubit`.
    pub fn left_mul_root_x_inverse(&mut self, qubit: usize) {
        self.apply_one_qubit(qubit, [[Some(7), Some(1)], [Some(1), Some(7)]], |clifford| {
            clifford.left_mul_root_x_inverse(qubit);
        });
    }

    /// Left-multiplies by `√Y` on `qubit`.
    pub fn left_mul_root_y(&mut self, qubit: usize) {
        self.apply_one_qubit(qubit, [[Some(1), Some(5)], [Some(1), Some(1)]], |clifford| {
            clifford.left_mul_root_y(qubit);
        });
    }

    /// Left-multiplies by `√Y†` on `qubit`.
    pub fn left_mul_root_y_inverse(&mut self, qubit: usize) {
        self.apply_one_qubit(qubit, [[Some(7), Some(7)], [Some(3), Some(7)]], |clifford| {
            clifford.left_mul_root_y_inverse(qubit);
        });
    }

    /// Left-multiplies by a controlled-`X` gate with the given control and target qubits.
    pub fn left_mul_cx(&mut self, control: usize, target: usize) {
        self.apply_two_qubit(
            control,
            target,
            |control_bit, target_bit| (control_bit, control_bit ^ target_bit, 0),
            |clifford| {
                clifford.left_mul_cx(control, target);
            },
        );
    }

    /// Left-multiplies by a controlled-`Z` gate on the two given qubits.
    pub fn left_mul_cz(&mut self, qubit_a: usize, qubit_b: usize) {
        self.apply_two_qubit(
            qubit_a,
            qubit_b,
            |bit_a, bit_b| (bit_a, bit_b, if bit_a && bit_b { 4 } else { 0 }),
            |clifford| {
                clifford.left_mul_cz(qubit_a, qubit_b);
            },
        );
    }

    /// Left-multiplies by a swap of the two given qubits.
    pub fn left_mul_swap(&mut self, qubit_a: usize, qubit_b: usize) {
        self.apply_two_qubit(
            qubit_a,
            qubit_b,
            |bit_a, bit_b| (bit_b, bit_a, 0),
            |clifford| {
                clifford.left_mul_swap(qubit_a, qubit_b);
            },
        );
    }

    /// Left-multiplies by the named elementary [`UnitaryOp`] on `support`.
    pub fn left_mul(&mut self, unitary_op: UnitaryOp, support: &[usize]) {
        use UnitaryOp::{
            ControlledX, ControlledZ, Hadamard, I, PrepareBell, SqrtX, SqrtXInv, SqrtY, SqrtYInv, SqrtZ, SqrtZInv,
            Swap, X, Y, Z,
        };
        match unitary_op {
            I => {}
            X => self.left_mul_x(support[0]),
            Y => self.left_mul_y(support[0]),
            Z => self.left_mul_z(support[0]),
            SqrtX => self.left_mul_root_x(support[0]),
            SqrtXInv => self.left_mul_root_x_inverse(support[0]),
            SqrtY => self.left_mul_root_y(support[0]),
            SqrtYInv => self.left_mul_root_y_inverse(support[0]),
            SqrtZ => self.left_mul_root_z(support[0]),
            SqrtZInv => self.left_mul_root_z_inverse(support[0]),
            Hadamard => self.left_mul_hadamard(support[0]),
            Swap => self.left_mul_swap(support[0], support[1]),
            ControlledX => self.left_mul_cx(support[0], support[1]),
            ControlledZ => self.left_mul_cz(support[0], support[1]),
            PrepareBell => self.left_mul_prepare_bell(support[0], support[1]),
        }
    }

    /// Left-multiplies by the qubit permutation `permutation` acting on `support`.
    ///
    /// A permutation of the computational basis labels has trivial global phase, so only the
    /// underlying tableau and the reference basis string are relabelled while the tracked phase is
    /// left unchanged. The convention matches [`CliffordMutable::left_mul_permutation`]: the qubit
    /// `support[i]` takes the role previously played by `support[permutation[i]]`.
    pub fn left_mul_permutation(&mut self, permutation: &[usize], support: &[usize]) {
        let previous: Vec<bool> = support
            .iter()
            .map(|&qubit| self.reference_string.index(qubit))
            .collect();
        self.clifford.left_mul_permutation(permutation, support);
        for (index, &qubit) in support.iter().enumerate() {
            self.reference_string.assign_index(qubit, previous[permutation[index]]);
        }
    }

    /// Left-multiplies by the Bell-state preparation Clifford on the two given qubits.
    pub fn left_mul_prepare_bell(&mut self, qubit_a: usize, qubit_b: usize) {
        self.left_mul_hadamard(qubit_a);
        self.left_mul_cx(qubit_a, qubit_b);
    }

    /// Left-multiplies by the Pauli operator `pauli` (including its sign).
    pub fn left_mul_pauli<PauliLike: Pauli<PhaseExponentValue = u8>>(&mut self, pauli: &PauliLike) {
        let phase = pauli.xz_phase_exponent();
        for qubit in pauli.z_bits().support() {
            self.left_mul_z(qubit);
        }
        for qubit in pauli.x_bits().support() {
            self.left_mul_x(qubit);
        }
        if phase != 0 {
            self.reference_phase_exponent =
                normalize_exponent(i64::from(self.reference_phase_exponent) + 2 * i64::from(phase));
        }
    }

    /// Left-multiplies by `exp(iπ/4 · pauli)`, the square root of `pauli` up to phase.
    pub fn left_mul_pauli_exp<PauliLike: Pauli<PhaseExponentValue = u8>>(&mut self, pauli: &PauliLike) {
        if self.num_qubits() == 0 {
            return;
        }
        let pauli_phase = i64::from(pauli.xz_phase_exponent());

        let mut shifted = self.reference_string.clone();
        for qubit in pauli.x_bits().support() {
            shifted.assign_index(qubit, !shifted.index(qubit));
        }
        let candidates = [self.reference_string.clone(), shifted];

        for candidate in candidates {
            let mut terms = [0i64; 2];
            let mut count = 0usize;
            if let Some(relative) = self.relative_phase(&candidate) {
                terms[count] = relative;
                count += 1;
            }
            let mut source = candidate.clone();
            let mut sign_parity = false;
            for qubit in pauli.z_bits().support() {
                let flipped = source.index(qubit) ^ pauli.x_bits().index(qubit);
                if flipped {
                    sign_parity = !sign_parity;
                }
            }
            for qubit in pauli.x_bits().support() {
                source.assign_index(qubit, !source.index(qubit));
            }
            if let Some(relative) = self.relative_phase(&source) {
                let coefficient = 2 + 2 * pauli_phase + if sign_parity { 4 } else { 0 };
                terms[count] = relative + coefficient;
                count += 1;
            }
            if let Some(increment) = combine_zeta8_powers(&terms[..count]) {
                self.reference_phase_exponent =
                    normalize_exponent(i64::from(self.reference_phase_exponent) + i64::from(increment));
                self.reference_string = candidate;
                self.clifford.left_mul_pauli_exp(pauli);
                return;
            }
        }
        unreachable!("a unitary maps a nonzero state to a nonzero state");
    }

    /// Grows or shrinks the operator to `new_num_qubits`, tensoring with identity on `|0⟩` ancillas.
    pub fn resize(&mut self, new_num_qubits: usize) {
        let old_num_qubits = self.num_qubits();
        self.clifford.resize(new_num_qubits);
        if new_num_qubits == old_num_qubits {
            return;
        }
        let mut reference = AlignedBitVec::zeros(new_num_qubits);
        for qubit in 0..old_num_qubits.min(new_num_qubits) {
            reference.assign_index(qubit, self.reference_string.index(qubit));
        }
        self.reference_string = reference;
    }
}
