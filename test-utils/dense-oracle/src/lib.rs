//! Dense (full state vector) simulation oracle shared by the `paulimer` and
//! `pauliverse` integration tests.
//!
//! This crate exists only to remove the duplicated brute-force statevector
//! simulator that both test suites used to carry inline. It is never published
//! (`publish = false`) and is depended on only as a `dev-dependency`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::should_implement_trait,
    clippy::must_use_candidate,
    clippy::return_self_not_must_use
)]

use binar::{Bitwise, BitwiseMut};
use num_complex::Complex;
use paulimer::clifford::{Clifford, PhasedCliffordUnitary};
use paulimer::pauli::Pauli;
use paulimer::{DensePauli, UnitaryOp};

/// Complex amplitude type used throughout the dense oracle.
pub type C = Complex<f64>;

/// `exp(i * pi/4 * k)`, i.e. the `k`-th power of the primitive 8th root of unity.
pub fn zeta8(k: i64) -> C {
    Complex::cis(std::f64::consts::FRAC_PI_4 * k.rem_euclid(8) as f64)
}

/// `1 / sqrt(2)`.
pub const ROOT_HALF: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Dense amplitude-vector state used as a correctness oracle.
pub struct Dense {
    pub qubit_count: usize,
    pub amp: Vec<C>,
}

impl Dense {
    pub fn zero(qubit_count: usize) -> Dense {
        let mut amp = vec![C::ZERO; 1 << qubit_count];
        amp[0] = C::new(1.0, 0.0);
        Dense { qubit_count, amp }
    }
    pub fn apply1(&mut self, qubit: usize, matrix: [[C; 2]; 2]) {
        let bit = 1usize << (self.qubit_count - 1 - qubit);
        for base in 0..(1 << self.qubit_count) {
            if base & bit == 0 {
                let amplitude_0 = self.amp[base];
                let amplitude_1 = self.amp[base | bit];
                self.amp[base] = matrix[0][0] * amplitude_0 + matrix[0][1] * amplitude_1;
                self.amp[base | bit] = matrix[1][0] * amplitude_0 + matrix[1][1] * amplitude_1;
            }
        }
    }
    pub fn apply_cx(&mut self, control: usize, target: usize) {
        let control_bit = 1usize << (self.qubit_count - 1 - control);
        let target_bit = 1usize << (self.qubit_count - 1 - target);
        let mut out = self.amp.clone();
        for base in 0..(1 << self.qubit_count) {
            let src = if base & control_bit != 0 {
                base ^ target_bit
            } else {
                base
            };
            out[base] = self.amp[src];
        }
        self.amp = out;
    }
    pub fn apply_cz(&mut self, first_qubit: usize, second_qubit: usize) {
        let first_bit = 1usize << (self.qubit_count - 1 - first_qubit);
        let second_bit = 1usize << (self.qubit_count - 1 - second_qubit);
        for base in 0..(1 << self.qubit_count) {
            if base & first_bit != 0 && base & second_bit != 0 {
                self.amp[base] = -self.amp[base];
            }
        }
    }
    pub fn apply_swap(&mut self, first_qubit: usize, second_qubit: usize) {
        let first_bit = 1usize << (self.qubit_count - 1 - first_qubit);
        let second_bit = 1usize << (self.qubit_count - 1 - second_qubit);
        let mut out = self.amp.clone();
        for base in 0..(1 << self.qubit_count) {
            let bit_first = usize::from(base & first_bit != 0);
            let bit_second = usize::from(base & second_bit != 0);
            let mut src = base & !first_bit & !second_bit;
            if bit_second != 0 {
                src |= first_bit;
            }
            if bit_first != 0 {
                src |= second_bit;
            }
            out[base] = self.amp[src];
        }
        self.amp = out;
    }
    pub fn pauli_applied(&self, x_bits: &[bool], z_bits: &[bool], phase: i64) -> Vec<C> {
        let mut out = vec![C::ZERO; self.amp.len()];
        let x_mask: usize = (0..self.qubit_count)
            .filter(|&qubit| x_bits[qubit])
            .map(|qubit| 1usize << (self.qubit_count - 1 - qubit))
            .sum();
        for base in 0..(1 << self.qubit_count) {
            let target = base ^ x_mask;
            let mut sign_parity = 0i64;
            for qubit in 0..self.qubit_count {
                if z_bits[qubit] && (base >> (self.qubit_count - 1 - qubit)) & 1 == 1 {
                    sign_parity ^= 1;
                }
            }
            let coeff = zeta8(2 * phase + 4 * sign_parity);
            out[target] += self.amp[base] * coeff;
        }
        out
    }
    pub fn apply_pauli(&mut self, x_bits: &[bool], z_bits: &[bool], phase: i64) {
        self.amp = self.pauli_applied(x_bits, z_bits, phase);
    }
    pub fn apply_pauli_exp(&mut self, x_bits: &[bool], z_bits: &[bool], phase: i64) {
        let pauli_applied = self.pauli_applied(x_bits, z_bits, phase);
        for base in 0..self.amp.len() {
            self.amp[base] = (self.amp[base] + pauli_applied[base] * Complex::I) * ROOT_HALF;
        }
    }
    pub fn apply_controlled_pauli(
        &mut self,
        first_pauli: &(Vec<bool>, Vec<bool>, i64),
        second_pauli: &(Vec<bool>, Vec<bool>, i64),
    ) {
        // controlled_pauli(first, second) = (I + first)/2 + (I - first)/2 * second
        let first_pauli_applied = self.pauli_applied(&first_pauli.0, &first_pauli.1, first_pauli.2);
        let plus: Vec<C> = (0..self.amp.len())
            .map(|index| (self.amp[index] + first_pauli_applied[index]) * 0.5)
            .collect();
        let minus = Dense {
            qubit_count: self.qubit_count,
            amp: (0..self.amp.len())
                .map(|index| (self.amp[index] - first_pauli_applied[index]) * 0.5)
                .collect(),
        };
        let second_pauli_applied_to_minus = minus.pauli_applied(&second_pauli.0, &second_pauli.1, second_pauli.2);
        for index in 0..self.amp.len() {
            self.amp[index] = plus[index] + second_pauli_applied_to_minus[index];
        }
    }
    pub fn project(&mut self, x_bits: &[bool], z_bits: &[bool], phase: i64, outcome: bool) {
        let pauli_applied = self.pauli_applied(x_bits, z_bits, phase);
        let sign = if outcome { -1.0 } else { 1.0 };
        for index in 0..self.amp.len() {
            self.amp[index] = (self.amp[index] + pauli_applied[index] * sign) * 0.5;
        }
        normalize(&mut self.amp);
    }
}

/// Renormalises an amplitude vector to unit norm.
///
/// # Panics
/// Panics if the state has (near) zero norm.
pub fn normalize(amp: &mut [C]) {
    let norm = amp.iter().map(Complex::norm_sqr).sum::<f64>().sqrt();
    assert!(norm > 1e-9, "attempted to normalize a vanishing state");
    let inv = 1.0 / norm;
    for amplitude in amp.iter_mut() {
        *amplitude *= inv;
    }
}

/// Returns the `2x2` matrix of a single-qubit unitary operation.
///
/// # Panics
/// Panics if `op` is not a single-qubit operation.
pub fn gate_matrix(op: UnitaryOp) -> [[C; 2]; 2] {
    let root_half = ROOT_HALF;
    match op {
        UnitaryOp::Hadamard => [
            [C::new(root_half, 0.0), C::new(root_half, 0.0)],
            [C::new(root_half, 0.0), C::new(-root_half, 0.0)],
        ],
        UnitaryOp::X => [[C::ZERO, C::new(1.0, 0.0)], [C::new(1.0, 0.0), C::ZERO]],
        UnitaryOp::Y => [[C::ZERO, C::new(0.0, -1.0)], [C::new(0.0, 1.0), C::ZERO]],
        UnitaryOp::Z => [[C::new(1.0, 0.0), C::ZERO], [C::ZERO, C::new(-1.0, 0.0)]],
        UnitaryOp::SqrtZ => [[C::new(1.0, 0.0), C::ZERO], [C::ZERO, C::new(0.0, 1.0)]],
        UnitaryOp::SqrtZInv => [[C::new(1.0, 0.0), C::ZERO], [C::ZERO, C::new(0.0, -1.0)]],
        UnitaryOp::SqrtX => [
            [zeta8(1) * root_half, zeta8(7) * root_half],
            [zeta8(7) * root_half, zeta8(1) * root_half],
        ],
        UnitaryOp::SqrtXInv => [
            [zeta8(7) * root_half, zeta8(1) * root_half],
            [zeta8(1) * root_half, zeta8(7) * root_half],
        ],
        UnitaryOp::SqrtY => [
            [zeta8(1) * root_half, zeta8(5) * root_half],
            [zeta8(1) * root_half, zeta8(1) * root_half],
        ],
        UnitaryOp::SqrtYInv => [
            [zeta8(7) * root_half, zeta8(7) * root_half],
            [zeta8(3) * root_half, zeta8(7) * root_half],
        ],
        other => panic!("gate_matrix called on multi-qubit op {other:?}"),
    }
}

/// Materialises the dense statevector produced by a phased Clifford unitary
/// acting on `|0...0>`.
pub fn statevector(phased: &PhasedCliffordUnitary) -> Vec<C> {
    use binar::BitMatrix;
    use binar::matrix::AlignedBitMatrix;
    let qubit_count = phased.num_qubits();
    let mut matrix = AlignedBitMatrix::zeros(qubit_count, qubit_count);
    for generator in 0..qubit_count {
        let image: DensePauli = phased.clifford().image_z(generator);
        for qubit in image.x_bits().support() {
            matrix.row_mut(generator).assign_index(qubit, true);
        }
    }
    let rank = BitMatrix::from_aligned(matrix).rank();
    let mag = (0.5f64).powf(rank as f64 / 2.0);
    let mut out = vec![C::ZERO; 1 << qubit_count];
    for idx in 0..(1usize << qubit_count) {
        let mut value = 0usize;
        for qubit in 0..qubit_count {
            if (idx >> (qubit_count - 1 - qubit)) & 1 == 1 {
                value |= 1usize << qubit;
            }
        }
        if let Some(exp) = phased.state_amplitude_phase_exponent_usize(value) {
            out[idx] = zeta8(i64::from(exp)) * mag;
        }
    }
    out
}

/// Decomposes a [`DensePauli`] into `(x_bits, z_bits, xz_phase_exponent)` arrays
/// sized to `qubit_count`.
pub fn pauli_arrays(pauli: &DensePauli, qubit_count: usize) -> (Vec<bool>, Vec<bool>, i64) {
    let mut x_bits = vec![false; qubit_count];
    let mut z_bits = vec![false; qubit_count];
    for qubit in pauli.x_bits().support() {
        x_bits[qubit] = true;
    }
    for qubit in pauli.z_bits().support() {
        z_bits[qubit] = true;
    }
    (x_bits, z_bits, i64::from(pauli.xz_phase_exponent()))
}

/// Approximate equality of two amplitude vectors, including global phase.
pub fn close(left: &[C], right: &[C]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left_value, right_value)| (left_value - right_value).norm_sqr() < 1e-6)
}
