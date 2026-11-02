use binar::matrix::AlignedBitMatrix;
use binar::{Bitwise, BitwisePair, BitwisePairMut};

use super::{DensePauli, Pauli, SparsePauli};

/// Minimum batch size `b` before `zx_upper_couplings` and `self_anti_commutation_matrix`
/// switch from evaluating the `b * (b - 1) / 2` upper-triangle pairs directly to full
/// `b × b` `AlignedBitMatrix::mul_transpose` calls.
///
/// ### Derivation
/// `AlignedBitMatrix::mul_transpose` transposes the right operand in `64 × 64` blocks before
/// running row-XOR multiplication. Because strictly upper-triangular evaluations (`0 <= i < j < b`)
/// compute only `b * (b - 1) / 2 ≈ (1 / 2) * b^2` row dot products, direct pair evaluation is
/// faster below `2 * 64 = 128` rows.
const DIRECT_UPPER_TRIANGLE_THRESHOLD: usize = 128;

/// Structure-of-arrays batch of `N`-qubit Pauli operators stored as two `(batch_size, N)`
/// [`AlignedBitMatrix`]es (`x_bits` and `z_bits`) and a `Vec<u8>` of `xz` phase exponents.
#[must_use]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PauliTensorBatch {
    x_bits: AlignedBitMatrix,
    z_bits: AlignedBitMatrix,
    phases: Vec<u8>,
}

impl PauliTensorBatch {
    /// Creates a batch of `batch_size` identity Paulis on `num_qubits` qubits.
    pub fn zeros(batch_size: usize, num_qubits: usize) -> Self {
        Self {
            x_bits: AlignedBitMatrix::zeros(batch_size, num_qubits),
            z_bits: AlignedBitMatrix::zeros(batch_size, num_qubits),
            phases: vec![0u8; batch_size],
        }
    }

    /// Creates a batch from preconstructed `x_bits`, `z_bits`, and `phases`.
    ///
    /// # Panics
    ///
    /// Panics if `x_bits.shape() != z_bits.shape()` or `phases.len() != x_bits.row_count()`.
    pub fn from_parts(x_bits: AlignedBitMatrix, z_bits: AlignedBitMatrix, phases: Vec<u8>) -> Self {
        assert_eq!(x_bits.shape(), z_bits.shape());
        assert_eq!(phases.len(), x_bits.row_count());
        Self { x_bits, z_bits, phases }
    }

    /// Packs a slice of [`DensePauli`] operators into a contiguous [`PauliTensorBatch`].
    pub fn from_dense(paulis: &[DensePauli], num_qubits: usize) -> Self {
        let batch_size = paulis.len();
        let mut x_bits = AlignedBitMatrix::zeros(batch_size, num_qubits);
        let mut z_bits = AlignedBitMatrix::zeros(batch_size, num_qubits);
        let mut phases = Vec::with_capacity(batch_size);
        for (row_index, pauli) in paulis.iter().enumerate() {
            x_bits.row_mut(row_index).assign(pauli.x_bits());
            z_bits.row_mut(row_index).assign(pauli.z_bits());
            phases.push(pauli.xz_phase_exponent());
        }
        Self { x_bits, z_bits, phases }
    }

    /// Packs a slice of [`SparsePauli`] operators into a contiguous [`PauliTensorBatch`].
    pub fn from_sparse(paulis: &[SparsePauli], num_qubits: usize) -> Self {
        let batch_size = paulis.len();
        let mut x_bits = AlignedBitMatrix::zeros(batch_size, num_qubits);
        let mut z_bits = AlignedBitMatrix::zeros(batch_size, num_qubits);
        let mut phases = Vec::with_capacity(batch_size);
        for (row_index, pauli) in paulis.iter().enumerate() {
            for qubit in pauli.x_bits().support() {
                x_bits.set((row_index, qubit), true);
            }
            for qubit in pauli.z_bits().support() {
                z_bits.set((row_index, qubit), true);
            }
            phases.push(pauli.xz_phase_exponent());
        }
        Self { x_bits, z_bits, phases }
    }

    /// Returns the number of Pauli operators in the batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.x_bits.row_count()
    }

    /// Returns `true` if the batch contains no Pauli operators.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of qubits `N` in each Pauli operator.
    #[must_use]
    pub fn qubit_count(&self) -> usize {
        self.x_bits.column_count()
    }

    /// Borrows the `(batch_size, N)` matrix of X bit vectors.
    #[must_use]
    pub fn x_matrix(&self) -> &AlignedBitMatrix {
        &self.x_bits
    }

    /// Mutably borrows the `(batch_size, N)` matrix of X bit vectors.
    pub fn x_matrix_mut(&mut self) -> &mut AlignedBitMatrix {
        &mut self.x_bits
    }

    /// Borrows the `(batch_size, N)` matrix of Z bit vectors.
    #[must_use]
    pub fn z_matrix(&self) -> &AlignedBitMatrix {
        &self.z_bits
    }

    /// Mutably borrows the `(batch_size, N)` matrix of Z bit vectors.
    pub fn z_matrix_mut(&mut self) -> &mut AlignedBitMatrix {
        &mut self.z_bits
    }

    /// Borrows the slice of `xz` phase exponents (`mod 4`).
    #[must_use]
    pub fn phases(&self) -> &[u8] {
        &self.phases
    }

    /// Mutably borrows the slice of `xz` phase exponents (`mod 4`).
    pub fn phases_mut(&mut self) -> &mut [u8] {
        &mut self.phases
    }

    /// Computes the `(self.len(), other.len())` pairwise symplectic inner-product
    /// (anti-commutation) matrix `X_self * Z_other^T ^ Z_self * X_other^T`.
    ///
    /// # Panics
    ///
    /// Panics if `self.qubit_count() != other.qubit_count()`.
    pub fn anti_commutation_matrix(&self, other: &Self) -> AlignedBitMatrix {
        &self.x_bits.mul_transpose(&other.z_bits) ^ &self.z_bits.mul_transpose(&other.x_bits)
    }

    /// Computes the symmetric `(b, b)` pairwise anti-commutation matrix within `self`.
    pub fn self_anti_commutation_matrix(&self) -> AlignedBitMatrix {
        let batch_size = self.x_bits.row_count();
        if batch_size < DIRECT_UPPER_TRIANGLE_THRESHOLD {
            let mut out = AlignedBitMatrix::zeros(batch_size, batch_size);
            for i in 0..batch_size {
                let xi = self.x_bits.row(i);
                let zi = self.z_bits.row(i);
                for j in (i + 1)..batch_size {
                    if xi.dot(&self.z_bits.row(j)) != zi.dot(&self.x_bits.row(j)) {
                        out.set((i, j), true);
                        out.set((j, i), true);
                    }
                }
            }
            out
        } else {
            self.anti_commutation_matrix(self)
        }
    }

    /// Computes the strictly upper-triangular `(b, b)` Z-to-X coupling matrix where entry
    /// `(i, j)` for `i < j` is `z_bits[i] · x_bits[j] (mod 2)`.
    pub fn zx_upper_couplings(&self) -> AlignedBitMatrix {
        let batch_size = self.x_bits.row_count();
        if batch_size < DIRECT_UPPER_TRIANGLE_THRESHOLD {
            let mut out = AlignedBitMatrix::zeros(batch_size, batch_size);
            for i in 0..batch_size {
                let zi = self.z_bits.row(i);
                for j in (i + 1)..batch_size {
                    if zi.dot(&self.x_bits.row(j)) {
                        out.set((i, j), true);
                    }
                }
            }
            out
        } else {
            let coupled = self.z_bits.mul_transpose(&self.x_bits);
            let mut out = AlignedBitMatrix::zeros(batch_size, batch_size);
            for i in 0..batch_size {
                for j in (i + 1)..batch_size {
                    if coupled.get((i, j)) {
                        out.set((i, j), true);
                    }
                }
            }
            out
        }
    }
}
