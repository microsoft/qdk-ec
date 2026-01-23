//! Frame-based Pauli error propagation with *O(n_gates × n_qubits)* complexity.
//!
//! This module provides [`FramePropagator`], which tracks accumulated Pauli errors
//! across multiple shots as they propagate through a quantum circuit.
//!
//! ## Layout
//!
//! Frames are stored as `(n_qubits × n_shots)` bit matrices, so that gate operations
//! become efficient row XORs across all shots simultaneously. Each row `i` contains
//! the X (or Z) component of qubit `i` across all shots.
//!
//! ## Algorithm
//!
//! Instead of pre-computing a fault→outcome matrix (*O(n_gates²)* for surface codes),
//! this module propagates Pauli error frames through the circuit gate-by-gate.
//! Each frame tracks the accumulated error for one shot as a Pauli on `n_qubits`.

// Allow mathematical notation like X_a → X_a Z_b in doc comments
#![allow(clippy::doc_markdown)]

use binar::matrix::AlignedBitMatrix;
use binar::vec::AlignedBitVec;
use binar::{BitMatrix, Bitwise, BitwisePairMut};
use paulimer::clifford::CliffordUnitary;
use paulimer::pauli::Pauli;
use paulimer::UnitaryOp;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::circuit::{Instruction, OutcomeId, QubitId};
use crate::noise::{sample_non_identity_pauli_bits, PauliFault};
use crate::sampling::GeometricSampler;

/// Pauli error frame propagator for batched multi-shot simulation.
///
/// Tracks accumulated Pauli errors across all shots as two bit matrices
/// (X and Z components), propagating them through Clifford gates via conjugation.
pub(crate) struct FramePropagator {
    x_frames: AlignedBitMatrix,
    z_frames: AlignedBitMatrix,
    outcome_deltas: AlignedBitMatrix,
    shot_count: usize,
    next_outcome_id: OutcomeId,
    /// Reusable scratch space for temporary bitvector operations.
    scratch: AlignedBitVec,
}

impl FramePropagator {
    /// Create a new frame propagator with all frames initialized to identity (no error).
    #[must_use]
    pub fn new(qubit_count: usize, outcome_count: usize, shot_count: usize) -> Self {
        Self {
            x_frames: AlignedBitMatrix::with_shape(qubit_count, shot_count),
            z_frames: AlignedBitMatrix::with_shape(qubit_count, shot_count),
            outcome_deltas: AlignedBitMatrix::with_shape(outcome_count, shot_count),
            shot_count,
            next_outcome_id: 0,
            scratch: AlignedBitVec::zeros(shot_count),
        }
    }

    /// Consume the propagator and return the outcome deltas matrix.
    ///
    /// Layout: `(n_outcomes × n_shots)` - each row is the error delta for one outcome.
    pub fn into_outcome_deltas(self) -> AlignedBitMatrix {
        self.outcome_deltas
    }

    // ========== Anti-commutation ==========

    /// Compute which shots have frames that anti-commute with the given Pauli.
    ///
    /// Returns a bitvector where bit `i` is set if the frame for shot `i`
    /// anti-commutes with `pauli`. This is the core primitive for measurement
    /// and Pauli exponential propagation.
    ///
    /// Anti-commutation: `{P_frame, P_obs} = 0` when `x_frame · z_obs + z_frame · x_obs = 1 (mod 2)`
    fn anticommutation_mask<P: Pauli>(&self, pauli: &P) -> AlignedBitVec
    where
        P::Bits: Bitwise,
    {
        let mut mask = AlignedBitVec::zeros(self.shot_count);
        for qubit in pauli.z_bits().support() {
            mask.bitxor_assign(&self.x_frames.row(qubit));
        }
        for qubit in pauli.x_bits().support() {
            mask.bitxor_assign(&self.z_frames.row(qubit));
        }
        mask
    }

    /// Save copies of frame rows for the given qubits.
    ///
    /// Used before in-place transformations that need the old values.
    fn snapshot_rows(&self, qubits: &[QubitId]) -> (Vec<AlignedBitVec>, Vec<AlignedBitVec>) {
        let x_rows = qubits
            .iter()
            .map(|&q| AlignedBitVec::from_view(&self.x_frames.row(q)))
            .collect();
        let z_rows = qubits
            .iter()
            .map(|&q| AlignedBitVec::from_view(&self.z_frames.row(q)))
            .collect();
        (x_rows, z_rows)
    }

    // ========== Gate Propagation ==========
    //
    // Pauli frames transform under Clifford conjugation:
    // If the circuit applies gate G, and we have error P before G,
    // then after G the effective error is G P G†.
    //
    // For Paulis: X → G X G†, Z → G Z G†
    // We track these transformations on the frame.

    /// Apply Hadamard on qubit q: X ↔ Z
    pub fn apply_h(&mut self, qubit: QubitId) {
        let x_temp = AlignedBitVec::from_view(&self.x_frames.row(qubit));
        let z_row = self.z_frames.row(qubit);
        self.x_frames.row_mut(qubit).assign(&z_row);
        self.z_frames.row_mut(qubit).assign(&x_temp);
    }

    /// Apply S (√Z) on qubit q: X → Y = iXZ, so X → XZ (mod phase), Z → Z
    ///
    /// Note: S† has the same effect on Pauli frames (we track mod phase).
    pub fn apply_s(&mut self, qubit: QubitId) {
        let z_row = self.z_frames.row(qubit);
        self.x_frames.row_mut(qubit).bitxor_assign(&z_row);
    }

    /// Apply CNOT(control, target): X_c → X_c X_t, Z_t → Z_c Z_t
    pub fn apply_cnot(&mut self, control: QubitId, target: QubitId) {
        {
            let (control_row, mut target_row) = self.x_frames.rows_mut(control, target);
            target_row.bitxor_assign(&control_row);
        }
        {
            let (mut control_row, target_row) = self.z_frames.rows_mut(control, target);
            control_row.bitxor_assign(&target_row);
        }
    }

    /// Apply CZ(a, b): X_a → X_a Z_b, X_b → Z_a X_b, Z unchanged
    pub fn apply_cz(&mut self, qubit_a: QubitId, qubit_b: QubitId) {
        let z_b = self.z_frames.row(qubit_b);
        self.x_frames.row_mut(qubit_a).bitxor_assign(&z_b);

        let z_a = self.z_frames.row(qubit_a);
        self.x_frames.row_mut(qubit_b).bitxor_assign(&z_a);
    }

    /// Apply SWAP(a, b): swap both X and Z rows
    pub fn apply_swap(&mut self, qubit_a: QubitId, qubit_b: QubitId) {
        self.x_frames.swap_rows(qubit_a, qubit_b);
        self.z_frames.swap_rows(qubit_a, qubit_b);
    }

    /// Apply √X on qubit q: Z → -Y = ZX, X → X
    ///
    /// Note: √X† has the same effect on Pauli frames (we track mod phase).
    pub fn apply_sqrt_x(&mut self, qubit: QubitId) {
        let x_row = self.x_frames.row(qubit);
        self.z_frames.row_mut(qubit).bitxor_assign(&x_row);
    }

    /// Apply a `UnitaryOp` to the frames.
    pub fn apply_unitary_op(&mut self, op: UnitaryOp, qubits: &[QubitId]) {
        match op {
            UnitaryOp::I | UnitaryOp::X | UnitaryOp::Y | UnitaryOp::Z => {
                // Paulis commute with themselves (mod phase), no frame change
            }
            UnitaryOp::Hadamard => {
                debug_assert_eq!(qubits.len(), 1);
                self.apply_h(qubits[0]);
            }
            UnitaryOp::SqrtZ | UnitaryOp::SqrtZInv => {
                debug_assert_eq!(qubits.len(), 1);
                self.apply_s(qubits[0]);
            }
            UnitaryOp::SqrtX | UnitaryOp::SqrtXInv => {
                debug_assert_eq!(qubits.len(), 1);
                self.apply_sqrt_x(qubits[0]);
            }
            UnitaryOp::SqrtY | UnitaryOp::SqrtYInv => {
                // √Y conjugation swaps X ↔ Z (with signs we don't track).
                debug_assert_eq!(qubits.len(), 1);
                self.apply_h(qubits[0]);
            }
            UnitaryOp::ControlledX => {
                debug_assert_eq!(qubits.len(), 2);
                self.apply_cnot(qubits[0], qubits[1]);
            }
            UnitaryOp::ControlledZ => {
                debug_assert_eq!(qubits.len(), 2);
                self.apply_cz(qubits[0], qubits[1]);
            }
            UnitaryOp::Swap => {
                debug_assert_eq!(qubits.len(), 2);
                self.apply_swap(qubits[0], qubits[1]);
            }
            UnitaryOp::PrepareBell => {
                debug_assert_eq!(qubits.len(), 2);
                self.apply_h(qubits[0]);
                self.apply_cnot(qubits[0], qubits[1]);
            }
        }
    }

    /// Apply a general Clifford unitary to the frames.
    #[allow(clippy::similar_names)]
    pub fn apply_clifford(&mut self, clifford: &CliffordUnitary, qubits: &[QubitId]) {
        use paulimer::clifford::{Clifford, PreimageViews};

        let gate_qubit_count = qubits.len();
        debug_assert_eq!(gate_qubit_count, clifford.num_qubits());

        let (old_x_rows, old_z_rows) = self.snapshot_rows(qubits);

        for (local_q, &global_q) in qubits.iter().enumerate() {
            let mut new_x_row = AlignedBitVec::zeros(self.shot_count);
            let mut new_z_row = AlignedBitVec::zeros(self.shot_count);

            for local_src in 0..gate_qubit_count {
                let x_image = clifford.x_image_view_up_to_phase(local_src);
                let z_image = clifford.z_image_view_up_to_phase(local_src);

                if x_image.x_bits().index(local_q) {
                    new_x_row.bitxor_assign(&old_x_rows[local_src]);
                }
                if x_image.z_bits().index(local_q) {
                    new_z_row.bitxor_assign(&old_x_rows[local_src]);
                }
                if z_image.x_bits().index(local_q) {
                    new_x_row.bitxor_assign(&old_z_rows[local_src]);
                }
                if z_image.z_bits().index(local_q) {
                    new_z_row.bitxor_assign(&old_z_rows[local_src]);
                }
            }

            self.x_frames.row_mut(global_q).assign(&new_x_row);
            self.z_frames.row_mut(global_q).assign(&new_z_row);
        }
    }

    /// Apply a Pauli exponential exp(iπ/4·P) to the frames.
    ///
    /// For frame Pauli Q:
    /// - If `[P, Q] = 0` (commute): Q → Q (no change)
    /// - If `{P, Q} = 0` (anti-commute): Q → P·Q (mod phase)
    pub fn apply_pauli_exp<P: Pauli>(&mut self, pauli: &P)
    where
        P::Bits: Bitwise,
    {
        let anticommutes = self.anticommutation_mask(pauli);

        for qubit in pauli.x_bits().support() {
            self.x_frames.row_mut(qubit).bitxor_assign(&anticommutes);
        }
        for qubit in pauli.z_bits().support() {
            self.z_frames.row_mut(qubit).bitxor_assign(&anticommutes);
        }
    }

    /// Apply a qubit permutation to the frames.
    #[allow(clippy::similar_names)]
    pub fn apply_permutation(&mut self, permutation: &[usize], qubits: &[QubitId]) {
        let gate_qubit_count = qubits.len();
        debug_assert_eq!(gate_qubit_count, permutation.len());

        let (old_x_rows, old_z_rows) = self.snapshot_rows(qubits);

        for (src_local, &dst_local) in permutation.iter().enumerate() {
            let dst_global = qubits[dst_local];
            self.x_frames.row_mut(dst_global).assign(&old_x_rows[src_local]);
            self.z_frames.row_mut(dst_global).assign(&old_z_rows[src_local]);
        }
    }

    /// Apply a controlled-Pauli gate to the frames.
    ///
    /// For frame Pauli Q:
    /// - Q anti-commuting with control → multiply by target
    /// - Q anti-commuting with target → multiply by control
    pub fn apply_controlled_pauli<P: Pauli>(&mut self, control: &P, target: &P)
    where
        P::Bits: Bitwise,
    {
        let anticommutes_control = self.anticommutation_mask(control);
        let anticommutes_target = self.anticommutation_mask(target);

        // Apply target to frames that anti-commute with control
        for qubit in target.x_bits().support() {
            self.x_frames.row_mut(qubit).bitxor_assign(&anticommutes_control);
        }
        for qubit in target.z_bits().support() {
            self.z_frames.row_mut(qubit).bitxor_assign(&anticommutes_control);
        }

        // Apply control to frames that anti-commute with target
        for qubit in control.x_bits().support() {
            self.x_frames.row_mut(qubit).bitxor_assign(&anticommutes_target);
        }
        for qubit in control.z_bits().support() {
            self.z_frames.row_mut(qubit).bitxor_assign(&anticommutes_target);
        }
    }

    // ========== Measurement ==========

    /// Record a measurement outcome.
    ///
    /// Computes the anti-commutation of the current frame with the observable
    /// and XORs the result into the outcome delta for this measurement.
    pub fn measure<P: Pauli>(&mut self, observable: &P)
    where
        P::Bits: Bitwise,
    {
        let outcome_id = self.next_outcome_id;
        self.next_outcome_id += 1;

        let anticommutes = self.anticommutation_mask(observable);
        self.outcome_deltas.row_mut(outcome_id).bitxor_assign(&anticommutes);
    }

    /// Apply a conditional Pauli gate based on outcome parity.
    ///
    /// In frame simulation, if the condition evaluates differently due to outcome
    /// errors (flipped bits in `outcome_deltas`), the Pauli fires when it shouldn't
    /// (or vice versa). We correct for this by XORing the Pauli into the frame for
    /// shots where the condition was flipped.
    ///
    /// This is essential for MRZ (measure-and-reset): when an X error flips the
    /// measurement outcome, the conditional X fires unexpectedly, and we need to
    /// XOR X into the frame to absorb the original error.
    pub fn apply_conditional_pauli<P: Pauli>(&mut self, pauli: &P, outcomes: &[OutcomeId])
    where
        P::Bits: Bitwise,
    {
        // Compute which shots have their condition flipped due to outcome errors
        self.scratch.clear();
        for &outcome_id in outcomes {
            self.scratch.bitxor_assign(&self.outcome_deltas.row(outcome_id));
        }

        // For those shots, XOR the Pauli into the frame
        for qubit in pauli.x_bits().support() {
            self.x_frames.row_mut(qubit).bitxor_assign(&self.scratch);
        }
        for qubit in pauli.z_bits().support() {
            self.z_frames.row_mut(qubit).bitxor_assign(&self.scratch);
        }
    }

    /// Advance the outcome counter without computing anti-commutation.
    ///
    /// Used for `AllocateRandomBit` instructions.
    pub fn skip_outcome(&mut self) {
        self.next_outcome_id += 1;
    }

    // ========== Noise Injection ==========

    /// Inject faults according to a noise specification.
    ///
    /// Handles:
    /// - Geometric sampling for fault occurrence
    /// - Sampling from the `PauliDistribution`
    /// - Correlated faults via shared RNG seed (`correlation_id`)
    /// - Conditional faults based on noiseless outcomes (`condition`)
    #[allow(clippy::cast_possible_truncation)]
    pub fn inject_noise<R: Rng>(
        &mut self,
        fault: &PauliFault,
        base_seed: u64,
        noiseless_outcomes: &BitMatrix,
        rng: &mut R,
    ) {
        if fault.probability <= 0.0 {
            return;
        }

        match fault.correlation_id {
            Some(correlation_id) => {
                let seed = base_seed.wrapping_add(correlation_id);
                let mut correlated_rng = SmallRng::seed_from_u64(seed);
                self.inject_noise_with_rng(fault, noiseless_outcomes, &mut correlated_rng);
            }
            None => {
                self.inject_noise_with_rng(fault, noiseless_outcomes, rng);
            }
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn inject_noise_with_rng<R: Rng>(&mut self, fault: &PauliFault, noiseless_outcomes: &BitMatrix, rng: &mut R) {
        // Fast path: uncorrelated depolarizing noise without condition
        if fault.condition.is_none() {
            if let crate::noise::PauliDistribution::DepolarizingOnQubits(qubits) = &fault.distribution {
                let mut sampler = GeometricSampler::new(fault.probability);
                self.inject_depolarizing_faults(qubits, &mut sampler, rng);
                return;
            }
        }

        // General path: handles conditions and non-depolarizing distributions
        let mut sampler = GeometricSampler::new(fault.probability);
        let mut shot: usize = 0;

        loop {
            let skip = sampler.next_skip(rng);
            shot = shot.saturating_add(skip);
            if shot >= self.shot_count {
                break;
            }

            if let Some(ref condition) = fault.condition {
                if !condition.is_satisfied(noiseless_outcomes, shot) {
                    shot += 1;
                    continue;
                }
            }

            let pauli = fault.distribution.sample(rng);
            self.apply_pauli_to_shot(shot, &pauli);

            shot += 1;
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn inject_depolarizing_faults<R: Rng>(&mut self, qubits: &[QubitId], sampler: &mut GeometricSampler, rng: &mut R) {
        if qubits.is_empty() {
            return;
        }

        let gate_qubit_count = qubits.len();

        let mut shot: usize = 0;

        loop {
            let skip = sampler.next_skip(rng);
            shot = shot.saturating_add(skip);

            if shot >= self.shot_count {
                break;
            }

            let mut pauli_bits = sample_non_identity_pauli_bits(gate_qubit_count, rng);

            for &qubit in qubits {
                // SAFETY: `shot < self.shot_count` is checked above.
                // `qubit` comes from circuit construction which validates qubit count.
                if pauli_bits & 1 != 0 {
                    unsafe { self.x_frames.negate_unchecked((qubit, shot)) };
                }
                if pauli_bits & 2 != 0 {
                    unsafe { self.z_frames.negate_unchecked((qubit, shot)) };
                }
                pauli_bits >>= 2;
            }

            shot += 1;
        }
    }

    /// Apply a sparse Pauli to the frame at a specific shot index.
    ///
    /// # Safety
    ///
    /// Callers must ensure:
    /// - `shot < self.shot_count`
    /// - All qubits in `pauli.support()` are less than `self.qubit_count()`
    fn apply_pauli_to_shot(&mut self, shot: usize, pauli: &paulimer::pauli::SparsePauli) {
        use paulimer::pauli::Pauli as PauliTrait;

        for qubit in pauli.support() {
            if pauli.x_bits().index(qubit) {
                unsafe { self.x_frames.negate_unchecked((qubit, shot)) };
            }
            if pauli.z_bits().index(qubit) {
                unsafe { self.z_frames.negate_unchecked((qubit, shot)) };
            }
        }
    }

    // ========== Instruction Dispatch ==========

    /// Execute a single instruction, propagating its effect through the frames.
    ///
    /// This method handles all instruction types and is the main entry point
    /// for circuit simulation.
    pub(crate) fn execute<R: Rng>(
        &mut self,
        instruction: &Instruction,
        base_seed: u64,
        noiseless_outcomes: &BitMatrix,
        rng: &mut R,
    ) {
        match instruction {
            Instruction::Unitary { opcode, qubits } => {
                self.apply_unitary_op(*opcode, qubits);
            }
            Instruction::Clifford { clifford, qubits } => {
                self.apply_clifford(clifford, qubits);
            }
            Instruction::Pauli { .. } => {
                // Pauli gates commute with all Paulis (mod phase), no frame change
            }
            Instruction::ConditionalPauli { pauli, outcomes, .. } => {
                self.apply_conditional_pauli(pauli, outcomes);
            }
            Instruction::PauliExp { pauli } => {
                self.apply_pauli_exp(pauli);
            }
            Instruction::Permute { permutation, qubits } => {
                self.apply_permutation(permutation, qubits);
            }
            Instruction::ControlledPauli { control, target } => {
                self.apply_controlled_pauli(control, target);
            }
            Instruction::Measure { observable, .. } => {
                self.measure(observable);
            }
            Instruction::AllocateRandomBit { .. } => {
                self.skip_outcome();
            }
            Instruction::Noise { fault } => {
                self.inject_noise(fault, base_seed, noiseless_outcomes, rng);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistical_testing::{
        assert_rate_within_tolerance, assert_uniform_distribution, TOLERANCE_HIGH_SAMPLES, TOLERANCE_LOW_SAMPLES,
    };
    use paulimer::clifford::{Clifford, CliffordMutable, CliffordUnitary};
    use paulimer::pauli::SparsePauli;
    use rand::rngs::SmallRng;
    use rand::SeedableRng;
    use smallvec::smallvec;
    use std::str::FromStr;

    #[test]
    fn test_cnot_propagation() {
        let mut propagator = FramePropagator::new(2, 0, 64);
        propagator.x_frames.set((0, 0), true);

        propagator.apply_cnot(0, 1);

        assert!(propagator.x_frames.get((0, 0)), "X still on qubit 0");
        assert!(propagator.x_frames.get((1, 0)), "X propagated to qubit 1");
    }

    #[test]
    fn test_hadamard_x_to_z() {
        let mut propagator = FramePropagator::new(1, 0, 64);
        propagator.x_frames.set((0, 0), true);

        propagator.apply_h(0);

        assert!(!propagator.x_frames.get((0, 0)), "X cleared");
        assert!(propagator.z_frames.get((0, 0)), "Z set");
    }

    #[test]
    fn test_hadamard_z_to_x() {
        let mut propagator = FramePropagator::new(1, 0, 64);
        propagator.z_frames.set((0, 0), true);

        propagator.apply_h(0);

        assert!(propagator.x_frames.get((0, 0)), "X set after H(Z)");
        assert!(!propagator.z_frames.get((0, 0)), "Z cleared after H(Z)");
    }

    #[test]
    fn test_clifford_hadamard() {
        let mut propagator = FramePropagator::new(1, 0, 64);
        propagator.z_frames.set((0, 0), true);

        let mut h = CliffordUnitary::identity(1);
        h.left_mul_hadamard(0);
        propagator.apply_clifford(&h, &[0]);

        assert!(propagator.x_frames.get((0, 0)), "X set after Clifford H(Z)");
        assert!(!propagator.z_frames.get((0, 0)), "Z cleared after Clifford H(Z)");
    }

    #[test]
    fn test_measurement_anticommutation() {
        let mut propagator = FramePropagator::new(1, 1, 64);
        propagator.x_frames.set((0, 0), true);

        let z_obs = SparsePauli::from_str("Z").unwrap();
        propagator.measure(&z_obs);

        assert!(propagator.outcome_deltas.get((0, 0)), "Outcome flipped by X error");
    }

    #[test]
    fn test_anticommutation_mask() {
        let mut propagator = FramePropagator::new(2, 0, 4);
        // Shot 0: X on qubit 0
        propagator.x_frames.set((0, 0), true);
        // Shot 1: Z on qubit 1
        propagator.z_frames.set((1, 1), true);
        // Shot 2: X on qubit 0 AND Z on qubit 1
        propagator.x_frames.set((0, 2), true);
        propagator.z_frames.set((1, 2), true);
        // Shot 3: identity

        let z0 = SparsePauli::from_str("ZI").unwrap();
        let mask = propagator.anticommutation_mask(&z0);

        // Z0 anti-commutes with X0 (shots 0 and 2)
        assert!(mask.index(0), "Shot 0 should anti-commute");
        assert!(!mask.index(1), "Shot 1 should commute");
        assert!(mask.index(2), "Shot 2 should anti-commute");
        assert!(!mask.index(3), "Shot 3 should commute");
    }

    #[test]
    fn test_fault_injection_rate_accuracy() {
        use crate::noise::{PauliDistribution, PauliFault};

        let shot_count = 100_000;
        let probability = 0.15;

        let mut propagator = FramePropagator::new(1, 0, shot_count);
        let empty_outcomes = BitMatrix::zeros(shot_count, 0);
        let mut rng = SmallRng::seed_from_u64(42);

        let fault = PauliFault {
            probability,
            distribution: PauliDistribution::single(SparsePauli::from_str("X").unwrap()),
            correlation_id: None,
            condition: None,
        };
        propagator.inject_noise(&fault, 0, &empty_outcomes, &mut rng);

        let fault_count: usize = (0..shot_count).filter(|&s| propagator.x_frames.get((0, s))).count();

        assert_rate_within_tolerance(
            fault_count,
            shot_count,
            probability,
            TOLERANCE_HIGH_SAMPLES,
            "Single X fault injection",
        );
    }

    #[test]
    fn test_depolarizing_fast_path_produces_correct_rate() {
        use crate::noise::PauliFault;

        let shot_count = 100_000;
        let probability = 0.1;

        let mut propagator = FramePropagator::new(2, 0, shot_count);
        let empty_outcomes = BitMatrix::zeros(shot_count, 0);
        let mut rng = SmallRng::seed_from_u64(42);

        propagator.inject_noise(
            &PauliFault::depolarizing(&[0, 1], probability),
            0,
            &empty_outcomes,
            &mut rng,
        );

        let mut fault_count = 0usize;
        for shot in 0..shot_count {
            let x_q0 = propagator.x_frames.get((0, shot));
            let z_q0 = propagator.z_frames.get((0, shot));
            let x_q1 = propagator.x_frames.get((1, shot));
            let z_q1 = propagator.z_frames.get((1, shot));
            if x_q0 || z_q0 || x_q1 || z_q1 {
                fault_count += 1;
            }
        }

        assert_rate_within_tolerance(
            fault_count,
            shot_count,
            probability,
            TOLERANCE_HIGH_SAMPLES,
            "2-qubit depolarizing fast path",
        );
    }

    #[test]
    fn test_depolarizing_fast_path_produces_uniform_paulis() {
        use crate::noise::PauliFault;

        let shot_count = 150_000;
        let probability = 1.0;

        let mut propagator = FramePropagator::new(2, 0, shot_count);
        let empty_outcomes = BitMatrix::zeros(shot_count, 0);
        let mut rng = SmallRng::seed_from_u64(42);

        propagator.inject_noise(
            &PauliFault::depolarizing(&[0, 1], probability),
            0,
            &empty_outcomes,
            &mut rng,
        );

        let mut counts = [0u32; 15];
        for shot in 0..shot_count {
            let x0 = u32::from(propagator.x_frames.get((0, shot)));
            let z0 = u32::from(propagator.z_frames.get((0, shot)));
            let x1 = u32::from(propagator.x_frames.get((1, shot)));
            let z1 = u32::from(propagator.z_frames.get((1, shot)));
            let bits = x0 | (z0 << 1) | (x1 << 2) | (z1 << 3);
            assert!(bits > 0, "Sampled identity at shot {shot}");
            counts[(bits - 1) as usize] += 1;
        }

        assert_uniform_distribution(
            &counts,
            shot_count,
            TOLERANCE_LOW_SAMPLES,
            "2-qubit depolarizing fast path uniformity",
        );
    }

    #[test]
    fn test_correlated_faults_hit_same_shots() {
        use crate::noise::{PauliDistribution, PauliFault};

        let shot_count = 10000;
        let base_seed = 12345_u64;
        let correlation_id = 99;

        let mut propagator_a = FramePropagator::new(1, 0, shot_count);
        let mut propagator_b = FramePropagator::new(1, 0, shot_count);
        let empty_outcomes = BitMatrix::zeros(shot_count, 0);

        let fault_x = PauliFault {
            probability: 0.1,
            distribution: PauliDistribution::single(SparsePauli::from_str("X").unwrap()),
            correlation_id: Some(correlation_id),
            condition: None,
        };

        let fault_z = PauliFault {
            probability: 0.1,
            distribution: PauliDistribution::single(SparsePauli::from_str("Z").unwrap()),
            correlation_id: Some(correlation_id),
            condition: None,
        };

        let mut rng_a = SmallRng::seed_from_u64(1);
        let mut rng_b = SmallRng::seed_from_u64(2);

        propagator_a.inject_noise(&fault_x, base_seed, &empty_outcomes, &mut rng_a);
        propagator_b.inject_noise(&fault_z, base_seed, &empty_outcomes, &mut rng_b);

        let shots_a: Vec<usize> = (0..shot_count).filter(|&s| propagator_a.x_frames.get((0, s))).collect();
        let shots_b: Vec<usize> = (0..shot_count).filter(|&s| propagator_b.z_frames.get((0, s))).collect();

        assert_eq!(shots_a, shots_b, "Correlated faults should hit the same shots");
        assert!(!shots_a.is_empty(), "Should have some faults");
    }

    #[test]
    fn test_conditional_fault_parity_true() {
        use crate::noise::{OutcomeCondition, PauliDistribution, PauliFault};

        let shot_count = 100;
        let mut propagator = FramePropagator::new(1, 0, shot_count);

        let mut outcomes = BitMatrix::zeros(shot_count, 1);
        for shot in (0..shot_count).step_by(2) {
            outcomes.set((shot, 0), true);
        }

        let fault = PauliFault {
            probability: 1.0,
            distribution: PauliDistribution::single(SparsePauli::from_str("X").unwrap()),
            correlation_id: None,
            condition: Some(OutcomeCondition {
                outcomes: smallvec![0],
                parity: true,
            }),
        };

        let mut rng = SmallRng::seed_from_u64(42);
        propagator.inject_noise(&fault, 0, &outcomes, &mut rng);

        for shot in 0..shot_count {
            let has_fault = propagator.x_frames.get((0, shot));
            let expected = shot % 2 == 0;
            assert_eq!(has_fault, expected, "Shot {shot} should have fault={expected}");
        }
    }

    #[test]
    fn test_conditional_fault_parity_false_suppresses_when_true() {
        use crate::noise::{OutcomeCondition, PauliDistribution, PauliFault};

        let shot_count = 100;
        let mut propagator = FramePropagator::new(1, 0, shot_count);

        let mut outcomes = BitMatrix::zeros(shot_count, 1);
        for shot in (0..shot_count).step_by(2) {
            outcomes.set((shot, 0), true);
        }

        let fault = PauliFault {
            probability: 1.0,
            distribution: PauliDistribution::single(SparsePauli::from_str("X").unwrap()),
            correlation_id: None,
            condition: Some(OutcomeCondition {
                outcomes: smallvec![0],
                parity: false,
            }),
        };

        let mut rng = SmallRng::seed_from_u64(42);
        propagator.inject_noise(&fault, 0, &outcomes, &mut rng);

        for shot in 0..shot_count {
            let has_fault = propagator.x_frames.get((0, shot));
            let expected = shot % 2 == 1;
            assert_eq!(
                has_fault, expected,
                "Shot {shot}: parity=false should fire when outcome is 0"
            );
        }
    }

    #[test]
    fn test_execute_dispatch() {
        let mut propagator = FramePropagator::new(2, 1, 64);
        propagator.x_frames.set((0, 0), true);

        let empty_outcomes = BitMatrix::zeros(64, 1);
        let mut rng = SmallRng::seed_from_u64(42);

        // Execute H gate via dispatch
        propagator.execute(
            &Instruction::Unitary {
                opcode: UnitaryOp::Hadamard,
                qubits: vec![0],
            },
            0,
            &empty_outcomes,
            &mut rng,
        );

        assert!(!propagator.x_frames.get((0, 0)), "X cleared after H");
        assert!(propagator.z_frames.get((0, 0)), "Z set after H");

        // Execute measurement via dispatch
        propagator.execute(
            &Instruction::Measure {
                observable: SparsePauli::from_str("Z").unwrap(),
                outcome_id: 0,
            },
            0,
            &empty_outcomes,
            &mut rng,
        );

        // Z error commutes with Z measurement, so no flip
        assert!(
            !propagator.outcome_deltas.get((0, 0)),
            "Z error should not flip Z measurement"
        );
    }

    #[test]
    fn test_conditional_pauli_absorbs_error_on_mrz() {
        // Simulates MRZ: Measure Z + Conditional X (reset to |0⟩)
        // An X error before measurement should be absorbed by the conditional X.

        let mut propagator = FramePropagator::new(1, 1, 64);

        // Shot 0: X error on qubit 0
        propagator.x_frames.set((0, 0), true);
        // Shot 1: no error (control case)

        // Measure Z: X error anticommutes, flips outcome for shot 0
        propagator.measure(&SparsePauli::from_str("Z").unwrap());
        assert!(
            propagator.outcome_deltas.get((0, 0)),
            "X error should flip Z measurement"
        );
        assert!(!propagator.outcome_deltas.get((0, 1)), "No error, no flip");

        // Conditional X (reset behavior): fires if outcome=1
        // For shot 0: outcome was flipped to 1, so X fires unexpectedly
        // This should XOR X into the frame, clearing the original X error
        propagator.apply_conditional_pauli(&SparsePauli::from_str("X").unwrap(), &[0]);

        // After conditional X, the error should be absorbed
        assert!(
            !propagator.x_frames.get((0, 0)),
            "X error should be absorbed by conditional X"
        );
        assert!(
            !propagator.x_frames.get((0, 1)),
            "Shot without error should remain clean"
        );
    }

    #[test]
    fn test_conditional_pauli_z_error_survives_mrz() {
        // Z errors commute with Z measurement and are unaffected by conditional X
        // They should survive through MRZ (which is physically correct - Z errors
        // don't affect the computational basis state after reset)

        let mut propagator = FramePropagator::new(1, 1, 64);

        // Shot 0: Z error on qubit 0
        propagator.z_frames.set((0, 0), true);

        // Measure Z: Z error commutes, no flip
        propagator.measure(&SparsePauli::from_str("Z").unwrap());
        assert!(
            !propagator.outcome_deltas.get((0, 0)),
            "Z error should not flip Z measurement"
        );

        // Conditional X: doesn't fire (outcome=0)
        propagator.apply_conditional_pauli(&SparsePauli::from_str("X").unwrap(), &[0]);

        // Z error still present (but this is actually fine - |0⟩ is Z eigenstate,
        // so a Z error on |0⟩ is just a global phase)
        assert!(
            propagator.z_frames.get((0, 0)),
            "Z error survives MRZ (physically correct)"
        );
    }
}
