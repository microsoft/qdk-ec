//! High-level noisy simulation using frame-based error propagation.
//!
//! This module provides [`FaultySimulation`], which combines noiseless outcome
//! sampling with [`FramePropagator`] for O(n\_gates × n\_qubits) noisy simulation.

use binar::BitMatrix;
use paulimer::clifford::CliffordUnitary;
use paulimer::pauli::SparsePauli;
use paulimer::UnitaryOp;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::circuit::{Circuit, Instruction};
use crate::frame_propagator::FramePropagator;
use crate::noise::PauliFault;
use crate::outcome_complete_simulation::OutcomeCompleteSimulation;
use crate::Simulation;

/// Noisy stabilizer simulation.
///
/// This simulator combines noiseless outcome sampling ([`OutcomeCompleteSimulation`])
/// with efficient frame-based error propagation ([`FramePropagator`]). Noise is
/// represented as Pauli errors that propagate through Clifford gates, enabling
/// O(n_gates × n_qubits) complexity for multi-shot noisy simulation.
///

/// Rather than tracking full noisy quantum states, errors are represented as Pauli
/// frames that commute through gates. This allows efficient simulation of realistic
/// noise models while maintaining the O(n²) scaling of stabilizer simulation.
///
/// # Use Cases
///
/// - **Logical error rates**: Estimate logical error rates via Monte Carlo sampling  
/// - **Noise characterization**: Study error propagation under different noise models
/// - **Decoder validation**: Test decoder performance with realistic noise
///
/// # Performance
///
/// - **Complexity**: O(n_gates × n_qubits²) worst-case, same as noiseless simulators
/// - **Frame propagation**: Tracks Pauli errors through gates efficiently
/// - **Sampling cost**: O(shots × (n_gates × n_qubits + n_measurements × n_random)) total for Monte Carlo
/// - **Based on**: Uses [`OutcomeCompleteSimulation`] for noiseless part, then applies noise via frames
/// - **Space**: O(n_qubits² + n_measurements² + shots × n_measurements) for simulation and samples
///
/// # Noise Models
///
/// Supports various noise distributions via [`PauliFault`]:
/// - Single Pauli errors: `PauliDistribution::Single`
/// - Depolarizing noise: `PauliDistribution::depolarizing`
/// - Weighted distributions: `PauliDistribution::Weighted`
/// - Conditional errors: Based on measurement outcomes
/// - Correlated errors: Via correlation IDs
///
/// # Examples
///
/// ```ignore
/// use pauliverse::{FaultySimulation, PauliFault, PauliDistribution, Simulation};
/// use paulimer::UnitaryOp;
/// use paulimer::pauli::SparsePauli;
///
/// // Define noise model
/// let mut sim = FaultySimulation::default();
/// sim.reserve_qubits(3);
///
/// let noise = PauliFault {
///     probability: 0.01,
///     distribution: PauliDistribution::depolarizing(&[0]),
///     correlation_id: None,
///     condition: None,
/// };
///
/// // Build circuit with noise
/// sim.unitary_op(UnitaryOp::Hadamard, &[0]);
/// sim.apply_fault(noise.clone());
/// sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
/// sim.apply_fault(noise);
///
/// // Measure syndrome
/// let stabilizer: SparsePauli = "ZZ".parse().unwrap();
/// sim.measure(&stabilizer);
///
/// // Sample outcomes
/// let outcomes = sim.sample(1000);
/// ```
///
/// # Alternatives
///
/// - Use [`crate::OutcomeSpecificSimulation`] for noiseless Monte Carlo sampling
/// - Use [`crate::OutcomeCompleteSimulation`] for noiseless exhaustive enumeration
/// - Use [`crate::OutcomeFreeSimulation`] when outcomes don't matter
pub struct FaultySimulation {
    circuit: Circuit,
    noiseless: OutcomeCompleteSimulation,
}

impl FaultySimulation {
    /// Create a new empty noisy simulation.
    ///
    /// Use [`reserve_qubits`](Self::reserve_qubits) to allocate qubits before use.
    #[must_use]
    pub fn new() -> Self {
        FaultySimulation {
            circuit: Circuit::default(),
            noiseless: OutcomeCompleteSimulation::new(0),
        }
    }

    /// Create a new simulation with pre-allocated capacity.
    ///
    /// Pre-allocating capacity for expected qubit count, outcome count, and
    /// instruction count can improve performance by avoiding reallocations.
    ///
    /// # Arguments
    /// * `qubit_count` - Expected number of qubits
    /// * `outcome_count` - Expected number of measurements
    /// * `instruction_count` - Expected number of gates and noise instructions
    #[must_use]
    pub fn with_capacity(qubit_count: usize, outcome_count: usize, instruction_count: usize) -> Self {
        FaultySimulation {
            circuit: Circuit::with_capacity(instruction_count),
            noiseless: OutcomeCompleteSimulation::with_capacity(qubit_count, outcome_count, outcome_count),
        }
    }

    /// Add a fault (noise) instruction to the circuit.
    ///
    /// The fault will be applied during noisy simulation via frame propagation.
    pub fn apply_fault(&mut self, fault: PauliFault) {
        self.circuit.push(Instruction::noise(fault));
    }

    /// Returns the number of fault locations in the circuit.
    #[must_use]
    pub fn fault_count(&self) -> usize {
        self.circuit.fault_count()
    }

    // ========== Sampling ==========

    /// Sample noisy measurement outcomes using frame simulation.
    ///
    /// Returns a matrix where each row is one shot's outcome vector.
    /// Noise is applied via frame propagation for efficiency.
    pub fn sample(&self, shots: usize) -> BitMatrix {
        let mut rng = SmallRng::from_entropy();
        self.sample_with_rng(shots, &mut rng)
    }

    /// Sample noisy outcomes with a provided RNG.
    ///
    /// Useful for reproducible simulations with seeded random number generation.
    pub fn sample_with_rng<R: Rng>(&self, shots: usize, rng: &mut R) -> BitMatrix {
        let base_seed: u64 = rng.gen();

        let mut outcomes = self.noiseless.sample_with_rng(shots, rng);
        let deltas = self.simulate_circuit(shots, &outcomes, base_seed, rng);
        let deltas_transposed: BitMatrix = deltas.transposed().into();
        outcomes ^= &deltas_transposed;

        outcomes
    }

    // ========== Internal methods ==========

    fn simulate_circuit<R: Rng>(
        &self,
        shot_count: usize,
        noiseless_outcomes: &BitMatrix,
        base_seed: u64,
        rng: &mut R,
    ) -> binar::matrix::AlignedBitMatrix {
        let outcome_count = self.noiseless.outcome_count();
        let mut propagator = FramePropagator::new(self.noiseless.qubit_count(), outcome_count, shot_count);

        for instruction in self.circuit.iter() {
            propagator.execute(instruction, base_seed, noiseless_outcomes, rng);
        }

        propagator.into_outcome_deltas()
    }
}

impl Default for FaultySimulation {
    fn default() -> Self {
        Self::new()
    }
}

impl Simulation for FaultySimulation {
    fn allocate_random_bit(&mut self) -> usize {
        let outcome_id = self.noiseless.allocate_random_bit();
        self.circuit.push(Instruction::AllocateRandomBit { outcome_id });
        outcome_id
    }

    fn clifford(&mut self, clifford: &CliffordUnitary, support: &[usize]) {
        self.noiseless.clifford(clifford, support);
        self.circuit.push(Instruction::Clifford {
            clifford: clifford.clone(),
            qubits: support.to_vec(),
        });
    }

    fn conditional_pauli(&mut self, observable: &SparsePauli, outcomes: &[usize], parity: bool) {
        self.noiseless.conditional_pauli(observable, outcomes, parity);
        self.circuit.push(Instruction::ConditionalPauli {
            pauli: observable.clone(),
            outcomes: outcomes.to_vec(),
            parity,
        });
    }

    fn controlled_pauli(&mut self, observable1: &SparsePauli, observable2: &SparsePauli) {
        self.noiseless.controlled_pauli(observable1, observable2);
        self.circuit.push(Instruction::ControlledPauli {
            control: observable1.clone(),
            target: observable2.clone(),
        });
    }

    fn pauli(&mut self, observable: &SparsePauli) {
        self.noiseless.pauli(observable);
        self.circuit.push(Instruction::Pauli {
            pauli: observable.clone(),
        });
    }

    fn pauli_exp(&mut self, sparse_pauli: &SparsePauli) {
        self.noiseless.pauli_exp(sparse_pauli);
        self.circuit.push(Instruction::PauliExp {
            pauli: sparse_pauli.clone(),
        });
    }

    fn permute(&mut self, permutation: &[usize], support: &[usize]) {
        self.noiseless.permute(permutation, support);
        self.circuit.push(Instruction::Permute {
            permutation: permutation.to_vec(),
            qubits: support.to_vec(),
        });
    }

    fn unitary_op(&mut self, operation: UnitaryOp, support: &[usize]) {
        self.noiseless.unitary_op(operation, support);
        self.circuit.push(Instruction::Unitary {
            opcode: operation,
            qubits: support.to_vec(),
        });
    }

    fn is_stabilizer(&self, observable: &SparsePauli) -> bool {
        self.noiseless.is_stabilizer(observable)
    }

    fn is_stabilizer_up_to_sign(&self, observable: &SparsePauli) -> bool {
        self.noiseless.is_stabilizer_up_to_sign(observable)
    }

    fn is_stabilizer_with_conditional_sign(&self, observable: &SparsePauli, outcomes: &[usize]) -> bool {
        self.noiseless.is_stabilizer_with_conditional_sign(observable, outcomes)
    }

    fn measure(&mut self, observable: &SparsePauli) -> usize {
        let outcome_id = self.noiseless.measure(observable);
        self.circuit.push(Instruction::Measure {
            observable: observable.clone(),
            outcome_id,
        });
        outcome_id
    }

    fn measure_with_hint(&mut self, observable: &SparsePauli, anti_commuting_stabilizer: &SparsePauli) -> usize {
        let outcome_id = self.noiseless.measure_with_hint(observable, anti_commuting_stabilizer);
        self.circuit.push(Instruction::Measure {
            observable: observable.clone(),
            outcome_id,
        });
        outcome_id
    }

    fn qubit_count(&self) -> usize {
        self.noiseless.qubit_count()
    }

    fn outcome_count(&self) -> usize {
        self.noiseless.outcome_count()
    }

    fn random_outcome_count(&self) -> usize {
        self.noiseless.random_outcome_count()
    }

    fn random_outcome_indicator(&self) -> &[bool] {
        self.noiseless.random_outcome_indicator()
    }

    fn with_capacity(qubit_count: usize, outcome_count: usize, random_outcome_count: usize) -> Self {
        FaultySimulation {
            circuit: Circuit::with_capacity(outcome_count),
            noiseless: OutcomeCompleteSimulation::with_capacity(qubit_count, outcome_count, random_outcome_count),
        }
    }

    fn qubit_capacity(&self) -> usize {
        self.noiseless.qubit_capacity()
    }

    fn reserve_qubits(&mut self, new_qubit_capacity: usize) {
        self.noiseless.reserve_qubits(new_qubit_capacity);
    }

    fn outcome_capacity(&self) -> usize {
        self.noiseless.outcome_capacity()
    }

    fn random_outcome_capacity(&self) -> usize {
        self.noiseless.random_outcome_capacity()
    }

    fn reserve_outcomes(&mut self, new_outcome_capacity: usize, new_random_outcome_capacity: usize) {
        self.noiseless
            .reserve_outcomes(new_outcome_capacity, new_random_outcome_capacity);
    }
}
