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

/// Frame-based noisy simulation combining noiseless outcome sampling with
/// O(n\_gates × n\_qubits) frame error propagation.
///
/// Use the `Simulation` trait methods to construct a circuit, then call `sample()` to
/// generate noisy measurement outcomes.
///
/// # Example
///
/// ```ignore
/// use pauliverse::{FaultySimulation, Simulation};
/// use paulimer::UnitaryOp;
/// use paulimer::pauli::SparsePauli;
///
/// let mut sim = FaultySimulation::new();
/// sim.unitary_op(UnitaryOp::Hadamard, &[0]);
/// sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
/// sim.measure(&SparsePauli::from_str("ZZ").unwrap());
/// let outcomes = sim.sample(1000);
/// ```
pub struct FaultySimulation {
    circuit: Circuit,
    noiseless: OutcomeCompleteSimulation,
}

impl FaultySimulation {
    /// Create a new empty simulation.
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
    #[must_use]
    pub fn with_capacity(qubit_count: usize, outcome_count: usize, instruction_count: usize) -> Self {
        FaultySimulation {
            circuit: Circuit::with_capacity(instruction_count),
            noiseless: OutcomeCompleteSimulation::with_capacity(qubit_count, outcome_count, outcome_count),
        }
    }

    /// Add a fault (noise) instruction.
    pub fn apply_fault(&mut self, fault: PauliFault) {
        self.circuit.push(Instruction::noise(fault));
    }

    /// Returns the number of fault locations in the simulation.
    #[must_use]
    pub fn fault_count(&self) -> usize {
        self.circuit.fault_count()
    }

    // ========== Sampling ==========

    /// Sample noisy outcomes using frame simulation.
    pub fn sample(&self, shots: usize) -> BitMatrix {
        let mut rng = SmallRng::from_entropy();
        self.sample_with_rng(shots, &mut rng)
    }

    /// Sample noisy outcomes with a provided RNG.
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
