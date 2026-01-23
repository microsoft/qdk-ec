//! Fast stabilizer simulation for quantum error correction research.
//!
//! `pauliverse` provides multiple stabilizer simulation implementations, each optimized
//! for different use cases in quantum computing and quantum error correction.
//!
//! These simulation algorithms are based on the framework described in
//! [arXiv:2309.08676](https://arxiv.org/abs/2309.08676).
//!
//! # Overview
//!
//! This crate offers four simulation modes:
//!
//! - **[`OutcomeSpecificSimulation`]**: Traditional simulation with random (or caller supplied) measurement outcomes.
//!   Best for Monte Carlo sampling and estimating error rates.
//!
//! - **[`OutcomeCompleteSimulation`]**: Tracks all possible measurement outcomes simultaneously.
//!   Achieves asymptotic speedup when enumerating outcomes.
//!
//! - **[`OutcomeFreeSimulation`]**: Simulation without tracking specific outcomes.
//!   Minimal overhead when you only care about stabilizer state evolution up to global phase.
//!
//! - **[`FaultySimulation`]**: Noisy simulation, combining outcome-complete simulation with
//!  Pauli frame propagation.
//!
//! All simulators implement the [`Simulation`] trait.
//!
//! # Quick Start
//!
//! ```
//! use pauliverse::{OutcomeSpecificSimulation, Simulation};
//! use paulimer::{UnitaryOp, SparsePauli};
//!
//! // Create a simple Bell state simulation
//! let mut sim = OutcomeSpecificSimulation::new_with_random_outcomes(2);
//!
//! // Apply gates
//! sim.unitary_op(UnitaryOp::Hadamard, &[0]);
//! sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
//!
//! // Measure
//! let observable: SparsePauli = "ZI".parse().unwrap();
//! let outcome = sim.measure(&observable);
//! ```
//!
//! # Choosing a Simulator
//!
//! All simulators have worst-case complexity O(n_gates × n_qubits²), though actual cost
//! depends on circuit composition.
//!
//! | Simulator | Best For | Key Advantage |
//! |-----------|----------|---------------|
//! | [`OutcomeSpecificSimulation`] | Monte Carlo sampling with few shots | Single concrete execution path |
//! | [`OutcomeCompleteSimulation`] | Whole-circuit analysis, enumerating outcomes | Avoids re-simulating for each outcome sample |
//! | [`OutcomeFreeSimulation`] | Stabilizer queries without outcomes | Minimal overhead, no outcome tracking |
//! | [`FaultySimulation`] | Noisy simulation with error correction | Efficient frame-based noise propagation |
//!
//! **When to use outcome-complete over outcome-specific**: If you need many samples and the
//! circuit has n_random random measurements, outcome-complete saves a factor of n_random by computing
//! the outcome distribution once, then sampling efficiently without re-running the circuit.
//! Outcome-complete simulation is also ideal for circuit characterizations, like verification or outcome codes.
//!
//! # Performance Features
//!
//! - **SIMD acceleration**: Via [`binar`] for bit matrix operations
//! - **Cache alignment**: For efficient memory access
//! - **Batch processing**: Simulate millions of shots with minimal overhead
//! - **Amortized enumeration**: [`OutcomeCompleteSimulation`] pays circuit cost once,
//!   then permits enumeration of outcomes without re-simulating.
//!
//! # Architecture
//!
//! Built on [`paulimer`] for Pauli and Clifford operations, which in turn uses
//! [`binar`] for efficient bit matrix operations.
//!
//! The simulation algorithms implement the general form representation described in
//! the paper, allowing efficient tracking of measurement outcomes and their correlations
//! with stabilizer states. This enables verification of stabilizer circuits by
//! exhaustively characterizing their behavior across all possible measurement outcomes.
//!
//! # Features
//!
//! Optional cargo features:
//! - `python`: Python bindings (included with paulimer Python package)
//!
//! # See Also
//!
//! - [`paulimer`]: Pauli and Clifford algebra primitives
//! - [`binar`]: Efficient bit vector and matrix operations

// pub mod bipartite_normal_form;
pub(crate) mod circuit;
pub mod faulty_simulation;
pub mod frame_propagator;
pub mod noise;
#[macro_use]
pub mod outcome_complete_simulation;

pub mod outcome_free_simulation;
pub mod outcome_specific_simulation;
pub mod sampling;
#[cfg(test)]
pub(crate) mod statistical_testing;

pub use circuit::{OutcomeId, QubitId};
pub use faulty_simulation::FaultySimulation;
pub use noise::{OutcomeCondition, PauliDistribution, PauliFault};
pub use outcome_complete_simulation::OutcomeCompleteSimulation;
pub use outcome_free_simulation::OutcomeFreeSimulation;
pub use outcome_specific_simulation::OutcomeSpecificSimulation;

type Pauli = paulimer::pauli::SparsePauli;
type Unitary = paulimer::clifford::CliffordUnitary;
type Operation = paulimer::UnitaryOp;

/// Common interface for stabilizer simulation.
///
/// This trait defines the core operations supported by all stabilizer simulators:
/// unitary gates, Pauli operators, measurements, and stabilizer queries.
///
/// All simulators maintain a stabilizer state and track measurement outcomes,
/// though different implementations handle outcomes differently (see individual
/// simulator documentation).
pub trait Simulation: Default {
    // ========== Measurement Outcome Allocation ==========
    
    /// Allocate a new measurement outcome with a random value.
    ///
    /// Returns the outcome ID for the newly allocated outcome.
    fn allocate_random_bit(&mut self) -> OutcomeId;

    // ========== Unitary Operations ==========
    
    /// Apply a Clifford unitary to specified qubits.
    ///
    /// # Arguments
    /// * `clifford` - The Clifford unitary to apply
    /// * `support` - The qubit indices to apply the unitary to
    fn clifford(&mut self, clifford: &Unitary, support: &[QubitId]);
    
    /// Apply a Pauli operator conditioned on measurement outcome parity.
    ///
    /// Applies `observable` if the XOR of the specified outcomes equals `parity`.
    ///
    /// # Arguments
    /// * `observable` - The Pauli operator to conditionally apply
    /// * `outcomes` - The outcome IDs to check
    /// * `parity` - The target parity (true = odd, false = even)
    fn conditional_pauli(&mut self, observable: &Pauli, outcomes: &[OutcomeId], parity: bool);
    
    /// Apply a controlled-Pauli operation.
    ///
    /// Applies `observable2` conditioned on `observable1` being in the +1 eigenstate.
    fn controlled_pauli(&mut self, observable1: &Pauli, observable2: &Pauli);
    
    /// Apply a Pauli operator to the state.
    fn pauli(&mut self, observable: &Pauli);
    
    /// Apply a Pauli exponential (Pauli rotation by π).
    fn pauli_exp(&mut self, sparse_pauli: &Pauli);
    
    /// Permute qubit indices according to the given permutation.
    ///
    /// # Arguments
    /// * `permutation` - Maps old qubit index to new qubit index
    /// * `support` - The qubits involved in the permutation
    fn permute(&mut self, permutation: &[usize], support: &[QubitId]);
    
    /// Apply a standard Clifford gate operation.
    ///
    /// # Arguments
    /// * `operation` - The gate type (Hadamard, ControlledX, etc.)
    /// * `support` - The qubit indices to apply the gate to
    fn unitary_op(&mut self, operation: Operation, support: &[QubitId]);

    // ========== Stabilizer Queries ==========
    
    /// Check if a Pauli operator is a stabilizer of the current state.
    ///
    /// Returns true if the operator commutes with all stabilizers and has eigenvalue +1.
    fn is_stabilizer(&self, observable: &Pauli) -> bool;
    
    /// Check if a Pauli operator is a stabilizer up to a global phase.
    ///
    /// Returns true if the operator commutes with all stabilizers (eigenvalue may be ±1).
    fn is_stabilizer_up_to_sign(&self, observable: &Pauli) -> bool;
    
    /// Check if a Pauli operator is a stabilizer with outcome-dependent sign.
    ///
    /// Returns true if the operator's eigenvalue depends on specified measurement outcomes.
    fn is_stabilizer_with_conditional_sign(&self, observable: &Pauli, outcomes: &[OutcomeId]) -> bool;

    // ========== Measurement ==========
    
    /// Measure a Pauli observable and return the outcome ID.
    ///
    /// For deterministic measurements, returns an existing outcome ID.
    /// For random measurements, allocates a new outcome ID.
    fn measure(&mut self, observable: &Pauli) -> OutcomeId;
    
    /// Measure a Pauli observable with a hint for optimization.
    ///
    /// The hint should be a stabilizer that anticommutes with the observable.
    /// This can improve measurement performance.
    ///
    /// # Arguments
    /// * `observable` - The Pauli operator to measure
    /// * `anti_commuting_stabilizer` - A stabilizer that anticommutes with observable
    fn measure_with_hint(&mut self, observable: &Pauli, anti_commuting_stabilizer: &Pauli) -> OutcomeId;

    // ========== State Queries ==========
    
    /// Get the number of qubits in the simulation.
    fn qubit_count(&self) -> usize;
    
    /// Get the total number of measurement outcomes (deterministic + random).
    fn outcome_count(&self) -> usize;
    
    /// Get the number of random (non-deterministic) measurement outcomes.
    fn random_outcome_count(&self) -> usize;
    
    /// Get indicators for which outcomes are random.
    ///
    /// Returns a slice where `[i]` is true if outcome `i` was random.
    fn random_outcome_indicator(&self) -> &[bool];

    // ========== Construction ==========
    
    /// Create a new simulator with the specified number of qubits.
    #[must_use]
    fn new(qubit_count: usize) -> Self
    where
        Self: Sized,
    {
        Self::with_capacity(qubit_count, 0, 0)
    }
    
    /// Create a new simulator with pre-allocated capacity.
    ///
    /// Pre-allocating can improve performance by avoiding reallocations.
    ///
    /// # Arguments
    /// * `qubit_count` - Number of qubits to simulate
    /// * `outcome_count` - Expected number of measurement outcomes
    /// * `random_outcome_count` - Expected number of random outcomes
    fn with_capacity(qubit_count: usize, outcome_count: usize, random_outcome_count: usize) -> Self
    where
        Self: Sized;

    // ========== Capacity Management ==========
    
    /// Get the current qubit capacity (may be larger than qubit count).
    fn qubit_capacity(&self) -> usize;
    
    /// Reserve capacity for additional qubits.
    ///
    /// Increases capacity to at least `new_qubit_capacity`. Does nothing if
    /// current capacity is already sufficient.
    fn reserve_qubits(&mut self, new_qubit_capacity: usize);
    
    /// Get the current outcome capacity.
    fn outcome_capacity(&self) -> usize;
    
    /// Get the current random outcome capacity.
    fn random_outcome_capacity(&self) -> usize;
    
    /// Reserve capacity for additional outcomes.
    ///
    /// # Arguments
    /// * `new_outcome_capacity` - Total outcome capacity
    /// * `new_random_outcome_capacity` - Random outcome capacity
    fn reserve_outcomes(&mut self, new_outcome_capacity: usize, new_random_outcome_capacity: usize);
}
