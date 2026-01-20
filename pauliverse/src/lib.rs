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

type Pauli = paulimer::pauli::SparsePauli;
type Unitary = paulimer::clifford::CliffordUnitary;
type Operation = paulimer::UnitaryOp;

pub trait Simulation: Default {
    fn allocate_random_bit(&mut self) -> OutcomeId;

    fn clifford(&mut self, clifford: &Unitary, support: &[QubitId]);
    fn conditional_pauli(&mut self, observable: &Pauli, outcomes: &[OutcomeId], parity: bool);
    fn controlled_pauli(&mut self, observable1: &Pauli, observable2: &Pauli);
    fn pauli(&mut self, observable: &Pauli);
    fn pauli_exp(&mut self, sparse_pauli: &Pauli);
    fn permute(&mut self, permutation: &[usize], support: &[QubitId]);
    fn unitary_op(&mut self, operation: Operation, support: &[QubitId]);

    fn is_stabilizer(&self, observable: &Pauli) -> bool;
    fn is_stabilizer_up_to_sign(&self, observable: &Pauli) -> bool;
    fn is_stabilizer_with_conditional_sign(&self, observable: &Pauli, outcomes: &[OutcomeId]) -> bool;

    fn measure(&mut self, observable: &Pauli) -> OutcomeId;
    fn measure_with_hint(&mut self, observable: &Pauli, anti_commuting_stabilizer: &Pauli) -> OutcomeId;

    fn qubit_count(&self) -> usize;
    fn outcome_count(&self) -> usize;
    fn random_outcome_count(&self) -> usize;
    fn random_outcome_indicator(&self) -> &[bool];

    #[must_use]
    fn new(qubit_count: usize) -> Self
    where
        Self: Sized,
    {
        Self::with_capacity(qubit_count, 0, 0)
    }
    fn with_capacity(qubit_count: usize, outcome_count: usize, random_outcome_count: usize) -> Self
    where
        Self: Sized;

    fn qubit_capacity(&self) -> usize;
    fn reserve_qubits(&mut self, new_qubit_capacity: usize);
    fn outcome_capacity(&self) -> usize;
    fn random_outcome_capacity(&self) -> usize;
    fn reserve_outcomes(&mut self, new_outcome_capacity: usize, new_random_outcome_capacity: usize);
}
