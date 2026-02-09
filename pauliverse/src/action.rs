use binar::{BitMatrix, BitVec};
use paulimer::{SparsePauli, pauli::SparsePauliProjective};
use crate::circuit::Circuit;


pub struct CircuitAction {
}

type QubitId = usize;

pub struct SignedObservable {
    observable: SparsePauli,
    /// the sign of observable is (-1)^<outcomes_sign_mask, outcome> where outcome is the bit vector of circuit outcomes
    outcomes_sign_mask: BitVec,
}

pub enum ActionError {
    AuxiliaryQubitsEntangled
}

pub enum ActionsInequivalenceReason {
    DifferentObservables,
    DifferentStabilizers,
    DifferentChoiStateStabilizers,
}

pub fn action_of( circuit: &Circuit, input_qubits: &[QubitId], output_qubits: &[QubitId] ) -> Result<CircuitAction, ActionError> {
    todo!()
}

impl CircuitAction {
    /// Canonical choice of circuit observables, that is Paulis measured by the circuit
    pub fn observables(&self) -> Vec<SparsePauliProjective> { 
        todo!()
    }

    /// Canonical choice of circuit stabilizers, that is Paulis that stabilize output state of the circuit 
    /// for all circuit inputs
    pub fn stabilizers(&self) -> Vec<SparsePauliProjective> { 
        todo!()
    }

    /// Canonical choice of circuit choi state stabilizers
    pub fn choi_state_stabilizers(&self) -> Vec<SparsePauliProjective> { 
        todo!()
    }
    
    pub fn equivalent_up_to_signs(&self, other: &CircuitAction) ->  Result<(), Vec<ActionsInequivalenceReason>> {
        let mut reasons = Vec::new();
        if self.observables() != other.observables() {
            reasons.push(ActionsInequivalenceReason::DifferentObservables);
        }
        if self.stabilizers() != other.stabilizers() {
            reasons.push(ActionsInequivalenceReason::DifferentStabilizers);
        }
        if self.choi_state_stabilizers() != other.choi_state_stabilizers() {
            reasons.push(ActionsInequivalenceReason::DifferentChoiStateStabilizers);
        }
        if reasons.is_empty() {
            Ok(())
        } else {
            Err(reasons)
        }
    }

    /// Check if two actions are equivalent when outcomes self o_r
    /// are related to outcomes of other o_l as o_r = M o_l + s for `outcome_map_matrix` M and `outcome_map_shift` s.
    pub fn equivalent_with_outcome_map(&self, other: &CircuitAction, outcome_map_matrix: &BitMatrix, outcome_map_shift: &BitVec) -> Result<(), Vec<ActionsInequivalenceReason>> {
        todo!()
    }

    /// Canonical stabilizers of auxiliary qubits used by the circuit
    pub fn auxiliary_stabilizers(&self) -> Vec<SparsePauliProjective> { 
        todo!()
    }

    /// Same as [CircuitAction::observables] but with signs as a function of circuit outcomes.
    pub fn signed_observables(&self) -> Vec<SignedObservable> { 
        todo!()
    }

    /// Same as [CircuitAction::stabilizers] but with signs as a function of circuit outcomes.
    pub fn signed_stabilizers(&self) -> Vec<SignedObservable> { 
        todo!()
    }

    /// Same as [CircuitAction::choi_state_stabilizers] but with signs as a function of circuit outcomes.
    pub fn signed_choi_state_stabilizers(&self) -> Vec<SignedObservable> { 
        todo!()
    }

    /// Same as [CircuitAction::auxiliary_stabilizers] but with signs as a function of circuit outcomes.
    pub fn signed_auxiliary_stabilizers(&self) -> Vec<SignedObservable> { 
        todo!()
    }

    fn reference_qubits(&self) -> &[QubitId] {
        todo!()
    }

    fn output_qubits(&self) -> &[QubitId] {
        todo!()
    }

    fn auxiliary_qubits(&self) -> &[QubitId] {
        todo!()
    }

    fn outcome_count(&self) -> usize {
        todo!()
    }

    fn outcome_shift(&self) -> BitVec {
        todo!()
    }

    fn choi_state_sign_matrix(&self) -> &BitMatrix {
        todo!()
    }

    fn observables_sign_matrix(&self) -> &BitMatrix {
        todo!()
    }

    fn stabilizers_sign_matrix(&self) -> &BitMatrix {
        todo!()
    }

    fn auxiliary_stabilizes_sign_matrix(&self) -> &BitMatrix {
        todo!()
    }

    // A matrix that maps circuit outcomes to inner random bits describing its choi state 
    fn random_bit_output_map(&self) -> &BitMatrix {
        todo!()
    }

    // Number of random bits needed to describe the choi state of the circuit
    fn random_bit_count(&self) -> usize {
        todo!()
    }
}
