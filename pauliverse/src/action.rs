use binar::{BitMatrix, BitVec};
use paulimer::{SparsePauli, pauli::SparsePauliProjective};
use crate::circuit::Circuit;


pub struct CircuitAction {
}

type QubitId = usize;

pub struct SignedObservable {
    observable: SparsePauli,
    outcomes_sign_mask: BitVec,
}



pub fn action_of( circuit: &Circuit, input_qubits: &[QubitId], output_qubits: &[QubitId] ) -> CircuitAction {
    // 
    CircuitAction {}
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

    /// Canonical stabilizers of auxiliary qubits used by the circuit
    pub fn auxiliary_stabilizers(&self) -> Vec<SparsePauliProjective> { 
        todo!()
    }

    /// Same as `observables` but with signs as a function of circuit outcomes.
    pub fn signed_observables(&self) -> Vec<SignedObservable> { 
        todo!()
    }

    /// Same as `stabilizers` but with signs as a function of circuit outcomes.
    pub fn signed_stabilizers(&self) -> Vec<SignedObservable> { 
        todo!()
    }

    /// Same as `choi_state_stabilizers` but with signs as a function of circuit outcomes.
    pub fn signed_choi_state_stabilizers(&self) -> Vec<SignedObservable> { 
        todo!()
    }

    /// Same as `auxiliary_stabilizers` but with signs as a function of circuit outcomes.
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
