use crate::{
    circuit::{Circuit, SimulationError},
    OutcomeCompleteSimulation, Simulation,
};
use binar::{BitMatrix, BitVec, Bitwise, IndexSet};
use paulimer::{
    clifford::standard_restriction_with_sign_matrix,
    pauli::{as_sparse_projective, remapped_sparse, SparsePauliProjective},
    Clifford, CliffordUnitary, Pauli, PauliMutable, SparsePauli,
};

type QubitId = usize;
type QubitIndexMap = Vec<QubitId>;

pub struct CircuitAction {
    /// The observables measured by the circuit, that is Paulis whose measurement outcomes are part of circuit outcomes
    observables: GeneratorsWithSignMatrix,
    /// The stabilizers of the output state of the circuit for all inputs
    stabilizers: GeneratorsWithSignMatrix,
    /// The stabilizers of the choi state of the circuit
    choi_state_stabilizers: GeneratorsWithSignMatrix,
    /// The stabilizers of auxiliary qubits used by the circuit
    auxiliary_stabilizers: GeneratorsWithSignMatrix,
    /// A matrix that maps circuit outcomes to inner random bits describing its choi state
    random_bit_map_matrix: BitMatrix,
    /// A vector offsets circuit outcomes to inner random bits describing its choi state
    random_bit_map_shift: BitVec,
}

struct GeneratorsWithSignMatrix {
    generators: Vec<SparsePauli>,
    /// The sign of generator i is determined by the inner product of `sign_matrix` row and outcome.
    sign_matrix: BitMatrix,
    sign_matrix_transposed: BitMatrix,
    qubits: Vec<QubitId>,
    qubit_index: QubitIndexMap,
}

impl GeneratorsWithSignMatrix {
    fn new(generators: Vec<SparsePauli>, sign_matrix: BitMatrix, qubits: &[QubitId], qubit_count: usize) -> Self {
        assert_eq!(generators.len(), sign_matrix.row_count());
        let sign_matrix_transposed = sign_matrix.transposed();
        let mut qubit_index = vec![usize::MAX; qubit_count];
        for (index, qubit) in qubits.iter().enumerate() {
            qubit_index[*qubit] = index;
        }
        Self {
            generators,
            sign_matrix,
            sign_matrix_transposed,
            qubits: qubits.to_vec(),
            qubit_index,
        }
    }

    fn from_restriction(clifford: &CliffordUnitary, sign_matrix: &BitMatrix, support: &[QubitId]) -> Self {
        let qubit_count = clifford.num_qubits();
        let (paulis, sign_matrix) = standard_restriction_with_sign_matrix(clifford, sign_matrix, support);
        Self::new(paulis, sign_matrix, support, qubit_count)
    }

    fn abs(&self) -> Vec<SparsePauliProjective> {
        self.generators.iter().map(as_sparse_projective).collect()
    }

    fn equivalent_up_to_signs(&self, other: &Self) -> bool {
        for (left, right) in self.generators.iter().zip(other.generators.iter()) {
            if left.x_bits() != right.x_bits() || left.z_bits() != right.z_bits() {
                return false;
            }
        }
        true
    }

    fn with_transformed_signs(
        &self,
        random_bit_map_matrix: &BitMatrix,
        random_bit_map_shift: &BitVec,
    ) -> Vec<SignedObservable> {
        let transformed_sign_matrix = &self.sign_matrix * random_bit_map_matrix; // is self.sign_matrix.dot(random_bit_map_matrix) more ergonomic ?
        let transformed_shift = self
            .sign_matrix_transposed
            .right_multiply(&random_bit_map_shift.as_view());
        let mut result = Vec::new();
        for (index, generator) in self.generators.iter().enumerate() {
            let mut observable = remapped_sparse(generator, &self.qubit_index);
            if transformed_shift.index(index) {
                observable.add_assign_phase_exp(2);
            }
            let outcomes_sign_mask = (&(transformed_sign_matrix.row(index))).into();
            result.push(SignedObservable {
                observable,
                outcomes_sign_mask,
            });
        }
        result
    }
}

#[must_use]
pub struct SignedObservable {
    pub observable: SparsePauli,
    /// The sign of observable is determined by the inner product of `outcomes_sign_mask` and outcome.
    pub outcomes_sign_mask: BitVec,
}

#[derive(Debug, derive_more::From)]
pub enum ActionError {
    AuxiliaryQubitsEntangled {
        state_encoder: CliffordUnitary,
        auxiliary_qubits: Vec<QubitId>,
    },
    #[from]
    SimulationFailed(SimulationError),
}

pub enum ActionsInequivalenceReason {
    DifferentObservables,
    DifferentStabilizers,
    DifferentChoiStateStabilizers,
}

/// [`Circuit`]s in pauliverse include fixed number of qubits and do not have prepare and destroy instructions.
/// For this reason, we provide indexes of input and output qubits via `input_qubits` and `output_qubits`.
/// The qubits that are not `output_qubits` at the end of circuit execution are considered auxiliary qubits (see [`CircuitAction::auxiliary_qubits`]).
/// If they are entangled with reference qubits in the choi state, then action is undefined.
///
/// # Errors
///
/// Returns [`ActionError`] if action calculation fails.
pub fn action_of(
    circuit: &Circuit,
    input_qubits: &[QubitId],
    output_qubits: &[QubitId],
) -> Result<CircuitAction, ActionError> {
    let qubit_count = circuit.qubit_count();
    let reference_qubits: Vec<QubitId> = (qubit_count..qubit_count + input_qubits.len()).collect();
    let outcome_count = circuit.outcome_count();
    let mut simulation =
        OutcomeCompleteSimulation::with_capacity(qubit_count + input_qubits.len(), outcome_count, outcome_count);

    for (input_qubit, reference_qubit) in input_qubits.iter().zip(reference_qubits.iter()) {
        simulation.unitary_op(paulimer::UnitaryOp::PrepareBell, &[*input_qubit, *reference_qubit]);
    }

    circuit.simulate(&mut simulation)?;
    let sign_matrix = simulation.sign_matrix();
    let state_encoder = simulation.state_encoder();

    let auxiliary_qubits: Vec<QubitId> = output_qubits
        .iter()
        .copied()
        .collect::<IndexSet>()
        .complement(qubit_count)
        .into_iter()
        .collect();
    let auxiliary_stabilizers =
        GeneratorsWithSignMatrix::from_restriction(&state_encoder, &sign_matrix, &auxiliary_qubits);
    if auxiliary_stabilizers.generators.len() < auxiliary_qubits.len() {
        return Err(ActionError::AuxiliaryQubitsEntangled {
            state_encoder,
            auxiliary_qubits,
        });
    }

    let observables = GeneratorsWithSignMatrix::from_restriction(&state_encoder, &sign_matrix, &reference_qubits);
    let stabilizers = GeneratorsWithSignMatrix::from_restriction(&state_encoder, &sign_matrix, output_qubits);
    let choi_state_stabilizers = GeneratorsWithSignMatrix::from_restriction(
        &state_encoder,
        &sign_matrix,
        &(reference_qubits
            .iter()
            .chain(output_qubits.iter())
            .copied()
            .collect::<Vec<_>>()),
    );

    // TODO:
    let random_bit_map_matrix = BitMatrix::identity(outcome_count);
    let random_bit_map_shift = BitVec::zeros(outcome_count);

    let action = CircuitAction {
        observables,
        stabilizers,
        choi_state_stabilizers,
        auxiliary_stabilizers,
        random_bit_map_matrix,
        random_bit_map_shift,
    };
    Ok(action)
}

impl CircuitAction {
    /// Canonical choice of circuit observables, that is Paulis measured by the circuit
    #[must_use]
    pub fn observables(&self) -> Vec<SparsePauliProjective> {
        self.observables.abs()
    }

    /// Canonical choice of circuit stabilizers, that is Paulis that stabilize output state of the circuit
    /// for all circuit inputs
    #[must_use]
    pub fn stabilizers(&self) -> Vec<SparsePauliProjective> {
        self.stabilizers.abs()
    }

    /// Canonical choice of circuit choi state stabilizers
    #[must_use]
    pub fn choi_state_stabilizers(&self) -> Vec<SparsePauliProjective> {
        self.choi_state_stabilizers.abs()
    }

    /// Returns `Ok(())` if actions are equivalent up to signs, otherwise returns reasons for inequivalence.
    ///
    /// # Errors
    ///
    /// Returns a list of [`ActionsInequivalenceReason`] if the actions differ.
    pub fn equivalent_up_to_signs(&self, other: &CircuitAction) -> Result<(), Vec<ActionsInequivalenceReason>> {
        let mut reasons = Vec::new();
        if !self.observables.equivalent_up_to_signs(&other.observables) {
            reasons.push(ActionsInequivalenceReason::DifferentObservables);
        }
        if !self.stabilizers.equivalent_up_to_signs(&other.stabilizers) {
            reasons.push(ActionsInequivalenceReason::DifferentStabilizers);
        }
        if !self
            .choi_state_stabilizers
            .equivalent_up_to_signs(&other.choi_state_stabilizers)
        {
            reasons.push(ActionsInequivalenceReason::DifferentChoiStateStabilizers);
        }
        if reasons.is_empty() {
            Ok(())
        } else {
            Err(reasons)
        }
    }

    /// Check if two actions are equivalent when outcomes are remapped.
    ///
    /// # Errors
    ///
    /// Returns a list of [`ActionsInequivalenceReason`] if the actions differ.
    pub fn equivalent_with_outcome_map(
        &self,
        _other: &CircuitAction,
        _outcome_map_matrix: &BitMatrix,
        _outcome_map_shift: &BitVec,
    ) -> Result<(), Vec<ActionsInequivalenceReason>> {
        todo!()
    }

    /// Canonical stabilizers of auxiliary qubits used by the circuit
    #[must_use]
    pub fn auxiliary_stabilizers(&self) -> Vec<SparsePauliProjective> {
        self.auxiliary_stabilizers.abs()
    }

    /// Same as [`CircuitAction::observables`] but with signs as a function of circuit outcomes.
    #[must_use]
    pub fn signed_observables(&self) -> Vec<SignedObservable> {
        self.observables
            .with_transformed_signs(&self.random_bit_map_matrix, &self.random_bit_map_shift)
    }

    /// Same as [`CircuitAction::stabilizers`] but with signs as a function of circuit outcomes.
    #[must_use]
    pub fn signed_stabilizers(&self) -> Vec<SignedObservable> {
        self.stabilizers
            .with_transformed_signs(&self.random_bit_map_matrix, &self.random_bit_map_shift)
    }

    /// Same as [`CircuitAction::choi_state_stabilizers`] but with signs as a function of circuit outcomes.
    #[must_use]
    pub fn signed_choi_state_stabilizers(&self) -> Vec<SignedObservable> {
        self.choi_state_stabilizers
            .with_transformed_signs(&self.random_bit_map_matrix, &self.random_bit_map_shift)
    }

    /// Same as [`CircuitAction::auxiliary_stabilizers`] but with signs as a function of circuit outcomes.
    #[must_use]
    pub fn signed_auxiliary_stabilizers(&self) -> Vec<SignedObservable> {
        self.auxiliary_stabilizers
            .with_transformed_signs(&self.random_bit_map_matrix, &self.random_bit_map_shift)
    }

    #[must_use]
    pub fn input_qubits(&self) -> &[QubitId] {
        &self.observables.qubits
    }

    #[must_use]
    pub fn output_qubits(&self) -> &[QubitId] {
        &self.stabilizers.qubits
    }

    #[must_use]
    pub fn auxiliary_qubits(&self) -> &[QubitId] {
        &self.auxiliary_stabilizers.qubits
    }
}
