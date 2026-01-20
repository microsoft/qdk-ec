use paulimer::clifford::CliffordUnitary;
use paulimer::pauli::{Pauli, SparsePauli};
use paulimer::UnitaryOp;

use crate::noise::PauliFault;

pub type OutcomeId = usize;
pub type QubitId = usize;

#[derive(Debug, Clone)]
#[allow(dead_code)] // Some fields are unused currently. But will be needed when circuits are made public.
pub(crate) enum Instruction {
    Unitary {
        opcode: UnitaryOp,
        qubits: Vec<QubitId>,
    },
    Clifford {
        clifford: CliffordUnitary,
        qubits: Vec<QubitId>,
    },
    Pauli {
        pauli: SparsePauli,
    },
    PauliExp {
        pauli: SparsePauli,
    },
    Permute {
        permutation: Vec<usize>,
        qubits: Vec<QubitId>,
    },
    ControlledPauli {
        control: SparsePauli,
        target: SparsePauli,
    },
    Measure {
        observable: SparsePauli,
        outcome_id: OutcomeId,
    },
    AllocateRandomBit {
        outcome_id: OutcomeId,
    },
    ConditionalPauli {
        pauli: SparsePauli,
        outcomes: Vec<OutcomeId>,
        parity: bool,
    },
    /// Noise instruction: injects faults according to a fault specification.
    Noise {
        fault: PauliFault,
    },
}

impl Instruction {
    /// Returns the qubits affected by this instruction for noise injection.
    ///
    /// Used by `Circuit::with_depolarizing_noise` to determine where to inject faults.
    #[must_use]
    pub fn noise_qubits(&self) -> Vec<QubitId> {
        match self {
            Instruction::Unitary { qubits, .. }
            | Instruction::Clifford { qubits, .. }
            | Instruction::Permute { qubits, .. } => qubits.clone(),
            Instruction::Pauli { pauli } | Instruction::PauliExp { pauli } => pauli.support().collect(),
            Instruction::ControlledPauli { control, target } => control.support().chain(target.support()).collect(),
            Instruction::Measure { observable, .. } => observable.support().collect(),
            Instruction::AllocateRandomBit { .. }
            | Instruction::ConditionalPauli { .. }
            | Instruction::Noise { .. } => Vec::new(),
        }
    }

    /// Returns true if noise should be inserted BEFORE this instruction.
    ///
    /// For measurements, faults are inserted before so they can flip outcomes.
    #[must_use]
    pub fn noise_before(&self) -> bool {
        matches!(self, Instruction::Measure { .. })
    }

    /// Create a noise instruction from a fault specification.
    #[must_use]
    pub fn noise(fault: PauliFault) -> Self {
        Instruction::Noise { fault }
    }

    #[must_use]
    pub fn num_fault_locations(&self) -> usize {
        match self {
            Instruction::Unitary { qubits, .. }
            | Instruction::Clifford { qubits, .. }
            | Instruction::Permute { qubits, .. } => 2 * qubits.len(),
            Instruction::PauliExp { pauli } | Instruction::Pauli { pauli } => 2 * pauli.weight(),
            Instruction::ControlledPauli { control, target } => 2 * (control.weight() + target.weight()),
            Instruction::Measure { observable, .. } => 2 * observable.weight(),
            Instruction::AllocateRandomBit { .. }
            | Instruction::ConditionalPauli { .. }
            | Instruction::Noise { .. } => 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
#[must_use]
pub(crate) struct Circuit {
    pub instructions: Vec<Instruction>,
}

#[allow(dead_code)]
impl Circuit {
    /// Create a new empty circuit.
    pub fn new() -> Self {
        Circuit::default()
    }

    /// Create a new circuit with pre-allocated instruction capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Circuit {
            instructions: Vec::with_capacity(capacity),
        }
    }

    /// Push an instruction to the circuit.
    pub fn push(&mut self, instruction: Instruction) {
        self.instructions.push(instruction);
    }

    pub fn iter(&self) -> impl Iterator<Item = &Instruction> {
        self.instructions.iter()
    }

    pub fn iter_rev(&self) -> impl Iterator<Item = &Instruction> {
        self.instructions.iter().rev()
    }

    pub fn fault_count(&self) -> usize {
        self.instructions.iter().map(Instruction::num_fault_locations).sum()
    }

    pub fn outcome_count(&self) -> usize {
        self.instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Measure { .. } | Instruction::AllocateRandomBit { .. }))
            .count()
    }

    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }

    /// Create a noisy version of this circuit with depolarizing noise after each gate.
    ///
    /// Inserts `Instruction::Noise` with depolarizing noise at the appropriate locations:
    /// - After unitary gates (so faults propagate through subsequent operations)
    /// - Before measurements (so faults can flip measurement outcomes)
    pub fn with_depolarizing_noise(&self, p_error: f64) -> Circuit {
        use crate::noise::PauliFault;

        let mut noisy_circuit = Circuit::with_capacity(self.instructions.len() * 2);

        for instruction in &self.instructions {
            let qubits = instruction.noise_qubits();

            if instruction.noise_before() && !qubits.is_empty() {
                noisy_circuit.push(Instruction::Noise {
                    fault: PauliFault::depolarizing(&qubits, p_error),
                });
            }

            noisy_circuit.push(instruction.clone());

            if !instruction.noise_before() && !qubits.is_empty() {
                noisy_circuit.push(Instruction::Noise {
                    fault: PauliFault::depolarizing(&qubits, p_error),
                });
            }
        }

        noisy_circuit
    }

    /// Validate that correlated faults (those sharing a `fault_set` ID) have matching distribution sizes.
    ///
    /// Correlated faults rely on selecting the same index from their distributions. For this to work
    /// correctly, all faults with the same `fault_set` must have distributions of equal length.
    ///
    /// # Errors
    ///
    /// Returns an error message if any `correlation_id` contains faults with mismatched distribution sizes.
    pub fn validate_correlated_faults(&self) -> Result<(), String> {
        use std::collections::HashMap;

        let mut correlation_id_sizes: HashMap<u64, usize> = HashMap::new();

        for instruction in &self.instructions {
            if let Instruction::Noise { fault } = instruction {
                if let Some(correlation_id) = fault.correlation_id {
                    let size = fault.distribution.len();
                    match correlation_id_sizes.entry(correlation_id) {
                        std::collections::hash_map::Entry::Occupied(entry) => {
                            let expected = *entry.get();
                            if size != expected {
                                return Err(format!(
                                    "Correlated faults with correlation_id {correlation_id} have mismatched distribution sizes: {expected} vs {size}"
                                ));
                            }
                        }
                        std::collections::hash_map::Entry::Vacant(entry) => {
                            entry.insert(size);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use paulimer::pauli::SparsePauli;
    use proptest::prelude::*;
    use rand::seq::SliceRandom;
    use rand::Rng;
    use std::str::FromStr;

    /// Configuration for circuit generation.
    #[derive(Debug, Clone, Copy)]
    struct CircuitConfig {
        qubit_count: usize,
        min_instructions: usize,
        max_instructions: usize,
        allow_measurements: bool,
        max_pauli_weight: usize,
    }

    impl Default for CircuitConfig {
        fn default() -> Self {
            Self {
                qubit_count: 4,
                min_instructions: 1,
                max_instructions: 20,
                allow_measurements: true,
                max_pauli_weight: 3,
            }
        }
    }

    /// Generate a random `SparsePauli` with weight at most `max_weight` on `qubit_count` qubits.
    fn random_sparse_pauli(qubit_count: usize, max_weight: usize, rng: &mut impl Rng) -> SparsePauli {
        if qubit_count == 0 {
            return SparsePauli::from_str("I").unwrap_or_else(|_| SparsePauli::from_str("").unwrap());
        }
        let weight = rng.gen_range(1..=max_weight.min(qubit_count));
        let mut positions: Vec<usize> = (0..qubit_count).collect();
        positions.shuffle(rng);
        positions.truncate(weight);
        positions.sort_unstable();

        let mut pauli_chars = vec!['I'; qubit_count];
        for &pos in &positions {
            pauli_chars[pos] = match rng.gen_range(0..3) {
                0 => 'X',
                1 => 'Y',
                _ => 'Z',
            };
        }
        let pauli_str: String = pauli_chars.into_iter().collect();
        SparsePauli::from_str(&pauli_str).expect("Valid Pauli string")
    }

    /// Generate a random instruction for a circuit with the given configuration.
    fn random_instruction(config: &CircuitConfig, outcome_counter: &mut OutcomeId, rng: &mut impl Rng) -> Instruction {
        let qubit_count = config.qubit_count;

        let instruction_type = if config.allow_measurements {
            rng.gen_range(0..10)
        } else {
            rng.gen_range(0..7)
        };

        match instruction_type {
            0..=3 => {
                let qubit = rng.gen_range(0..qubit_count);
                let opcode = match rng.gen_range(0..7) {
                    0 => UnitaryOp::Hadamard,
                    1 => UnitaryOp::SqrtZ,
                    2 => UnitaryOp::SqrtZInv,
                    3 => UnitaryOp::SqrtX,
                    4 => UnitaryOp::SqrtXInv,
                    5 => UnitaryOp::SqrtY,
                    _ => UnitaryOp::SqrtYInv,
                };
                Instruction::Unitary {
                    opcode,
                    qubits: vec![qubit],
                }
            }
            4..=6 => {
                if qubit_count < 2 {
                    return Instruction::Unitary {
                        opcode: UnitaryOp::Hadamard,
                        qubits: vec![0],
                    };
                }
                let qubit_a = rng.gen_range(0..qubit_count);
                let mut qubit_b = rng.gen_range(0..qubit_count);
                while qubit_b == qubit_a {
                    qubit_b = rng.gen_range(0..qubit_count);
                }
                let opcode = match rng.gen_range(0..3) {
                    0 => UnitaryOp::ControlledX,
                    1 => UnitaryOp::ControlledZ,
                    _ => UnitaryOp::Swap,
                };
                Instruction::Unitary {
                    opcode,
                    qubits: vec![qubit_a, qubit_b],
                }
            }
            7..=8 => {
                let observable = random_sparse_pauli(qubit_count, config.max_pauli_weight, rng);
                let outcome_id = *outcome_counter;
                *outcome_counter += 1;
                Instruction::Measure { observable, outcome_id }
            }
            _ => {
                let outcome_id = *outcome_counter;
                *outcome_counter += 1;
                Instruction::AllocateRandomBit { outcome_id }
            }
        }
    }

    /// Generate a random circuit with the given configuration and RNG.
    fn random_circuit_with_rng(config: CircuitConfig, rng: &mut impl Rng) -> Circuit {
        let num_instructions = rng.gen_range(config.min_instructions..=config.max_instructions);
        let mut circuit = Circuit::with_capacity(num_instructions);
        let mut outcome_counter = 0;

        for _ in 0..num_instructions {
            circuit.push(random_instruction(&config, &mut outcome_counter, rng));
        }

        circuit
    }

    /// Generate a random circuit with the given configuration.
    fn random_circuit(config: CircuitConfig) -> Circuit {
        let mut rng = rand::thread_rng();
        random_circuit_with_rng(config, &mut rng)
    }

    /// Create a simple Bell state preparation circuit.
    fn bell_state_circuit() -> Circuit {
        let mut circuit = Circuit::new();
        circuit.push(Instruction::Unitary {
            opcode: UnitaryOp::Hadamard,
            qubits: vec![0],
        });
        circuit.push(Instruction::Unitary {
            opcode: UnitaryOp::ControlledX,
            qubits: vec![0, 1],
        });
        circuit.push(Instruction::Measure {
            observable: SparsePauli::from_str("ZZ").expect("Valid Pauli"),
            outcome_id: 0,
        });
        circuit.push(Instruction::Measure {
            observable: SparsePauli::from_str("XX").expect("Valid Pauli"),
            outcome_id: 1,
        });
        circuit
    }

    /// Create a simple GHZ state preparation circuit.
    fn ghz_circuit(qubit_count: usize) -> Circuit {
        let mut circuit = Circuit::new();
        circuit.push(Instruction::Unitary {
            opcode: UnitaryOp::Hadamard,
            qubits: vec![0],
        });
        for i in 0..qubit_count - 1 {
            circuit.push(Instruction::Unitary {
                opcode: UnitaryOp::ControlledX,
                qubits: vec![i, i + 1],
            });
        }
        let zz_str: String = "Z".repeat(qubit_count);
        circuit.push(Instruction::Measure {
            observable: SparsePauli::from_str(&zz_str).expect("Valid Pauli"),
            outcome_id: 0,
        });
        circuit
    }

    /// Create a repetition code circuit (distance d).
    fn repetition_code_circuit(distance: usize, rounds: usize) -> Circuit {
        let qubit_count = distance;
        let mut circuit = Circuit::new();
        let mut outcome_id = 0;

        for _ in 0..rounds {
            for i in 0..qubit_count - 1 {
                let mut pauli_chars = vec!['I'; qubit_count];
                pauli_chars[i] = 'Z';
                pauli_chars[i + 1] = 'Z';
                let pauli_str: String = pauli_chars.into_iter().collect();
                circuit.push(Instruction::Measure {
                    observable: SparsePauli::from_str(&pauli_str).expect("Valid Pauli"),
                    outcome_id,
                });
                outcome_id += 1;
            }
        }

        circuit
    }

    /// Proptest strategy for generating arbitrary circuits.
    fn arbitrary_circuit_strategy(
        num_qubits_range: std::ops::Range<usize>,
        instructions_range: std::ops::Range<usize>,
    ) -> impl Strategy<Value = Circuit> {
        (num_qubits_range, instructions_range).prop_map(|(num_qubits, max_instructions)| {
            let config = CircuitConfig {
                qubit_count: num_qubits.max(1),
                min_instructions: 1,
                max_instructions: max_instructions.max(1),
                allow_measurements: true,
                max_pauli_weight: 3,
            };
            random_circuit(config)
        })
    }

    // ========== Basic Unit Tests ==========

    #[test]
    fn empty_circuit() {
        let circuit = Circuit::new();
        assert_eq!(circuit.fault_count(), 0);
        assert!(circuit.is_empty());
    }

    #[test]
    fn empty_circuit_has_zero_faults() {
        let circuit = Circuit::new();
        assert_eq!(circuit.fault_count(), 0);
        assert_eq!(circuit.outcome_count(), 0);
        assert!(circuit.is_empty());
    }

    #[test]
    fn circuit_with_gates() {
        let mut circuit = Circuit::new();
        circuit.push(Instruction::Unitary {
            opcode: UnitaryOp::Hadamard,
            qubits: vec![0],
        });
        circuit.push(Instruction::Unitary {
            opcode: UnitaryOp::ControlledX,
            qubits: vec![0, 1],
        });

        assert_eq!(circuit.len(), 2);
        assert_eq!(circuit.fault_count(), 2 + 4);
    }

    #[test]
    fn circuit_with_measure() {
        let mut circuit = Circuit::new();
        let obs = SparsePauli::from_str("ZZ").unwrap();
        circuit.push(Instruction::Measure {
            observable: obs,
            outcome_id: 0,
        });

        assert_eq!(circuit.fault_count(), 4);
        assert_eq!(circuit.outcome_count(), 1);
    }

    #[test]
    fn single_qubit_gate_has_two_fault_locations() {
        let mut circuit = Circuit::new();
        circuit.push(Instruction::Unitary {
            opcode: UnitaryOp::Hadamard,
            qubits: vec![0],
        });
        assert_eq!(circuit.fault_count(), 2);
    }

    #[test]
    fn two_qubit_gate_has_four_fault_locations() {
        let mut circuit = Circuit::new();
        circuit.push(Instruction::Unitary {
            opcode: UnitaryOp::ControlledX,
            qubits: vec![0, 1],
        });
        assert_eq!(circuit.fault_count(), 4);
    }

    #[test]
    fn measurement_has_fault_locations_per_weight() {
        let mut circuit = Circuit::new();
        let observable = SparsePauli::from_str("ZZ").unwrap();
        circuit.push(Instruction::Measure {
            observable,
            outcome_id: 0,
        });
        assert_eq!(circuit.fault_count(), 4);
        assert_eq!(circuit.outcome_count(), 1);
    }

    #[test]
    fn allocate_random_bit_has_no_faults() {
        let mut circuit = Circuit::new();
        circuit.push(Instruction::AllocateRandomBit { outcome_id: 0 });
        assert_eq!(circuit.fault_count(), 0);
        assert_eq!(circuit.outcome_count(), 1);
    }

    #[test]
    fn conditional_pauli_has_no_faults() {
        let mut circuit = Circuit::new();
        circuit.push(Instruction::ConditionalPauli {
            pauli: SparsePauli::from_str("XI").unwrap(),
            outcomes: vec![0],
            parity: true,
        });
        assert_eq!(circuit.fault_count(), 0);
        assert_eq!(circuit.outcome_count(), 0);
    }

    #[test]
    fn bell_circuit_properties() {
        let circuit = bell_state_circuit();
        assert_eq!(circuit.outcome_count(), 2);
        assert_eq!(circuit.fault_count(), 2 + 4 + 4 + 4);
    }

    #[test]
    fn ghz_circuit_properties() {
        for n in 2..=8 {
            let circuit = ghz_circuit(n);
            assert_eq!(circuit.outcome_count(), 1);
            let expected_faults = 2 + 4 * (n - 1) + 2 * n;
            assert_eq!(circuit.fault_count(), expected_faults);
        }
    }

    #[test]
    fn repetition_code_circuit_properties() {
        let circuit = repetition_code_circuit(5, 3);
        assert_eq!(circuit.outcome_count(), 12);
        assert_eq!(circuit.fault_count(), 12 * 4);
    }

    #[test]
    fn circuit_iter_and_len() {
        let mut circuit = Circuit::new();
        circuit.push(Instruction::Unitary {
            opcode: UnitaryOp::Hadamard,
            qubits: vec![0],
        });
        circuit.push(Instruction::Unitary {
            opcode: UnitaryOp::ControlledX,
            qubits: vec![0, 1],
        });
        circuit.push(Instruction::Measure {
            observable: SparsePauli::from_str("ZII").unwrap(),
            outcome_id: 0,
        });

        assert_eq!(circuit.len(), 3);
        assert!(!circuit.is_empty());

        let count = circuit.iter().count();
        assert_eq!(count, 3);

        let rev_count = circuit.iter_rev().count();
        assert_eq!(rev_count, 3);
    }

    #[test]
    fn very_deep_circuit() {
        let config = CircuitConfig {
            qubit_count: 4,
            min_instructions: 1000,
            max_instructions: 1000,
            allow_measurements: true,
            max_pauli_weight: 2,
        };
        let circuit = random_circuit(config);
        assert_eq!(circuit.len(), 1000);
        let _ = circuit.fault_count();
        let _ = circuit.outcome_count();
    }

    #[test]
    fn single_qubit_circuit() {
        let config = CircuitConfig {
            qubit_count: 1,
            min_instructions: 10,
            max_instructions: 10,
            allow_measurements: true,
            max_pauli_weight: 1,
        };
        let circuit = random_circuit(config);
        for instruction in circuit.iter() {
            for qubit in instruction.noise_qubits() {
                assert_eq!(qubit, 0);
            }
        }
    }

    #[test]
    fn circuit_with_capacity() {
        let circuit = Circuit::with_capacity(100);
        assert!(circuit.is_empty());
    }

    #[test]
    fn with_depolarizing_noise_inserts_noise_after_gates() {
        let mut circuit = Circuit::new();
        circuit.push(Instruction::Unitary {
            opcode: UnitaryOp::Hadamard,
            qubits: vec![0],
        });
        circuit.push(Instruction::Unitary {
            opcode: UnitaryOp::ControlledX,
            qubits: vec![0, 1],
        });
        let obs = SparsePauli::from_str("ZZ").unwrap();
        circuit.push(Instruction::Measure {
            observable: obs,
            outcome_id: 0,
        });

        let noisy = circuit.with_depolarizing_noise(0.01);

        assert_eq!(circuit.len(), 3);
        assert_eq!(noisy.len(), 6);

        let instructions: Vec<_> = noisy.iter().collect();

        assert!(matches!(instructions[0], Instruction::Unitary { .. }));
        assert!(matches!(instructions[1], Instruction::Noise { .. }));
        assert!(matches!(instructions[2], Instruction::Unitary { .. }));
        assert!(matches!(instructions[3], Instruction::Noise { .. }));
        assert!(matches!(instructions[4], Instruction::Noise { .. }));
        assert!(matches!(instructions[5], Instruction::Measure { .. }));
    }

    // ========== Property-Based Tests ==========

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn num_outcomes_matches_outcome_instructions(circuit in arbitrary_circuit_strategy(1..20, 1..50)) {
            let circuit: Circuit = circuit;
            let expected_outcomes = circuit.iter().filter(|i| {
                matches!(i, Instruction::Measure { .. } | Instruction::AllocateRandomBit { .. })
            }).count();
            prop_assert_eq!(circuit.outcome_count(), expected_outcomes);
        }

        #[test]
        fn num_faults_is_sum_of_instruction_faults(circuit in arbitrary_circuit_strategy(1..20, 1..50)) {
            let circuit: Circuit = circuit;
            let expected_faults: usize = circuit.iter().map(Instruction::num_fault_locations).sum();
            prop_assert_eq!(circuit.fault_count(), expected_faults);
        }

        #[test]
        fn len_equals_instruction_count(circuit in arbitrary_circuit_strategy(1..20, 1..50)) {
            let circuit: Circuit = circuit;
            prop_assert_eq!(circuit.len(), circuit.instructions.len());
            prop_assert_eq!(circuit.iter().count(), circuit.len());
        }

        #[test]
        fn unitary_instruction_fault_locations(weight in 1usize..5) {
            let instruction = Instruction::Unitary {
                opcode: UnitaryOp::ControlledX,
                qubits: (0..weight).collect(),
            };
            prop_assert_eq!(instruction.num_fault_locations(), 2 * weight);
        }

        #[test]
        fn measure_instruction_fault_locations(weight in 1usize..10) {
            let pauli_str: String = "Z".repeat(weight);
            let observable = SparsePauli::from_str(&pauli_str).unwrap();
            let instruction = Instruction::Measure {
                observable,
                outcome_id: 0,
            };
            prop_assert_eq!(instruction.num_fault_locations(), 2 * weight);
        }
    }
}
