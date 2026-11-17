//! Compute a noiseless reference without a tableau for CSS circuits.
//!
//! X/Z preparation, measurement, and CX preserve an all-plus CSS trajectory
//! when random projections select the positive branch. Pauli gates change its
//! signs; representing them as probability-one errors lets Stim's frame sampler
//! draw a valid noiseless reference. Other operations retain Stim's ordinary
//! reference algorithm. The original noisy circuit is never rewritten.

use stim::{Circuit, CircuitItem, Gate};

fn css_reference_circuit(circuit: &Circuit) -> Option<Circuit> {
    let mut reference = Circuit::new();
    for item in circuit.without_noise() {
        let CircuitItem::Instruction(instruction) = item else {
            return None;
        };
        let gate = instruction.gate();
        if matches!(
            gate,
            Gate::TICK | Gate::QUBIT_COORDS | Gate::SHIFT_COORDS | Gate::DETECTOR | Gate::OBSERVABLE_INCLUDE
        ) {
            continue;
        }
        if !matches!(
            gate,
            Gate::R
                | Gate::RX
                | Gate::M
                | Gate::MX
                | Gate::MR
                | Gate::MRX
                | Gate::CX
                | Gate::I
                | Gate::X
                | Gate::Y
                | Gate::Z
        ) || instruction
            .targets()
            .iter()
            .any(|target| !target.is_qubit_target() || target.is_inverted_result_target())
            || instruction.gate_args().iter().any(|&argument| argument != 0.0)
        {
            return None;
        }
        let pauli_error = match gate {
            Gate::X => Some(Gate::X_ERROR),
            Gate::Y => Some(Gate::Y_ERROR),
            Gate::Z => Some(Gate::Z_ERROR),
            _ => None,
        };
        if let Some(pauli_error) = pauli_error {
            reference
                .append_gate_targets(pauli_error, instruction.targets(), &[1.0])
                .expect("Pauli gates and their deterministic errors have the same targets");
        } else {
            reference
                .append_instruction(&instruction)
                .expect("copying a validated Stim instruction preserves validity");
        }
    }
    assert_eq!(reference.num_measurements(), circuit.num_measurements());
    Some(reference)
}

/// Return a valid noiseless outcome in the original measurement order.
pub(super) fn reference_sample(circuit: &Circuit) -> Vec<bool> {
    if circuit.num_measurements() == 0 {
        return vec![];
    }
    let Some(reference) = css_reference_circuit(circuit) else {
        #[cfg(feature = "cli")]
        log::debug!("Using Stim's tableau reference algorithm for a non-CSS circuit");
        return circuit.reference_sample();
    };
    reference.compile_sampler_with_seed(true, 0).sample(1).row(0).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_keeps_pauli_signs_but_not_physical_noise() {
        let circuit: Circuit = "X 0\nX_ERROR(1) 0\nM 0 1\nRX 2\nZ 2\nMX 2".parse().unwrap();
        assert!(css_reference_circuit(&circuit).is_some());
        assert_eq!(reference_sample(&circuit), vec![true, false, true]);
    }

    #[test]
    fn unsupported_operations_use_original_reference() {
        for text in [
            "H 0\nM 0",
            "S 0\nMX 0",
            "RY 0\nMY 0",
            "X 0\nM !0",
            "M 0\nCX rec[-1] 1\nM 1",
            "CX sweep[0] 0\nM 0",
            "MPP X0*Z1",
            "MXX 0 1",
            "MPAD 1 0",
            "REPEAT 2 {\nM 0\n}",
            "HERALDED_ERASE(0.1) 0\nM 0",
        ] {
            let circuit: Circuit = text.parse().unwrap();
            assert!(css_reference_circuit(&circuit).is_none(), "{text}");
            assert_eq!(reference_sample(&circuit), circuit.reference_sample(), "{text}");
        }
    }

    #[test]
    fn empty_measurement_record_needs_no_reference() {
        assert!(reference_sample(&"H 0".parse().unwrap()).is_empty());
    }

    #[test]
    fn references_are_valid_postselected_trajectories() {
        use rand::{RngExt, SeedableRng};

        let mut rng = rand_xoshiro::Xoshiro256StarStar::seed_from_u64(73);
        let gates = [
            Gate::X,
            Gate::Y,
            Gate::Z,
            Gate::CX,
            Gate::R,
            Gate::RX,
            Gate::M,
            Gate::MX,
            Gate::MR,
            Gate::MRX,
        ];
        for _ in 0..100 {
            let mut circuit = Circuit::new();
            for _ in 0..100 {
                let gate = gates[rng.random_range(0..gates.len())];
                let qubit = rng.random_range(0..6);
                let targets = if gate == Gate::CX {
                    vec![qubit, (qubit + rng.random_range(1..6)) % 6]
                } else {
                    vec![qubit]
                };
                // Record reset projections so the test can also follow their hidden branches.
                let projection = match gate {
                    Gate::R => Some(Gate::M),
                    Gate::RX => Some(Gate::MX),
                    _ => None,
                };
                if let Some(projection) = projection {
                    circuit.append(projection, &targets, &[]).unwrap();
                }
                circuit.append(gate, &targets, &[]).unwrap();
            }
            assert!(css_reference_circuit(&circuit).is_some());
            let reference = reference_sample(&circuit);
            let mut simulator = stim::TableauSimulator::new();
            let mut values = reference.into_iter();
            for item in circuit {
                let CircuitItem::Instruction(instruction) = item else {
                    unreachable!()
                };
                let gate = instruction.gate();
                if gate.produces_measurements() {
                    for target in instruction.targets() {
                        let qubit = target.qubit_value().unwrap() as usize;
                        let value = values.next().unwrap();
                        match gate {
                            Gate::M | Gate::MR => simulator.postselect_z(&[qubit], value),
                            Gate::MX | Gate::MRX => simulator.postselect_x(&[qubit], value),
                            _ => unreachable!(),
                        }
                        .expect("reference must be a possible trajectory of the original circuit");
                        let reset = match gate {
                            Gate::MR => Some(Gate::R),
                            Gate::MRX => Some(Gate::RX),
                            _ => None,
                        };
                        if let Some(reset) = reset {
                            let mut step = Circuit::new();
                            step.append_gate_targets(reset, &[*target], &[]).unwrap();
                            simulator.do_circuit(&step);
                        }
                    }
                } else {
                    let mut step = Circuit::new();
                    step.append_instruction(&instruction).unwrap();
                    simulator.do_circuit(&step);
                }
            }
            assert!(values.next().is_none());
        }
    }
}
