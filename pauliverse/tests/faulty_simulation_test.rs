//! Tests for `FaultySimulation` using only the public API.

use binar::Bitwise;
use paulimer::clifford::{Clifford, CliffordMutable, CliffordUnitary};
use paulimer::pauli::SparsePauli;
use paulimer::UnitaryOp;
use pauliverse::faulty_simulation::FaultySimulation;
use pauliverse::noise::{PauliDistribution, PauliFault};
use pauliverse::Simulation;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::str::FromStr;

/// Create a Bell state simulation: H(0), CNOT(0,1), measure ZZ and XX.
fn bell_state_simulation() -> FaultySimulation {
    let mut sim = FaultySimulation::new();
    sim.unitary_op(UnitaryOp::Hadamard, &[0]);
    sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
    sim.measure(&SparsePauli::from_str("ZZ").unwrap());
    sim.measure(&SparsePauli::from_str("XX").unwrap());
    sim
}

/// Create a GHZ state simulation: H(0), CNOT chain, measure ZZZ...Z.
fn ghz_simulation(qubit_count: usize) -> FaultySimulation {
    let mut sim = FaultySimulation::new();
    sim.unitary_op(UnitaryOp::Hadamard, &[0]);
    for i in 0..qubit_count - 1 {
        sim.unitary_op(UnitaryOp::ControlledX, &[i, i + 1]);
    }
    let zz_str: String = "Z".repeat(qubit_count);
    sim.measure(&SparsePauli::from_str(&zz_str).unwrap());
    sim
}

/// Create a repetition code simulation with ZZ stabilizer measurements.
fn repetition_code_simulation(distance: usize, rounds: usize) -> FaultySimulation {
    let mut sim = FaultySimulation::new();
    for _ in 0..rounds {
        for i in 0..distance - 1 {
            let mut pauli_chars = vec!['I'; distance];
            pauli_chars[i] = 'Z';
            pauli_chars[i + 1] = 'Z';
            let pauli_str: String = pauli_chars.into_iter().collect();
            sim.measure(&SparsePauli::from_str(&pauli_str).unwrap());
        }
    }
    sim
}

// ========== Basic Tests ==========

#[test]
fn empty_simulation_produces_empty_outcomes() {
    let sim = FaultySimulation::new();
    let outcomes = sim.sample(100);
    assert_eq!(outcomes.row_count(), 100);
    assert_eq!(outcomes.column_count(), 0);
}

#[test]
fn single_shot_works() {
    let sim = bell_state_simulation();
    let outcomes = sim.sample(1);
    assert_eq!(outcomes.row_count(), 1);
    assert_eq!(outcomes.column_count(), 2);
}

#[test]
fn noiseless_bell_state_produces_deterministic_outcomes() {
    let sim = bell_state_simulation();
    let outcomes = sim.sample(1000);
    assert_eq!(outcomes.row_count(), 1000);
    assert_eq!(outcomes.column_count(), 2);

    for shot in 0..1000 {
        assert!(
            !outcomes.row(shot).index(0),
            "ZZ outcome should be 0 for Bell state at shot {shot}"
        );
        assert!(
            !outcomes.row(shot).index(1),
            "XX outcome should be 0 for Bell state at shot {shot}"
        );
    }
}

#[test]
fn noiseless_ghz_produces_correct_outcomes() {
    for n in [2, 4, 6] {
        let sim = ghz_simulation(n);
        let outcomes = sim.sample(100);
        assert_eq!(outcomes.row_count(), 100);
        assert_eq!(outcomes.column_count(), 1);

        for shot in 0..100 {
            assert!(
                !outcomes.row(shot).index(0),
                "ZZZ...Z outcome should be 0 for GHZ-{n} state"
            );
        }
    }
}

#[test]
fn seeded_sampling_is_deterministic() {
    let mut sim = repetition_code_simulation(5, 3);
    sim.apply_fault(PauliFault::depolarizing(&[0], 0.01));

    let mut rng1 = SmallRng::seed_from_u64(12345);
    let mut rng2 = SmallRng::seed_from_u64(12345);

    let outcomes1 = sim.sample_with_rng(1000, &mut rng1);
    let outcomes2 = sim.sample_with_rng(1000, &mut rng2);

    assert_eq!(outcomes1.row_count(), outcomes2.row_count());
    assert_eq!(outcomes1.column_count(), outcomes2.column_count());

    for shot in 0..outcomes1.row_count() {
        for outcome in 0..outcomes1.column_count() {
            assert_eq!(
                outcomes1.row(shot).index(outcome),
                outcomes2.row(shot).index(outcome),
                "Outcomes differ at shot {shot}, outcome {outcome}"
            );
        }
    }
}

#[test]
fn zero_shots_produces_empty_matrix() {
    let sim = bell_state_simulation();
    let outcomes = sim.sample(0);
    assert_eq!(outcomes.row_count(), 0);
}

#[test]
fn repetition_code_properties() {
    let sim = repetition_code_simulation(5, 3);
    assert_eq!(sim.qubit_count(), 5);
    assert_eq!(sim.outcome_count(), 12);
}

// ========== Fault Injection Tests ==========

#[test]
fn x_fault_before_z_measurement_flips_outcome() {
    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::single(SparsePauli::from_str("X").unwrap()),
        correlation_id: None,
        condition: None,
    });
    sim.measure(&SparsePauli::from_str("Z").unwrap());

    let outcomes = sim.sample(100);
    for shot in 0..100 {
        assert!(
            outcomes.row(shot).index(0),
            "X error should flip Z measurement outcome at shot {shot}"
        );
    }
}

#[test]
fn z_fault_before_z_measurement_does_not_flip_outcome() {
    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::single(SparsePauli::from_str("Z").unwrap()),
        correlation_id: None,
        condition: None,
    });
    sim.measure(&SparsePauli::from_str("Z").unwrap());

    let outcomes = sim.sample(100);
    for shot in 0..100 {
        assert!(
            !outcomes.row(shot).index(0),
            "Z error should NOT flip Z measurement outcome at shot {shot}"
        );
    }
}

#[test]
fn fault_after_cnot_propagates_to_target() {
    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::single(SparsePauli::from_str("XI").unwrap()),
        correlation_id: None,
        condition: None,
    });
    sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
    sim.measure(&SparsePauli::from_str("IZ").unwrap());

    let outcomes = sim.sample(100);
    for shot in 0..100 {
        assert!(
            outcomes.row(shot).index(0),
            "X fault before CNOT should propagate to target and flip Z_1 at shot {shot}"
        );
    }
}

#[test]
fn depolarizing_on_measurement_produces_expected_flip_rate() {
    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault::depolarizing(&[0], 0.3));
    sim.measure(&SparsePauli::from_str("Z").unwrap());

    let mut rng = SmallRng::seed_from_u64(42);
    let outcomes = sim.sample_with_rng(100_000, &mut rng);

    let flip_count: usize = (0..100_000).filter(|&shot| outcomes.row(shot).index(0)).count();
    #[allow(clippy::cast_precision_loss)]
    let flip_rate = flip_count as f64 / 100_000.0;

    // Expected: 0.3 * 2/3 ≈ 0.2
    assert!(
        (0.18..0.22).contains(&flip_rate),
        "Depolarizing flip rate {flip_rate:.4} outside expected range [0.18, 0.22]"
    );
}

// ========== Gate Instruction Tests ==========

#[test]
fn clifford_instruction_works() {
    let mut sim = FaultySimulation::new();
    sim.clifford(&CliffordUnitary::identity(2), &[0, 1]);
    sim.measure(&SparsePauli::from_str("ZI").unwrap());
    let outcomes = sim.sample(10);
    assert_eq!(outcomes.row_count(), 10);
    assert_eq!(outcomes.column_count(), 1);
}

#[test]
fn pauli_exp_instruction_works() {
    let mut sim = FaultySimulation::new();
    sim.pauli_exp(&SparsePauli::from_str("XX").unwrap());
    sim.measure(&SparsePauli::from_str("ZZ").unwrap());
    let outcomes = sim.sample(10);
    assert_eq!(outcomes.row_count(), 10);
    assert_eq!(outcomes.column_count(), 1);
}

#[test]
fn permute_instruction_works() {
    let mut sim = FaultySimulation::new();
    sim.permute(&[2, 0, 1], &[0, 1, 2]);
    sim.measure(&SparsePauli::from_str("ZII").unwrap());
    let outcomes = sim.sample(10);
    assert_eq!(outcomes.row_count(), 10);
    assert_eq!(outcomes.column_count(), 1);
}

#[test]
fn controlled_pauli_instruction_works() {
    let mut sim = FaultySimulation::new();
    sim.controlled_pauli(
        &SparsePauli::from_str("ZI").unwrap(),
        &SparsePauli::from_str("IX").unwrap(),
    );
    sim.measure(&SparsePauli::from_str("ZZ").unwrap());
    let outcomes = sim.sample(10);
    assert_eq!(outcomes.row_count(), 10);
    assert_eq!(outcomes.column_count(), 1);
}

// ========== Clifford Frame Propagation Tests ==========

#[test]
fn clifford_s_gate_transforms_x_to_y() {
    let mut s_gate = CliffordUnitary::identity(1);
    s_gate.left_mul_root_z(0);

    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::single(SparsePauli::from_str("X").unwrap()),
        correlation_id: None,
        condition: None,
    });
    sim.clifford(&s_gate, &[0]);
    sim.measure(&SparsePauli::from_str("Z").unwrap());

    let outcomes = sim.sample(100);
    for shot in 0..100 {
        assert!(
            outcomes.row(shot).index(0),
            "X before S becomes Y, which flips Z at shot {shot}"
        );
    }
}

#[test]
fn clifford_hadamard_swaps_x_and_z() {
    let mut h_gate = CliffordUnitary::identity(1);
    h_gate.left_mul_hadamard(0);

    let mut sim = FaultySimulation::new();
    sim.clifford(&h_gate, &[0]);
    sim.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::single(SparsePauli::from_str("Z").unwrap()),
        correlation_id: None,
        condition: None,
    });
    sim.measure(&SparsePauli::from_str("X").unwrap());

    let outcomes = sim.sample(100);
    for shot in 0..100 {
        assert!(
            outcomes.row(shot).index(0),
            "Z fault anti-commutes with X measurement, should flip at shot {shot}"
        );
    }
}

// ========== PauliExp Frame Propagation Test ==========

#[test]
fn pauli_exp_correctly_propagates_faults() {
    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::single(SparsePauli::from_str("XI").unwrap()),
        correlation_id: None,
        condition: None,
    });
    sim.pauli_exp(&SparsePauli::from_str("ZZ").unwrap());
    sim.measure(&SparsePauli::from_str("ZI").unwrap());

    let outcomes = sim.sample(100);
    for shot in 0..100 {
        assert!(
            outcomes.row(shot).index(0),
            "X fault before exp(ZZ) should flip Z_0 measurement at shot {shot}"
        );
    }
}

// ========== Permutation Frame Propagation Test ==========

#[test]
fn permute_correctly_reorders_frame() {
    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::single(SparsePauli::from_str("XI").unwrap()),
        correlation_id: None,
        condition: None,
    });
    sim.permute(&[1, 0], &[0, 1]);
    sim.measure(&SparsePauli::from_str("IZ").unwrap());

    let outcomes = sim.sample(100);
    for shot in 0..100 {
        assert!(
            outcomes.row(shot).index(0),
            "X fault before permute should appear on q1 after swap at shot {shot}"
        );
    }
}

#[test]
fn permute_leaves_other_qubit_unchanged() {
    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::single(SparsePauli::from_str("XI").unwrap()),
        correlation_id: None,
        condition: None,
    });
    sim.permute(&[1, 0], &[0, 1]);
    sim.measure(&SparsePauli::from_str("ZI").unwrap());

    let outcomes = sim.sample(100);
    for shot in 0..100 {
        assert!(
            !outcomes.row(shot).index(0),
            "X fault moved to q1, so Z_0 should not flip at shot {shot}"
        );
    }
}

// ========== ControlledPauli Frame Propagation Test ==========

#[test]
fn controlled_pauli_propagates_x_fault_correctly() {
    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::single(SparsePauli::from_str("XI").unwrap()),
        correlation_id: None,
        condition: None,
    });
    sim.controlled_pauli(
        &SparsePauli::from_str("ZI").unwrap(),
        &SparsePauli::from_str("IX").unwrap(),
    );
    sim.measure(&SparsePauli::from_str("IZ").unwrap());

    let outcomes = sim.sample(100);
    for shot in 0..100 {
        assert!(
            outcomes.row(shot).index(0),
            "X_0 before ControlledPauli(Z_0, X_1) should produce X_1, flipping Z_1 at shot {shot}"
        );
    }
}

#[test]
fn controlled_pauli_z_fault_does_not_trigger_target() {
    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::single(SparsePauli::from_str("ZI").unwrap()),
        correlation_id: None,
        condition: None,
    });
    sim.controlled_pauli(
        &SparsePauli::from_str("ZI").unwrap(),
        &SparsePauli::from_str("IX").unwrap(),
    );
    sim.measure(&SparsePauli::from_str("IZ").unwrap());

    let outcomes = sim.sample(100);
    for shot in 0..100 {
        assert!(
            !outcomes.row(shot).index(0),
            "Z_0 commutes with control, so no X_1 applied, Z_1 should not flip at shot {shot}"
        );
    }
}

// ========== AllocateRandomBit Test ==========

#[test]
fn allocate_random_bit_increments_outcome_counter() {
    let mut sim = FaultySimulation::new();
    Simulation::allocate_random_bit(&mut sim);
    sim.measure(&SparsePauli::from_str("Z").unwrap());

    let outcomes = sim.sample(100);
    assert_eq!(outcomes.column_count(), 2);

    for shot in 0..100 {
        assert!(!outcomes.row(shot).index(1), "Z measurement should be 0 at shot {shot}");
    }
}

// ========== Two-Qubit Depolarizing Tests ==========

#[test]
fn two_qubit_depolarizing_produces_correlated_errors() {
    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault::depolarizing(&[0, 1], 1.0));
    sim.measure(&SparsePauli::from_str("ZZ").unwrap());

    let mut rng = SmallRng::seed_from_u64(42);
    let outcomes = sim.sample_with_rng(10_000, &mut rng);

    let flip_count: usize = (0..10_000).filter(|&shot| outcomes.row(shot).index(0)).count();
    #[allow(clippy::cast_precision_loss)]
    let flip_rate = flip_count as f64 / 10_000.0;

    // 2-qubit depolarizing: 15 Paulis total
    // ZZ measurement flips when X appears on odd number of qubits: 8 out of 15
    // Expected flip rate ≈ 8/15 ≈ 0.533
    assert!(
        (0.48..0.58).contains(&flip_rate),
        "2-qubit depolarizing ZZ flip rate {flip_rate:.4} outside expected range [0.48, 0.58]"
    );
}

// ========== Pauli Instruction Frame Test ==========

#[test]
fn pauli_instruction_does_not_affect_frame() {
    let mut sim_with_pauli = FaultySimulation::new();
    sim_with_pauli.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::single(SparsePauli::from_str("X").unwrap()),
        correlation_id: None,
        condition: None,
    });
    sim_with_pauli.pauli(&SparsePauli::from_str("Z").unwrap());
    sim_with_pauli.measure(&SparsePauli::from_str("Z").unwrap());

    let mut sim_without_pauli = FaultySimulation::new();
    sim_without_pauli.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::single(SparsePauli::from_str("X").unwrap()),
        correlation_id: None,
        condition: None,
    });
    sim_without_pauli.measure(&SparsePauli::from_str("Z").unwrap());

    let outcomes_with = sim_with_pauli.sample(100);
    let outcomes_without = sim_without_pauli.sample(100);

    for shot in 0..100 {
        assert_eq!(
            outcomes_with.row(shot).index(0),
            outcomes_without.row(shot).index(0),
            "Pauli instruction should not change frame outcome at shot {shot}"
        );
    }
}

// ========== Noise Distribution Tests ==========

#[test]
fn uniform_distribution_samples_from_list() {
    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::uniform(vec![
            SparsePauli::from_str("X").unwrap(),
            SparsePauli::from_str("Y").unwrap(),
        ]),
        correlation_id: None,
        condition: None,
    });
    sim.measure(&SparsePauli::from_str("Z").unwrap());

    let outcomes = sim.sample(100);
    for shot in 0..100 {
        assert!(
            outcomes.row(shot).index(0),
            "Uniform(X,Y) should always flip Z measurement at shot {shot}"
        );
    }
}

#[test]
fn weighted_distribution_respects_weights() {
    let mut sim = FaultySimulation::new();
    sim.apply_fault(PauliFault {
        probability: 1.0,
        distribution: PauliDistribution::weighted(vec![
            (SparsePauli::from_str("X").unwrap(), 0.9),
            (SparsePauli::from_str("Z").unwrap(), 0.1),
        ]),
        correlation_id: None,
        condition: None,
    });
    sim.measure(&SparsePauli::from_str("Z").unwrap());

    let mut rng = SmallRng::seed_from_u64(42);
    let outcomes = sim.sample_with_rng(10_000, &mut rng);

    let flip_count: usize = (0..10_000).filter(|&shot| outcomes.row(shot).index(0)).count();
    #[allow(clippy::cast_precision_loss)]
    let flip_rate = flip_count as f64 / 10_000.0;

    // Expected ≈ 0.9
    assert!(
        (0.85..0.95).contains(&flip_rate),
        "Weighted(90% X, 10% Z) flip rate {flip_rate:.4} outside expected range [0.85, 0.95]"
    );
}
