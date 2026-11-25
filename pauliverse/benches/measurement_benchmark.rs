//! Criterion benchmarks for circuits with frequent single-qubit measurements and resets.
//!
//! Each layer of a circuit applies S or √X to every qubit, then CZ to disjoint pairs of qubits, then measures Z on a
//! random subset of qubits and resets each of them. Many of these measurements have a random outcome.

use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main};
use paulimer::UnitaryOp;
use paulimer::pauli::{DensePauli, PauliMutable, SparsePauli};
use paulimer::traits::NeutralElement;
use pauliverse::{
    OutcomeCompleteSimulation, OutcomeFreeSimulation, OutcomeSpecificSimulation, PhasedOutcomeCompleteSimulation,
    Simulation,
};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};

const SEED: u64 = 42;
const LAYER_COUNT: usize = 100;
/// The probability that a qubit is measured and reset in a layer.
const MEASUREMENT_PROBABILITY: f64 = 0.05;
const QUBIT_COUNTS: [usize; 2] = [256, 1024];
/// `PhasedOutcomeCompleteSimulation` takes seconds per circuit at 256 qubits, so it runs on a smaller circuit.
const PHASED_QUBIT_COUNT: usize = 64;
const PAIRINGS: [Pairing; 2] = [Pairing::Ring, Pairing::Random];

/// How the CZ gates of a layer pair up the qubits.
#[derive(Clone, Copy)]
enum Pairing {
    /// Neighbors on a ring, alternating between the two ways to pair them up from one layer to the next.
    Ring,
    /// A new random pairing in each layer.
    Random,
}

impl Pairing {
    fn name(self) -> &'static str {
        match self {
            Pairing::Ring => "ring",
            Pairing::Random => "random_pairs",
        }
    }
}

enum Step {
    Gate(UnitaryOp, Vec<usize>),
    MeasureAndReset { observable: SparsePauli, flip: SparsePauli },
}

fn single_qubit_x(qubit: usize, qubit_count: usize) -> SparsePauli {
    let mut dense = DensePauli::neutral_element_of_size(qubit_count);
    dense.mul_assign_right_x(qubit);
    dense.into()
}

fn single_qubit_z(qubit: usize, qubit_count: usize) -> SparsePauli {
    let mut dense = DensePauli::neutral_element_of_size(qubit_count);
    dense.mul_assign_right_z(qubit);
    dense.into()
}

/// Builds a circuit of [`LAYER_COUNT`] layers on an even number of qubits.
fn layered_circuit(qubit_count: usize, pairing: Pairing) -> Vec<Step> {
    let mut rng = SmallRng::seed_from_u64(SEED);
    let mut circuit = Vec::new();
    let mut qubits: Vec<usize> = (0..qubit_count).collect();
    for layer in 0..LAYER_COUNT {
        for qubit in 0..qubit_count {
            let gate = if rng.random_bool(0.5) {
                UnitaryOp::SqrtZ
            } else {
                UnitaryOp::SqrtX
            };
            circuit.push(Step::Gate(gate, vec![qubit]));
        }
        match pairing {
            Pairing::Ring => {
                for first in (layer % 2..qubit_count).step_by(2) {
                    circuit.push(Step::Gate(
                        UnitaryOp::ControlledZ,
                        vec![first, (first + 1) % qubit_count],
                    ));
                }
            }
            Pairing::Random => {
                qubits.shuffle(&mut rng);
                for pair in qubits.as_chunks::<2>().0 {
                    circuit.push(Step::Gate(UnitaryOp::ControlledZ, pair.to_vec()));
                }
            }
        }
        for qubit in 0..qubit_count {
            if rng.random_bool(MEASUREMENT_PROBABILITY) {
                circuit.push(Step::MeasureAndReset {
                    observable: single_qubit_z(qubit, qubit_count),
                    flip: single_qubit_x(qubit, qubit_count),
                });
            }
        }
    }
    circuit
}

fn run_circuit<S: Simulation>(simulation: &mut S, circuit: &[Step]) {
    for step in circuit {
        match step {
            Step::Gate(gate, support) => simulation.unitary_op(*gate, support),
            Step::MeasureAndReset { observable, flip } => {
                let outcome_id = simulation.measure(observable);
                simulation.conditional_pauli(flip, &[outcome_id], true);
            }
        }
    }
}

fn bench_simulation<S: Simulation>(
    group: &mut BenchmarkGroup<WallTime>,
    simulation_name: &str,
    parameter: &str,
    circuit: &[Step],
    new_simulation: impl Fn() -> S,
) {
    group.bench_function(BenchmarkId::new(simulation_name, parameter), |bencher| {
        bencher.iter(|| {
            let mut simulation = new_simulation();
            run_circuit(&mut simulation, circuit);
            simulation
        });
    });
}

fn measurement_and_reset_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("measurement_and_reset");
    group.sample_size(10);

    for qubit_count in QUBIT_COUNTS {
        for pairing in PAIRINGS {
            let circuit = layered_circuit(qubit_count, pairing);
            let parameter = format!("{}_{qubit_count}", pairing.name());

            bench_simulation(&mut group, "OutcomeSpecificSimulation", &parameter, &circuit, || {
                OutcomeSpecificSimulation::new_with_seeded_random_samples(qubit_count, SEED)
            });
            bench_simulation(&mut group, "OutcomeCompleteSimulation", &parameter, &circuit, || {
                OutcomeCompleteSimulation::new(qubit_count)
            });
            bench_simulation(&mut group, "OutcomeFreeSimulation", &parameter, &circuit, || {
                OutcomeFreeSimulation::new(qubit_count)
            });
        }
    }
    for pairing in PAIRINGS {
        let circuit = layered_circuit(PHASED_QUBIT_COUNT, pairing);
        let parameter = format!("{}_{PHASED_QUBIT_COUNT}", pairing.name());

        bench_simulation(
            &mut group,
            "PhasedOutcomeCompleteSimulation",
            &parameter,
            &circuit,
            || PhasedOutcomeCompleteSimulation::new(PHASED_QUBIT_COUNT),
        );
    }
    group.finish();
}

criterion_group!(benches, measurement_and_reset_benchmark);
criterion_main!(benches);
