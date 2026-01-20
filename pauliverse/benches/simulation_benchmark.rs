//! Criterion benchmarks for `FaultySimulation`.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use paulimer::pauli::{Pauli, SparsePauli};
use pauliverse::faulty_simulation::FaultySimulation;
use pauliverse::noise::PauliFault;
use pauliverse::Simulation;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::str::FromStr;

const SEED: u64 = 42;

/// Build a repetition code simulation (distance d, r rounds) with optional depolarizing noise.
fn repetition_code_simulation(distance: usize, rounds: usize, p_error: Option<f64>) -> FaultySimulation {
    let mut sim = FaultySimulation::new();
    for _ in 0..rounds {
        for i in 0..distance - 1 {
            let mut pauli_chars = vec!['I'; distance];
            pauli_chars[i] = 'Z';
            pauli_chars[i + 1] = 'Z';
            let pauli_str: String = pauli_chars.into_iter().collect();
            let observable = SparsePauli::from_str(&pauli_str).expect("Valid Pauli");

            if let Some(p) = p_error {
                let qubits: Vec<usize> = observable.support().collect();
                sim.apply_fault(PauliFault::depolarizing(&qubits, p));
            }

            sim.measure(&observable);
        }
    }
    sim
}

fn construction_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("construction");

    for (distance, rounds) in [(5, 3), (11, 5)] {
        let param = format!("d{distance}_r{rounds}");

        group.bench_function(BenchmarkId::new("FaultySimulation", &param), |bencher| {
            bencher.iter(|| repetition_code_simulation(distance, rounds, None));
        });
    }
    group.finish();
}

fn sampling_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("sampling");
    group.sample_size(20);

    let faulty = repetition_code_simulation(11, 5, Some(0.01));

    for shots in [10_000, 100_000] {
        group.bench_with_input(
            BenchmarkId::new("FaultySimulation", shots),
            &shots,
            |bencher, &shots| {
                bencher.iter_with_setup(
                    || SmallRng::seed_from_u64(SEED),
                    |mut rng| faulty.sample_with_rng(shots, &mut rng),
                );
            },
        );
    }
    group.finish();
}

fn error_rate_scaling_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("error_rate_scaling");
    group.sample_size(20);

    let shots = 50_000;

    for p_error in [0.001, 0.01, 0.1] {
        let param = format!("p{p_error}");
        let faulty = repetition_code_simulation(11, 5, Some(p_error));

        group.bench_with_input(BenchmarkId::new("FaultySimulation", &param), &p_error, |bencher, &_| {
            bencher.iter_with_setup(
                || SmallRng::seed_from_u64(SEED),
                |mut rng| faulty.sample_with_rng(shots, &mut rng),
            );
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    construction_benchmark,
    sampling_benchmark,
    error_rate_scaling_benchmark,
);
criterion_main!(benches);
