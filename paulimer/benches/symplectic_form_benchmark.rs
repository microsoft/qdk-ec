extern crate criterion;
use binar::{BitwiseMut, IndexSet};
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use paulimer::pauli::{Pauli, SparsePauli};
use paulimer::pauli_group::symplectic_form_of;
use rand::prelude::*;

/// # Panics
/// Will panic if benchmarking fails
pub fn symplectic_form_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("symplectic_form_of");

    // Configure for faster benchmarking
    group.sample_size(20); // Reduce from default 100 to 20 samples
    group.measurement_time(std::time::Duration::from_secs(2)); // Reduce measurement time

    // Test different numbers of generators using realistic Pauli structures
    for num_generators in [64usize, 512usize, 1024usize] {
        group.bench_with_input(
            BenchmarkId::new("permuted_basis", num_generators),
            &num_generators,
            |bencher, &num_generators| {
                bencher.iter_batched(
                    || generate_permuted_basis_paulis(num_generators),
                    |generators| symplectic_form_of(&generators),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    // Test random products of basis elements
    for num_generators in [64usize, 512usize, 1024usize] {
        group.bench_with_input(
            BenchmarkId::new("random_products", num_generators),
            &num_generators,
            |bencher, &num_generators| {
                bencher.iter_batched(
                    || generate_random_product_paulis(num_generators),
                    |generators| symplectic_form_of(&generators),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    // // Test different qubit counts with fixed number of generators (reduced set)
    // for qubit_count in [10usize, 20usize] {
    //     group.bench_with_input(
    //         BenchmarkId::new("qubit_count", qubit_count),
    //         &qubit_count,
    //         |bencher, &qubit_count| {
    //             bencher.iter_batched(
    //                 || generate_random_paulis(8, qubit_count), // 8 generators
    //                 |generators| symplectic_form_of(&generators),
    //                 BatchSize::SmallInput,
    //             );
    //         },
    //     );
    // }

    // // Test different weights (sparsity) of Pauli operators (reduced set)
    // for max_weight in [1usize, 5usize] {
    //     group.bench_with_input(
    //         BenchmarkId::new("max_weight", max_weight),
    //         &max_weight,
    //         |bencher, &max_weight| {
    //             bencher.iter_batched(
    //                 || generate_sparse_paulis(8, 20, max_weight), // 8 generators, 20 qubits
    //                 |generators| symplectic_form_of(&generators),
    //                 BatchSize::SmallInput,
    //             );
    //         },
    //     );
    // }

    group.finish();
}

#[cfg(not(unix))]
criterion_group!(benches, symplectic_form_benchmark);

#[cfg(unix)]
criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = symplectic_form_benchmark
}

criterion_main!(benches);

/// Generate a full X/Z basis for the given number of qubits
fn generate_full_basis(num_qubits: usize) -> Vec<SparsePauli> {
    let mut basis = Vec::with_capacity(2 * num_qubits);

    // Add X operators for each qubit
    for qubit_id in 0..num_qubits {
        let mut x_bits = IndexSet::new();
        x_bits.assign_index(qubit_id, true);
        basis.push(SparsePauli::from_bits(x_bits, IndexSet::new(), 0));
    }

    // Add Z operators for each qubit
    for qubit_id in 0..num_qubits {
        let mut z_bits = IndexSet::new();
        z_bits.assign_index(qubit_id, true);
        basis.push(SparsePauli::from_bits(IndexSet::new(), z_bits, 0));
    }

    basis
}

/// Generate permuted basis Paulis - randomly select and permute from full X/Z basis
fn generate_permuted_basis_paulis(count: usize) -> Vec<SparsePauli> {
    let num_qubits = (count / 2).max(1) + 2; // Ensure we have enough qubits
    let full_basis = generate_full_basis(num_qubits);

    let mut rng = thread_rng();
    let mut selected: Vec<SparsePauli> = full_basis
        .choose_multiple(&mut rng, count.min(full_basis.len()))
        .cloned()
        .collect();

    // Randomly permute the selected basis elements
    selected.shuffle(&mut rng);

    // Add random phases to make it more realistic
    for pauli in &mut selected {
        let x_bits = pauli.x_bits().clone();
        let z_bits = pauli.z_bits().clone();
        let new_pauli = SparsePauli::from_bits(x_bits, z_bits, rng.gen::<u8>() % 4);
        *pauli = new_pauli;
    }

    selected
}

/// Generate random products of basis elements
fn generate_random_product_paulis(count: usize) -> Vec<SparsePauli> {
    let num_qubits = (count / 4).max(1); // Ensure we have enough qubits
    let full_basis = generate_full_basis(num_qubits);

    let mut rng = thread_rng();
    let mut products = Vec::with_capacity(count);

    for _ in 0..count {
        // Start with identity
        let mut product = SparsePauli::from_bits(IndexSet::new(), IndexSet::new(), 0);

        // Multiply with random number of basis elements (1-4 for reasonable complexity)
        let num_factors = rng.gen_range(1..=4.min(full_basis.len()));
        let factors = full_basis.choose_multiple(&mut rng, num_factors);

        for factor in factors {
            product = &product * factor;
        }

        products.push(product);
    }

    products
}

//
//
//
//
//
// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.
#[cfg(unix)]
use pprof::flamegraph::Options as FlamegraphOptions;

#[cfg(unix)]
use criterion::profiler::Profiler;
#[cfg(unix)]
use pprof::ProfilerGuard;

#[cfg(unix)]
use std::fs::File;
#[cfg(unix)]
use std::os::raw::c_int;
#[cfg(unix)]
use std::path::Path;

#[cfg(unix)]
#[allow(clippy::large_enum_variant)]
pub enum Output<'a> {
    Flamegraph(Option<FlamegraphOptions<'a>>),
}
#[cfg(unix)]
pub struct PProfProfiler<'a, 'b> {
    frequency: c_int,
    output: Output<'b>,
    active_profiler: Option<ProfilerGuard<'a>>,
}
#[cfg(unix)]
impl<'b> PProfProfiler<'_, 'b> {
    #[must_use]
    pub fn new(frequency: c_int, output: Output<'b>) -> Self {
        Self {
            frequency,
            output,
            active_profiler: None,
        }
    }
}
#[cfg(unix)]
impl Profiler for PProfProfiler<'_, '_> {
    fn start_profiling(&mut self, _benchmark_id: &str, _benchmark_dir: &Path) {
        self.active_profiler = Some(ProfilerGuard::new(self.frequency).unwrap());
    }

    fn stop_profiling(&mut self, _benchmark_id: &str, benchmark_dir: &Path) {
        std::fs::create_dir_all(benchmark_dir).unwrap();

        let filename = match self.output {
            Output::Flamegraph(_) => "flamegraph.svg",
        };
        let output_path = benchmark_dir.join(filename);
        let output_file = File::create(&output_path)
            .unwrap_or_else(|_| panic!("File system error while creating {}", output_path.display()));

        if let Some(profiler) = self.active_profiler.take() {
            match &mut self.output {
                Output::Flamegraph(options) => {
                    let default_options = &mut FlamegraphOptions::default();
                    let options = options.as_mut().unwrap_or(default_options);

                    profiler
                        .report()
                        .build()
                        .unwrap()
                        .flamegraph_with_options(output_file, options)
                        .expect("Error while writing flamegraph");
                }
            }
        }
    }
}
