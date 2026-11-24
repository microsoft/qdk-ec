extern crate criterion;
use std::hint::black_box;
use std::time::Duration;

use binar::vec::AlignedBitVec;
use binar::{BitBlock, BitLength, BitVec};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use paulimer::pauli::{DensePauli, Pauli, PauliUnitary};
use paulimer::pauli_group::as_sparse_pauli;
use rand::prelude::*;

pub fn multiply_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("Pauli::multiply");
    for size in [100usize, 1000usize, 10000usize] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |bencher, size| {
            bencher.iter_batched(
                || (random_pauli(*size), random_pauli(*size)),
                |mut pair| pair.0 *= &pair.1,
                BatchSize::LargeInput,
            );
        });
    }
    group.finish();
}

type BitPredicate = fn(usize) -> bool;

fn patterned_pauli(length: usize, predicate: BitPredicate) -> DensePauli {
    let x_bits: AlignedBitVec = (0..length).map(predicate).collect();
    let z_bits: AlignedBitVec = (0..length).map(|index| predicate(index + 63)).collect();
    DensePauli::from_bits(x_bits, z_bits, 3u8)
}

fn patterned_symplectic_bits(length: usize, predicate: BitPredicate) -> AlignedBitVec {
    (0..2 * length)
        .map(|index| predicate(if index < length { index } else { index - length + 63 }))
        .collect()
}

fn support_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("pauli_support");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(1));
    let patterns: [(&str, BitPredicate); 4] = [
        ("zero", |_| false),
        ("sparse", |index| index.is_multiple_of(257)),
        ("mixed", |index| (index * 37) % 67 < 33),
        ("dense", |_| true),
    ];
    for block_count in [1, 16] {
        let length = block_count * BitBlock::BLOCK_BIT_LEN;
        let mapping: Vec<_> = (0..length).map(|index| 2 * (length - index) + 1).collect();
        for (density, predicate) in patterns {
            let pauli = patterned_pauli(length, predicate);
            let combined = patterned_symplectic_bits(length, predicate);
            group.bench_with_input(
                BenchmarkId::new(format!("{density}/sum"), block_count),
                &pauli,
                |bencher, pauli| bencher.iter(|| black_box(pauli).support().sum::<usize>()),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{density}/collect"), block_count),
                &pauli,
                |bencher, pauli| bencher.iter(|| black_box(pauli).support().collect::<Vec<_>>()),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{density}/to_sparse"), block_count),
                &combined,
                |bencher, combined| {
                    bencher.iter(|| as_sparse_pauli(&black_box(combined).as_view(), black_box(3), black_box(&mapping)));
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, multiply_benchmark, support_benchmark);
criterion_main!(benches);

fn random_pauli(dimension: usize) -> PauliUnitary<BitVec, u8> {
    let x_bits = std::iter::from_fn(move || Some(rand::rng().random::<bool>())).take(dimension);
    let z_bits = std::iter::from_fn(move || Some(rand::rng().random::<bool>())).take(dimension);
    PauliUnitary::from_bits(x_bits.collect(), z_bits.collect(), 0u8)
}
