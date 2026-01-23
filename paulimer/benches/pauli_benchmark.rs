extern crate criterion;
use binar::BitVec;
use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion};
use paulimer::pauli::PauliUnitary;
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

criterion_group!(benches, multiply_benchmark);
criterion_main!(benches);

fn random_pauli(dimension: usize) -> PauliUnitary<BitVec, u8> {
    let x_bits = std::iter::from_fn(move || Some(thread_rng().gen::<bool>())).take(dimension);
    let z_bits = std::iter::from_fn(move || Some(thread_rng().gen::<bool>())).take(dimension);
    PauliUnitary::from_bits(x_bits.collect(), z_bits.collect(), 0u8)
}
