use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use paulimer::clifford::{
    Clifford, CliffordMutable, CliffordUnitary, clifford_residue_rank, clifford_to_transvections_signed,
    try_clifford_to_transvections_minimal,
};
use paulimer::{Pauli, SparsePauli};
use rand::{SeedableRng, rngs::StdRng};

fn synthesis(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("clifford_synthesis");
    let mut random = StdRng::seed_from_u64(54);
    for width in [6, 12, 24, 48] {
        let clifford = CliffordUnitary::random(width, &mut random);
        group.bench_with_input(BenchmarkId::new("residue_rank", width), &clifford, |bencher, input| {
            bencher.iter(|| clifford_residue_rank(black_box(input)));
        });
        group.bench_with_input(BenchmarkId::new("signed_greedy", width), &clifford, |bencher, input| {
            bencher.iter(|| clifford_to_transvections_signed(black_box(input)).unwrap());
        });
        if width <= 12 {
            assert!(try_clifford_to_transvections_minimal(&clifford, 10_000).is_ok());
            group.bench_with_input(BenchmarkId::new("bounded_exact", width), &clifford, |bencher, input| {
                bencher.iter(|| try_clifford_to_transvections_minimal(black_box(input), 10_000).unwrap());
            });
        }
    }
    group.finish();
}

fn frame_updates(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("clifford_frame");
    for width in [128, 512, 1727] {
        let mut frame = CliffordUnitary::identity(width);
        for qubit in 1..width {
            frame.left_mul_hadamard(qubit);
            frame.left_mul_cx(qubit - 1, qubit);
        }
        let observable = SparsePauli::z(width - 1, width);
        group.bench_function(BenchmarkId::new("preimage", width), |bencher| {
            bencher.iter(|| frame.preimage(black_box(&observable)));
        });
        group.bench_function(BenchmarkId::new("cx", width), |bencher| {
            bencher.iter_batched(
                || frame.clone(),
                |mut copy| {
                    copy.left_mul_cx(0, width - 1);
                    black_box(copy)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, synthesis, frame_updates);
criterion_main!(benches);
