use std::hint::black_box;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use paulimer::CliffordUnitary;
use paulimer::clifford::{Clifford, CliffordMutable, clifford_fixed_space, clifford_to_transvections};
use rand::{SeedableRng, rngs::StdRng};

fn transvection_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("transvections");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(1));
    for qubit_count in [1, 2, 8, 32, 128] {
        let identity = CliffordUnitary::identity(qubit_count);
        let mut hadamards = identity.clone();
        let mut swaps = identity.clone();
        for qubit in 0..qubit_count {
            hadamards.left_mul_hadamard(qubit);
        }
        for qubit in (0..qubit_count.saturating_sub(1)).step_by(2) {
            swaps.left_mul_swap(qubit, qubit + 1);
        }
        let random = CliffordUnitary::random(qubit_count, &mut StdRng::seed_from_u64(118));
        for (family, clifford) in [
            ("identity", identity),
            ("hadamards", hadamards),
            ("swaps", swaps),
            ("random", random),
        ] {
            group.bench_with_input(
                BenchmarkId::new(format!("greedy_{family}"), qubit_count),
                &clifford,
                |bencher, clifford| bencher.iter(|| clifford_to_transvections(black_box(clifford))),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("fixed_space_{family}"), qubit_count),
                &clifford,
                |bencher, clifford| bencher.iter(|| clifford_fixed_space(black_box(clifford))),
            );
        }
    }
    group.finish();
}

criterion_group!(benches, transvection_benchmark);
criterion_main!(benches);
