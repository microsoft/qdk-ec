use std::hint::black_box;
use std::time::Duration;

use binar::vec::AlignedBitVec;
use binar::{Bitwise, BitwiseMut, BitwisePairMut};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

fn patterned_vector(length: usize) -> AlignedBitVec {
    (0..length).map(|index| (index * 37) % 67 < 33).collect()
}

fn interval_copy_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("interval_copy");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(1));
    for (offset, length) in [
        (0, 1),
        (1, 1),
        (1, 2),
        (8, 8),
        (32, 32),
        (63, 65),
        (64, 128),
        (65, 128),
        (511, 513),
        (512, 1024),
        (513, 1024),
        (31, 4096),
    ] {
        let source = patterned_vector(offset + length);
        let mut destination = AlignedBitVec::ones(length + 65);
        for packed in [false, true] {
            group.bench_function(
                BenchmarkId::new(if packed { "packed" } else { "scalar" }, format!("{offset}_{length}")),
                |bencher| {
                    bencher.iter(|| {
                        let source = black_box(&source);
                        let destination = black_box(&mut destination);
                        let offset = black_box(offset);
                        let length = black_box(length);
                        if packed {
                            destination.assign_from_interval(source, offset, length);
                        } else {
                            for index in 0..length {
                                destination.assign_index(index, source.index(offset + index));
                            }
                        }
                        black_box(destination);
                    });
                },
            );
        }
    }
    group.finish();
}

fn half_swap_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("half_swap");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(250));
    group.measurement_time(Duration::from_secs(1));
    for half_length in [1, 2, 8, 32, 63, 64, 65, 128, 257, 512] {
        let source = patterned_vector(2 * half_length);
        for packed in [false, true] {
            group.bench_function(
                BenchmarkId::new(if packed { "packed" } else { "scalar" }, half_length),
                |bencher| {
                    bencher.iter(|| {
                        let source = black_box(&source);
                        let half_length = black_box(half_length);
                        let result: AlignedBitVec = if packed {
                            let mut result = AlignedBitVec::zeros(2 * half_length);
                            result.assign_from_interval(source, half_length, half_length);
                            result.assign_with_offset(source, half_length, half_length);
                            result
                        } else {
                            (half_length..2 * half_length)
                                .chain(0..half_length)
                                .map(|index| source.index(index))
                                .collect()
                        };
                        black_box(result)
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, interval_copy_benchmark, half_swap_benchmark);
criterion_main!(benches);
