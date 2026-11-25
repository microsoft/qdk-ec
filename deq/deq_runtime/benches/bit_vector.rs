use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use deq_runtime::misc::bit_vector::{append, bit_vector_len, extend_num_bits, get_bit, set_bit, slice};
use deq_runtime::util::BitVector;
use std::hint::black_box;

fn scalar_slice(source: &BitVector, start: usize, length: usize) -> BitVector {
    let size = u64::try_from(length).unwrap();
    let mut result = BitVector {
        size,
        data: vec![0; bit_vector_len(size)],
    };
    for index in 0..length {
        if get_bit(source, u64::try_from(start + index).unwrap()) {
            set_bit(&mut result, u64::try_from(index).unwrap(), true);
        }
    }
    result
}

fn scalar_append(destination: &mut BitVector, source: &BitVector) {
    let start = destination.size;
    extend_num_bits(destination, source.size);
    for index in 0..source.size {
        if get_bit(source, index) {
            set_bit(destination, start + index, true);
        }
    }
}

fn packed_bits(criterion: &mut Criterion) {
    let size = 545_000;
    let source = BitVector {
        size: u64::try_from(size + 8).unwrap(),
        data: (0..bit_vector_len(u64::try_from(size + 8).unwrap()))
            .map(|index| u8::try_from(index % 251).unwrap())
            .collect(),
    };
    let mut group = criterion.benchmark_group("bit_vector_slice");
    group.throughput(Throughput::Bytes(
        u64::try_from(bit_vector_len(u64::try_from(size).unwrap())).unwrap(),
    ));
    for offset in [0, 3, 7] {
        assert_eq!(slice(&source, offset, size), scalar_slice(&source, offset, size));
        group.bench_with_input(BenchmarkId::new("packed", offset), &offset, |bench, &offset| {
            bench.iter(|| black_box(slice(black_box(&source), offset, size)));
        });
        group.bench_with_input(BenchmarkId::new("scalar", offset), &offset, |bench, &offset| {
            bench.iter(|| black_box(scalar_slice(black_box(&source), offset, size)));
        });
    }
    group.finish();

    let appended = slice(&source, 0, size);
    let mut group = criterion.benchmark_group("bit_vector_append");
    group.throughput(Throughput::Bytes(u64::try_from(appended.data.len()).unwrap()));
    for offset in [0, 3] {
        let initial = slice(&source, 0, offset);
        for (name, operation) in [
            ("packed", append as fn(&mut BitVector, &BitVector)),
            ("scalar", scalar_append as fn(&mut BitVector, &BitVector)),
        ] {
            group.bench_function(BenchmarkId::new(name, offset), |bench| {
                bench.iter_batched(
                    || initial.clone(),
                    |mut destination| {
                        operation(&mut destination, black_box(&appended));
                        black_box(destination)
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }
    group.finish();
}

criterion_group!(benches, packed_bits);
criterion_main!(benches);
