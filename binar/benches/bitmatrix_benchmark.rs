use binar::{BitMatrix, BitVec, BitwiseMut, EchelonForm};
use criterion::{BatchSize, BenchmarkId, Criterion, criterion_group, criterion_main};
use rand::prelude::*;

struct Parameters((f64, usize));

pub fn echelonize_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("Bitmatrix::echelonize");
    for sparsity in [0.5, 0.1, 0.01, 0.001] {
        for size in [100usize, 1000usize, 10000usize] {
            group.sample_size(10);
            let parameters = Parameters((sparsity, size));
            group.bench_with_input(
                BenchmarkId::from_parameter(&parameters),
                &parameters,
                |bencher, parameters| {
                    let (sparsity, size) = parameters.0;
                    bencher.iter_batched(
                        || random_bitmatrix(size, size, sparsity),
                        |mut matrix| matrix.echelonize(),
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

pub fn transpose_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("Bitmatrix::transpose");
    for sparsity in [0.5, 0.1, 0.01, 0.001] {
        for size in [64usize, 100usize, 1000usize, 10000usize] {
            group.sample_size(10);
            let parameters = Parameters((sparsity, size));
            group.bench_with_input(
                BenchmarkId::from_parameter(&parameters),
                &parameters,
                |bencher, parameters| {
                    let (sparsity, size) = parameters.0;
                    bencher.iter_batched(
                        || random_bitmatrix(size, size, sparsity),
                        |matrix| matrix.transposed(),
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

pub fn dot_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("Bitmatrix::mul");
    for sparsity in [0.5, 0.01] {
        for size in [100usize, 1000usize, 10000usize] {
            group.sample_size(10);
            let parameters = Parameters((sparsity, size));
            group.bench_with_input(
                BenchmarkId::from_parameter(&parameters),
                &parameters,
                |bencher, parameters| {
                    let (sparsity, size) = parameters.0;
                    bencher.iter_batched(
                        || {
                            let matrix_a = random_bitmatrix(size, size, sparsity);
                            let matrix_b = random_bitmatrix(size, size, sparsity);
                            (matrix_a, matrix_b)
                        },
                        |(matrix_a, matrix_b)| &matrix_a * &matrix_b,
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

pub fn solve_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("EchelonForm::solve");
    for sparsity in [0.5, 0.1, 0.01] {
        // Skip very sparse for now
        for size in [100usize, 1000usize] {
            // Smaller sizes to avoid the bug
            group.sample_size(10);
            let parameters = Parameters((sparsity, size));
            group.bench_with_input(
                BenchmarkId::from_parameter(&parameters),
                &parameters,
                |bencher, parameters| {
                    let (sparsity, size) = parameters.0;
                    bencher.iter_batched(
                        || {
                            let matrix = random_bitmatrix(size, size, sparsity);
                            let echelon_form = EchelonForm::new(matrix.clone());
                            let target = random_bitvec(matrix.row_count(), sparsity);
                            (echelon_form, target)
                        },
                        |(echelon_form, target)| echelon_form.solve(&target.as_view()),
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }
    group.finish();
}

impl std::fmt::Display for Parameters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (sparsity, size) = self.0;
        write!(f, "(sparsity={sparsity}, size={size})")?;
        Ok(())
    }
}

criterion_group!(
    benches,
    echelonize_benchmark,
    dot_benchmark,
    transpose_benchmark,
    solve_benchmark,
    mul_transpose_benchmark,
);
criterion_main!(benches);

pub fn mul_transpose_benchmark(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("Bitmatrix::mul_transpose");
    for size in [100usize, 1000usize, 5000usize] {
        group.sample_size(10);
        let parameters = Parameters((0.5, size));
        group.bench_with_input(
            BenchmarkId::from_parameter(&parameters),
            &parameters,
            |bencher, parameters| {
                let (_sparsity, size) = parameters.0;
                bencher.iter_batched(
                    || {
                        let matrix_a = random_bitmatrix(size, size, 0.5);
                        let matrix_b = random_bitmatrix(size, size, 0.5);
                        (matrix_a, matrix_b)
                    },
                    |(matrix_a, matrix_b)| matrix_a.mul_transpose(&matrix_b),
                    BatchSize::SmallInput,
                );
            },
        );
    }
    group.finish();
}

fn random_bitmatrix(row_count: usize, column_count: usize, sparsity: f64) -> BitMatrix {
    let mut matrix = BitMatrix::with_shape(row_count, column_count);
    let mut bits = std::iter::from_fn(move || Some(thread_rng().gen_bool(sparsity)));
    for row_index in 0..row_count {
        for column_index in 0..column_count {
            matrix.set((row_index, column_index), bits.next().expect("boom"));
        }
    }
    matrix
}

fn random_bitvec(length: usize, sparsity: f64) -> BitVec {
    let mut bitvec = BitVec::zeros(length);
    let mut bits = std::iter::from_fn(move || Some(thread_rng().gen_bool(sparsity)));
    for index in 0..length {
        bitvec.assign_index(index, bits.next().expect("boom"));
    }
    bitvec
}
