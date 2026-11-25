use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use deq_runtime::misc::sync::ReadMostlyRwLock;
use std::hint::black_box;
use std::sync::Arc;
use tokio::sync::{Barrier, RwLock};
use tokio::task::JoinSet;

enum SharedLock {
    Tokio(RwLock<u64>),
    ReadMostly(ReadMostlyRwLock<u64>),
}

async fn read_batch(lock: Arc<SharedLock>, tasks: usize) {
    let ready = Arc::new(Barrier::new(tasks + 1));
    let mut readers = JoinSet::new();
    for _ in 0..tasks {
        let lock = Arc::clone(&lock);
        let ready = Arc::clone(&ready);
        readers.spawn(async move {
            ready.wait().await;
            for _ in 0..512 {
                let value = match lock.as_ref() {
                    SharedLock::Tokio(lock) => *lock.read().await,
                    SharedLock::ReadMostly(lock) => *lock.read().await,
                };
                black_box(value);
            }
        });
    }
    ready.wait().await;
    while let Some(result) = readers.join_next().await {
        result.unwrap();
    }
}

fn shared_readers(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("shared_readers");
    for workers in [1, 8, 24] {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(workers)
            .enable_all()
            .build()
            .unwrap();
        let tasks = workers * 2;
        group.throughput(Throughput::Elements(u64::try_from(tasks * 512).unwrap()));
        for (name, lock) in [
            ("tokio", SharedLock::Tokio(RwLock::new(7))),
            ("read_mostly", SharedLock::ReadMostly(ReadMostlyRwLock::new(7))),
        ] {
            let lock = Arc::new(lock);
            group.bench_function(BenchmarkId::new(name, workers), |bench| {
                bench.iter(|| runtime.block_on(read_batch(Arc::clone(&lock), tasks)));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, shared_readers);
criterion_main!(benches);
