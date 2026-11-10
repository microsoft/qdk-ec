//! Benchmark the shared JIT compiler path without any decoding or simulation.

mod support;

use criterion::{BenchmarkId, Criterion, SamplingMode, Throughput};
use deq_runtime::jit::benchmark::{CompilationSummary, CompilerHarness, PreparedProgram};
use serde_json::json;
use std::collections::BTreeSet;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};
use support::CaseMetadata;
use tokio::runtime::Runtime;

fn check(summary: &CompilationSummary, expected: &CaseMetadata) {
    assert_eq!(summary.gadgets, expected.gadgets, "every instruction must compile");
    assert_eq!(summary.checks, expected.checks, "every check must be expanded");
    assert_eq!(summary.errors, expected.errors, "every deferred error model must resolve");
    assert_eq!(summary.retained_gadgets, 0, "completed compiler state must be collected");
}

fn run(runtime: &Runtime, compiler: &CompilerHarness, program: PreparedProgram, expected: &CaseMetadata) -> Duration {
    let started = Instant::now();
    let summary = runtime
        .block_on(async {
            tokio::time::timeout(Duration::from_mins(3), compiler.run(program))
                .await
                .expect("JIT compilation exceeded the 180-second safety timeout")
        })
        .expect("JIT workload failed");
    let elapsed = started.elapsed();
    check(&summary, expected);
    black_box(summary);
    elapsed
}

fn main() {
    let started_unix_seconds = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .expect("system clock precedes Unix epoch")
        .as_secs_f64();
    let fixture = support::load().expect("load the deterministic surface-code fixture");
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target"), PathBuf::from);
    let output = target.join("criterion");
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(fixture.manifest.compiler_threads)
        .enable_all()
        .build()
        .expect("create the benchmark runtime");
    let mut criterion = Criterion::default().output_directory(&output).configure_from_args();
    let mut group = criterion.benchmark_group("jit_compiler");
    let mut executed = BTreeSet::new();
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(5))
        .sampling_mode(SamplingMode::Flat);
    for (metadata, program) in &fixture.programs {
        group.throughput(Throughput::Elements(metadata.gadgets));
        group.bench_function(BenchmarkId::new("cold_templates", &metadata.name), |bencher| {
            executed.insert(format!("cold_templates/{}", metadata.name));
            bencher.iter_custom(|iterations| {
                let mut measured = Duration::ZERO;
                for _ in 0..iterations {
                    let compiler = runtime
                        .block_on(CompilerHarness::new(fixture.library.clone()))
                        .expect("register fixture types");
                    let prepared = PreparedProgram::new(program.clone()).expect("prepare dependency notifications");
                    measured += run(&runtime, &compiler, prepared, metadata);
                }
                measured
            });
        });
        group.bench_function(BenchmarkId::new("warm_templates", &metadata.name), |bencher| {
            executed.insert(format!("warm_templates/{}", metadata.name));
            let compiler = runtime
                .block_on(CompilerHarness::new(fixture.library.clone()))
                .expect("register fixture types");
            run(
                &runtime,
                &compiler,
                PreparedProgram::new(program.clone()).expect("prepare warmup"),
                metadata,
            );
            bencher.iter_custom(|iterations| {
                let mut measured = Duration::ZERO;
                for _ in 0..iterations {
                    runtime.block_on(compiler.reset());
                    let prepared = PreparedProgram::new(program.clone()).expect("prepare dependency notifications");
                    measured += run(&runtime, &compiler, prepared, metadata);
                }
                measured
            });
        });
    }
    group.finish();
    criterion.final_summary();
    let metadata = json!({
        "schema_version": 1,
        "started_unix_seconds": started_unix_seconds,
        "executed": executed,
        "compiler_threads": fixture.manifest.compiler_threads,
        "maximum_mean_epoch_ms": fixture.manifest.maximum_mean_epoch_ms,
        "optimized": !cfg!(debug_assertions),
        "rustflags": option_env!("RUSTFLAGS").unwrap_or(""),
        "fixture_manifest_crc32": fixture.manifest_crc32,
        "compiled_source_crc32": {
            "Cargo.toml": crc32fast::hash(include_bytes!("../../../Cargo.toml")),
            "Cargo.lock": crc32fast::hash(include_bytes!("../../../Cargo.lock")),
            "deq/deq_runtime/Cargo.toml": crc32fast::hash(include_bytes!("../Cargo.toml")),
            "deq/deq_runtime/src/jit.rs": crc32fast::hash(include_bytes!("../src/jit.rs")),
            "deq/deq_runtime/src/jit/jit_compiler.rs": crc32fast::hash(include_bytes!("../src/jit/jit_compiler.rs")),
            "deq/deq_runtime/src/jit/benchmark.rs": crc32fast::hash(include_bytes!("../src/jit/benchmark.rs")),
            "deq/deq_runtime/benches/jit_compiler.rs": crc32fast::hash(include_bytes!("jit_compiler.rs")),
            "deq/deq_runtime/benches/support/mod.rs": crc32fast::hash(include_bytes!("support/mod.rs"))
        }
    });
    std::fs::create_dir_all(&output).expect("create Criterion output directory");
    std::fs::write(
        output.join("jit_compiler_run.json"),
        serde_json::to_vec_pretty(&metadata).expect("encode benchmark metadata"),
    )
    .expect("write benchmark provenance");
}
