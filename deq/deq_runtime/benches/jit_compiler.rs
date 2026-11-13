//! Benchmark the shared JIT compiler path without any decoding or simulation.

mod support;

use criterion::{BenchmarkId, Criterion, SamplingMode, Throughput};
use deq_runtime::jit::benchmark::{CompilationSummary, CompilerHarness, PreparedProgram};
use deq_runtime::jit::jit_compiler::{DEFAULT_CACHE_LIMIT_RATIO, cache_capacity};
use serde_json::json;
use std::collections::BTreeSet;
use std::hint::black_box;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant, SystemTime};
use support::CaseMetadata;
use tokio::runtime::Runtime;

const COMPILATION_TIMEOUT: Duration = Duration::from_secs(360);

fn check(summary: &CompilationSummary, expected: &CaseMetadata) {
    assert_eq!(summary.gadgets, expected.gadgets, "every instruction must compile");
    assert_eq!(summary.checks, expected.checks, "every check must be expanded");
    assert_eq!(summary.errors, expected.errors, "every deferred error model must resolve");
    assert_eq!(summary.retained_gadgets, 0, "completed compiler state must be collected");
}

fn run(
    runtime: &Runtime,
    compiler: &CompilerHarness,
    program: PreparedProgram,
    expected: &CaseMetadata,
) -> Result<Duration, tokio::time::error::Elapsed> {
    let started = Instant::now();
    let summary = runtime
        .block_on(async { tokio::time::timeout(COMPILATION_TIMEOUT, compiler.run(program)).await })?
        .expect("JIT workload failed");
    let elapsed = started.elapsed();
    check(&summary, expected);
    black_box(summary);
    Ok(elapsed)
}

fn write_provenance(output: &Path, metadata: &serde_json::Value) {
    std::fs::create_dir_all(output).expect("create Criterion output directory");
    std::fs::write(
        output.join("jit_compiler_run.json"),
        serde_json::to_vec_pretty(metadata).expect("encode benchmark metadata"),
    )
    .expect("write benchmark provenance");
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
    let mut provenance = json!({
        "schema_version": 4,
        "started_unix_seconds": started_unix_seconds,
        "executed": [],
        "timed_out": [],
        "timeout_seconds": COMPILATION_TIMEOUT.as_secs(),
        "cache_limit_ratio": DEFAULT_CACHE_LIMIT_RATIO,
        "registered_gadget_types": fixture.library.gadget_types.len(),
        "error_template_cache_capacity": cache_capacity(fixture.library.gadget_types.len(), DEFAULT_CACHE_LIMIT_RATIO),
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
    write_provenance(&output, &provenance);
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
        for (mode, ratio) in [("cache_enabled", DEFAULT_CACHE_LIMIT_RATIO), ("cache_disabled", 0)] {
            group.bench_function(BenchmarkId::new(mode, &metadata.name), |bencher| {
                let run_name = format!("{mode}/{}", metadata.name);
                executed.insert(run_name.clone());
                bencher.iter_custom(|iterations| {
                    let mut measured = Duration::ZERO;
                    for _ in 0..iterations {
                        let compiler = runtime
                            .block_on(CompilerHarness::new(fixture.library.clone(), ratio))
                            .expect("register fixture types");
                        let prepared = PreparedProgram::new(program.clone()).expect("prepare dependency notifications");
                        match run(&runtime, &compiler, prepared, metadata) {
                            Ok(elapsed) => measured += elapsed,
                            Err(error) => {
                                executed.remove(&run_name);
                                provenance["executed"] = json!(executed);
                                provenance["timed_out"] = json!([run_name]);
                                write_provenance(&output, &provenance);
                                panic!(
                                    "{run_name}: compilation exceeded {} seconds: {error}; timeout provenance recorded",
                                    COMPILATION_TIMEOUT.as_secs()
                                );
                            }
                        }
                    }
                    measured
                });
            });
        }
    }
    group.finish();
    criterion.final_summary();
    provenance["executed"] = json!(executed);
    write_provenance(&output, &provenance);
}
