//! The benchmark must resolve the real surface-code program and retire its state.

#[path = "../benches/support/mod.rs"]
mod support;

use deq_runtime::bin;
use deq_runtime::jit::benchmark::{CompilerHarness, PreparedProgram};
use deq_runtime::jit::jit_compiler::{DEFAULT_CACHE_LIMIT_RATIO, cache_capacity};
use deq_runtime::jit::{JitInstruction, JitLibrary};
use std::sync::OnceLock;
use std::time::Duration;

fn fixture() -> &'static support::Fixture {
    static FIXTURE: OnceLock<support::Fixture> = OnceLock::new();
    FIXTURE.get_or_init(|| support::load().expect("benchmark fixture must decode and match its checksum"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cached_and_uncached_compilation_cover_all_checks_errors_and_retirement() {
    let fixture = fixture();
    assert_eq!(
        fixture.manifest_crc32,
        crc32fast::hash(&std::fs::read(support::fixture_directory().join("manifest.json")).unwrap())
    );
    assert_eq!(fixture.manifest.compiler_threads, 24);
    assert_eq!(fixture.manifest.maximum_mean_epoch_ms.to_bits(), 390.0_f64.to_bits());
    let (expected, program) = &fixture.programs[0];
    assert_eq!(expected.name, "d7_w4_l2");
    assert_eq!(fixture.programs[1].0.name, "d7_w200_l100");
    assert_eq!(fixture.programs[1].0.gadgets, 110_150);
    assert_eq!(fixture.programs[1].0.epochs, 302);
    assert_eq!(fixture.library.gadget_types.len(), 25);
    assert_eq!(
        cache_capacity(fixture.library.gadget_types.len(), DEFAULT_CACHE_LIMIT_RATIO),
        250
    );
    for ratio in [DEFAULT_CACHE_LIMIT_RATIO, 0] {
        let compiler = CompilerHarness::new(fixture.library.clone(), ratio).await.unwrap();
        for _ in 0..3 {
            compiler.reset().await;
            let summary = tokio::time::timeout(
                Duration::from_secs(30),
                compiler.run(PreparedProgram::new(program.clone()).unwrap()),
            )
            .await
            .expect("future-dependent compilation must not deadlock")
            .unwrap();
            assert_eq!(summary.gadgets, expected.gadgets);
            assert_eq!(summary.checks, expected.checks);
            assert_eq!(summary.errors, expected.errors);
            assert!(summary.peak_live_gadgets > 0);
            assert_eq!(summary.retained_gadgets, 0);
        }
    }
}

#[test]
fn missing_generated_fixtures_explain_how_to_prepare_them() {
    let directory = tempfile::tempdir().unwrap();
    let error = support::load_from(directory.path()).err().expect("fixtures are absent");
    assert!(
        error
            .to_string()
            .contains("python deq/documents/tutorial/examples/jit-basics/generate_jit_benchmark_fixture.py")
    );
}

#[test]
fn malformed_dependency_order_is_rejected_before_measurement() {
    let instruction = |gid: u64, predecessor: Option<u64>| JitInstruction {
        gadget: Some(bin::Gadget {
            gid,
            connectors: predecessor
                .map(|gid| bin::gadget::Connector { gid, port: 0 })
                .into_iter()
                .collect(),
            ..Default::default()
        }),
        ..Default::default()
    };
    assert!(PreparedProgram::new(vec![JitInstruction::default()]).is_err());
    assert!(PreparedProgram::new(vec![instruction(0, None)]).is_err());
    assert!(PreparedProgram::new(vec![instruction(1, None), instruction(1, None)]).is_err());
    assert!(PreparedProgram::new(vec![instruction(2, Some(1))]).is_err());
}

#[tokio::test]
async fn executable_instructions_are_not_hidden_in_untimed_library_setup() {
    let library = JitLibrary {
        program: vec![JitInstruction::default()],
        ..Default::default()
    };
    for ratio in [DEFAULT_CACHE_LIMIT_RATIO, 0] {
        assert!(CompilerHarness::new(library.clone(), ratio).await.is_err());
    }
}
