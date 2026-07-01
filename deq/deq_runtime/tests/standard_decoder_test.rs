//! Integration tests for the standard decoder test suite.
//!
//! Each decoder is exercised with a closure declaring the expected pass/fail
//! outcome for every (problem, case, path) entry. The closure is the *test's*
//! policy — the `test_problems` and `test_harness` modules do not assume
//! anything about which cases should pass.

use std::sync::Arc;

use deq_runtime::decoder::test_harness::{Outcome, Path, SuiteReport, run_standard_suite};
use deq_runtime::decoder::test_problems::standard_test_problems;
use deq_runtime::decoder::{BlackBoxDecoderClient, DynBlackBoxDecoder, MockDecoder, NaiveDecoder};

type ExpectedPassFn = fn(problem: &str, case: &str, path: Path) -> bool;

/// Compare a [`SuiteReport`] against an expected-pass policy. Panics with a
/// readable summary on any discrepancy.
fn assert_matches_policy(report: &SuiteReport, expected_pass: ExpectedPassFn) {
    let mut mismatches: Vec<String> = Vec::new();
    for result in &report.results {
        let expected = expected_pass(result.problem, result.case, result.path);
        let actual = result.outcome.is_pass();
        if expected != actual {
            let detail = match &result.outcome {
                Outcome::Pass => "Pass".to_string(),
                Outcome::InvalidSubgraph { returned } => format!("InvalidSubgraph(returned={returned:?})"),
                Outcome::RpcError(msg) => format!("RpcError({msg})"),
            };
            mismatches.push(format!(
                "  {}/{}/{}: expected {}, got {}",
                result.problem,
                result.case,
                result.path.as_str(),
                if expected { "Pass" } else { "Fail" },
                detail,
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "decoder outcome did not match expectations:\n{}",
        mismatches.join("\n"),
    );
}

/// Every standard case must appear in the report exactly once per API path.
fn assert_full_coverage(report: &SuiteReport) {
    for problem in standard_test_problems() {
        for case in &problem.cases {
            for path in [Path::Decode, Path::DecodeLoaded] {
                assert!(
                    report.get(problem.name, case.name, path).is_some(),
                    "missing entry for {}/{}/{}",
                    problem.name,
                    case.name,
                    path.as_str(),
                );
            }
        }
    }
}

/// Policy shared by decoders that always return an empty subgraph:
/// only the zero-syndrome cases satisfy the parity-factor check.
fn always_empty_subgraph_policy(_problem: &str, case: &str, _path: Path) -> bool {
    case == "zero"
}

/// Policy for real decoders that should solve every standard problem.
fn always_pass_policy(_problem: &str, _case: &str, _path: Path) -> bool {
    true
}

#[tokio::test]
async fn test_naive_decoder() {
    let decoder = Arc::new(NaiveDecoder::new(serde_json::json!({})));
    let mut client = BlackBoxDecoderClient::Local(DynBlackBoxDecoder::BlackBoxNaive(decoder));
    let report = run_standard_suite(&mut client).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_empty_subgraph_policy);
}

#[tokio::test]
async fn test_mock_decoder() {
    let decoder = Arc::new(MockDecoder::new());
    let mut client = BlackBoxDecoderClient::Local(DynBlackBoxDecoder::MockDecoder(decoder));
    let report = run_standard_suite(&mut client).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_empty_subgraph_policy);
}

#[tokio::test]
async fn test_relay_bp_decoder() {
    use deq_runtime::decoder::RelayBPDecoder;
    let decoder = Arc::new(RelayBPDecoder::new(serde_json::json!({})));
    let mut client = BlackBoxDecoderClient::Local(DynBlackBoxDecoder::BlackBoxRelayBP(decoder));
    let report = run_standard_suite(&mut client).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_pass_policy);
}

#[cfg(feature = "tesseract")]
#[tokio::test]
async fn test_tesseract_decoder() {
    use deq_runtime::decoder::TesseractDecoder;
    let decoder = Arc::new(TesseractDecoder::new(serde_json::json!({})));
    let mut client = BlackBoxDecoderClient::Local(DynBlackBoxDecoder::BlackBoxTesseract(decoder));
    let report = run_standard_suite(&mut client).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_pass_policy);
}

#[cfg(feature = "python")]
#[tokio::test]
async fn test_python_naive_decoder() {
    use deq_runtime::decoder::PythonDecoder;
    let config = serde_json::json!({ "file": "@naive_decoder" });
    let decoder = Arc::new(PythonDecoder::new(config));
    let mut client = BlackBoxDecoderClient::Local(DynBlackBoxDecoder::BlackBoxPython(decoder));
    let report = run_standard_suite(&mut client).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_empty_subgraph_policy);
}

/// Skip the test (with an explanatory message) when one of the listed Python
/// modules is not importable in the embedded interpreter. Returns `true` when
/// every module is available.
#[cfg(feature = "python")]
fn python_modules_available(test_name: &str, modules: &[&str]) -> bool {
    use pyo3::Python;
    let missing = Python::attach(|py| {
        modules
            .iter()
            .filter(|name| py.import(**name).is_err())
            .copied()
            .map(String::from)
            .collect::<Vec<_>>()
    });
    if !missing.is_empty() {
        eprintln!(
            "{test_name}: skipping — required Python module(s) not importable in embedded interpreter: {missing:?}. \
             Hint: install the packages where the embedded libpython can find them, or run with \
             `LD_LIBRARY_PATH=<conda-env>/lib` to load a libpython that has them."
        );
        return false;
    }
    true
}

#[cfg(feature = "python")]
#[tokio::test]
async fn test_python_relay_bp_decoder() {
    use deq_runtime::decoder::PythonDecoder;
    if !python_modules_available("test_python_relay_bp_decoder", &["numpy", "scipy.sparse", "relay_bp"]) {
        return;
    }
    let config = serde_json::json!({ "file": "@relay_bp_decoder" });
    let decoder = Arc::new(PythonDecoder::new(config));
    let mut client = BlackBoxDecoderClient::Local(DynBlackBoxDecoder::BlackBoxPython(decoder));
    let report = run_standard_suite(&mut client).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_pass_policy);
}

#[cfg(feature = "python")]
#[tokio::test]
async fn test_python_tesseract_decoder() {
    use deq_runtime::decoder::PythonDecoder;
    if !python_modules_available("test_python_tesseract_decoder", &["numpy", "stim", "tesseract_decoder"]) {
        return;
    }
    let config = serde_json::json!({ "file": "@tesseract_decoder" });
    let decoder = Arc::new(PythonDecoder::new(config));
    let mut client = BlackBoxDecoderClient::Local(DynBlackBoxDecoder::BlackBoxPython(decoder));
    let report = run_standard_suite(&mut client).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_pass_policy);
}
