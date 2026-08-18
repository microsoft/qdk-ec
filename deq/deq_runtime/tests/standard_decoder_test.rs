//! Integration tests for the standard decoder test suite.
//!
//! Each decoder is exercised with a closure declaring the expected pass/fail
//! outcome for every (problem, case, path) entry. The closure is the *test's*
//! policy — the `test_problems` and `test_harness` modules do not assume
//! anything about which cases should pass.

use std::sync::Arc;

#[cfg(feature = "python")]
use std::io::Write;

use deq_runtime::decoder::blackbox_decoder::{
    DecodingHypergraph, DecodingProblem, EdgeReweight, Hyperedge, LoadedDecodingProblem, LossInfo, LossSite,
};
use deq_runtime::decoder::test_harness::{Outcome, Path, SuiteReport, run_standard_suite};
use deq_runtime::decoder::test_problems::standard_test_problems;
use deq_runtime::decoder::{DecoderFeatures, DynDecoder, MockDecoder, NaiveDecoder};
use deq_runtime::util::BitVector;

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

async fn assert_accepts_all_features(decoder: &DynDecoder) {
    assert_eq!(decoder.features(), DecoderFeatures::REWEIGHTS | DecoderFeatures::LOSS);
    let hypergraph = DecodingHypergraph {
        vertex_num: 1,
        hyperedges: vec![Hyperedge {
            vertices: vec![0],
            probability: 0.1,
        }],
    };
    let syndrome = BitVector {
        size: 1,
        data: vec![0b1000_0000],
    };
    let loss = LossInfo {
        sites: vec![LossSite {
            source_edges: vec![0],
            probability: 0.2,
            ..Default::default()
        }],
    };

    let parity_factor = decoder
        .decode(DecodingProblem {
            hypergraph: Some(hypergraph.clone()),
            syndrome: Some(syndrome.clone()),
            loss: Some(loss.clone()),
        })
        .await
        .unwrap();
    assert!(parity_factor.subgraph.is_empty());

    let hid = decoder.load_hypergraph(hypergraph).await.unwrap().hid;
    let parity_factor = decoder
        .decode_loaded(LoadedDecodingProblem {
            hid,
            syndrome: Some(syndrome),
            reweights: vec![EdgeReweight {
                edge: 0,
                probability: 0.25,
            }],
            loss: Some(loss),
        })
        .await
        .unwrap();
    assert!(parity_factor.subgraph.is_empty());
}

async fn assert_accepts_isolated_zero_vertex(decoder: &DynDecoder) {
    let hypergraph = DecodingHypergraph {
        vertex_num: 2,
        hyperedges: vec![Hyperedge {
            vertices: vec![0],
            probability: 0.1,
        }],
    };
    let syndrome = BitVector {
        size: 2,
        data: vec![0b1000_0000],
    };

    decoder
        .decode(DecodingProblem {
            hypergraph: Some(hypergraph.clone()),
            syndrome: Some(syndrome.clone()),
            loss: None,
        })
        .await
        .unwrap();

    let hid = decoder.load_hypergraph(hypergraph).await.unwrap().hid;
    decoder
        .decode_loaded(LoadedDecodingProblem {
            hid,
            syndrome: Some(syndrome),
            ..Default::default()
        })
        .await
        .unwrap();
}

#[tokio::test]
async fn test_naive_decoder() {
    let decoder = DynDecoder::BlackBoxNaive(Arc::new(NaiveDecoder::new(serde_json::json!({}))));
    assert_accepts_all_features(&decoder).await;
    assert_accepts_isolated_zero_vertex(&decoder).await;
    let report = run_standard_suite(&decoder).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_empty_subgraph_policy);
}

#[tokio::test]
async fn test_mock_decoder() {
    let decoder = DynDecoder::Mock(Arc::new(MockDecoder::new()));
    assert_accepts_isolated_zero_vertex(&decoder).await;
    let report = run_standard_suite(&decoder).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_empty_subgraph_policy);
}

#[tokio::test]
async fn test_relay_bp_decoder() {
    use deq_runtime::decoder::RelayBPDecoder;
    let decoder = DynDecoder::BlackBoxRelayBP(Arc::new(RelayBPDecoder::new(serde_json::json!({}))));
    assert_accepts_isolated_zero_vertex(&decoder).await;
    let report = run_standard_suite(&decoder).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_pass_policy);
}

#[cfg(feature = "tesseract")]
#[tokio::test]
async fn test_tesseract_decoder() {
    use deq_runtime::decoder::TesseractDecoder;
    let decoder = DynDecoder::BlackBoxTesseract(Arc::new(TesseractDecoder::new(serde_json::json!({}))));
    assert_accepts_isolated_zero_vertex(&decoder).await;
    let report = run_standard_suite(&decoder).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_pass_policy);
}

#[cfg(feature = "python")]
#[tokio::test]
async fn test_python_naive_decoder() {
    use deq_runtime::decoder::PythonDecoder;
    let config = serde_json::json!({ "file": "@naive_decoder" });
    let decoder = DynDecoder::BlackBoxPython(Arc::new(PythonDecoder::new(config)));
    assert_accepts_all_features(&decoder).await;
    assert_accepts_isolated_zero_vertex(&decoder).await;
    let report = run_standard_suite(&decoder).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_empty_subgraph_policy);
}

#[cfg(feature = "python")]
#[tokio::test]
async fn test_python_named_decoder_without_supported_features() {
    use deq_runtime::decoder::DecoderFeatures;
    use deq_runtime::decoder::PythonDecoder;

    let mut decoder_file = tempfile::Builder::new().suffix(".py").tempfile().unwrap();
    decoder_file
        .write_all(
            br#"
class LegacyDecoder:
    def __init__(self, hypergraph, config):
        pass

    def decode(self, syndrome):
        return []

    def reset(self):
        pass
"#,
        )
        .unwrap();

    let decoder = DynDecoder::BlackBoxPython(Arc::new(PythonDecoder::new(serde_json::json!({
        "file": decoder_file.path(),
        "name": "LegacyDecoder",
    }))));

    assert_eq!(decoder.features(), DecoderFeatures::empty());
    assert_accepts_isolated_zero_vertex(&decoder).await;
    let report = run_standard_suite(&decoder).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_empty_subgraph_policy);
}

#[cfg(feature = "python")]
#[tokio::test]
async fn test_python_decoder_receives_reweights_and_loss_together() {
    use deq_runtime::decoder::PythonDecoder;

    let mut decoder_file = tempfile::Builder::new().suffix(".py").tempfile().unwrap();
    decoder_file
        .write_all(
            br#"
class CombinedDecoder:
    @staticmethod
    def supported_features():
        return ["reweights", "loss"]

    def __init__(self, hypergraph, config):
        assert hypergraph.vertex_num == 1, hypergraph.vertex_num
        assert config == {}, config

    def decode(self, syndrome, *, reweights=None, loss=None):
        assert syndrome == [0], syndrome
        if reweights is None:
            assert loss is not None, loss
            assert list(loss.sites) == [], loss.sites
            return []
        assert reweights == [(0, 0.25)], reweights
        assert loss is not None, loss
        assert len(loss.sites) == 1, loss.sites
        assert loss.sites[0].source_edges == [0], loss.sites[0].source_edges
        assert loss.sites[0].probability == 0.2, loss.sites[0].probability
        assert loss.sites[0].heralds == [4, 7], loss.sites[0].heralds
        return [0]

    def reset(self):
        pass
"#,
        )
        .unwrap();

    let config = serde_json::json!({
        "file": decoder_file.path(),
        "name": "CombinedDecoder",
    });
    let decoder = DynDecoder::BlackBoxPython(Arc::new(PythonDecoder::new(config)));
    assert_eq!(decoder.features(), DecoderFeatures::REWEIGHTS | DecoderFeatures::LOSS);

    let hid = decoder
        .load_hypergraph(DecodingHypergraph {
            vertex_num: 1,
            hyperedges: vec![Hyperedge {
                vertices: vec![0],
                probability: 0.1,
            }],
        })
        .await
        .unwrap()
        .hid;
    let parity_factor = decoder
        .decode_loaded(LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector {
                size: 1,
                data: vec![0b1000_0000],
            }),
            reweights: vec![EdgeReweight {
                edge: 0,
                probability: 0.25,
            }],
            loss: Some(LossInfo {
                sites: vec![LossSite {
                    source_edges: vec![0],
                    probability: 0.2,
                    heralds: vec![4, 7],
                    ..Default::default()
                }],
            }),
        })
        .await
        .unwrap();

    assert_eq!(parity_factor.subgraph, vec![0]);

    let parity_factor = decoder
        .decode_loaded(LoadedDecodingProblem {
            hid,
            syndrome: Some(BitVector {
                size: 1,
                data: vec![0b1000_0000],
            }),
            loss: Some(LossInfo { sites: vec![] }),
            ..Default::default()
        })
        .await
        .unwrap();

    assert!(parity_factor.subgraph.is_empty());
}

#[cfg(feature = "python")]
#[test]
#[should_panic(expected = "unsupported Python decoder feature")]
fn test_python_decoder_rejects_unknown_supported_feature() {
    use deq_runtime::decoder::PythonDecoder;

    let mut decoder_file = tempfile::Builder::new().suffix(".py").tempfile().unwrap();
    decoder_file
        .write_all(
            br#"
class Decoder:
    @staticmethod
    def supported_features():
        return ["unknown"]
"#,
        )
        .unwrap();

    let _ = PythonDecoder::new(serde_json::json!({ "file": decoder_file.path() }));
}

#[cfg(feature = "python")]
#[test]
#[should_panic(expected = "invalid PythonDecoderConfig")]
fn test_python_decoder_rejects_supported_features_in_config() {
    use deq_runtime::decoder::PythonDecoder;

    let _ = PythonDecoder::new(serde_json::json!({
        "file": "@naive_decoder",
        "supported_features": ["loss"],
    }));
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
    let decoder = DynDecoder::BlackBoxPython(Arc::new(PythonDecoder::new(config)));
    assert_accepts_isolated_zero_vertex(&decoder).await;
    let report = run_standard_suite(&decoder).await;
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
    let decoder = DynDecoder::BlackBoxPython(Arc::new(PythonDecoder::new(config)));
    assert_accepts_isolated_zero_vertex(&decoder).await;
    let report = run_standard_suite(&decoder).await;
    assert_full_coverage(&report);
    assert_matches_policy(&report, always_pass_policy);
}

#[cfg(feature = "python")]
#[tokio::test]
async fn test_python_mle_loss_decoder_accepts_isolated_zero_vertex() {
    use deq_runtime::decoder::PythonDecoder;
    if !python_modules_available(
        "test_python_mle_loss_decoder_accepts_isolated_zero_vertex",
        &["numpy", "scipy.optimize", "scipy.sparse"],
    ) {
        return;
    }
    let config = serde_json::json!({ "file": "@mle_loss_decoder" });
    let decoder = DynDecoder::BlackBoxPython(Arc::new(PythonDecoder::new(config)));
    assert_accepts_isolated_zero_vertex(&decoder).await;
}
