//! Standard decoder test harness
//!
//! Runs the curated set of [`StandardTestProblem`]s against any
//! [`BlackBoxDecoderClient`] (local or remote), exercising both the one-shot
//! `decode` path and the `load_hypergraph` + `decode_loaded` path for every
//! case. Returns a [`SuiteReport`] describing the actual outcome of each call;
//! callers decide what is acceptable.

use crate::decoder::BlackBoxDecoderClient;
use crate::decoder::blackbox_decoder::{self, DecodingHypergraph, ParityFactor};
use crate::decoder::blackbox_util::is_parity_factor;
use crate::decoder::test_problems::{StandardTestProblem, TestCase, case_id, standard_test_problems};
use crate::util::BitVector;

/// Which API path produced a [`CaseResult`].
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Path {
    /// One-shot `decode(hypergraph, syndrome)`.
    Decode,
    /// `load_hypergraph(hypergraph)` followed by `decode_loaded(hid, syndrome)`.
    DecodeLoaded,
}

impl Path {
    pub fn as_str(self) -> &'static str {
        match self {
            Path::Decode => "decode",
            Path::DecodeLoaded => "decode_loaded",
        }
    }
}

/// Outcome of running one case on one API path.
#[derive(Clone, Debug, PartialEq)]
pub enum Outcome {
    /// The decoder returned a subgraph that explains the syndrome.
    Pass,
    /// The decoder returned a subgraph that does not explain the syndrome.
    InvalidSubgraph { returned: Vec<u64> },
    /// The RPC failed (panic, error status, etc.).
    RpcError(String),
}

impl Outcome {
    pub fn is_pass(&self) -> bool {
        matches!(self, Outcome::Pass)
    }
}

/// Outcome of a single (problem, case, path) entry.
#[derive(Clone, Debug)]
pub struct CaseResult {
    pub problem: &'static str,
    pub case: &'static str,
    pub path: Path,
    pub outcome: Outcome,
}

/// Aggregated outcomes for one suite run.
#[derive(Clone, Debug, Default)]
pub struct SuiteReport {
    pub results: Vec<CaseResult>,
}

impl SuiteReport {
    /// Look up the outcome for a specific (problem, case, path) entry.
    pub fn get(&self, problem: &str, case: &str, path: Path) -> Option<&Outcome> {
        self.results
            .iter()
            .find(|r| r.problem == problem && r.case == case && r.path == path)
            .map(|r| &r.outcome)
    }

    /// Number of entries that passed.
    pub fn pass_count(&self) -> usize {
        self.results.iter().filter(|r| r.outcome.is_pass()).count()
    }

    /// Total number of entries.
    pub fn total(&self) -> usize {
        self.results.len()
    }

    /// One human-readable line per entry, suitable for CLI output.
    pub fn summary_lines(&self) -> Vec<String> {
        self.results
            .iter()
            .map(|r| {
                let id = case_id(r.problem, r.case);
                match &r.outcome {
                    Outcome::Pass => format!("[PASS] {}/{}", id, r.path.as_str()),
                    Outcome::InvalidSubgraph { returned } => {
                        format!("[FAIL] {}/{}  invalid subgraph: returned {:?}", id, r.path.as_str(), returned)
                    }
                    Outcome::RpcError(msg) => {
                        format!("[FAIL] {}/{}  rpc error: {}", id, r.path.as_str(), msg)
                    }
                }
            })
            .collect()
    }
}

fn classify(hypergraph: &DecodingHypergraph, syndrome: &BitVector, response: &ParityFactor) -> Outcome {
    if is_parity_factor(hypergraph, response, syndrome) {
        Outcome::Pass
    } else {
        Outcome::InvalidSubgraph {
            returned: response.subgraph.clone(),
        }
    }
}

async fn run_one_problem(client: &mut BlackBoxDecoderClient, problem: &StandardTestProblem, out: &mut Vec<CaseResult>) {
    for case in &problem.cases {
        out.push(run_decode_path(client, problem, case).await);
    }

    let load_outcome = client.load_hypergraph(problem.hypergraph.clone()).await;
    let hid = match load_outcome {
        Ok(response) => Some(response.hid),
        Err(status) => {
            let msg = format!("load_hypergraph failed: {status}");
            for case in &problem.cases {
                out.push(CaseResult {
                    problem: problem.name,
                    case: case.name,
                    path: Path::DecodeLoaded,
                    outcome: Outcome::RpcError(msg.clone()),
                });
            }
            None
        }
    };

    if let Some(hid) = hid {
        for case in &problem.cases {
            out.push(run_decode_loaded_path(client, problem, case, hid).await);
        }
    }

    // Best-effort reset between problems. Failures are recorded as a synthetic
    // entry so callers see the issue but do not crash the rest of the suite.
    if let Err(status) = client
        .reset(blackbox_decoder::ResetRequest {
            reset_hypergraphs: true,
            ..Default::default()
        })
        .await
    {
        out.push(CaseResult {
            problem: problem.name,
            case: "reset",
            path: Path::DecodeLoaded,
            outcome: Outcome::RpcError(format!("reset failed: {status}")),
        });
    }
}

async fn run_decode_path(client: &mut BlackBoxDecoderClient, problem: &StandardTestProblem, case: &TestCase) -> CaseResult {
    let problem_payload = blackbox_decoder::DecodingProblem {
        hypergraph: Some(problem.hypergraph.clone()),
        syndrome: Some(case.syndrome.clone()),
    };
    let outcome = match client.decode(problem_payload).await {
        Ok(response) => classify(&problem.hypergraph, &case.syndrome, &response),
        Err(status) => Outcome::RpcError(status.to_string()),
    };
    CaseResult {
        problem: problem.name,
        case: case.name,
        path: Path::Decode,
        outcome,
    }
}

async fn run_decode_loaded_path(
    client: &mut BlackBoxDecoderClient,
    problem: &StandardTestProblem,
    case: &TestCase,
    hid: u64,
) -> CaseResult {
    let problem_payload = blackbox_decoder::LoadedDecodingProblem {
        hid,
        syndrome: Some(case.syndrome.clone()),
    };
    let outcome = match client.decode_loaded(problem_payload).await {
        Ok(response) => classify(&problem.hypergraph, &case.syndrome, &response),
        Err(status) => Outcome::RpcError(status.to_string()),
    };
    CaseResult {
        problem: problem.name,
        case: case.name,
        path: Path::DecodeLoaded,
        outcome,
    }
}

/// Run the full standard suite against `client` and collect outcomes.
pub async fn run_standard_suite(client: &mut BlackBoxDecoderClient) -> SuiteReport {
    let mut results: Vec<CaseResult> = Vec::new();
    for problem in standard_test_problems() {
        run_one_problem(client, &problem, &mut results).await;
    }
    SuiteReport { results }
}
