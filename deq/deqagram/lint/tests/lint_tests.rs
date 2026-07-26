//! End-to-end tests: parse `.deq` source, lint it, and assert on the rule codes
//! produced. Positive cases (valid codes) must be clean; each negative case must
//! surface its target rule.

use deqagram::ast::DeqFile;
use deqagram_lint::{Rule, Severity, lint};

/// Parses and lints `src`, returning the set of rule codes reported.
fn codes(src: &str) -> Vec<&'static str> {
    let file: DeqFile = src.parse().expect("source should parse");
    lint(&file).into_iter().map(|d| d.rule.code()).collect()
}

fn has(src: &str, rule: Rule) -> bool {
    codes(src).contains(&rule.code())
}

// ── Positive cases: valid codes lint clean ───────────────────────────

#[test]
fn repetition_code_is_clean() {
    let src = "\
CODE RepetitionCode [[3,1,1]] {
LOGICAL X0*X1*X2 Z0*Z1*Z2
STABILIZER Z0*Z1 Z1*Z2
}
";
    assert_eq!(codes(src), Vec::<&str>::new());
}

#[test]
fn steane_code_is_clean() {
    // The [[7,1,3]] Steane code: CSS from the [7,4,3] Hamming code, so the X and
    // Z stabilizers share supports; logical X/Z act on all seven qubits.
    let src = "\
CODE Steane [[7,1,3]] {
LOGICAL X0*X1*X2*X3*X4*X5*X6 Z0*Z1*Z2*Z3*Z4*Z5*Z6
STABILIZER X0*X2*X4*X6 X1*X2*X5*X6 X3*X4*X5*X6
STABILIZER Z0*Z2*Z4*Z6 Z1*Z2*Z5*Z6 Z3*Z4*Z5*Z6
}
";
    assert_eq!(codes(src), Vec::<&str>::new());
}

// ── QEC negative cases ───────────────────────────────────────────────

#[test]
fn rank_inconsistent_with_k() {
    // Same stabilizers as the repetition code (rank 2 => n - k = 1) but declared
    // k = 2, so rank 2 exceeds n - k = 1.
    let src = "\
CODE Bad [[3,2,3]] {
LOGICAL X0*X1*X2 Z0*Z1*Z2
STABILIZER Z0*Z1 Z1*Z2
}
";
    assert!(has(src, Rule::StabilizerRankTooHigh));
}

#[test]
fn subsystem_code_rank_below_is_a_warning() {
    // Bacon-Shor [[9,1,3]] is a subsystem code: 4 stabilizer generators (rank 4)
    // with 4 gauge qubits, so rank 4 < n - k = 8. This is expected, not an
    // error, so the linter warns rather than erroring.
    let src = "\
CODE BaconShor [[9,1,3]] {
LOGICAL X0*X1*X2*X3*X4*X5*X6*X7*X8 Z0*Z1*Z2*Z3*Z4*Z5*Z6*Z7*Z8
STABILIZER Z0*Z1*Z3*Z4*Z6*Z7 Z1*Z2*Z4*Z5*Z7*Z8
STABILIZER X0*X1*X2*X3*X4*X5 X3*X4*X5*X6*X7*X8
}
";
    let got = codes(src);
    assert!(got.contains(&Rule::StabilizerRankTooLow.code()));
    assert!(!got.contains(&Rule::StabilizerRankTooHigh.code()));
    assert_eq!(Rule::StabilizerRankTooLow.severity(), Severity::Warning);
}

#[test]
fn non_commuting_stabilizers() {
    let src = "\
CODE Bad [[1,0]] {
STABILIZER X0
STABILIZER Z0
}
";
    assert!(has(src, Rule::StabilizersNotCommuting));
}

#[test]
fn redundant_stabilizer_warns() {
    // Third generator Z0*Z2 is the product of the first two: rank stays 2.
    let src = "\
CODE Rep [[3,1,1]] {
LOGICAL X0*X1*X2 Z0*Z1*Z2
STABILIZER Z0*Z1 Z1*Z2 Z0*Z2
}
";
    let got = codes(src);
    assert!(got.contains(&Rule::RedundantStabilizer.code()));
    // It is only a warning; the code is otherwise consistent (rank still == n-k).
    assert!(!got.contains(&Rule::StabilizerRankTooHigh.code()));
    assert!(!got.contains(&Rule::StabilizerRankTooLow.code()));
    assert_eq!(Rule::RedundantStabilizer.severity(), Severity::Warning);
}

#[test]
fn logical_count_mismatch() {
    let src = "\
CODE Bad [[3,1,1]] {
STABILIZER Z0*Z1 Z1*Z2
}
";
    // Rep-code stabilizers give rank 2 => n - k = 1, consistent with k = 1, but
    // zero logicals are declared.
    assert!(has(src, Rule::LogicalCountMismatch));
}

#[test]
fn logical_rank_deficient_when_operators_are_dependent() {
    // A [[2,2]] code with no stabilizers: two logical qubits need four
    // independent logical operators. Here both pairs are identical (X0/Z0), so
    // they span only two independent operators, not four — rank-deficient even
    // though the pair count (2) matches k.
    let src = "\
CODE Bad [[2,2]] {
LOGICAL X0 Z0
LOGICAL X0 Z0
}
";
    let got = codes(src);
    assert!(got.contains(&Rule::LogicalRankDeficient.code()));
    // The count check is satisfied (2 pairs == k), so this is caught only by the
    // rank check, not by counting.
    assert!(!got.contains(&Rule::LogicalCountMismatch.code()));
}

#[test]
fn independent_logicals_are_not_rank_deficient() {
    // A genuine [[2,2]] code: X0/Z0 and X1/Z1 are four independent operators.
    let src = "\
CODE Good [[2,2]] {
LOGICAL X0 Z0
LOGICAL X1 Z1
}
";
    assert!(!codes(src).contains(&Rule::LogicalRankDeficient.code()));
}

#[test]
fn logical_anticommutes_stabilizer() {
    // Logical Z = X0 anticommutes with stabilizer Z0*Z1.
    let src = "\
CODE Bad [[3,1,1]] {
LOGICAL X0*X1*X2 X0
STABILIZER Z0*Z1 Z1*Z2
}
";
    assert!(has(src, Rule::LogicalStabilizerAnticommute));
}

#[test]
fn logical_canonical_form_violation() {
    // Logical X and Z are both X-type, so they commute where they must
    // anticommute.
    let src = "\
CODE Bad [[3,1,1]] {
LOGICAL X0*X1*X2 X0*X1*X2
STABILIZER Z0*Z1 Z1*Z2
}
";
    assert!(has(src, Rule::LogicalCanonicalForm));
}

#[test]
fn trivial_logical() {
    // Both logical operators equal the stabilizer Z0*Z1, so they are in the
    // stabilizer group.
    let src = "\
CODE Bad [[2,1]] {
LOGICAL Z0*Z1 Z0*Z1
STABILIZER Z0*Z1
}
";
    assert!(has(src, Rule::TrivialLogical));
}

#[test]
fn qubit_index_out_of_range() {
    let src = "\
CODE Bad [[2,0]] {
STABILIZER Z0*Z5
}
";
    assert!(has(src, Rule::QubitIndexOutOfRange));
}

#[test]
fn repeated_qubit_reduces_without_panic() {
    // `X0*Z0` reduces to -iY0; multiplying single-qubit factors must not panic.
    // -iY0 squares to -I, so the stabilizer group contains -I and is rejected as
    // a valid stabilizer group (rather than crashing the linter).
    let src = "\
CODE Reduce [[2,0]] {
STABILIZER X0*Z0
}
";
    assert!(has(src, Rule::StabilizerGroupContainsMinusIdentity));
}

// ── Structural negative cases ────────────────────────────────────────

#[test]
fn code_param_bounds() {
    assert!(has("CODE C [[0,0]] {\n}\n", Rule::CodeParamN));
    assert!(has("CODE C [[2,3]] {\n}\n", Rule::CodeParamKGreaterThanN));
    assert!(has("CODE C [[3,1,0]] {\nSTABILIZER Z0*Z1 Z1*Z2\n}\n", Rule::CodeParamD));
}

#[test]
fn repeat_count_zero() {
    let src = "\
GADGET G {
REPEAT 0 { M 0 }
}
";
    assert!(has(src, Rule::RepeatCount));
}

#[test]
fn error_probability_out_of_range() {
    let src = "\
GADGET G {
ERROR(1.5) C0
}
";
    assert!(has(src, Rule::ErrorProbability));
}

#[test]
fn dangling_decorator() {
    let src = "\
GADGET G {
M 0
@FOO
}
";
    assert!(has(src, Rule::DanglingDecorator));
}
