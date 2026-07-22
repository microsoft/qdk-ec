//! QEC well-formedness checks for `CODE` blocks, backed by [`paulimer`].
//!
//! Pauli operators are built directly from the parsed AST via
//! [`PositionedPauliObservable`] — the established qdk-ec construction path — not
//! by re-serializing to a string. Commutation is `paulimer`'s symplectic product
//! over GF(2); stabilizer rank is [`PauliGroup::binary_rank`].

use deqagram::Span;
use deqagram::ast::{CodeDefinition, Pauli, PauliProduct};
use paulimer::core::{PauliObservable, PositionedPauliObservable};
use paulimer::pauli::{SparsePauli, anti_commutes_with};
use paulimer::pauli_group::PauliGroup;

use crate::{Diagnostic, Rule};

/// The GF(2) rank of a set of Pauli operators (the rank of their symplectic
/// check matrix). An empty set has rank 0; [`PauliGroup`] is not constructed
/// from zero generators.
fn pauli_rank(paulis: &[SparsePauli]) -> usize {
    if paulis.is_empty() {
        0
    } else {
        PauliGroup::new(paulis).binary_rank()
    }
}

/// Builds a [`SparsePauli`] from an AST Pauli product by multiplying its
/// single-qubit factors, so a qubit that appears more than once reduces
/// naturally (`X0*X0 = I`, `X0*Z0 = -iY0`) with the correct phase.
///
/// Emits a diagnostic and returns `None` only if a qubit index is `>= n`, since
/// that operator cannot be interpreted on `n` qubits. Identity terms contribute
/// nothing and are skipped.
fn build_pauli(
    product: &PauliProduct,
    n: u64,
    span: Span,
    context: &str,
    out: &mut Vec<Diagnostic>,
) -> Option<SparsePauli> {
    let terms: &[deqagram::ast::PauliTerm] = match product {
        PauliProduct::Identity => &[],
        PauliProduct::Terms(terms) => terms,
    };

    let mut operator = SparsePauli::from([].as_slice());
    let mut in_range = true;
    for term in terms {
        if term.index >= n {
            out.push(Diagnostic::new(
                Rule::QubitIndexOutOfRange,
                span,
                format!("{context}: qubit index {} is out of range for n = {n}", term.index),
            ));
            in_range = false;
            continue;
        }
        let observable = match term.pauli {
            Pauli::I => continue,
            Pauli::X => PauliObservable::PlusX,
            Pauli::Y => PauliObservable::PlusY,
            Pauli::Z => PauliObservable::PlusZ,
        };
        let factor = SparsePauli::from(
            [PositionedPauliObservable {
                qubit_id: usize::try_from(term.index).expect("qubit index fits in usize"),
                observable,
            }]
            .as_slice(),
        );
        operator = &operator * &factor;
    }

    in_range.then_some(operator)
}

/// Runs every QEC check on one `CODE` definition. `span` anchors code-level
/// diagnostics; operator-level diagnostics use the operator's own span.
pub(crate) fn check_code(code: &CodeDefinition, span: Span, out: &mut Vec<Diagnostic>) {
    // n == 0 is already reported structurally; every index would be out of
    // range, so there is nothing meaningful to check here.
    if code.n == 0 {
        return;
    }

    // Build all operators up front. If any qubit index is out of range the
    // corresponding builder emits a diagnostic and yields None; we then skip the
    // algebra to avoid reporting derivative, misleading errors.
    let mut all_in_range = true;

    let stabilizers: Vec<SparsePauli> = code
        .stabilizers
        .iter()
        .filter_map(|stab| {
            let built = build_pauli(&stab.node, code.n, stab.span, "stabilizer", out);
            all_in_range &= built.is_some();
            built
        })
        .collect();

    let logicals: Vec<(SparsePauli, SparsePauli)> = code
        .logicals
        .iter()
        .enumerate()
        .filter_map(|(i, logical)| {
            let x = build_pauli(
                &logical.x_operator.node,
                code.n,
                logical.x_operator.span,
                &format!("logical X{i}"),
                out,
            );
            let z = build_pauli(
                &logical.z_operator.node,
                code.n,
                logical.z_operator.span,
                &format!("logical Z{i}"),
                out,
            );
            if let (Some(x), Some(z)) = (x, z) {
                Some((x, z))
            } else {
                all_in_range = false;
                None
            }
        })
        .collect();

    if !all_in_range {
        return;
    }

    check_stabilizers(code, span, &stabilizers, out);
    check_logicals(code, span, &stabilizers, &logicals, out);
}

/// Checks that the stabilizers commute, are sign-consistent, and have rank
/// matching `n - k`.
fn check_stabilizers(code: &CodeDefinition, span: Span, stabilizers: &[SparsePauli], out: &mut Vec<Diagnostic>) {
    // An empty stabilizer set has rank 0; only the rank/parameter check applies.
    let rank: u64 = if stabilizers.is_empty() {
        0
    } else {
        let group = PauliGroup::new(stabilizers);
        if !group.is_abelian() {
            out.push(Diagnostic::new(
                Rule::StabilizersNotCommuting,
                span,
                format!("CODE {}: stabilizer generators do not all commute", code.name),
            ));
            // Rank-based checks presuppose a valid (commuting) stabilizer group.
            return;
        }
        if !group.is_stabilizer_group() {
            out.push(Diagnostic::new(
                Rule::StabilizerGroupContainsMinusIdentity,
                span,
                format!(
                    "CODE {}: stabilizer group contains -I (a product of \
                     generators equals -I), so it has no +1 eigenspace",
                    code.name
                ),
            ));
            return;
        }
        group.binary_rank() as u64
    };

    if code.k <= code.n {
        let expected = code.n - code.k;
        let encoded = code.n - rank;
        if rank > expected {
            // More logical degrees of freedom than the declared k: the operators
            // cannot support k logical qubits. This is a genuine inconsistency.
            out.push(Diagnostic::new(
                Rule::StabilizerRankTooHigh,
                span,
                format!(
                    "CODE {}: stabilizer rank {rank} exceeds n - k = {expected} \
                     for [[n={}, k={}]]; the stabilizers leave room for only \
                     k = {encoded} logical qubit(s), fewer than declared",
                    code.name, code.n, code.k,
                ),
            ));
        } else if rank < expected {
            // Fewer independent stabilizers than a pure [[n,k]] stabilizer code
            // would have. Either k is wrong, or this is a subsystem/gauge code
            // (whose stabilizer rank is n - k - g for g gauge qubits) — which
            // this linter does not model, hence a warning rather than an error.
            out.push(Diagnostic::new(
                Rule::StabilizerRankTooLow,
                span,
                format!(
                    "CODE {}: stabilizer rank {rank} is below n - k = {expected} \
                     for [[n={}, k={}]]; the stabilizers alone encode \
                     k = {encoded} logical qubit(s). This is expected for a \
                     subsystem/gauge code (not modeled here); otherwise k or the \
                     stabilizer set is wrong",
                    code.name, code.n, code.k,
                ),
            ));
        }
    }

    if stabilizers.len() as u64 > rank {
        out.push(Diagnostic::new(
            Rule::RedundantStabilizer,
            span,
            format!(
                "CODE {}: {} stabilizer generators but rank is only {rank}; \
                 {} generator(s) are dependent",
                code.name,
                stabilizers.len(),
                stabilizers.len() as u64 - rank,
            ),
        ));
    }
}

/// Checks that the declared logical operators are linearly independent modulo
/// the stabilizer group.
///
/// A valid `[[n, k, d]]` code has exactly `2k` logical operators that are
/// independent modulo the stabilizers, so the GF(2) rank of `stabilizers +
/// logicals` must exceed the rank of `stabilizers` alone by 2 per declared pair.
/// A smaller increase means a declared logical operator is a product of the
/// others and the stabilizers — it names no fresh logical degree of freedom.
fn check_logical_rank(
    code: &CodeDefinition,
    span: Span,
    stabilizers: &[SparsePauli],
    logicals: &[(SparsePauli, SparsePauli)],
    out: &mut Vec<Diagnostic>,
) {
    let stabilizer_rank = pauli_rank(stabilizers);
    let mut combined: Vec<SparsePauli> = stabilizers.to_vec();
    for (lx, lz) in logicals {
        combined.push(lx.clone());
        combined.push(lz.clone());
    }
    let independent_logicals = pauli_rank(&combined) - stabilizer_rank;
    let expected = 2 * logicals.len();
    if independent_logicals < expected {
        out.push(Diagnostic::new(
            Rule::LogicalRankDeficient,
            span,
            format!(
                "CODE {}: the {expected} declared logical operator(s) span only \
                 {independent_logicals} independent operator(s) modulo the \
                 stabilizers (expected 2 * {} = {expected}); some logical \
                 operators are redundant",
                code.name,
                logicals.len(),
            ),
        ));
    }
}

/// Checks the logical-operator count, their commutation with the stabilizers,
/// their symplectic canonical form, and that they are nontrivial.
fn check_logicals(
    code: &CodeDefinition,
    span: Span,
    stabilizers: &[SparsePauli],
    logicals: &[(SparsePauli, SparsePauli)],
    out: &mut Vec<Diagnostic>,
) {
    if logicals.len() as u64 != code.k {
        out.push(Diagnostic::new(
            Rule::LogicalCountMismatch,
            span,
            format!(
                "CODE {}: {} logical operator pair(s) declared but k = {}",
                code.name,
                logicals.len(),
                code.k,
            ),
        ));
    }

    check_logical_rank(code, span, stabilizers, logicals, out);

    // Each logical operator must commute with every stabilizer (it lies in the
    // normalizer of the stabilizer group).
    for (i, (lx, lz)) in logicals.iter().enumerate() {
        for (op, label) in [(lx, 'X'), (lz, 'Z')] {
            for (s, stab) in stabilizers.iter().enumerate() {
                if anti_commutes_with(op, stab) {
                    out.push(Diagnostic::new(
                        Rule::LogicalStabilizerAnticommute,
                        span,
                        format!(
                            "CODE {}: logical {label}{i} anticommutes with \
                             stabilizer {s} (logicals must commute with all \
                             stabilizers)",
                            code.name
                        ),
                    ));
                }
            }
        }
    }

    // Symplectic canonical form: LXi anticommutes LZi, and every other pairing
    // among distinct logical operators commutes.
    for (i, (lxi, lzi)) in logicals.iter().enumerate() {
        if !anti_commutes_with(lxi, lzi) {
            out.push(Diagnostic::new(
                Rule::LogicalCanonicalForm,
                span,
                format!("CODE {}: logical X{i} and Z{i} must anticommute", code.name),
            ));
        }
        for (j, (lxj, lzj)) in logicals.iter().enumerate().skip(i + 1) {
            let pairs = [
                (lxi, lxj, 'X', i, 'X', j),
                (lzi, lzj, 'Z', i, 'Z', j),
                (lxi, lzj, 'X', i, 'Z', j),
                (lxj, lzi, 'X', j, 'Z', i),
            ];
            for (a, b, la, ia, lb, ib) in pairs {
                if anti_commutes_with(a, b) {
                    out.push(Diagnostic::new(
                        Rule::LogicalCanonicalForm,
                        span,
                        format!(
                            "CODE {}: logical {la}{ia} and {lb}{ib} must commute \
                             (distinct logical qubits)",
                            code.name
                        ),
                    ));
                }
            }
        }
    }

    // A logical operator that lies in the stabilizer group is not a genuine
    // logical: it is a mere product of stabilizer generators and so acts as the
    // identity on the code space.
    if !stabilizers.is_empty() {
        let group = PauliGroup::new(stabilizers);
        if group.is_abelian() {
            for (i, (lx, lz)) in logicals.iter().enumerate() {
                for (op, label) in [(lx, 'X'), (lz, 'Z')] {
                    if group.contains(op) {
                        out.push(Diagnostic::new(
                            Rule::TrivialLogical,
                            span,
                            format!(
                                "CODE {}: logical {label}{i} is a product of \
                                 stabilizer generators, so it acts trivially on \
                                 the code space (a logical operator must not be a \
                                 mere combination of stabilizers)",
                                code.name
                            ),
                        ));
                    }
                }
            }
        }
    }
}
