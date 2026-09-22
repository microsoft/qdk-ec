//! Tests for the Clifford -> transvection decomposition (arXiv:2102.11380).
//!
//! The decomposition reproduces the *symplectic action* (ignoring Pauli-image signs and the global
//! phase) with a linear number of factors. It is not guaranteed to hit the strict `r`/`r + 1`
//! minimum, so these tests validate the symplectic-action round trip, the linear factor bound, and
//! the centralizer contract, rather than exact minimality.

use binar::Bitwise;
use paulimer::UnitaryOp;
use paulimer::clifford::{Clifford, CliffordMutable, CliffordUnitary, clifford_centralizer, clifford_to_transvections};
use paulimer::pauli::{Pauli, SparsePauli};
use proptest::collection::vec;
use proptest::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Rebuilds a Clifford's symplectic action by replaying transvections on the identity.
fn symplectic_action_from_transvections(transvections: &[SparsePauli], qubit_count: usize) -> CliffordUnitary {
    let mut rebuilt = CliffordUnitary::identity(qubit_count);
    for transvection in transvections {
        rebuilt.left_mul_pauli_exp(transvection);
    }
    rebuilt
}

/// Whether conjugation by `clifford` fixes `pauli` as a symplectic vector (ignoring sign).
fn is_conjugation_fixed(clifford: &CliffordUnitary, pauli: &SparsePauli) -> bool {
    let image = clifford.image(pauli);
    image.x_bits() == pauli.x_bits() && image.z_bits() == pauli.z_bits()
}

fn is_non_identity(pauli: &SparsePauli) -> bool {
    !(pauli.x_bits().is_zero() && pauli.z_bits().is_zero())
}

/// The strict minimal factor count `r = 2n - dim Fix(F)`; the greedy decomposition returns `r` or a
/// little more.
fn residue_rank(clifford: &CliffordUnitary) -> usize {
    2 * clifford.num_qubits() - clifford_centralizer(clifford).len()
}

fn assert_valid_decomposition(clifford: &CliffordUnitary) {
    let qubit_count = clifford.num_qubits();
    let transvections = clifford_to_transvections(clifford);

    let rebuilt = symplectic_action_from_transvections(&transvections, qubit_count);
    assert_eq!(
        rebuilt.symplectic_matrix(),
        clifford.symplectic_matrix(),
        "replayed transvections must reproduce the symplectic action"
    );

    for transvection in &transvections {
        assert_eq!(transvection.xz_phase_exponent(), 0, "factors carry no phase");
        assert!(is_non_identity(transvection), "factors are non-identity Paulis");
    }

    let minimum = residue_rank(clifford);
    assert!(
        transvections.len() >= minimum,
        "a decomposition cannot be shorter than the minimum {minimum}, got {}",
        transvections.len()
    );
    assert!(
        transvections.len() <= 4 * qubit_count + 2,
        "the decomposition must be linear in the qubit count, got {}",
        transvections.len()
    );
}

#[test]
fn identity_decomposes_to_no_transvections() {
    for qubit_count in 0..5 {
        let identity = CliffordUnitary::identity(qubit_count);
        let transvections = clifford_to_transvections(&identity);
        assert!(
            transvections.is_empty(),
            "identity has no transvections (qubit_count {qubit_count})"
        );
        let centralizer = clifford_centralizer(&identity);
        assert_eq!(
            centralizer.len(),
            2 * qubit_count,
            "identity commutes with all {qubit_count} Pauli generators"
        );
    }
}

#[test]
fn single_qubit_gates_reproduce_symplectic_action() {
    let mut s_gate = CliffordUnitary::identity(1);
    s_gate.left_mul_root_z(0);
    assert_valid_decomposition(&s_gate);
    assert_eq!(clifford_to_transvections(&s_gate).len(), 1, "S is one transvection T_Z");

    let mut hadamard = CliffordUnitary::identity(1);
    hadamard.left_mul_hadamard(0);
    assert_valid_decomposition(&hadamard);
    assert_eq!(
        clifford_to_transvections(&hadamard).len(),
        1,
        "H is the transvection T_Y"
    );
}

#[test]
fn pauli_gates_are_conjugation_trivial() {
    // Pauli operators act trivially by conjugation (sign-only), so their symplectic action is the
    // identity and no transvections are needed.
    for axis in 0..3 {
        let mut clifford = CliffordUnitary::identity(1);
        match axis {
            0 => clifford.left_mul_pauli(&SparsePauli::x(0, 1)),
            1 => clifford.left_mul_pauli(&SparsePauli::z(0, 1)),
            _ => clifford.left_mul_pauli(&SparsePauli::y(0, 1)),
        }
        assert!(
            clifford_to_transvections(&clifford).is_empty(),
            "Pauli axis {axis} needs no factor"
        );
        assert_eq!(
            clifford_centralizer(&clifford).len(),
            2,
            "a Pauli commutes with all generators"
        );
    }
}

#[test]
fn swap_exercises_the_hyperbolic_branch() {
    // SWAP is hyperbolic (its residue space is totally isotropic), so the greedy reduction returns
    // r + 1 = 3 transvections, where r = 2n - dim Fix = 4 - 2 = 2.
    let mut swap = CliffordUnitary::identity(2);
    swap.left_mul_swap(0, 1);
    assert_valid_decomposition(&swap);
    assert_eq!(residue_rank(&swap), 2);
    assert_eq!(clifford_to_transvections(&swap).len(), 3);
    assert_eq!(clifford_centralizer(&swap).len(), 2);
}

#[test]
fn two_qubit_gates_reproduce_symplectic_action() {
    let mut cx = CliffordUnitary::identity(2);
    cx.left_mul_cx(0, 1);
    assert_valid_decomposition(&cx);

    let mut cz = CliffordUnitary::identity(2);
    cz.left_mul_cz(0, 1);
    assert_valid_decomposition(&cz);
}

#[test]
fn composite_circuit_reproduces_symplectic_action() {
    let mut clifford = CliffordUnitary::identity(4);
    clifford.left_mul_hadamard(0);
    clifford.left_mul_cx(0, 1);
    clifford.left_mul_root_z(2);
    clifford.left_mul_cz(1, 3);
    clifford.left_mul_swap(2, 3);
    clifford.left_mul_hadamard(3);
    assert_valid_decomposition(&clifford);
}

#[test]
fn centralizer_generators_are_conjugation_fixed_and_independent() {
    let mut clifford = CliffordUnitary::identity(3);
    clifford.left_mul_hadamard(0);
    clifford.left_mul_cx(0, 1);
    clifford.left_mul_root_z(2);

    let centralizer = clifford_centralizer(&clifford);
    assert!(centralizer.iter().all(|pauli| is_conjugation_fixed(&clifford, pauli)));
    assert!(centralizer.iter().all(is_non_identity));
    assert_eq!(
        centralizer.len(),
        2 * clifford.num_qubits() - residue_rank(&clifford),
        "the centralizer dimension is 2n - r"
    );
}

fn random_clifford(qubit_count: usize, seed: u64) -> CliffordUnitary {
    let mut random_number_generator = StdRng::seed_from_u64(seed);
    CliffordUnitary::random(qubit_count, &mut random_number_generator)
}

#[test]
fn many_random_cliffords_reproduce_symplectic_action() {
    // A deterministic sweep giving broad coverage independent of the proptest shrink budget.
    for qubit_count in 0..7 {
        for seed in 0..200 {
            assert_valid_decomposition(&random_clifford(qubit_count, seed));
        }
    }
}

/// A single Clifford generator, modeled as an operation so proptest can shrink a failing input down
/// to a minimal gate sequence (unlike an opaque RNG seed).
#[derive(Clone, Debug)]
enum Gate {
    Single { op: UnitaryOp, qubit: usize },
    Two { op: UnitaryOp, first: usize, second: usize },
}

fn distinct_pair(qubit_count: usize) -> impl Strategy<Value = (usize, usize)> {
    (0..qubit_count, 0..qubit_count - 1)
        .prop_map(|(first, second)| (first, if second < first { second } else { second + 1 }))
}

fn gate_strategy(qubit_count: usize) -> BoxedStrategy<Gate> {
    use UnitaryOp::{ControlledX, ControlledZ, Hadamard, SqrtX, SqrtZ, Swap, X, Y, Z};
    let single = (
        prop::sample::select(vec![Hadamard, SqrtZ, SqrtX, X, Y, Z]),
        0..qubit_count,
    )
        .prop_map(|(op, qubit)| Gate::Single { op, qubit });
    if qubit_count < 2 {
        return single.boxed();
    }
    let two = (
        prop::sample::select(vec![ControlledX, ControlledZ, Swap]),
        distinct_pair(qubit_count),
    )
        .prop_map(|(op, (first, second))| Gate::Two { op, first, second });
    prop_oneof![3 => single, 1 => two].boxed()
}

fn clifford_from_gates(qubit_count: usize, gates: &[Gate]) -> CliffordUnitary {
    let mut clifford = CliffordUnitary::identity(qubit_count);
    for gate in gates {
        match *gate {
            Gate::Single { op, qubit } => clifford.left_mul(op, &[qubit]),
            Gate::Two { op, first, second } => clifford.left_mul(op, &[first, second]),
        }
    }
    clifford
}

/// A qubit count paired with a random gate sequence acting on it.
fn scenario() -> impl Strategy<Value = (usize, Vec<Gate>)> {
    (1usize..7).prop_flat_map(|qubit_count| {
        vec(gate_strategy(qubit_count), 0..=3 * qubit_count).prop_map(move |gates| (qubit_count, gates))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn reproduces_symplectic_action((qubit_count, gates) in scenario()) {
        let clifford = clifford_from_gates(qubit_count, &gates);
        let transvections = clifford_to_transvections(&clifford);
        let rebuilt = symplectic_action_from_transvections(&transvections, qubit_count);
        prop_assert_eq!(rebuilt.symplectic_matrix(), clifford.symplectic_matrix());
    }

    #[test]
    fn decomposition_is_linear_and_no_shorter_than_minimum((qubit_count, gates) in scenario()) {
        let clifford = clifford_from_gates(qubit_count, &gates);
        let transvection_count = clifford_to_transvections(&clifford).len();
        let minimum = residue_rank(&clifford);
        prop_assert!(
            transvection_count >= minimum,
            "got {transvection_count} factors, below the minimum {minimum}"
        );
        prop_assert!(
            transvection_count <= 4 * qubit_count + 2,
            "got {transvection_count} factors, above the linear bound"
        );
    }

    #[test]
    fn centralizer_is_conjugation_fixed((qubit_count, gates) in scenario()) {
        let clifford = clifford_from_gates(qubit_count, &gates);
        for generator in clifford_centralizer(&clifford) {
            prop_assert!(is_conjugation_fixed(&clifford, &generator));
            prop_assert!(is_non_identity(&generator), "centralizer generators must be non-identity");
        }
    }
}
