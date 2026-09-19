//! Tests for the Clifford -> transvection decomposition (arXiv:2102.11380).
//!
//! The greedy decomposition reproduces the *symplectic action* (ignoring Pauli-image signs and the
//! global phase) with a linear number of factors. It is not guaranteed to hit the strict minimum,
//! so its tests validate the symplectic-action round trip, the linear factor bound, and the
//! centralizer contract. Separate tests cover the minimal decomposition.

use binar::{Bitwise, IndexSet};
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

/// The residue rank `r = 2n - dim Fix(F)`, a lower bound on every decomposition.
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

    let lower_bound = residue_rank(clifford);
    assert!(
        transvections.len() >= lower_bound,
        "a decomposition cannot be shorter than the residue rank {lower_bound}, got {}",
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
        let lower_bound = residue_rank(&clifford);
        prop_assert!(
            transvection_count >= lower_bound,
            "got {transvection_count} factors, below the residue rank {lower_bound}"
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

use paulimer::clifford::clifford_to_transvections_minimal;
use std::collections::HashMap;

/// A symplectic action matrix over GF(2) as a row-major boolean grid (test-local, used only by the
/// brute-force minimality oracle).
type ActionMatrix = Vec<Vec<bool>>;

/// The image-convention symplectic action of `clifford`: row `k` is the image of the `k`-th standard
/// basis Pauli. The minimal transvection length is a conjugation invariant, so any faithful matrix
/// realization yields the same brute-force minimum.
fn action_of(clifford: &CliffordUnitary) -> ActionMatrix {
    let qubit_count = clifford.num_qubits();
    let dimension = 2 * qubit_count;
    let basis: Vec<SparsePauli> = (0..qubit_count)
        .map(|qubit| SparsePauli::x(qubit, qubit_count))
        .chain((0..qubit_count).map(|qubit| SparsePauli::z(qubit, qubit_count)))
        .collect();
    let mut matrix = vec![vec![false; dimension]; dimension];
    for (row, pauli) in basis.iter().enumerate() {
        let image = clifford.image(pauli);
        for qubit in 0..qubit_count {
            matrix[row][qubit] = image.x_bits().index(qubit);
            matrix[row][qubit_count + qubit] = image.z_bits().index(qubit);
        }
    }
    matrix
}

fn multiply(left: &ActionMatrix, right: &ActionMatrix) -> ActionMatrix {
    let dimension = left.len();
    let mut product = vec![vec![false; dimension]; dimension];
    for i in 0..dimension {
        for k in 0..dimension {
            if left[i][k] {
                for j in 0..dimension {
                    product[i][j] ^= right[k][j];
                }
            }
        }
    }
    product
}

fn transvection(vector: &[bool], qubit_count: usize) -> ActionMatrix {
    let dimension = 2 * qubit_count;
    let mut matrix = vec![vec![false; dimension]; dimension];
    for (row, output) in matrix.iter_mut().enumerate() {
        output[row] = true;
        let coupling = if row < qubit_count {
            vector[qubit_count + row]
        } else {
            vector[row - qubit_count]
        };
        if coupling {
            for (column, slot) in output.iter_mut().enumerate() {
                *slot ^= vector[column];
            }
        }
    }
    matrix
}

fn encode(matrix: &ActionMatrix) -> u32 {
    let mut key = 0u32;
    let mut bit = 0;
    for row in matrix {
        for &value in row {
            if value {
                key |= 1 << bit;
            }
            bit += 1;
        }
    }
    key
}

/// The exact minimal transvection length of `clifford`'s symplectic action, by breadth-first search
/// over the symplectic group. Only tractable for small qubit counts (`n <= 2`).
fn minimal_length_oracle(clifford: &CliffordUnitary) -> usize {
    let qubit_count = clifford.num_qubits();
    let dimension = 2 * qubit_count;
    let identity: ActionMatrix = (0..dimension)
        .map(|i| (0..dimension).map(|j| i == j).collect())
        .collect();
    let target = encode(&action_of(clifford));
    let generators: Vec<ActionMatrix> = (1..(1u32 << dimension))
        .map(|mask| {
            let vector: Vec<bool> = (0..dimension).map(|bit| mask & (1 << bit) != 0).collect();
            transvection(&vector, qubit_count)
        })
        .collect();
    let mut distances: HashMap<u32, usize> = HashMap::new();
    distances.insert(encode(&identity), 0);
    let mut frontier = vec![identity];
    let mut distance = 0;
    while !frontier.is_empty() {
        if distances.contains_key(&target) {
            break;
        }
        let mut next = Vec::new();
        for current in &frontier {
            for generator in &generators {
                let product = multiply(current, generator);
                let key = encode(&product);
                if let std::collections::hash_map::Entry::Vacant(entry) = distances.entry(key) {
                    entry.insert(distance + 1);
                    next.push(product);
                }
            }
        }
        frontier = next;
        distance += 1;
    }
    distances[&target]
}

fn assert_valid_minimal_decomposition(clifford: &CliffordUnitary) {
    let qubit_count = clifford.num_qubits();
    let transvections = clifford_to_transvections_minimal(clifford);

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

    let rank = residue_rank(clifford);
    assert!(
        transvections.len() == rank || transvections.len() == rank + 1,
        "the minimal count is r or r + 1 (r = {rank}), got {}",
        transvections.len()
    );
    assert!(
        transvections.len() <= clifford_to_transvections(clifford).len(),
        "the minimal decomposition cannot exceed the greedy one"
    );
}

#[test]
fn minimal_identity_decomposes_to_no_transvections() {
    for qubit_count in 0..5 {
        assert!(clifford_to_transvections_minimal(&CliffordUnitary::identity(qubit_count)).is_empty());
    }
}

#[test]
fn minimal_single_qubit_gates() {
    let mut s_gate = CliffordUnitary::identity(1);
    s_gate.left_mul_root_z(0);
    assert_valid_minimal_decomposition(&s_gate);
    assert_eq!(clifford_to_transvections_minimal(&s_gate).len(), 1);

    let mut hadamard = CliffordUnitary::identity(1);
    hadamard.left_mul_hadamard(0);
    assert_valid_minimal_decomposition(&hadamard);
    assert_eq!(clifford_to_transvections_minimal(&hadamard).len(), 1);
}

#[test]
fn minimal_swap_needs_r_plus_one() {
    let mut swap = CliffordUnitary::identity(2);
    swap.left_mul_swap(0, 1);
    assert_valid_minimal_decomposition(&swap);
    assert_eq!(residue_rank(&swap), 2);
    assert_eq!(clifford_to_transvections_minimal(&swap).len(), 3);
}

#[test]
fn minimal_callan_class_a_needs_r_plus_one() {
    let centers = [
        SparsePauli::x(0, 2),
        SparsePauli::x(1, 2),
        SparsePauli::from_bits([0, 1].into_iter().collect(), IndexSet::new(), 0),
        SparsePauli::z(0, 2),
    ];
    let clifford = symplectic_action_from_transvections(&centers, 2);
    let action = action_of(&clifford);

    assert_eq!(
        action,
        vec![
            vec![true, false, true, false],
            vec![false, true, false, false],
            vec![false, true, true, false],
            vec![true, false, true, true],
        ]
    );
    assert!(action[0][2], "⟨X₀, X₀F⟩ = 1, so F is non-hyperbolic");
    assert_eq!(residue_rank(&clifford), 3);
    assert_eq!(minimal_length_oracle(&clifford), 4);
    assert_eq!(clifford_to_transvections_minimal(&clifford).len(), 4);
}

#[test]
fn minimal_two_qubit_gates() {
    let mut cx = CliffordUnitary::identity(2);
    cx.left_mul_cx(0, 1);
    assert_valid_minimal_decomposition(&cx);

    let mut cz = CliffordUnitary::identity(2);
    cz.left_mul_cz(0, 1);
    assert_valid_minimal_decomposition(&cz);
}

#[test]
fn minimal_composite_circuit() {
    let mut clifford = CliffordUnitary::identity(4);
    clifford.left_mul_hadamard(0);
    clifford.left_mul_cx(0, 1);
    clifford.left_mul_root_z(2);
    clifford.left_mul_cz(1, 3);
    clifford.left_mul_swap(2, 3);
    clifford.left_mul_hadamard(3);
    assert_valid_minimal_decomposition(&clifford);
}

#[test]
fn minimal_matches_brute_force_oracle_on_one_and_two_qubits() {
    // Exact minimality against an independent breadth-first search over the symplectic group.
    for qubit_count in 0..=2 {
        for seed in 0..400 {
            let clifford = random_clifford(qubit_count, seed);
            let decomposed = clifford_to_transvections_minimal(&clifford);
            let rebuilt = symplectic_action_from_transvections(&decomposed, qubit_count);
            assert_eq!(rebuilt.symplectic_matrix(), clifford.symplectic_matrix());
            assert_eq!(
                decomposed.len(),
                minimal_length_oracle(&clifford),
                "decomposition length must equal the brute-force minimum (n={qubit_count}, seed={seed})"
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn minimal_reproduces_symplectic_action((qubit_count, gates) in scenario()) {
        let clifford = clifford_from_gates(qubit_count, &gates);
        let transvections = clifford_to_transvections_minimal(&clifford);
        let rebuilt = symplectic_action_from_transvections(&transvections, qubit_count);
        prop_assert_eq!(rebuilt.symplectic_matrix(), clifford.symplectic_matrix());
    }

    #[test]
    fn minimal_is_r_or_r_plus_one_and_at_most_greedy((qubit_count, gates) in scenario()) {
        let clifford = clifford_from_gates(qubit_count, &gates);
        let minimal = clifford_to_transvections_minimal(&clifford).len();
        let greedy = clifford_to_transvections(&clifford).len();
        let residue = residue_rank(&clifford);
        prop_assert!(minimal == residue || minimal == residue + 1, "got {minimal}, r = {residue}");
        prop_assert!(minimal <= greedy, "minimal {minimal} exceeded greedy {greedy}");
    }
}
