//! Tests for the Clifford -> transvection decomposition (arXiv:2102.11380).
//!
//! The greedy decomposition reproduces the *symplectic action* (ignoring Pauli-image signs and the
//! global phase) with a linear number of factors. It is a greedy reduction rather than a
//! minimal-length algorithm, so its tests validate the symplectic-action round trip, the validity
//! of the replayed tableau, the residue-rank lower bound, the linear upper bound, and the
//! fixed-space contract, rather than exact minimality. Separate tests cover the minimal
//! decomposition.

use binar::matrix::AlignedBitMatrix;
use binar::{Bitwise, IndexSet};
use paulimer::UnitaryOp;
use paulimer::clifford::{Clifford, CliffordMutable, CliffordUnitary, clifford_fixed_space, clifford_to_transvections};
use paulimer::pauli::{Pauli, PauliMutable, SparsePauli};
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

/// The residue rank `r = rank(I + F)` of the symplectic action.
///
/// This is computed from the symplectic matrix alone so that it stays independent of
/// [`clifford_fixed_space`]; deriving it from the fixed-space dimension would make the fixed-space
/// dimension assertions tautological.
fn residue_rank(clifford: &CliffordUnitary) -> usize {
    let mut residue = clifford.symplectic_matrix();
    residue ^= &AlignedBitMatrix::identity(2 * clifford.num_qubits());
    residue.rank()
}

/// The rank over GF(2) of the symplectic vectors of `paulis`.
fn binary_rank(paulis: &[SparsePauli], qubit_count: usize) -> usize {
    let mut matrix = AlignedBitMatrix::zeros(paulis.len(), 2 * qubit_count);
    for (row, pauli) in paulis.iter().enumerate() {
        for qubit in 0..qubit_count {
            matrix.set((row, qubit), pauli.x_bits().index(qubit));
            matrix.set((row, qubit_count + qubit), pauli.z_bits().index(qubit));
        }
    }
    matrix.rank()
}

fn assert_valid_decomposition(clifford: &CliffordUnitary) {
    let qubit_count = clifford.num_qubits();
    let transvections = clifford_to_transvections(clifford);

    let rebuilt = symplectic_action_from_transvections(&transvections, qubit_count);
    assert!(rebuilt.is_valid());
    assert_eq!(
        rebuilt.symplectic_matrix(),
        clifford.symplectic_matrix(),
        "replayed transvections must reproduce the symplectic action"
    );

    for transvection in &transvections {
        assert!(transvection.is_order_two(), "factors must be Hermitian");
        assert_eq!(transvection.xyz_phase_exponent(), 0, "factors carry no xyz phase");
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
        let fixed_space = clifford_fixed_space(&identity);
        assert_eq!(
            fixed_space.len(),
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
            clifford_fixed_space(&clifford).len(),
            2,
            "a Pauli fixes every generator up to sign"
        );
    }
}

#[test]
fn fixed_space_of_pauli_x_includes_z() {
    let mut clifford = CliffordUnitary::identity(1);
    clifford.left_mul_pauli(&SparsePauli::x(0, 1));
    let pauli_z = SparsePauli::z(0, 1);

    let fixed_space = clifford_fixed_space(&clifford);
    assert!(fixed_space.contains(&pauli_z));
    assert_eq!(SparsePauli::from(clifford.image(&pauli_z)), -pauli_z);
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
    assert_eq!(clifford_fixed_space(&swap).len(), 2);
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
fn fixed_space_generators_are_conjugation_fixed_and_independent() {
    let mut clifford = CliffordUnitary::identity(3);
    clifford.left_mul_hadamard(0);
    clifford.left_mul_cx(0, 1);
    clifford.left_mul_root_z(2);

    let fixed_space = clifford_fixed_space(&clifford);
    assert!(fixed_space.iter().all(|pauli| is_conjugation_fixed(&clifford, pauli)));
    assert!(fixed_space.iter().all(is_non_identity));
    assert_eq!(
        fixed_space.len(),
        2 * clifford.num_qubits() - residue_rank(&clifford),
        "the fixed space has dimension 2n - rank(I + F)"
    );
    assert_eq!(
        binary_rank(&fixed_space, clifford.num_qubits()),
        fixed_space.len(),
        "the generators must be independent"
    );
    assert!(
        fixed_space.iter().all(|pauli| pauli.xyz_phase_exponent() == 0),
        "fixed-space generators must be positive Hermitian observables"
    );
}

#[test]
fn fixed_space_generators_of_a_y_axis_rotation_are_hermitian() {
    let mut clifford = CliffordUnitary::identity(1);
    clifford.left_mul(UnitaryOp::SqrtY, &[0]);

    let fixed_space = clifford_fixed_space(&clifford);
    assert_eq!(fixed_space.len(), 1, "a sqrt(Y) rotation fixes exactly the Y axis");
    assert!(is_conjugation_fixed(&clifford, &fixed_space[0]));
    assert!(fixed_space[0].is_order_two(), "the generator must be Hermitian");
    assert_eq!(
        fixed_space[0].xyz_phase_exponent(),
        0,
        "the generator must be the positive Hermitian representative"
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
        assert_valid_decomposition(&clifford_from_gates(qubit_count, &gates));
    }

    #[test]
    fn fixed_space_is_conjugation_fixed((qubit_count, gates) in scenario()) {
        let clifford = clifford_from_gates(qubit_count, &gates);
        let fixed_space = clifford_fixed_space(&clifford);
        prop_assert_eq!(fixed_space.len(), 2 * qubit_count - residue_rank(&clifford));
        prop_assert_eq!(binary_rank(&fixed_space, qubit_count), fixed_space.len());
        for generator in fixed_space {
            prop_assert!(is_conjugation_fixed(&clifford, &generator));
            prop_assert!(is_non_identity(&generator), "fixed-space generators must be non-identity");
            prop_assert_eq!(generator.xyz_phase_exponent(), 0);
        }
    }
}

use paulimer::clifford::clifford_to_transvections_minimal;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

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

fn assert_valid_minimal_decomposition(clifford: &CliffordUnitary) {
    let qubit_count = clifford.num_qubits();
    let transvections = clifford_to_transvections_minimal(clifford);

    let rebuilt = symplectic_action_from_transvections(&transvections, qubit_count);
    assert!(rebuilt.is_valid());
    assert_eq!(
        rebuilt.symplectic_matrix(),
        clifford.symplectic_matrix(),
        "replayed transvections must reproduce the symplectic action"
    );

    for transvection in &transvections {
        assert!(transvection.is_order_two(), "factors must be Hermitian");
        assert_eq!(transvection.xyz_phase_exponent(), 0, "factors carry no xyz phase");
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
        let no_factors: Vec<SparsePauli> = Vec::new();
        assert_eq!(
            clifford_to_transvections_minimal(&CliffordUnitary::identity(qubit_count)),
            no_factors,
            "the identity needs no transvections"
        );
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
    assert_eq!(enumerate_symplectic_group(2)[&encode(&action)].0, 4);
    assert_eq!(clifford_to_transvections_minimal(&clifford).len(), 4);
}

#[test]
fn minimal_swap_layer_of_32_qubits_finishes() {
    let qubit_count = 32;
    let mut clifford = CliffordUnitary::identity(qubit_count);
    for qubit in (0..qubit_count).step_by(2) {
        clifford.left_mul_swap(qubit, qubit + 1);
    }
    let expected = clifford.symplectic_matrix();
    let (sender, receiver) = std::sync::mpsc::sync_channel(1);
    let worker = std::thread::spawn(move || {
        sender
            .send(clifford_to_transvections_minimal(&clifford))
            .expect("the test must receive the decomposition");
    });
    let factors = receiver
        .recv_timeout(std::time::Duration::from_secs(60))
        .expect("a 32-qubit SWAP layer must finish within 60 seconds");
    worker.join().expect("the decomposition worker must not panic");

    assert_eq!(factors.len(), qubit_count + 1);
    let rebuilt = symplectic_action_from_transvections(&factors, qubit_count);
    assert!(rebuilt.is_valid());
    assert_eq!(rebuilt.symplectic_matrix(), expected);
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

/// The Hermitian Pauli whose symplectic vector is `vector` (`x`-bits first, then `z`-bits).
fn pauli_of_vector(vector: &[bool], qubit_count: usize) -> SparsePauli {
    let x_bits: IndexSet = (0..qubit_count).filter(|&qubit| vector[qubit]).collect();
    let z_bits: IndexSet = (0..qubit_count).filter(|&qubit| vector[qubit_count + qubit]).collect();
    let mut pauli = SparsePauli::from_bits(x_bits, z_bits, 0);
    let phase = u8::try_from(pauli.y_weight() % 4).expect("phase exponent fits in u8");
    pauli.assign_phase_exp(phase);
    pauli
}

/// Every element of `Sp(2n;2)`, reached by breadth-first search over the transvection generators.
///
/// Maps the encoded symplectic action to its exact minimal transvection length and one Clifford
/// realizing it. The search is independent of [`clifford_to_transvections_minimal`].
fn enumerate_symplectic_group(qubit_count: usize) -> HashMap<u32, (usize, CliffordUnitary)> {
    let dimension = 2 * qubit_count;
    let generators: Vec<SparsePauli> = (1..(1u32 << dimension))
        .map(|mask| {
            let vector: Vec<bool> = (0..dimension).map(|bit| mask & (1 << bit) != 0).collect();
            pauli_of_vector(&vector, qubit_count)
        })
        .collect();

    let identity = CliffordUnitary::identity(qubit_count);
    let mut reached = HashMap::new();
    reached.insert(encode(&action_of(&identity)), (0, identity.clone()));
    let mut frontier = vec![identity];
    let mut distance = 0;
    while !frontier.is_empty() {
        distance += 1;
        let mut next = Vec::new();
        for clifford in &frontier {
            for generator in &generators {
                let mut candidate = clifford.clone();
                candidate.left_mul_pauli_exp(generator);
                let key = encode(&action_of(&candidate));
                if let Entry::Vacant(slot) = reached.entry(key) {
                    slot.insert((distance, candidate.clone()));
                    next.push(candidate);
                }
            }
        }
        frontier = next;
    }
    reached
}

#[test]
fn minimal_matches_brute_force_oracle_on_every_one_and_two_qubit_action() {
    for (qubit_count, group_order) in [(1usize, 6usize), (2, 720)] {
        let group = enumerate_symplectic_group(qubit_count);
        assert_eq!(
            group.len(),
            group_order,
            "the search must visit all of Sp({dimension};2)",
            dimension = 2 * qubit_count
        );

        for (minimum, clifford) in group.values() {
            let decomposed = clifford_to_transvections_minimal(clifford);
            let rebuilt = symplectic_action_from_transvections(&decomposed, qubit_count);
            assert!(rebuilt.is_valid());
            assert_eq!(rebuilt.symplectic_matrix(), clifford.symplectic_matrix());
            for transvection in &decomposed {
                assert!(transvection.is_order_two(), "factors must be Hermitian");
                assert_eq!(
                    transvection.xyz_phase_exponent(),
                    0,
                    "factors must be positive Hermitian representatives"
                );
            }
            assert_eq!(
                decomposed.len(),
                *minimum,
                "decomposition length must equal the brute-force minimum (n={qubit_count})"
            );
        }
    }
}

/// A three-qubit symplectic action packed as six six-bit rows, used only by the exhaustive
/// three-qubit oracle.
type PackedAction = u64;

const THREE_QUBIT_DIMENSION: usize = 6;
const THREE_QUBIT_ROW_MASK: u64 = (1 << THREE_QUBIT_DIMENSION) - 1;

fn symplectic_form(left: u64, right: u64) -> bool {
    let left_x = left & 0b111;
    let left_z = left >> 3;
    let right_x = right & 0b111;
    let right_z = right >> 3;
    ((left_x & right_z) ^ (left_z & right_x)).count_ones() % 2 == 1
}

fn apply_packed_transvection(action: PackedAction, vector: u64) -> PackedAction {
    let mut updated = 0;
    for row in 0..THREE_QUBIT_DIMENSION {
        let mut image = (action >> (THREE_QUBIT_DIMENSION * row)) & THREE_QUBIT_ROW_MASK;
        if symplectic_form(image, vector) {
            image ^= vector;
        }
        updated |= image << (THREE_QUBIT_DIMENSION * row);
    }
    updated
}

fn three_qubit_pauli_of_vector(vector: u64) -> SparsePauli {
    let x_bits: IndexSet = (0..3).filter(|&qubit| vector >> qubit & 1 == 1).collect();
    let z_bits: IndexSet = (0..3).filter(|&qubit| vector >> (3 + qubit) & 1 == 1).collect();
    let mut pauli = SparsePauli::from_bits(x_bits, z_bits, 0);
    let phase = u8::try_from(pauli.y_weight() % 4).expect("a Y weight modulo four fits in a byte");
    pauli.assign_phase_exp(phase);
    pauli
}

fn pack_action(clifford: &CliffordUnitary) -> PackedAction {
    let basis: Vec<SparsePauli> = (0..3)
        .map(|qubit| SparsePauli::x(qubit, 3))
        .chain((0..3).map(|qubit| SparsePauli::z(qubit, 3)))
        .collect();
    let mut packed = 0;
    for (row, generator) in basis.iter().enumerate() {
        let image = clifford.image(generator);
        let mut value = 0u64;
        for qubit in 0..3 {
            if image.x_bits().index(qubit) {
                value |= 1 << qubit;
            }
            if image.z_bits().index(qubit) {
                value |= 1 << (3 + qubit);
            }
        }
        packed |= value << (THREE_QUBIT_DIMENSION * row);
    }
    packed
}

/// Exhaustive three-qubit minimality check. It visits all 1,451,520 elements of Sp(6;2) and takes
/// a few minutes, so it is excluded from the default run. Invoke it with
/// `cargo test --profile ci-test -p paulimer --test transvection_test -- --ignored`.
#[test]
#[ignore = "visits all of Sp(6;2) and takes minutes"]
fn minimal_matches_brute_force_oracle_on_every_three_qubit_action() {
    let identity = pack_action(&CliffordUnitary::identity(3));
    let mut reached: HashMap<PackedAction, (usize, u64, PackedAction)> = HashMap::new();
    reached.insert(identity, (0, 0, identity));
    let mut frontier = vec![identity];
    let mut distance = 0usize;

    while !frontier.is_empty() {
        distance += 1;
        let mut next = Vec::new();
        for &action in &frontier {
            for vector in 1..(1u64 << THREE_QUBIT_DIMENSION) {
                let candidate = apply_packed_transvection(action, vector);
                if let Entry::Vacant(slot) = reached.entry(candidate) {
                    slot.insert((distance, vector, action));
                    next.push(candidate);
                }
            }
        }
        frontier = next;
    }
    assert_eq!(reached.len(), 1_451_520, "the search must visit all of Sp(6;2)");

    let mut census: HashMap<(usize, usize), usize> = HashMap::new();
    let mut greedy_excess = 0usize;
    for (&packed, &(minimum, _, _)) in &reached {
        let mut path = Vec::new();
        let mut current = packed;
        while current != identity {
            let (_, vector, parent) = reached[&current];
            path.push(vector);
            current = parent;
        }
        let mut clifford = CliffordUnitary::identity(3);
        for &vector in path.iter().rev() {
            clifford.left_mul_pauli_exp(&three_qubit_pauli_of_vector(vector));
        }
        assert_eq!(pack_action(&clifford), packed);

        let decomposed = clifford_to_transvections_minimal(&clifford);
        assert_eq!(decomposed.len(), minimum, "length must equal the brute-force minimum");
        let rebuilt = symplectic_action_from_transvections(&decomposed, 3);
        assert!(rebuilt.is_valid(), "replayed factors must form a valid tableau");
        assert_eq!(
            pack_action(&rebuilt),
            packed,
            "replayed action must match the BFS element"
        );
        for transvection in &decomposed {
            assert!(transvection.is_order_two(), "factors must be Hermitian");
        }
        let residue_rank = 6 - clifford_fixed_space(&clifford).len();
        *census.entry((residue_rank, minimum)).or_default() += 1;
        if clifford_to_transvections(&clifford).len() > minimum {
            greedy_excess += 1;
        }
    }

    let expected = [
        ((0, 0), 1),
        ((1, 1), 63),
        ((2, 2), 1617),
        ((2, 3), 315),
        ((3, 3), 21420),
        ((3, 4), 7560),
        ((4, 4), 151_284),
        ((4, 5), 51660),
        ((5, 5), 518_112),
        ((5, 6), 90720),
        ((6, 6), 608_768),
    ];
    let mut observed: Vec<((usize, usize), usize)> = census.into_iter().collect();
    observed.sort_unstable();
    assert_eq!(
        observed, expected,
        "the residue-rank and minimum-length census must match"
    );
    assert_eq!(
        greedy_excess, 428_447,
        "the greedy reduction must exceed the minimum on exactly this many elements"
    );
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
