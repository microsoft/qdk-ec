//! Noise instructions and Pauli distributions for faulty simulation.
//!
//! Noise is modeled as `Instruction::Noise` variants that can be placed anywhere
//! in a circuit. Each noise instruction specifies:
//! - A probability that a fault occurs
//! - A distribution over Paulis, given that a fault occurred
//! - An optional fault set ID for temporal/spatial correlations
//! - An optional condition based on measurement outcomes

// Allow mathematical notation like P_i, q_i in doc comments
#![allow(clippy::doc_markdown)]

use binar::{Bitwise, BitwiseMut};
use paulimer::pauli::SparsePauli;
use rand::Rng;
use smallvec::SmallVec;

use crate::circuit::{OutcomeId, QubitId};

/// Maximum number of qubits supported by depolarizing faults.
///
/// Limited so that `4^k - 1` fits into a `usize`, allowing distribution lengths
/// to remain `usize` while avoiding unchecked shifts.
pub(crate) const MAX_DEPOLARIZING_QUBITS: usize = (usize::BITS as usize - 1) / 2;

/// A probability distribution over a set of Pauli operators.
///
/// This defines a random variable $P$ taking values in a set of Pauli operators $\{P_i\}$
/// with probabilities $\{q_i\}$, such that $\sum_i q_i = 1$.
///
/// While typically used to describe the "error term" of a noisy channel, i.e., the `distribution` of a [`PauliFault`],
/// this type itself is strictly a distribution definition. It does not encode the probability of the channel
/// acting (the "fault rate").
///
/// # Variants
///
/// * [`Single`](PauliDistribution::Single): A distribution with a single element ($q_0 = 1$).
/// * [`DepolarizingOnQubits`](PauliDistribution::DepolarizingOnQubits): A uniform distribution over all
///   $4^n - 1$ non-identity Paulis on $n$ qubits.
/// * [`UniformOver`](PauliDistribution::UniformOver): A uniform distribution over an explicit list of Paulis.
/// * [`Weighted`](PauliDistribution::Weighted): An arbitrary distribution defined by pairs $(P_i, q_i)$.
#[derive(Debug, Clone)]
pub enum PauliDistribution {
    /// Single deterministic Pauli (no sampling needed).
    Single(SparsePauli),

    /// Uniform over all non-identity Paulis on the given qubits.
    /// Uses fast bit sampling: O(1) space, O(k) time for k qubits.
    DepolarizingOnQubits(SmallVec<[QubitId; 2]>),

    /// Uniform over an explicit list of Paulis.
    UniformOver(Vec<SparsePauli>),

    /// Weighted distribution with precomputed CDF for binary search.
    Weighted { paulis: Vec<SparsePauli>, cdf: Vec<f64> },
}

impl PauliDistribution {
    /// Create a single deterministic Pauli.
    #[must_use]
    pub fn single(pauli: SparsePauli) -> Self {
        Self::Single(pauli)
    }

    /// Create a depolarizing distribution on the given qubits.
    /// Samples uniformly from all non-identity Paulis on these qubits.
    #[must_use]
    pub fn depolarizing(qubits: &[QubitId]) -> Self {
        Self::DepolarizingOnQubits(qubits.iter().copied().collect())
    }

    /// Create a uniform distribution over an explicit list of Paulis.
    #[must_use]
    pub fn uniform(paulis: Vec<SparsePauli>) -> Self {
        Self::UniformOver(paulis.into_iter().collect())
    }

    /// Create a weighted distribution from (Pauli, weight) pairs.
    /// Weights are normalized to sum to 1.
    ///
    /// # Panics
    ///
    /// Panics if the weights do not sum to a positive value.
    #[must_use]
    pub fn weighted(mut pairs: Vec<(SparsePauli, f64)>) -> Self {
        let total: f64 = pairs.iter().map(|(_, w)| w).sum();
        assert!(total > 0.0, "Weights must sum to a positive value");

        // Build CDF
        let mut cumulative = 0.0;
        let mut cdf = Vec::with_capacity(pairs.len());
        for (_, w) in &pairs {
            cumulative += w / total;
            cdf.push(cumulative);
        }
        // Ensure last entry is exactly 1.0 to avoid floating point issues
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }

        let paulis = pairs.drain(..).map(|(p, _)| p).collect();
        Self::Weighted { paulis, cdf }
    }

    /// Sample a Pauli from this distribution.
    #[allow(clippy::cast_possible_truncation)]
    pub fn sample<R: Rng>(&self, rng: &mut R) -> SparsePauli {
        match self {
            Self::Single(p) => p.clone(),

            Self::DepolarizingOnQubits(qubits) => sample_depolarizing_pauli(qubits, rng),

            Self::UniformOver(list) => {
                let idx = rng.gen_range(0..list.len());
                list[idx].clone()
            }

            Self::Weighted { paulis, cdf } => {
                let u: f64 = rng.gen();
                let idx = cdf.partition_point(|&c| c < u);
                paulis[idx.min(paulis.len() - 1)].clone()
            }
        }
    }

    /// Returns the number of elements in this distribution.
    ///
    /// Note: A distribution is never empty (minimum is 1 for `Single` variant),
    /// so no `is_empty` method is provided.
    #[must_use]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Self::Single(_) => 1,
            Self::DepolarizingOnQubits(qubits) => depolarizing_pauli_count(qubits.len()),
            Self::UniformOver(list) => list.len(),
            Self::Weighted { paulis, .. } => paulis.len(),
        }
    }

    /// Returns all elements of this distribution as (Pauli, probability) pairs.
    ///
    /// For depolarizing noise, this enumerates all $4^n - 1$ non-identity Paulis
    /// with uniform probabilities.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn elements(&self) -> Vec<(SparsePauli, f64)> {
        match self {
            Self::Single(p) => vec![(p.clone(), 1.0)],
            Self::DepolarizingOnQubits(qubits) => {
                let count = self.len();
                let prob = 1.0 / count as f64;
                (0..count)
                    .map(|i| (enumerate_depolarizing_pauli(qubits, i), prob))
                    .collect()
            }
            Self::UniformOver(list) => {
                let prob = 1.0 / list.len() as f64;
                list.iter().map(|p| (p.clone(), prob)).collect()
            }
            Self::Weighted { paulis, cdf } => {
                // Convert CDF back to individual probabilities
                let mut probs = Vec::with_capacity(cdf.len());
                let mut prev = 0.0;
                for &c in cdf {
                    probs.push(c - prev);
                    prev = c;
                }
                paulis.iter().zip(probs).map(|(p, prob)| (p.clone(), prob)).collect()
            }
        }
    }
}

/// Sample a random non-identity Pauli on the given qubits.
/// Uses bit manipulation for efficiency: O(1) space, O(k) time.
#[allow(clippy::cast_possible_truncation)]
fn sample_depolarizing_pauli<R: Rng>(qubits: &[QubitId], rng: &mut R) -> SparsePauli {
    use binar::IndexSet;
    use paulimer::pauli::generic::PauliUnitary;

    let qubit_count = qubits.len();
    let bits = sample_non_identity_pauli_bits(qubit_count, rng);

    let mut x_bits = IndexSet::new();
    let mut z_bits = IndexSet::new();
    let mut remaining_bits = bits;

    for &qubit in qubits {
        if remaining_bits & 1 != 0 {
            BitwiseMut::assign_index(&mut x_bits, qubit, true);
        }
        if remaining_bits & 2 != 0 {
            BitwiseMut::assign_index(&mut z_bits, qubit, true);
        }
        remaining_bits >>= 2;
    }

    PauliUnitary::from_bits(x_bits, z_bits, 0)
}

/// Return the Pauli at a specific index in the depolarizing enumeration.
///
/// Maps index in `[0, 4^k - 2)` to a non-identity Pauli on the given qubits.
/// Index 0 maps to the Pauli with bit pattern `0b01` (X on first qubit),
/// and index `4^k - 2` maps to the Pauli with bit pattern `0b11...11` (Y on all qubits).
#[allow(clippy::cast_possible_truncation)]
fn enumerate_depolarizing_pauli(qubits: &[QubitId], index: usize) -> SparsePauli {
    use binar::IndexSet;
    use paulimer::pauli::generic::PauliUnitary;

    let count = depolarizing_pauli_count(qubits.len());
    debug_assert!(
        index < count,
        "index {index} out of range for depolarizing Pauli count {count}"
    );

    // index is in [0, 4^k - 2], add 1 to get non-identity Pauli bits
    let bits = (index + 1) as u64;

    let mut x_bits = IndexSet::new();
    let mut z_bits = IndexSet::new();
    let mut remaining_bits = bits;

    for &qubit in qubits {
        if remaining_bits & 1 != 0 {
            BitwiseMut::assign_index(&mut x_bits, qubit, true);
        }
        if remaining_bits & 2 != 0 {
            BitwiseMut::assign_index(&mut z_bits, qubit, true);
        }
        remaining_bits >>= 2;
    }

    PauliUnitary::from_bits(x_bits, z_bits, 0)
}

#[must_use]
fn depolarizing_pauli_count(qubit_count: usize) -> usize {
    assert!(qubit_count > 0, "Depolarizing faults require at least one qubit");
    assert!(
        qubit_count <= MAX_DEPOLARIZING_QUBITS,
        "Depolarizing faults support at most {MAX_DEPOLARIZING_QUBITS} qubits, got {qubit_count}"
    );
    let bit_length = qubit_count * 2;
    // Bit length is strictly less than `usize::BITS`, so the shift cannot overflow.
    (1usize << bit_length) - 1
}

#[must_use]
pub(crate) fn sample_non_identity_pauli_bits<R: Rng>(qubit_count: usize, rng: &mut R) -> u64 {
    debug_assert!(qubit_count > 0, "Depolarizing faults require at least one qubit");
    debug_assert!(
        qubit_count <= MAX_DEPOLARIZING_QUBITS,
        "Depolarizing faults support at most {MAX_DEPOLARIZING_QUBITS} qubits, got {qubit_count}"
    );
    let bit_length = qubit_count * 2;
    let pauli_count = (1u64 << bit_length) - 1;
    let limit = u64::MAX - (u64::MAX % pauli_count);

    loop {
        let sample = rng.gen::<u64>();
        if sample < limit {
            let index = sample % pauli_count;
            return index + 1;
        }
    }
}

/// Condition for applying noise based on measurement outcomes.
#[derive(Debug, Clone)]
pub struct OutcomeCondition {
    /// Outcome IDs to check.
    pub outcomes: SmallVec<[OutcomeId; 2]>,
    /// Required parity of XOR of the outcomes.
    pub parity: bool,
}

impl OutcomeCondition {
    /// Create a condition that triggers when the XOR of outcomes equals parity.
    #[must_use]
    pub fn new(outcomes: &[OutcomeId], parity: bool) -> Self {
        Self {
            outcomes: outcomes.iter().copied().collect(),
            parity,
        }
    }

    /// Check if the condition is satisfied for a given shot.
    #[must_use]
    pub fn is_satisfied(&self, outcomes: &binar::BitMatrix, shot: usize) -> bool {
        let mut xor_parity = false;
        for &outcome_id in &self.outcomes {
            xor_parity ^= outcomes.row(shot).index(outcome_id);
        }
        xor_parity == self.parity
    }
}

/// A stochastic Pauli noise channel.
///
/// A `PauliFault` models a quantum channel $\mathcal{E}$ acting on a state $\rho$ as:
///
/// $$ \mathcal{E}(\rho) = (1 - p)\rho + p \sum_i q_i P_i \rho P_i^\dagger $$
///
/// where:
/// - $p$ is the `probability` that the fault mechanism is triggered.
/// - $\{ (P_i, q_i) \}$ is the probability distribution defined by `distribution`.
///
/// # Semantics in Simulation
///
/// When simulating this fault:
/// 1. A "fault event" is sampled as a Bernoulli trial with parameter $p$.
/// 2. If the fault occurs, a Pauli operator $P$ is independently sampled from `distribution`.
/// 3. The operator $P$ is applied to the stabilizer state.
///
/// If `distribution` contains the identity operator, it is possible for a fault to "occur"
/// (step 1) without changing the state (step 3). However, the specialized
/// [`DepolarizingOnQubits`](PauliDistribution::DepolarizingOnQubits) variant explicitly excludes identity.
///
/// # Correlated faults
///
/// If multiple `PauliFault` instructions share the same `correlation_id`:
/// 1. **Trigger Coupling**: They are treated as a single probabilistic event. In any given
///    simulation shot, either *all* of them trigger or *none* of them trigger.
/// 2. **Sample Coupling**: If they trigger, they sample from their distributions using the
///    same random index. This allows modeling correlated errors (e.g., "XX on pair A OR
///    ZZ on pair B").
///
/// To ensure valid sample coupling, all faults with the same `correlation_id` must have the same `probability`
/// and have distributions of the same size (see [`PauliDistribution::len`]).
///
/// # Conditional faults
///
/// If present, the fault mechanism is active only if the `condition` is satisfied (i.e.
/// the XOR of the specified measurement outcomes matches the required parity).
/// - If the condition is **false**, the fault is suppressed (probability 0).
/// - If the condition is **true**, the fault occurs with `self.probability`.
///
#[derive(Debug, Clone)]
pub struct PauliFault {
    /// Probability that a fault occurs.
    pub probability: f64,
    /// Distribution over Paulis given a fault.
    pub distribution: PauliDistribution,
    /// Correlation ID: same ID at different locations → same shots affected.
    pub correlation_id: Option<u64>,
    /// Optional condition based on measurement outcomes.
    pub condition: Option<OutcomeCondition>,
}

impl PauliFault {
    /// Create a simple depolarizing noise on the given qubits.
    #[must_use]
    pub fn depolarizing(qubits: &[QubitId], probability: f64) -> Self {
        Self {
            probability,
            distribution: PauliDistribution::depolarizing(qubits),
            correlation_id: None,
            condition: None,
        }
    }

    /// Builder: set correlation ID for correlated faults.
    #[must_use]
    pub fn with_correlation_id(mut self, id: u64) -> Self {
        self.correlation_id = Some(id);
        self
    }

    /// Builder: set condition based on outcomes.
    #[must_use]
    pub fn with_condition(mut self, outcomes: &[OutcomeId], parity: bool) -> Self {
        self.condition = Some(OutcomeCondition::new(outcomes, parity));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statistical_testing::{
        assert_uniform_distribution, two_qubit_pauli_to_bit_index, TOLERANCE_HIGH_SAMPLES,
    };
    use binar::Bitwise;
    use paulimer::pauli::Pauli;
    use rand::rngs::{mock::StepRng, SmallRng};
    use rand::SeedableRng;
    use std::collections::HashSet;
    use std::str::FromStr;

    #[test]
    fn rejection_sampler_handles_retries() {
        let mut rng = StepRng::new(u64::MAX, 1);
        let value = sample_non_identity_pauli_bits(1, &mut rng);
        assert!((1..=3).contains(&value));
    }

    #[test]
    fn depolarizing_samples_non_identity() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dist = PauliDistribution::depolarizing(&[0]);

        for _ in 0..100 {
            let p = dist.sample(&mut rng);
            assert!(p.weight() > 0, "Should not sample identity");
        }
    }

    #[test]
    fn two_qubit_depolarizing_samples_non_identity() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dist = PauliDistribution::depolarizing(&[0, 1]);

        for _ in 0..100 {
            let p = dist.sample(&mut rng);
            assert!(p.weight() > 0, "Should not sample identity");
            for q in p.support() {
                assert!(q <= 1, "Support should be on qubits 0 or 1");
            }
        }
    }

    #[test]
    fn weighted_respects_probabilities() {
        let mut rng = SmallRng::seed_from_u64(42);
        let x = SparsePauli::from_str("X").unwrap();
        let z = SparsePauli::from_str("Z").unwrap();

        let dist = PauliDistribution::weighted(vec![(x.clone(), 0.9), (z.clone(), 0.1)]);

        let mut x_count = 0;
        let trials = 10000;

        for _ in 0..trials {
            let p = dist.sample(&mut rng);
            if p == x {
                x_count += 1;
            }
        }

        let x_ratio = f64::from(x_count) / f64::from(trials);
        assert!((0.85..0.95).contains(&x_ratio), "X ratio {x_ratio} should be ~0.9");
    }

    // ========== Sampling uniformity ==========

    fn one_qubit_pauli_to_index(pauli: &SparsePauli) -> usize {
        let has_x = pauli.x_bits().index(0);
        let has_z = pauli.z_bits().index(0);
        match (has_x, has_z) {
            (true, false) => 0,
            (false, true) => 1,
            (true, true) => 2,
            (false, false) => panic!("Sampled identity"),
        }
    }

    #[test]
    fn depolarizing_one_qubit_samples_uniformly() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dist = PauliDistribution::depolarizing(&[0]);
        let trials = 100_000;

        let mut counts = [0u32; 3];
        for _ in 0..trials {
            let p = dist.sample(&mut rng);
            counts[one_qubit_pauli_to_index(&p)] += 1;
        }

        assert_uniform_distribution(&counts, trials, TOLERANCE_HIGH_SAMPLES, "1-qubit depolarizing");
    }

    #[test]
    fn depolarizing_two_qubit_samples_uniformly() {
        let mut rng = SmallRng::seed_from_u64(42);
        let dist = PauliDistribution::depolarizing(&[0, 1]);
        let trials = 150_000;

        let mut counts = [0u32; 15];
        for _ in 0..trials {
            let p = dist.sample(&mut rng);
            let bits = two_qubit_pauli_to_bit_index(&p);
            assert!(bits > 0, "Sampled identity");
            counts[(bits - 1) as usize] += 1;
        }

        assert_uniform_distribution(&counts, trials, TOLERANCE_HIGH_SAMPLES, "2-qubit depolarizing");
    }

    #[test]
    fn uniform_over_samples_uniformly() {
        let mut rng = SmallRng::seed_from_u64(42);
        let paulis: Vec<SparsePauli> = ["X0", "Y0", "Z0", "X1", "Z1"]
            .iter()
            .map(|s| SparsePauli::from_str(s).unwrap())
            .collect();
        let dist = PauliDistribution::uniform(paulis.clone());
        let trials = 100_000;

        let mut counts = vec![0u32; paulis.len()];
        for _ in 0..trials {
            let p = dist.sample(&mut rng);
            let idx = paulis.iter().position(|q| *q == p).expect("Unknown Pauli sampled");
            counts[idx] += 1;
        }

        assert_uniform_distribution(&counts, trials, TOLERANCE_HIGH_SAMPLES, "UniformOver");
    }

    #[test]
    fn single_always_returns_same_pauli() {
        let mut rng = SmallRng::seed_from_u64(42);
        let pauli = SparsePauli::from_str("XYZ").unwrap();
        let dist = PauliDistribution::single(pauli.clone());

        for _ in 0..100 {
            assert_eq!(dist.sample(&mut rng), pauli);
        }
    }

    // ========== elements() correctness ==========

    #[test]
    fn elements_single_returns_pauli_with_probability_one() {
        let pauli = SparsePauli::from_str("XYZ").unwrap();
        let dist = PauliDistribution::single(pauli.clone());
        let elements = dist.elements();

        assert_eq!(elements.len(), 1);
        assert_eq!(elements[0].0, pauli);
        assert!((elements[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn elements_uniform_returns_equal_probabilities() {
        let paulis: Vec<SparsePauli> = ["X", "Y", "Z"]
            .iter()
            .map(|s| SparsePauli::from_str(s).unwrap())
            .collect();
        let dist = PauliDistribution::uniform(paulis.clone());
        let elements = dist.elements();

        assert_eq!(elements.len(), 3);
        for (pauli, prob) in &elements {
            assert!(paulis.contains(pauli));
            assert!((prob - 1.0 / 3.0).abs() < 1e-10);
        }
    }

    #[test]
    fn elements_weighted_recovers_input_probabilities() {
        let x = SparsePauli::from_str("X").unwrap();
        let y = SparsePauli::from_str("Y").unwrap();
        let z = SparsePauli::from_str("Z").unwrap();

        let dist = PauliDistribution::weighted(vec![(x.clone(), 0.5), (y.clone(), 0.3), (z.clone(), 0.2)]);
        let elements = dist.elements();

        assert_eq!(elements.len(), 3);
        let prob_sum: f64 = elements.iter().map(|(_, p)| p).sum();
        assert!((prob_sum - 1.0).abs() < 1e-10, "Probabilities must sum to 1");

        for (pauli, prob) in &elements {
            let expected = if *pauli == x {
                0.5
            } else if *pauli == y {
                0.3
            } else {
                0.2
            };
            assert!((prob - expected).abs() < 1e-10, "Probability mismatch for {pauli:?}");
        }
    }

    #[test]
    fn elements_depolarizing_one_qubit_returns_three_non_identity_paulis() {
        let dist = PauliDistribution::depolarizing(&[0]);
        let elements = dist.elements();

        assert_eq!(elements.len(), 3, "1-qubit depolarizing has 4^1 - 1 = 3 Paulis");

        let prob_sum: f64 = elements.iter().map(|(_, p)| p).sum();
        assert!((prob_sum - 1.0).abs() < 1e-10, "Probabilities must sum to 1");

        for (pauli, _) in &elements {
            assert!(pauli.weight() > 0, "No identity in depolarizing distribution");
        }

        let unique_paulis: HashSet<_> = elements.iter().map(|(p, _)| format!("{p:#}")).collect();
        assert_eq!(unique_paulis.len(), 3, "All Paulis must be distinct");
    }

    #[test]
    fn elements_depolarizing_two_qubit_returns_fifteen_non_identity_paulis() {
        let dist = PauliDistribution::depolarizing(&[0, 1]);
        let elements = dist.elements();

        assert_eq!(elements.len(), 15, "2-qubit depolarizing has 4^2 - 1 = 15 Paulis");

        let prob_sum: f64 = elements.iter().map(|(_, p)| p).sum();
        assert!((prob_sum - 1.0).abs() < 1e-10, "Probabilities must sum to 1");

        for (pauli, prob) in &elements {
            assert!(pauli.weight() > 0, "No identity in depolarizing distribution");
            assert!((prob - 1.0 / 15.0).abs() < 1e-10, "Uniform probability expected");
        }

        let unique_paulis: HashSet<_> = elements.iter().map(|(p, _)| format!("{p:#}")).collect();
        assert_eq!(unique_paulis.len(), 15, "All Paulis must be distinct");
    }

    // ========== len() consistency ==========

    #[test]
    fn len_matches_elements_count_for_all_variants() {
        let single = PauliDistribution::single(SparsePauli::from_str("X").unwrap());
        assert_eq!(single.len(), single.elements().len());

        let uniform = PauliDistribution::uniform(vec![
            SparsePauli::from_str("X").unwrap(),
            SparsePauli::from_str("Y").unwrap(),
        ]);
        assert_eq!(uniform.len(), uniform.elements().len());

        let weighted = PauliDistribution::weighted(vec![
            (SparsePauli::from_str("X").unwrap(), 0.7),
            (SparsePauli::from_str("Z").unwrap(), 0.3),
        ]);
        assert_eq!(weighted.len(), weighted.elements().len());

        let depol_1 = PauliDistribution::depolarizing(&[0]);
        assert_eq!(depol_1.len(), depol_1.elements().len());
        assert_eq!(depol_1.len(), 3);

        let depol_2 = PauliDistribution::depolarizing(&[0, 1]);
        assert_eq!(depol_2.len(), depol_2.elements().len());
        assert_eq!(depol_2.len(), 15);
    }

    // ========== OutcomeCondition ==========

    #[test]
    fn outcome_condition_parity_true_requires_odd_xor() {
        let mut outcomes = binar::BitMatrix::zeros(4, 3);
        outcomes.set((1, 0), true);
        outcomes.set((2, 0), true);
        outcomes.set((2, 1), true);
        outcomes.set((3, 0), true);
        outcomes.set((3, 1), true);
        outcomes.set((3, 2), true);

        let condition = OutcomeCondition::new(&[0, 1, 2], true);

        assert!(
            !condition.is_satisfied(&outcomes, 0),
            "shot 0: 0 ones → XOR=0 → parity false"
        );
        assert!(
            condition.is_satisfied(&outcomes, 1),
            "shot 1: 1 one → XOR=1 → parity true"
        );
        assert!(
            !condition.is_satisfied(&outcomes, 2),
            "shot 2: 2 ones → XOR=0 → parity false"
        );
        assert!(
            condition.is_satisfied(&outcomes, 3),
            "shot 3: 3 ones → XOR=1 → parity true"
        );
    }

    #[test]
    fn outcome_condition_parity_false_requires_even_xor() {
        let mut outcomes = binar::BitMatrix::zeros(4, 3);
        outcomes.set((1, 0), true);
        outcomes.set((2, 0), true);
        outcomes.set((2, 1), true);
        outcomes.set((3, 0), true);
        outcomes.set((3, 1), true);
        outcomes.set((3, 2), true);

        let condition = OutcomeCondition::new(&[0, 1, 2], false);

        assert!(
            condition.is_satisfied(&outcomes, 0),
            "shot 0: 0 ones → XOR=0 → parity false"
        );
        assert!(
            !condition.is_satisfied(&outcomes, 1),
            "shot 1: 1 one → XOR=1 → parity true"
        );
        assert!(
            condition.is_satisfied(&outcomes, 2),
            "shot 2: 2 ones → XOR=0 → parity false"
        );
        assert!(
            !condition.is_satisfied(&outcomes, 3),
            "shot 3: 3 ones → XOR=1 → parity true"
        );
    }
}
