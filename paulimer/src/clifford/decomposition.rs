//! Decomposition of Clifford unitaries into products of `π/4` Pauli exponents.

use binar::Bitwise;

use crate::clifford::{Clifford, CliffordMutable, PhasedCliffordUnitary};
use crate::{CliffordUnitary, Pauli, PauliMutable, SparsePauli};

/// Decomposes `clifford` into an ordered product of `π/4` Pauli exponents.
///
/// Returns a list of Pauli operators `[P₁, …, P_k]` such that left-multiplying the identity by
/// `exp(iπ/4·P₁)`, then `exp(iπ/4·P₂)`, …, then `exp(iπ/4·P_k)` reproduces `clifford` exactly,
/// including the Pauli-image signs (the full tableau, not merely the symplectic action). Each factor
/// is applied with [`CliffordMutable::left_mul_pauli_exp`]; the sign of a factor (its
/// [`Pauli::xz_phase_exponent`]) selects `exp(±iπ/4·P)`.
///
/// Because [`PhasedCliffordUnitary::left_mul_pauli_exp`](crate::clifford::PhasedCliffordUnitary::left_mul_pauli_exp)
/// applies the same factors with exact `ζ₈` phase tracking, replaying the returned list on a phased
/// operator yields a *well-defined* global phase. This is the building block used to recover the
/// global phase in the auxiliary-qubit separation of §4.3 of
/// [arXiv:2603.24717](https://arxiv.org/abs/2603.24717): decompose each Clifford factor and replay it
/// on a [`PhasedCliffordUnitary`](crate::clifford::PhasedCliffordUnitary).
///
/// # Algorithm
///
/// Reduce a working copy to the identity by left-multiplying `π/4` exponents `E₁, …, E_m`, so that
/// `E_m ⋯ E₁ · clifford = I` and hence `clifford = E₁⁻¹ ⋯ E_m⁻¹`. Replaying the inverses in reverse
/// order (each `exp(iπ/4·P)⁻¹ = exp(iπ/4·(−P))`) rebuilds `clifford` from the identity.
///
/// # Examples
///
/// ```
/// use paulimer::CliffordUnitary;
/// use paulimer::clifford::{clifford_to_pauli_exponents, Clifford, CliffordMutable};
///
/// let mut clifford = CliffordUnitary::identity(2);
/// clifford.left_mul_hadamard(0);
/// clifford.left_mul_cx(0, 1);
///
/// let exponents = clifford_to_pauli_exponents(&clifford);
///
/// let mut rebuilt = CliffordUnitary::identity(2);
/// for pauli in &exponents {
///     rebuilt.left_mul_pauli_exp(pauli);
/// }
/// assert_eq!(rebuilt, clifford);
/// ```
#[must_use]
pub fn clifford_to_pauli_exponents(clifford: &CliffordUnitary) -> Vec<SparsePauli> {
    let mut recorded = Reduction::new();
    let mut working = clifford.clone();
    for pivot in 0..working.num_qubits() {
        recorded.clear_x_image(&mut working, pivot);
        recorded.clear_z_image(&mut working, pivot);
    }
    debug_assert!(working.is_identity(), "Clifford reduction did not reach the identity");
    recorded.into_decomposition(clifford.num_qubits())
}

/// Builds the phased Clifford whose tableau equals `clifford`, with the canonical global phase fixed
/// by decomposing the whole Clifford into `π/4` Pauli exponents and replaying them.
pub(super) fn phased_from_clifford(clifford: &CliffordUnitary) -> PhasedCliffordUnitary {
    let mut recorded = Reduction::new();
    let mut working = clifford.clone();
    for pivot in 0..working.num_qubits() {
        recorded.clear_x_image(&mut working, pivot);
        recorded.clear_z_image(&mut working, pivot);
    }
    debug_assert!(working.is_identity(), "Clifford reduction did not reach the identity");
    recorded.into_phased_clifford(clifford.num_qubits())
}

#[derive(Clone, Copy)]
enum ReductionStep {
    Hadamard(usize),
    RootZ(usize),
    RootZInverse(usize),
    RootX(usize),
    ControlledX(usize, usize),
    ControlledZ(usize, usize),
    PauliZ(usize),
    PauliX(usize),
}

/// Accumulates the elementary reduction steps applied while reducing a Clifford to the identity.
struct Reduction {
    applied: Vec<ReductionStep>,
}

impl Reduction {
    fn new() -> Self {
        Reduction { applied: Vec::new() }
    }

    fn single_x(qubit: usize, qubit_count: usize) -> SparsePauli {
        SparsePauli::x(qubit, qubit_count)
    }

    fn single_z(qubit: usize, qubit_count: usize) -> SparsePauli {
        SparsePauli::z(qubit, qubit_count)
    }

    /// `Z_control · X_target`, the generator of a controlled-`X`.
    fn control_x(control: usize, target: usize, qubit_count: usize) -> SparsePauli {
        let mut pauli = SparsePauli::z(control, qubit_count);
        pauli.mul_assign_left_x(target);
        pauli
    }

    /// `Z_a · Z_b`, the generator of a controlled-`Z`.
    fn control_z(first: usize, second: usize, qubit_count: usize) -> SparsePauli {
        let mut pauli = SparsePauli::z(first, qubit_count);
        pauli.mul_assign_left_z(second);
        pauli
    }

    fn hadamard(&mut self, working: &mut CliffordUnitary, qubit: usize) {
        working.left_mul_hadamard(qubit);
        self.applied.push(ReductionStep::Hadamard(qubit));
    }

    fn root_z(&mut self, working: &mut CliffordUnitary, qubit: usize) {
        working.left_mul_root_z_inverse(qubit);
        self.applied.push(ReductionStep::RootZ(qubit));
    }

    fn root_z_inverse(&mut self, working: &mut CliffordUnitary, qubit: usize) {
        working.left_mul_root_z(qubit);
        self.applied.push(ReductionStep::RootZInverse(qubit));
    }

    fn root_x(&mut self, working: &mut CliffordUnitary, qubit: usize) {
        working.left_mul_root_x_inverse(qubit);
        self.applied.push(ReductionStep::RootX(qubit));
    }

    fn controlled_x(&mut self, working: &mut CliffordUnitary, control: usize, target: usize) {
        working.left_mul_cx(control, target);
        self.applied.push(ReductionStep::ControlledX(control, target));
    }

    fn controlled_z(&mut self, working: &mut CliffordUnitary, first: usize, second: usize) {
        working.left_mul_cz(first, second);
        self.applied.push(ReductionStep::ControlledZ(first, second));
    }

    /// Conjugation by the Pauli `Z_qubit`, flipping the sign of an `X`-type image on `qubit`.
    fn pauli_z(&mut self, working: &mut CliffordUnitary, qubit: usize) {
        working.left_mul_z(qubit);
        self.applied.push(ReductionStep::PauliZ(qubit));
    }

    /// Conjugation by the Pauli `X_qubit`, flipping the sign of a `Z`-type image on `qubit`.
    fn pauli_x(&mut self, working: &mut CliffordUnitary, qubit: usize) {
        working.left_mul_x(qubit);
        self.applied.push(ReductionStep::PauliX(qubit));
    }

    /// Turns the image of `X_pivot` into `+X_pivot` using gates supported on qubits `≥ pivot`.
    fn clear_x_image(&mut self, working: &mut CliffordUnitary, pivot: usize) {
        let count = working.num_qubits();
        let image = working.image_x(pivot);
        if !(pivot..count).any(|qubit| x_bit(&image, qubit)) {
            let qubit = (pivot..count)
                .find(|&qubit| z_bit(&image, qubit))
                .expect("a non-identity image has an X or Z component");
            self.hadamard(working, qubit);
        }
        let image = working.image_x(pivot);
        if !x_bit(&image, pivot) {
            let qubit = (pivot..count)
                .find(|&qubit| qubit != pivot && x_bit(&image, qubit))
                .expect("the image has an X component to move onto the pivot");
            self.controlled_x(working, qubit, pivot);
        }
        let image = working.image_x(pivot);
        for qubit in (pivot + 1)..count {
            if x_bit(&image, qubit) {
                self.controlled_x(working, pivot, qubit);
            }
        }
        let image = working.image_x(pivot);
        if z_bit(&image, pivot) {
            self.root_z_inverse(working, pivot);
        }
        let image = working.image_x(pivot);
        for qubit in (pivot + 1)..count {
            if z_bit(&image, qubit) {
                self.controlled_z(working, pivot, qubit);
            }
        }
        if working.image_x(pivot).xz_phase_exponent() != 0 {
            self.pauli_z(working, pivot);
        }
    }

    /// Turns the image of `Z_pivot` into `+Z_pivot`, assuming the image of `X_pivot` is already
    /// `+X_pivot`; every gate used fixes `X_pivot`.
    fn clear_z_image(&mut self, working: &mut CliffordUnitary, pivot: usize) {
        let count = working.num_qubits();
        let mut image = working.image_z(pivot);
        if x_bit(&image, pivot) {
            self.root_x(working, pivot);
            image = working.image_z(pivot);
        }
        let mut x_qubits = Vec::new();
        for qubit in (pivot + 1)..count {
            match (x_bit(&image, qubit), z_bit(&image, qubit)) {
                (false, false) => {}
                (false, true) => {
                    self.controlled_x(working, qubit, pivot);
                }
                (true, true) => {
                    self.root_z(working, qubit);
                    x_qubits.push(qubit);
                }
                (true, false) => {
                    x_qubits.push(qubit);
                }
            }
        }
        if let Some(&lead_qubit) = x_qubits.first() {
            for &target in &x_qubits[1..] {
                self.controlled_x(working, lead_qubit, target);
            }
            self.hadamard(working, lead_qubit);
            self.controlled_x(working, lead_qubit, pivot);
        }
        if working.image_z(pivot).xz_phase_exponent() != 0 {
            self.pauli_x(working, pivot);
        }
    }

    /// The exponents that rebuild the original Clifford from the identity.
    fn into_decomposition(self, qubit_count: usize) -> Vec<SparsePauli> {
        let mut exponents = Vec::new();
        for step in self.applied.into_iter().rev() {
            match step {
                ReductionStep::Hadamard(qubit) => {
                    exponents.push(negated(Self::single_x(qubit, qubit_count)));
                    exponents.push(negated(Self::single_z(qubit, qubit_count)));
                    exponents.push(negated(Self::single_x(qubit, qubit_count)));
                }
                ReductionStep::RootZ(qubit) => {
                    exponents.push(negated(Self::single_z(qubit, qubit_count)));
                }
                ReductionStep::RootZInverse(qubit) => {
                    exponents.push(Self::single_z(qubit, qubit_count));
                }
                ReductionStep::RootX(qubit) => {
                    exponents.push(negated(Self::single_x(qubit, qubit_count)));
                }
                ReductionStep::ControlledX(control, target) => {
                    exponents.push(Self::single_x(target, qubit_count));
                    exponents.push(Self::single_z(control, qubit_count));
                    exponents.push(negated(Self::control_x(control, target, qubit_count)));
                }
                ReductionStep::ControlledZ(first, second) => {
                    exponents.push(Self::single_z(second, qubit_count));
                    exponents.push(Self::single_z(first, qubit_count));
                    exponents.push(negated(Self::control_z(first, second, qubit_count)));
                }
                ReductionStep::PauliZ(qubit) => {
                    exponents.push(negated(Self::single_z(qubit, qubit_count)));
                    exponents.push(negated(Self::single_z(qubit, qubit_count)));
                }
                ReductionStep::PauliX(qubit) => {
                    exponents.push(negated(Self::single_x(qubit, qubit_count)));
                    exponents.push(negated(Self::single_x(qubit, qubit_count)));
                }
            }
        }
        exponents
    }

    /// Replays the inverse reduction steps directly onto a [`PhasedCliffordUnitary`], preserving
    /// the exact `ζ₈` global phase of the `π/4` Pauli exponent decomposition.
    fn into_phased_clifford(self, qubit_count: usize) -> PhasedCliffordUnitary {
        let mut phased = PhasedCliffordUnitary::identity(qubit_count);
        for step in self.applied.into_iter().rev() {
            match step {
                ReductionStep::Hadamard(qubit) => {
                    phased.left_mul_hadamard(qubit);
                    phased.left_mul_global_phase(6);
                }
                ReductionStep::RootZ(qubit) => {
                    phased.left_mul_root_z(qubit);
                    phased.left_mul_global_phase(7);
                }
                ReductionStep::RootZInverse(qubit) => {
                    phased.left_mul_root_z_inverse(qubit);
                    phased.left_mul_global_phase(1);
                }
                ReductionStep::RootX(qubit) => {
                    phased.left_mul_root_x(qubit);
                    phased.left_mul_global_phase(7);
                }
                ReductionStep::ControlledX(control, target) => {
                    phased.left_mul_cx(control, target);
                    phased.left_mul_global_phase(1);
                }
                ReductionStep::ControlledZ(first, second) => {
                    phased.left_mul_cz(first, second);
                    phased.left_mul_global_phase(1);
                }
                ReductionStep::PauliZ(qubit) => {
                    phased.left_mul_z(qubit);
                    phased.left_mul_global_phase(6);
                }
                ReductionStep::PauliX(qubit) => {
                    phased.left_mul_x(qubit);
                    phased.left_mul_global_phase(6);
                }
            }
        }
        phased
    }
}

/// `exp(iπ/4·P)⁻¹ = exp(iπ/4·(−P))`, with `−P` encoded as a phase-exponent shift of two.
fn negated(mut pauli: SparsePauli) -> SparsePauli {
    pauli.add_assign_phase_exp(2);
    pauli
}

fn x_bit<P: Pauli>(pauli: &P, qubit: usize) -> bool {
    pauli.x_bits().index(qubit)
}

fn z_bit<P: Pauli>(pauli: &P, qubit: usize) -> bool {
    pauli.z_bits().index(qubit)
}
