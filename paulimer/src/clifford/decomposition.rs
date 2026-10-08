//! Decomposition of Clifford unitaries into products of `π/4` Pauli exponents.

use binar::Bitwise;

use crate::clifford::{Clifford, CliffordMutable};
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
    recorded.into_decomposition()
}

/// Accumulates the `π/4` exponents applied while reducing a Clifford to the identity.
struct Reduction {
    applied: Vec<SparsePauli>,
}

impl Reduction {
    fn new() -> Self {
        Reduction { applied: Vec::new() }
    }

    /// Left-multiplies `working` by `exp(iπ/4·pauli)` and records the factor.
    fn exp(&mut self, working: &mut CliffordUnitary, pauli: SparsePauli) {
        working.left_mul_pauli_exp(&pauli);
        self.applied.push(pauli);
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
        let count = working.num_qubits();
        self.exp(working, Self::single_x(qubit, count));
        self.exp(working, Self::single_z(qubit, count));
        self.exp(working, Self::single_x(qubit, count));
    }

    fn root_z(&mut self, working: &mut CliffordUnitary, qubit: usize) {
        let count = working.num_qubits();
        self.exp(working, Self::single_z(qubit, count));
    }

    fn root_z_inverse(&mut self, working: &mut CliffordUnitary, qubit: usize) {
        let count = working.num_qubits();
        self.exp(working, negated(Self::single_z(qubit, count)));
    }

    fn root_x(&mut self, working: &mut CliffordUnitary, qubit: usize) {
        let count = working.num_qubits();
        self.exp(working, Self::single_x(qubit, count));
    }

    fn controlled_x(&mut self, working: &mut CliffordUnitary, control: usize, target: usize) {
        let count = working.num_qubits();
        self.exp(working, Self::control_x(control, target, count));
        self.exp(working, negated(Self::single_z(control, count)));
        self.exp(working, negated(Self::single_x(target, count)));
    }

    fn controlled_z(&mut self, working: &mut CliffordUnitary, first: usize, second: usize) {
        let count = working.num_qubits();
        self.exp(working, Self::control_z(first, second, count));
        self.exp(working, negated(Self::single_z(first, count)));
        self.exp(working, negated(Self::single_z(second, count)));
    }

    /// Conjugation by the Pauli `Z_qubit`, flipping the sign of an `X`-type image on `qubit`.
    fn pauli_z(&mut self, working: &mut CliffordUnitary, qubit: usize) {
        let count = working.num_qubits();
        self.exp(working, Self::single_z(qubit, count));
        self.exp(working, Self::single_z(qubit, count));
    }

    /// Conjugation by the Pauli `X_qubit`, flipping the sign of a `Z`-type image on `qubit`.
    fn pauli_x(&mut self, working: &mut CliffordUnitary, qubit: usize) {
        let count = working.num_qubits();
        self.exp(working, Self::single_x(qubit, count));
        self.exp(working, Self::single_x(qubit, count));
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
        if x_bit(&working.image_z(pivot), pivot) {
            self.root_x(working, pivot);
        }
        for qubit in (pivot + 1)..count {
            let image = working.image_z(pivot);
            if x_bit(&image, qubit) && z_bit(&image, qubit) {
                self.root_z(working, qubit);
            }
            if x_bit(&working.image_z(pivot), qubit) {
                self.hadamard(working, qubit);
            }
            if z_bit(&working.image_z(pivot), qubit) {
                self.controlled_x(working, qubit, pivot);
            }
        }
        if working.image_z(pivot).xz_phase_exponent() != 0 {
            self.pauli_x(working, pivot);
        }
    }

    /// The exponents that rebuild the original Clifford from the identity.
    fn into_decomposition(self) -> Vec<SparsePauli> {
        self.applied.into_iter().rev().map(negated).collect()
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
