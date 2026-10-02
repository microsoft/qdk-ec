//! Sign-correctness of hinted Pauli measurements (`measure_with_hint`).
//!
//! Defining property of a projective measurement: once `P` has been measured, `P` is a stabilizer of
//! the post-measurement state whose sign equals the reported outcome. In particular, this must hold
//! no matter which anti-commuting `hint` is supplied — including hints carrying a negative sign,
//! which drive the `(-1)^alpha` branch of case 5 of Algorithm 4.2.

use paulimer::UnitaryOp;
use paulimer::clifford::{Clifford, PhasedCliffordUnitary};
use paulimer::pauli::{SparsePauli, as_sparse};
use pauliverse::{PhasedOutcomeCompleteSimulation, Simulation};
use proptest::collection::vec;
use proptest::prelude::*;

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
    use UnitaryOp::{ControlledX, Hadamard, SqrtX, SqrtZ};
    let single = (prop::sample::select(vec![Hadamard, SqrtZ, SqrtX]), 0..qubit_count)
        .prop_map(|(op, qubit)| Gate::Single { op, qubit });
    if qubit_count >= 2 {
        let two = distinct_pair(qubit_count).prop_map(|(first, second)| Gate::Two {
            op: ControlledX,
            first,
            second,
        });
        prop_oneof![3 => single, 1 => two].boxed()
    } else {
        single.boxed()
    }
}

/// Generates a qubit count, a random Clifford preparation circuit, whether to negate the hint, and
/// the qubit whose stabilizer/destabilizer images drive the measurement.
fn scenario() -> impl Strategy<Value = (usize, Vec<Gate>, bool, usize)> {
    (1usize..4).prop_flat_map(|qubit_count| {
        (
            Just(qubit_count),
            vec(gate_strategy(qubit_count), 0..12),
            any::<bool>(),
            0..qubit_count,
        )
    })
}

fn apply(gate: &Gate, sim: &mut PhasedOutcomeCompleteSimulation, mirror: &mut PhasedCliffordUnitary) {
    match *gate {
        Gate::Single { op, qubit } => {
            sim.unitary_op(op, &[qubit]);
            mirror.left_mul(op, &[qubit]);
        }
        Gate::Two { op, first, second } => {
            sim.unitary_op(op, &[first, second]);
            mirror.left_mul(op, &[first, second]);
        }
    }
}

proptest! {
    /// For a random stabilizer state, measuring the destabilizer `X`-image of qubit `target` while
    /// hinting with the (optionally negated) stabilizer `Z`-image of `target` must leave the
    /// observable a stabilizer whose conditional sign matches the reported outcome.
    #[test]
    fn measure_with_hint_outcome_sign_is_correct((qubit_count, gates, negate_hint, target) in scenario()) {
        let mut sim = PhasedOutcomeCompleteSimulation::new(qubit_count);
        let mut mirror = PhasedCliffordUnitary::identity(qubit_count);
        for gate in &gates {
            apply(gate, &mut sim, &mut mirror);
        }

        // `image_z(target)` is a stabilizer of the prepared state; `image_x(target)` anti-commutes
        // with it, so measuring the latter is a genuine (random) case-5 measurement.
        let stabilizer: SparsePauli = as_sparse(&mirror.clifford().image_z(target));
        let observable: SparsePauli = as_sparse(&mirror.clifford().image_x(target));
        let hint = if negate_hint { -stabilizer } else { stabilizer };

        let outcome = sim.measure_with_hint(&observable, &hint);

        prop_assert!(
            sim.is_stabilizer_with_conditional_sign(&observable, &[outcome]),
            "measured observable {observable} is not a stabilizer with the reported outcome sign \
             (qubit_count={qubit_count}, negate_hint={negate_hint}, target={target})"
        );
    }
}
