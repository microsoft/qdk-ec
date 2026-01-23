use paulimer::pauli::SparsePauli;
use paulimer::UnitaryOp;
use pauliverse::{OutcomeSpecificSimulation, Simulation};

fn main() {
    let observable: SparsePauli = "ZII".parse().unwrap();

    // Run many independent shots to collect statistics
    let mut outcome_counts = [0, 0];

    for _ in 0..1000 {
        // Each shot is independent
        let mut sim = OutcomeSpecificSimulation::new_with_random_outcomes(3);

        sim.unitary_op(UnitaryOp::Hadamard, &[0]);
        sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);

        let outcome_id = sim.measure(&observable);

        // Access the outcome value from the internal vector
        if outcome_id < sim.outcome_count() {
            let outcome = sim.random_outcome_indicator()[outcome_id] as usize;
            outcome_counts[outcome] += 1;
        }
    }

    println!("Outcome 0: {} times", outcome_counts[0]);
    println!("Outcome 1: {} times", outcome_counts[1]);
    println!("\nUse case: Monte Carlo sampling for error rates");
}
