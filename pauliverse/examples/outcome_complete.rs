use paulimer::pauli::SparsePauli;
use paulimer::UnitaryOp;
use pauliverse::{OutcomeCompleteSimulation, Simulation};

fn main() {
    // Track all possible measurement outcomes in one simulation
    let mut sim = OutcomeCompleteSimulation::new(3);

    sim.unitary_op(UnitaryOp::Hadamard, &[0]);
    sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);

    let observable: SparsePauli = "ZII".parse().unwrap();
    let outcome_id = sim.measure(&observable);

    // After measurement, state has branched
    let num_outcomes = sim.outcome_count();
    let num_random = sim.random_outcome_count();

    println!("Total outcomes tracked: {}", num_outcomes);
    println!("Random outcomes: {}", num_random);
    println!("Number of branches: 2^{} = {}", num_random, 1 << num_random);

    // Each branch represents a different measurement outcome history
    // All branches tracked simultaneously without separate simulation runs

    println!("\nUse case: Efficient computation over all possible outcomes");
}
