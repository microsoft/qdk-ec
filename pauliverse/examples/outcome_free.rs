use paulimer::pauli::SparsePauli;
use paulimer::UnitaryOp;
use pauliverse::{OutcomeFreeSimulation, Simulation};

fn main() {
    // Simulation tracks state without specific outcomes
    let mut sim = OutcomeFreeSimulation::new(3);

    sim.unitary_op(UnitaryOp::Hadamard, &[0]);
    sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);

    let observable: SparsePauli = "ZII".parse().unwrap();
    sim.measure(&observable);

    // After measurement, query the stabilizer state
    let test_paulis = [
        "ZII".parse::<SparsePauli>().unwrap(),
        "IZI".parse::<SparsePauli>().unwrap(),
        "ZZI".parse::<SparsePauli>().unwrap(),
        "XII".parse::<SparsePauli>().unwrap(),
    ];

    println!("Stabilizer checks:");
    for pauli in &test_paulis {
        let is_stab = sim.is_stabilizer(pauli);
        println!("  {} is stabilizer: {}", pauli, is_stab);
    }

    println!("\nUse case: State verification without tracking outcomes");
}
