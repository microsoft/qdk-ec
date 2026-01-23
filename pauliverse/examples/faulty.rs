use paulimer::pauli::SparsePauli;
use paulimer::UnitaryOp;
use pauliverse::{FaultySimulation, PauliDistribution, PauliFault, Simulation};

fn main() {
    // Define noise model
    let noise = PauliFault {
        probability: 0.1, // 10% error rate for demonstration
        distribution: PauliDistribution::Single("XII".parse::<SparsePauli>().unwrap()),
        correlation_id: None,
        condition: None,
    };
    
    let stabilizer: SparsePauli = "ZZI".parse().unwrap();
    let mut syndrome_counts = [0, 0];
    
    // Collect syndromes over many shots with noise
    for _ in 0..1000 {
        let mut sim = FaultySimulation::default();
        sim.reserve_qubits(3);
        
        // Apply gate and noise
        sim.unitary_op(UnitaryOp::Hadamard, &[0]);
        sim.apply_fault(noise.clone());
        sim.unitary_op(UnitaryOp::ControlledX, &[0, 1]);
        
        // Measure syndrome
        let syndrome_id = sim.measure(&stabilizer);
        
        // Extract syndrome value for this shot
        if syndrome_id < sim.outcome_count() {
            let syndrome = sim.random_outcome_indicator()[syndrome_id] as usize;
            syndrome_counts[syndrome] += 1;
        }
    }
    
    println!("Syndrome statistics:");
    println!("  Syndrome 0: {} shots", syndrome_counts[0]);
    println!("  Syndrome 1: {} shots (error detected)", syndrome_counts[1]);
    println!("\nUse case: Error correction decoder training and testing");
}
