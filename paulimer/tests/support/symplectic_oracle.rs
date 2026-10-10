use std::collections::{HashMap, hash_map::Entry};

pub type PackedAction = u64;

fn apply_transvection(action: PackedAction, vector: u64, qubit_count: usize) -> PackedAction {
    let dimension = 2 * qubit_count;
    let row_mask = (1 << dimension) - 1;
    let half_mask = (1 << qubit_count) - 1;
    let mut updated = 0;
    for row in 0..dimension {
        let mut image = (action >> (dimension * row)) & row_mask;
        let pairing = ((image & half_mask) & (vector >> qubit_count)) ^ ((image >> qubit_count) & (vector & half_mask));
        if pairing.count_ones() % 2 == 1 {
            image ^= vector;
        }
        updated |= image << (dimension * row);
    }
    updated
}

pub fn enumerate_actions(qubit_count: usize) -> HashMap<PackedAction, (usize, u64, PackedAction)> {
    assert!(qubit_count <= 3, "the packed oracle supports at most three qubits");
    let dimension = 2 * qubit_count;
    let identity = (0..dimension).fold(0, |packed, row| packed | (1 << (row * (dimension + 1))));
    let mut reached = HashMap::from([(identity, (0, 0, identity))]);
    let mut frontier = vec![identity];
    let mut distance = 0;
    while !frontier.is_empty() {
        distance += 1;
        let mut next = Vec::new();
        for &action in &frontier {
            for vector in 1..(1 << dimension) {
                let candidate = apply_transvection(action, vector, qubit_count);
                if let Entry::Vacant(slot) = reached.entry(candidate) {
                    slot.insert((distance, vector, action));
                    next.push(candidate);
                }
            }
        }
        frontier = next;
    }
    reached
}
