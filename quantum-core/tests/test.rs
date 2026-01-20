use itertools::iproduct;
use quantum_core::{id, x, y, z, All, Axis, DirectedAxis, PauliMatrix, PauliObservable};

#[test]
fn iter_and_print_test() {
    for (a, b, p, po) in iproduct!(
        Axis::all(),
        DirectedAxis::all(),
        PauliMatrix::all(),
        PauliObservable::all()
    ) {
        println!("{:?} {:?} {:?} {:?} {:?}", a, b, p, po, (a, b, p, -po));
    }
}

#[test]
fn neg_neg_is_identity() {
    for a in PauliObservable::all() {
        assert_eq!(a, -(-a));
    }
    assert_eq!(x(0), -(-x(0)));
    assert_eq!(y(0), -(-y(0)));
    assert_eq!(z(0), -(-z(0)));
    assert_eq!(id(0), -(-id(0)));
}

#[test]
fn qubit_then_pauli_order() {
    assert!(x(1) < x(2));
    assert!(x(1) < y(1));
    assert!(y(1) < x(2));
}
