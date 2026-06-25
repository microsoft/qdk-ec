use crate::bin;
use crate::util;
use std::ops::BitXorAssign;

pub fn sparse_to_dense(sparse: &util::BitMatrix) -> binar::BitMatrix {
    let mut dense = binar::BitMatrix::zeros(sparse.rows as usize, sparse.cols as usize);
    for (&i, &j) in sparse.i.iter().zip(sparse.j.iter()) {
        dense.set((i as usize, j as usize), true);
    }
    dense
}

pub fn optional_sparse_to_dense(sparse: &Option<util::BitMatrix>) -> binar::BitMatrix {
    if let Some(sparse) = sparse {
        sparse_to_dense(sparse)
    } else {
        binar::BitMatrix::zeros(0, 0)
    }
}

pub fn dense_to_sparse(dense: &binar::BitMatrix) -> util::BitMatrix {
    let (rows, cols) = dense.shape();
    let mut sparse = util::BitMatrix {
        rows: rows as u64,
        cols: cols as u64,
        i: vec![],
        j: vec![],
    };
    for i in 0..rows {
        for j in 0..cols {
            if dense.get((i, j)) {
                sparse.i.push(i as u64);
                sparse.j.push(j as u64);
            }
        }
    }
    sparse
}

pub fn zeros(rows: usize, cols: usize) -> util::BitMatrix {
    util::BitMatrix {
        rows: rows as u64,
        cols: cols as u64,
        i: vec![],
        j: vec![],
    }
}

pub fn append_bit(matrix: &mut util::BitMatrix, i: usize, j: usize) {
    debug_assert!(
        !matrix
            .i
            .iter()
            .zip(matrix.j.iter())
            .any(|index| index == (&(i as u64), &(j as u64)))
    );
    matrix.i.push(i as u64);
    matrix.j.push(j as u64);
}

/// Apply a BitMatrixModifier to a dense BitMatrix.
/// Toggle (XOR) is applied first, then overwrite replaces the entire matrix.
pub fn apply_modifier(mut matrix: binar::BitMatrix, modifier: &bin::BitMatrixModifier) -> binar::BitMatrix {
    if let Some(toggle) = &modifier.toggle {
        let toggle_dense = sparse_to_dense(toggle);
        matrix.bitxor_assign(&toggle_dense);
    }
    if let Some(overwrite) = &modifier.overwrite {
        matrix = sparse_to_dense(overwrite);
    }
    matrix
}
