use crate::{
    py_bitvec::PyBitVec,
    utils::{try_py_index, try_py_indices},
};
use binar::{
    BitMatrix, BitVec, IndexSet,
    matrix::bitmatrix_complete_to_full_rank_row_basis,
    python::{bitmatrix_as_capsule, bitmatrix_from_capsule},
};
use derive_more::{Deref, DerefMut, From, Into};
use pyo3::{
    exceptions::{PyTypeError, PyValueError},
    prelude::*,
    pyclass,
    types::{PyBytes, PyCapsule},
};

#[pyclass]
#[pyo3(name = "RowIterator", module = "binar")]
pub struct PyRowIterator {
    // Store a concrete iterator that owns its data
    iter: Box<dyn Iterator<Item = PyBitVec> + Send + Sync>,
}

#[pymethods]
impl PyRowIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyBitVec> {
        slf.iter.next()
    }
}

#[derive(Clone, Deref, DerefMut, From, Into, PartialEq)]
#[must_use]
#[pyclass(name = "BitMatrix", module = "binar", eq)]
pub struct PyBitMatrix {
    #[from]
    #[into]
    pub(crate) matrix: BitMatrix,
}

impl<'life> From<&'life PyBitMatrix> for &'life BitMatrix {
    fn from(py_matrix: &'life PyBitMatrix) -> Self {
        &py_matrix.matrix
    }
}

#[pymethods]
impl PyBitMatrix {
    #[new]
    fn new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rows: Vec<BitVec> = data
            .try_iter()?
            .map(|item| parse_row(&item?))
            .collect::<PyResult<_>>()?;

        let columns = rows
            .first()
            .ok_or_else(|| PyValueError::new_err("Cannot create BitMatrix from empty iterable"))?
            .len();

        if let Some((i, row)) = rows.iter().enumerate().find(|(_, r)| r.len() != columns) {
            return Err(PyValueError::new_err(format!(
                "Row {i} has {} columns, expected {columns}",
                row.len()
            )));
        }

        Ok(BitMatrix::from_iter(rows.iter().map(BitVec::iter), columns).into())
    }

    /// Pickle support: used by __reduce__ to reconstruct the object.
    #[staticmethod]
    #[pyo3(name = "_from_pickle")]
    #[allow(clippy::needless_pass_by_value)]
    fn from_pickle(rows: usize, cols: usize, words: Vec<u64>) -> Self {
        let _ = rows; // rows is inferred from words length and cols
        BitMatrix::from_words(&words, cols).into()
    }

    /// Pickle support: returns (callable, args) for portable serialization.
    #[allow(clippy::type_complexity)]
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (usize, usize, Vec<u64>))> {
        let cls = py.get_type::<Self>();
        let from_pickle = cls.getattr("_from_pickle")?;
        Ok((
            from_pickle.into(),
            (
                self.matrix.row_count(),
                self.matrix.column_count(),
                self.matrix.as_words().to_vec(),
            ),
        ))
    }

    /// Serialize the matrix to bytes (native endianness).
    /// Faster than pickle but not portable across different endianness.
    #[must_use]
    #[pyo3(name = "_to_bytes")]
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.matrix.as_bytes())
    }

    /// Deserialize a matrix from bytes produced by `_to_bytes`.
    /// Faster than pickle but not portable across different endianness.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes length is not a multiple of 64.
    #[staticmethod]
    #[pyo3(name = "_from_bytes")]
    fn from_bytes(_rows: usize, columns: usize, data: &[u8]) -> PyResult<Self> {
        if !data.len().is_multiple_of(64) {
            return Err(PyValueError::new_err("Bytes length must be a multiple of 64"));
        }
        Ok(BitMatrix::from_bytes(data, columns).into())
    }

    /// Construct a `PyBitMatrix` by taking ownership from a `PyCapsule`.
    ///
    /// # Safety
    /// This consumes the capsule's contents. The capsule must not be used again.
    #[staticmethod]
    #[pyo3(name = "_from_capsule")]
    fn from_owned_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        let matrix = bitmatrix_from_capsule(capsule).ok_or_else(|| {
            PyValueError::new_err("Failed to extract BitMatrix from capsule: invalid capsule or wrong type.")
        })?;
        Ok(PyBitMatrix { matrix })
    }

    /// Returns a `PyCapsule` containing a pointer to the inner `BitMatrix`.
    /// Used for zero-copy access from other Rust-based Python extensions.
    ///
    /// # Safety
    /// The capsule is only valid while `self` is alive.
    /// The caller must ensure the `PyBitMatrix` outlives any use of the capsule.
    #[pyo3(name = "_as_capsule")]
    fn as_ptr_capsule<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        bitmatrix_as_capsule(&self.matrix, py)
    }

    #[staticmethod]
    fn identity(dimension: usize) -> Self {
        BitMatrix::identity(dimension).into()
    }

    #[staticmethod]
    fn zeros(rows: usize, columns: usize) -> Self {
        BitMatrix::zeros(rows, columns).into()
    }

    #[staticmethod]
    fn ones(rows: usize, columns: usize) -> Self {
        BitMatrix::ones(rows, columns).into()
    }

    /// Construct a matrix from sparse column descriptions.
    ///
    /// Each element of `columns` is an iterable of row indices where the bit is 1.
    #[staticmethod]
    fn from_sparse_columns(columns: &Bound<'_, PyAny>, row_count: usize, column_count: usize) -> PyResult<Self> {
        let index_sets = parse_index_sets(columns)?;
        BitMatrix::from_sparse_columns(&index_sets, row_count, column_count)
            .map(Into::into)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Construct a matrix from sparse row descriptions.
    ///
    /// Each element of `rows` is an iterable of column indices where the bit is 1.
    #[staticmethod]
    fn from_sparse_rows(rows: &Bound<'_, PyAny>, row_count: usize, column_count: usize) -> PyResult<Self> {
        let index_sets = parse_index_sets(rows)?;
        BitMatrix::from_sparse_rows(&index_sets, row_count, column_count)
            .map(Into::into)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Return each column as a sorted list of row indices where the bit is 1.
    fn sparse_columns(&self) -> Vec<Vec<usize>> {
        BitMatrix::sparse_columns(self)
            .into_iter()
            .map(|s| s.into_iter().collect())
            .collect()
    }

    /// Return each row as a sorted list of column indices where the bit is 1.
    fn sparse_rows(&self) -> Vec<Vec<usize>> {
        BitMatrix::sparse_rows(self)
            .into_iter()
            .map(|s| s.into_iter().collect())
            .collect()
    }

    #[getter]
    fn row_count(&self) -> usize {
        BitMatrix::row_count(self)
    }

    #[getter]
    fn column_count(&self) -> usize {
        BitMatrix::column_count(self)
    }

    #[getter]
    #[must_use]
    pub fn shape(&self) -> (usize, usize) {
        BitMatrix::shape(self)
    }

    #[getter]
    #[must_use]
    pub fn size(&self) -> usize {
        BitMatrix::row_count(self) * BitMatrix::column_count(self)
    }

    #[getter]
    fn is_zero(&self) -> bool {
        BitMatrix::is_zero(self)
    }

    #[getter]
    #[allow(non_snake_case)]
    pub fn T(&self) -> PyBitMatrix {
        BitMatrix::transposed(self).into()
    }

    /// # Errors
    ///
    /// Returns an error if either row index is out of range.
    pub fn swap_rows(&mut self, left_row_index: usize, right_row_index: usize) -> PyResult<()> {
        if left_row_index >= self.row_count() || right_row_index >= self.row_count() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Row index out of range.",
            ));
        }
        BitMatrix::swap_rows(&mut self.matrix, left_row_index, right_row_index);
        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if either column index is out of range.
    pub fn swap_columns(&mut self, left_column_index: usize, right_column_index: usize) -> PyResult<()> {
        if left_column_index >= self.column_count() || right_column_index >= self.column_count() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Column index out of range.",
            ));
        }
        BitMatrix::swap_columns(&mut self.matrix, left_column_index, right_column_index);
        Ok(())
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn permute_rows(&mut self, perm: Vec<usize>) {
        BitMatrix::permute_rows(&mut self.matrix, &perm);
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn permuted_rows(&self, perm: Vec<usize>) -> PyBitMatrix {
        let mut out = self.clone();
        BitMatrix::permute_rows(&mut out.matrix, &perm);
        out
    }

    /// Adds (XORs) one row into another: `row[to_index] ^= row[from_index]`.
    ///
    /// # Errors
    ///
    /// Returns an error if either index is out of range.
    pub fn add_into_row(&mut self, to_index: usize, from_index: usize) -> PyResult<()> {
        if to_index >= self.row_count() || from_index >= self.row_count() {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                "Row index out of range.",
            ));
        }
        BitMatrix::add_into_row(&mut self.matrix, to_index, from_index);
        Ok(())
    }

    #[getter]
    #[allow(clippy::unused_self)]
    fn ndim(&self) -> usize {
        2
    }

    #[getter]
    #[must_use]
    pub fn rows(&self) -> Vec<PyBitVec> {
        BitMatrix::rows(self).map(|row| BitVec::from(&row).into()).collect()
    }

    /// Return a copy of the row at the given index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of range.
    pub fn row(&self, index: isize) -> PyResult<PyBitVec> {
        let index = try_py_index(index, self.row_count())?;
        let row = BitMatrix::row(self, index);
        Ok(BitVec::from(&row).into())
    }

    /// Return a copy of the column at the given index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index is out of range.
    pub fn column(&self, index: isize) -> PyResult<PyBitVec> {
        let index = try_py_index(index, self.column_count())?;
        let column = BitMatrix::column(self, index);
        Ok(BitVec::from_iter(column.iter()).into())
    }

    pub fn copy(&self) -> PyBitMatrix {
        BitMatrix::clone(self).into()
    }

    /// # Errors
    ///
    /// Returns an error if the reshape operation fails.
    pub fn reshape(&mut self, rows: usize, columns: usize) -> PyResult<()> {
        BitMatrix::resize(self, rows, columns);
        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if `self.column_count != other.row_count`.
    pub fn dot(&self, other: &PyBitMatrix) -> PyResult<PyBitMatrix> {
        if self.column_count() != other.row_count() {
            return Err(PyValueError::new_err("Incompatible dimensions for matrix product."));
        }
        Ok((&self.matrix * &other.matrix).into())
    }

    pub fn echelonize(&mut self) -> Vec<usize> {
        BitMatrix::echelonize(self)
    }

    pub fn echelonized(&mut self) -> PyBitMatrix {
        let mut clone = BitMatrix::clone(self);
        clone.echelonize();
        clone.into()
    }

    /// Compute the rank of the matrix (dimension of the row/column space).
    #[must_use]
    pub fn rank(&self) -> usize {
        BitMatrix::rank(self)
    }

    /// Compute the inverse of this matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix is not square or not invertible.
    pub fn inverted(&self) -> PyResult<PyBitMatrix> {
        BitMatrix::try_inverted(self)
            .map(Into::into)
            .map_err(PyValueError::new_err)
    }

    pub fn kernel(&mut self) -> PyBitMatrix {
        BitMatrix::kernel(self).into()
    }

    pub fn row_space_intersection_with(&self, other: &PyBitMatrix) -> PyBitMatrix {
        BitMatrix::row_space_intersection_with(self, other).into()
    }

    /// Complete this matrix's (linearly independent) rows to a full-rank row basis.
    ///
    /// Returns a `column_count x column_count` matrix whose first `row_count`
    /// rows are `self`'s rows, with the remaining rows filled in so that the
    /// whole matrix has full row rank. Returns `None` if `self`'s rows are not
    /// already linearly independent.
    pub fn complete_to_full_rank_row_basis(&self) -> Option<PyBitMatrix> {
        bitmatrix_complete_to_full_rank_row_basis(self).map(Into::into)
    }

    /// # Errors
    ///
    /// Returns an error if any row or column index is out of range.
    #[allow(clippy::needless_pass_by_value)]
    pub fn submatrix(&self, rows: Vec<isize>, columns: Vec<isize>) -> PyResult<PyBitMatrix> {
        let rows = try_py_indices(rows, self.row_count())?;
        let columns = try_py_indices(columns, self.column_count())?;
        Ok(BitMatrix::submatrix(self, &rows, &columns).into())
    }

    /// # Errors
    ///
    /// Returns an error if either index is out of range.
    pub fn __getitem__(&self, index: (isize, isize)) -> PyResult<bool> {
        let index = (
            try_py_index(index.0, self.row_count())?,
            try_py_index(index.1, self.column_count())?,
        );
        Ok(self.get(index))
    }

    /// # Errors
    ///
    /// Returns an error if either index is out of range.
    pub fn __setitem__(&mut self, index: (isize, isize), to: bool) -> PyResult<()> {
        let index = (
            try_py_index(index.0, self.row_count())?,
            try_py_index(index.1, self.column_count())?,
        );
        self.set(index, to);
        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if the matrices have different dimensions.
    pub fn __add__(&self, other: &PyBitMatrix) -> PyResult<PyBitMatrix> {
        self.check_same_dimensions(other)?;
        Ok((&self.matrix + &other.matrix).into())
    }

    /// # Errors
    ///
    /// Returns an error if the matrices have different dimensions.
    pub fn __iadd__(&mut self, other: &PyBitMatrix) -> PyResult<()> {
        self.check_same_dimensions(other)?;
        self.matrix += &other.matrix;
        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if the matrices have different dimensions.
    pub fn __mul__(&self, other: &PyBitMatrix) -> PyResult<PyBitMatrix> {
        self.check_same_dimensions(other)?;
        Ok((&self.matrix & &other.matrix).into())
    }

    /// # Errors
    ///
    /// Returns an error if the matrix multiplication operation fails.
    pub fn __matmul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = other.py();

        if let Ok(matrix) = other.extract::<PyBitMatrix>() {
            let result = self.dot(&matrix)?;
            return Ok(Bound::new(py, result)?.into_any());
        }

        if let Ok(vector) = other.extract::<PyBitVec>() {
            if vector.len() != self.column_count() {
                return Err(PyValueError::new_err(
                    "Incompatible dimensions for matrix-vector multiplication",
                ));
            }
            let bitvec: &BitVec = &vector;
            let result: PyBitVec = (&self.matrix * &bitvec.as_view()).into();
            return Ok(Bound::new(py, result)?.into_any());
        }

        Ok(py.NotImplemented().into_bound(py))
    }

    /// # Errors
    ///
    /// Returns an error if the matrices have different dimensions.
    pub fn __xor__(&self, other: &PyBitMatrix) -> PyResult<PyBitMatrix> {
        self.check_same_dimensions(other)?;
        Ok((&self.matrix ^ &other.matrix).into())
    }

    /// # Errors
    ///
    /// Returns an error if the matrices have different dimensions.
    pub fn __ixor__(&mut self, other: &PyBitMatrix) -> PyResult<()> {
        self.check_same_dimensions(other)?;
        self.matrix ^= &other.matrix;
        Ok(())
    }

    /// # Errors
    ///
    /// Returns an error if the matrices have different dimensions.
    pub fn __and__(&self, other: &PyBitMatrix) -> PyResult<PyBitMatrix> {
        self.check_same_dimensions(other)?;
        Ok((&self.matrix & &other.matrix).into())
    }

    /// # Errors
    ///
    /// Returns an error if the matrices have different dimensions.
    pub fn __iand__(&mut self, other: &PyBitMatrix) -> PyResult<()> {
        self.check_same_dimensions(other)?;
        self.matrix &= &other.matrix;
        Ok(())
    }

    #[must_use]
    pub fn __str__(&self) -> String {
        self.to_string()
    }

    #[must_use]
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}

impl PyBitMatrix {
    fn check_same_dimensions(&self, other: &PyBitMatrix) -> PyResult<()> {
        if self.row_count() != other.row_count() || self.column_count() != other.column_count() {
            return Err(PyValueError::new_err("Matrices must have the same dimensions"));
        }
        Ok(())
    }
}

/// Collect an iterable of iterables of indices into `IndexSet`s, matching the
/// iterable-of-iterables input accepted by `BitMatrix.__new__`.
fn parse_index_sets(data: &Bound<'_, PyAny>) -> PyResult<Vec<IndexSet>> {
    data.try_iter()?
        .map(|inner| {
            inner?
                .try_iter()?
                .map(|index| index?.extract::<usize>())
                .collect::<PyResult<IndexSet>>()
        })
        .collect()
}

fn parse_row_from_string(s: &str) -> PyResult<BitVec> {
    s.chars()
        .map(|c| match c {
            '0' => Ok(false),
            '1' => Ok(true),
            _ => Err(PyValueError::new_err(
                "String rows must contain only '0' and '1' characters",
            )),
        })
        .collect::<PyResult<Vec<_>>>()
        .map(BitVec::from_iter)
}

fn parse_row_from_iterable(iter: pyo3::Bound<'_, pyo3::types::PyIterator>) -> PyResult<BitVec> {
    iter.map(|item| {
        let item = item?;
        item.extract::<bool>()
            .or_else(|_| {
                item.extract::<u64>().and_then(|n| match n {
                    0 => Ok(false),
                    1 => Ok(true),
                    _ => Err(PyValueError::new_err(format!("Integer values must be 0 or 1, not {n}"))),
                })
            })
            .map_err(|_| PyValueError::new_err("Row elements must be one of: True, False, 0, 1"))
    })
    .collect::<PyResult<Vec<_>>>()
    .map(BitVec::from_iter)
}

fn parse_row(row: &Bound<'_, PyAny>) -> PyResult<BitVec> {
    use pyo3::types::PyString;

    if let Ok(bitvec) = row.extract::<PyBitVec>() {
        return Ok((*bitvec).clone());
    }

    if let Ok(s) = row.cast::<PyString>() {
        return parse_row_from_string(&s.to_cow()?);
    }

    if let Ok(iter) = row.try_iter() {
        return parse_row_from_iterable(iter);
    }

    Err(PyTypeError::new_err(
        "Each row must be a BitVector, string of '0' and '1', or an iterable of booleans/integers",
    ))
}
