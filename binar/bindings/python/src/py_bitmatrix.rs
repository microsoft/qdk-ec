use crate::py_bitvec::PyBitVec;
use binar::{
    BitMatrix, BitVec,
    python::{bitmatrix_as_capsule, bitmatrix_from_capsule},
};
use derive_more::{Deref, DerefMut, From, Into};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyCapsule};

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
            .ok_or_else(|| py_value_err("Cannot create BitMatrix from empty iterable"))?
            .len();

        if let Some((i, row)) = rows.iter().enumerate().find(|(_, r)| r.len() != columns) {
            return Err(py_value_err(format!(
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
            return Err(py_value_err("Bytes length must be a multiple of 64"));
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
        let matrix = bitmatrix_from_capsule(capsule)
            .ok_or_else(|| py_value_err("Failed to extract BitMatrix from capsule: invalid capsule or wrong type."))?;
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
    #[allow(non_snake_case)]
    pub fn T(&self) -> PyBitMatrix {
        BitMatrix::transposed(self).into()
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

    pub fn dot(&self, other: &PyBitMatrix) -> PyBitMatrix {
        (&self.matrix * &other.matrix).into()
    }

    pub fn echelonize(&mut self) -> Vec<usize> {
        BitMatrix::echelonize(self)
    }

    pub fn echelonized(&mut self) -> PyBitMatrix {
        let mut clone = BitMatrix::clone(self);
        clone.echelonize();
        clone.into()
    }

    pub fn kernel(&mut self) -> PyBitMatrix {
        BitMatrix::kernel(self).into()
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn submatrix(&self, rows: Vec<usize>, columns: Vec<usize>) -> PyBitMatrix {
        BitMatrix::submatrix(self, &rows, &columns).into()
    }

    #[must_use]
    pub fn __getitem__(&self, index: (usize, usize)) -> bool {
        self[index]
    }

    pub fn __setitem__(&mut self, index: (usize, usize), to: bool) {
        self.set(index, to);
    }

    pub fn __add__(&self, other: &PyBitMatrix) -> PyBitMatrix {
        (&self.matrix + &other.matrix).into()
    }

    pub fn __iadd__(&mut self, other: &PyBitMatrix) {
        self.matrix += &other.matrix;
    }

    pub fn __mul__(&self, other: &PyBitMatrix) -> PyBitMatrix {
        (&self.matrix & &other.matrix).into()
    }

    /// # Errors
    ///
    /// Returns an error if the matrix multiplication operation fails.
    pub fn __matmul__<'py>(&self, other: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let py = other.py();

        if let Ok(matrix) = other.extract::<PyBitMatrix>() {
            let result: PyBitMatrix = (&self.matrix * &matrix.matrix).into();
            return Ok(Bound::new(py, result)?.into_any());
        }

        if let Ok(vector) = other.extract::<PyBitVec>() {
            let bitvec: &BitVec = &vector;
            let result: PyBitVec = (&self.matrix * &bitvec.as_view()).into();
            return Ok(Bound::new(py, result)?.into_any());
        }

        Ok(py.NotImplemented().into_bound(py))
    }

    pub fn __xor__(&self, other: &PyBitMatrix) -> PyBitMatrix {
        (&self.matrix ^ &other.matrix).into()
    }

    pub fn __ixor__(&mut self, other: &PyBitMatrix) {
        self.matrix += &other.matrix;
    }

    pub fn __and__(&self, other: &PyBitMatrix) -> PyBitMatrix {
        (&self.matrix & &other.matrix).into()
    }

    pub fn __iand__(&mut self, other: &PyBitMatrix) {
        self.matrix &= &other.matrix;
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

fn py_value_err(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(msg.into())
}

fn py_type_err(msg: impl Into<String>) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(msg.into())
}

fn parse_row_from_string(s: &str) -> PyResult<BitVec> {
    s.chars()
        .map(|c| match c {
            '0' => Ok(false),
            '1' => Ok(true),
            _ => Err(py_value_err("String rows must contain only '0' and '1' characters")),
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
                    _ => Err(py_value_err(format!("Integer values must be 0 or 1, not {n}"))),
                })
            })
            .map_err(|_| py_value_err("Row elements must be one of: True, False, 0, 1"))
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
        return parse_row_from_string(s.to_str()?);
    }

    if let Ok(iter) = row.try_iter() {
        return parse_row_from_iterable(iter);
    }

    Err(py_type_err(
        "Each row must be a BitVector, string of '0' and '1', or an iterable of booleans/integers",
    ))
}
