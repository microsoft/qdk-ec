use binar::{
    BitVec, Bitwise, BitwiseMut, BitwisePair, BitwisePairMut, IntoBitIterator,
    python::{bitvec_as_capsule, bitvec_from_capsule},
};
use derive_more::{Deref, DerefMut, From, Into};
use pyo3::{
    exceptions::{PyIndexError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyBytes, PyCapsule, PyInt, PyList, PySlice, PyTuple},
};

#[derive(Clone, PartialEq, Deref, DerefMut, From, Into)]
#[must_use]
#[pyclass(name = "BitVector", module = "binar", eq, sequence)]
pub struct PyBitVec(BitVec);

impl<'life> From<&'life PyBitVec> for &'life BitVec {
    fn from(py_bitvec: &'life PyBitVec) -> Self {
        &py_bitvec.0
    }
}

#[pymethods]
impl PyBitVec {
    #[new]
    fn new(bits: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(string) = bits.extract::<String>() {
            return try_from_string(&string);
        }

        if let Ok(list) = bits.cast::<PyList>() {
            return try_from_sequence(list.iter());
        }

        if let Ok(list) = bits.cast::<PyTuple>() {
            return try_from_sequence(list.iter());
        }

        if let Ok(vec) = bits.extract::<Vec<bool>>() {
            return Ok(BitVec::from_iter(vec).into());
        }

        if let Ok(vec) = bits.extract::<Vec<u8>>() {
            let bools = vec.into_iter().map(|value| value != 0);
            return Ok(bools.collect::<BitVec>().into());
        }

        bits.try_iter()?
            .map(|item| item?.is_truthy())
            .collect::<PyResult<BitVec>>()
            .map(PyBitVec::from)
    }

    #[staticmethod]
    fn zeros(length: usize) -> Self {
        BitVec::zeros(length).into()
    }

    #[staticmethod]
    fn ones(length: usize) -> Self {
        BitVec::ones(length).into()
    }

    /// Pickle support: used by __reduce__ to reconstruct the object.
    #[staticmethod]
    #[pyo3(name = "_from_pickle")]
    #[allow(clippy::needless_pass_by_value)]
    fn from_pickle(length: usize, words: Vec<u64>) -> Self {
        BitVec::from_words(length, &words).into()
    }

    /// Pickle support: returns (callable, args) for portable serialization.
    #[allow(clippy::type_complexity)]
    fn __reduce__(&self, py: Python<'_>) -> PyResult<(Py<PyAny>, (usize, Vec<u64>))> {
        let cls = py.get_type::<Self>();
        let from_pickle = cls.getattr("_from_pickle")?;
        Ok((from_pickle.into(), (self.0.len(), self.0.as_words().to_vec())))
    }

    /// Serialize the vector to bytes (native endianness).
    /// Faster than pickle but not portable across different endianness.
    #[must_use]
    #[pyo3(name = "_to_bytes")]
    fn to_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.0.as_bytes())
    }

    /// Deserialize a vector from bytes produced by `_to_bytes`.
    /// Faster than pickle but not portable across different endianness.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes length is not a multiple of 64.
    #[staticmethod]
    #[pyo3(name = "_from_bytes")]
    fn from_bytes(length: usize, data: &[u8]) -> PyResult<Self> {
        if !data.len().is_multiple_of(64) {
            return Err(PyValueError::new_err("Bytes length must be a multiple of 64"));
        }
        Ok(BitVec::from_bytes(length, data).into())
    }

    /// Construct a `PyBitVec` by taking ownership from a `PyCapsule`.
    ///
    /// # Safety
    /// This consumes the capsule's contents. The capsule must not be used again.
    #[staticmethod]
    #[pyo3(name = "_from_capsule")]
    fn from_owned_capsule(capsule: &Bound<'_, PyCapsule>) -> PyResult<Self> {
        let bitvec = bitvec_from_capsule(capsule).ok_or_else(|| {
            PyValueError::new_err("Failed to extract BitVec from capsule: invalid capsule or pointer.")
        })?;
        Ok(bitvec.into())
    }

    /// Returns a `PyCapsule` containing a pointer to the inner `BitVec`.
    /// Used for zero-copy access from other Rust-based Python extensions.
    ///
    /// # Safety
    /// The capsule is only valid while `self` is alive.
    /// The caller must ensure the `PyBitVec` outlives any use of the capsule.
    #[pyo3(name = "_as_capsule")]
    fn as_ptr_capsule<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        bitvec_as_capsule(&self.0, py)
    }

    #[getter]
    fn weight(&self) -> usize {
        BitVec::weight(self)
    }

    #[getter]
    fn parity(&self) -> bool {
        BitVec::parity(self)
    }

    #[getter]
    fn is_zero(&self) -> bool {
        BitVec::is_zero(self)
    }

    #[getter]
    fn support(&self) -> Vec<usize> {
        BitVec::support(self).collect()
    }

    pub fn resize(&mut self, new_length: usize) {
        BitVec::resize(self, new_length);
    }

    pub fn clear(&mut self) {
        BitVec::clear_bits(self);
    }

    pub fn negate_index(&mut self, index: usize) {
        BitVec::negate_index(self, index);
    }

    #[must_use]
    pub fn dot(&self, other: &PyBitVec) -> bool {
        <BitVec as BitwisePair<BitVec>>::dot(self, other)
    }

    #[must_use]
    pub fn and_weight(&self, other: &PyBitVec) -> usize {
        <BitVec as BitwisePair<BitVec>>::and_weight(self, other)
    }

    #[must_use]
    pub fn or_weight(&self, other: &PyBitVec) -> usize {
        <BitVec as BitwisePair<BitVec>>::or_weight(self, other)
    }

    /// # Errors
    ///
    /// Returns an error if the index is invalid or if slice step is zero.
    pub fn __getitem__<'py>(&self, py: Python<'py>, index: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if index.is_instance_of::<PyInt>() {
            let index = index.extract::<usize>()?;
            if index >= self.len() {
                return Err(PyIndexError::new_err(format!(
                    "index {} is out of bounds for BitVector of length {}",
                    index,
                    self.len()
                )));
            }

            let bool_value = self.index(index);
            let py_bool = bool_value.into_pyobject(py)?;
            return Ok(py_bool.as_any().clone());
        }

        if let Ok(slice) = index.cast::<PySlice>() {
            let result = self.slice(slice)?;
            return Ok(Bound::new(py, result)?.into_any());
        }

        Err(PyTypeError::new_err("Indices must be integers or slices"))
    }

    pub fn __setitem__(&mut self, index: usize, to: bool) {
        self.assign_index(index, to);
    }

    #[must_use]
    pub fn __len__(&self) -> usize {
        self.len()
    }

    pub fn __xor__(&self, other: &PyBitVec) -> PyBitVec {
        let mut result = self.clone();
        <BitVec as BitwisePairMut<BitVec>>::bitxor_assign(&mut result, other);
        result
    }

    pub fn __ixor__(&mut self, other: &PyBitVec) {
        <BitVec as BitwisePairMut<BitVec>>::bitxor_assign(self, other);
    }

    pub fn __and__(&self, other: &PyBitVec) -> PyBitVec {
        let mut result = self.clone();
        <BitVec as BitwisePairMut<BitVec>>::bitand_assign(&mut result, other);
        result
    }

    pub fn __iand__(&mut self, other: &PyBitVec) {
        <BitVec as BitwisePairMut<BitVec>>::bitand_assign(self, other);
    }

    pub fn __or__(&self, other: &PyBitVec) -> PyBitVec {
        let mut result = self.clone();
        <BitVec as BitwisePairMut<BitVec>>::bitor_assign(&mut result, other);
        result
    }

    pub fn __ior__(&mut self, other: &PyBitVec) {
        <BitVec as BitwisePairMut<BitVec>>::bitor_assign(self, other);
    }

    #[must_use]
    pub fn __iter__(&self) -> PyBitVecIterator {
        PyBitVecIterator {
            iter: <BitVec>::clone(self).iter_bits(),
        }
    }

    #[must_use]
    pub fn __str__(&self) -> String {
        let bits: String = BitVec::iter(self).map(|b| if b { '1' } else { '0' }).collect();
        format!("[{bits}]")
    }

    #[must_use]
    pub fn __repr__(&self) -> String {
        format!("BitVector('{}')", self.__str__())
    }
}

#[pyclass]
pub struct PyBitVecIterator {
    iter: <BitVec as IntoBitIterator>::BitIterator,
}

#[pymethods]
impl PyBitVecIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<bool> {
        self.iter.next()
    }
}

impl PyBitVec {
    fn slice(&self, slice: &Bound<'_, pyo3::types::PySlice>) -> PyResult<PyBitVec> {
        let indices = slice.indices(isize::try_from(self.len()).unwrap())?;

        if indices.step == 1 {
            let start = usize::try_from(indices.start).unwrap();
            let stop = usize::try_from(indices.stop).unwrap();
            return Ok(self.extract(start, stop).into());
        }

        let mut result = BitVec::zeros(indices.slicelength);
        let mut current = indices.start;
        for index in 0..indices.slicelength {
            #[allow(clippy::cast_sign_loss)]
            result.assign_index(index, self.index(current as usize));
            current += indices.step;
        }
        Ok(PyBitVec::from(result))
    }
}

fn try_from_string(string: &str) -> PyResult<PyBitVec> {
    let bools: Result<Vec<bool>, _> = string
        .chars()
        .map(|c| match c {
            '0' => Ok(false),
            '1' => Ok(true),
            _ => Err(PyValueError::new_err("String must contain only '0' and '1' characters")),
        })
        .collect();

    Ok(BitVec::from_iter(bools?).into())
}

fn try_from_sequence<'py>(iter: impl Iterator<Item = Bound<'py, PyAny>>) -> PyResult<PyBitVec> {
    let items: Vec<_> = iter.collect();

    #[allow(clippy::redundant_closure_for_method_calls)]
    let bools: Result<Vec<bool>, _> = items.iter().map(|item| item.extract::<bool>()).collect();

    if let Ok(vec) = bools {
        return Ok(BitVec::from_iter(vec).into());
    }

    #[allow(clippy::redundant_closure_for_method_calls)]
    let ints: Result<Vec<u8>, _> = items.iter().map(|item| item.extract::<u8>()).collect();

    if let Ok(vec) = ints {
        let bools: Result<Vec<bool>, _> = vec
            .into_iter()
            .map(|value| match value {
                0 => Ok(false),
                1 => Ok(true),
                _ => Err(PyValueError::new_err(format!(
                    "Integer values must be 0 or 1, got {value}"
                ))),
            })
            .collect();
        return Ok(BitVec::from_iter(bools?).into());
    }

    // If neither worked, use truthiness
    items
        .into_iter()
        .map(|item| item.is_truthy())
        .collect::<PyResult<Vec<bool>>>()
        .map(|bools| BitVec::from_iter(bools).into())
}
