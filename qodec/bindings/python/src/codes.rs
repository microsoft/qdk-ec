//! Python bindings for quantum-code definitions.
//!
//! Hosts `PyCode`.

use std::path::PathBuf;
use std::sync::Arc;

use pyo3::prelude::*;

// ── Codes ───────────────────────────────────────────────────────────────────

#[pyclass(name = "Code", module = "qodec")]
pub struct PyCode {
    pub(crate) inner: qodec::Code,
}

impl PyCode {
    /// Materialize a fresh `Arc<Code>` snapshot of the current state.
    pub fn to_arc(&self) -> Arc<qodec::Code> {
        Arc::new(self.inner.clone())
    }

    /// Build a `PyCode` cell from a Rust `Code` snapshot.
    pub fn from_inner(inner: qodec::Code) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyCode {
    #[new]
    #[pyo3(signature = (
        name,
        stabilizers,
        x,
        z,
        *,
        description = String::new(),
        metadata = None,
    ))]
    fn new(
        name: String,
        stabilizers: Vec<Bound<'_, PyAny>>,
        x: Vec<Bound<'_, PyAny>>,
        z: Vec<Bound<'_, PyAny>>,
        description: String,
        metadata: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let inner = qodec::Code {
            name,
            description,
            stabilizers: crate::pauli_strings(stabilizers)?,
            x: crate::pauli_strings(x)?,
            z: crate::pauli_strings(z)?,
            metadata: crate::metadata_from_py(metadata.as_ref())?,
        };
        inner.validate().map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Load and validate a standalone code-definition YAML file.
    ///
    /// Raises ``QodecLoadError`` if reading, parsing, or validation fails.
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        qodec::Code::load(&path)
            .map(Self::from_inner)
            .map_err(|error| crate::QodecLoadError::new_err(format!("{}: {error}", path.display())))
    }

    /// Validate and write a standalone code-definition YAML file, creating parent directories.
    ///
    /// Raises ``QodecSaveError`` if validation, serialization, or writing fails.
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner
            .save(&path)
            .map_err(|error| crate::QodecSaveError::new_err(format!("{}: {error}", path.display())))
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    #[setter]
    fn set_name(&mut self, value: String) {
        self.inner.name = value;
    }

    #[getter]
    fn description(&self) -> &str {
        &self.inner.description
    }

    #[setter]
    fn set_description(&mut self, value: String) {
        self.inner.description = value;
    }

    #[getter]
    fn stabilizers(&self) -> Vec<String> {
        self.inner.stabilizers.iter().map(|pauli| pauli.0.clone()).collect()
    }

    /// Number of declared logical X operators, without checking the Z list.
    #[getter]
    fn logical_count(&self) -> usize {
        self.inner.logical_count()
    }

    /// Number of physical qubits, inferred from the highest qubit index used.
    #[getter]
    fn physical_qubit_count(&self) -> usize {
        self.inner.physical_qubit_count()
    }

    #[setter]
    fn set_stabilizers(&mut self, value: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        self.inner.stabilizers = crate::pauli_strings(value)?;
        Ok(())
    }

    #[getter]
    fn x(&self) -> Vec<String> {
        self.inner.x.iter().map(|pauli| pauli.0.clone()).collect()
    }

    #[setter]
    fn set_x(&mut self, value: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        self.inner.x = crate::pauli_strings(value)?;
        Ok(())
    }

    #[getter]
    fn z(&self) -> Vec<String> {
        self.inner.z.iter().map(|pauli| pauli.0.clone()).collect()
    }

    #[setter]
    fn set_z(&mut self, value: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        self.inner.z = crate::pauli_strings(value)?;
        Ok(())
    }

    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::metadata_to_py(py, &self.inner.metadata)
    }

    #[setter]
    fn set_metadata(&mut self, value: &Bound<'_, PyAny>) -> PyResult<()> {
        self.inner.metadata = crate::metadata_from_py(Some(value))?;
        Ok(())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyCode>>()
            .is_ok_and(|other| self.inner == other.inner)
    }

    fn __repr__(&self) -> String {
        format!("Code({:?})", self.inner.name)
    }

    fn __str__(&self) -> String {
        crate::display::yaml(&self.inner)
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}
