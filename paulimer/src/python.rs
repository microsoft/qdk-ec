use crate::clifford::CliffordUnitary;
use crate::pauli::{DensePauli, Pauli, SparsePauli};
use crate::pauli_group::PauliGroup;
use crate::UnitaryOp;
use binar::{vec::AlignedBitVec, IndexSet};
use pyo3::conversion::IntoPyObject;
use pyo3::prelude::*;

impl<'py> IntoPyObject<'py> for UnitaryOp {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let paulimer = PyModule::import(py, "paulimer")?;
        let cls = paulimer.getattr("UnitaryOpcode")?;
        let name = format!("{self:?}");
        cls.getattr(name.as_str())
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for UnitaryOp {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let value: isize = ob.getattr("value")?.extract()?;
        value
            .try_into()
            .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid UnitaryOp value"))
    }
}

impl<'py> IntoPyObject<'py> for DensePauli {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let paulimer = PyModule::import(py, "paulimer")?;
        let cls = paulimer.getattr("DensePauli")?;
        let instance = cls.call0()?;

        let x_words: Vec<u64> = self.x_bits().as_words().to_vec();
        let z_words: Vec<u64> = self.z_bits().as_words().to_vec();
        let exponent = self.xz_phase_exponent();
        let size = self.size();

        instance.call_method1("__setstate__", ((x_words, z_words, exponent, size),))?;
        Ok(instance)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for DensePauli {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let state: (Vec<u64>, Vec<u64>, u8, usize) = ob.call_method0("__getstate__")?.extract()?;
        let (x_words, z_words, exponent, _size) = state;
        let x_bits = AlignedBitVec::from_words(&x_words);
        let z_bits = AlignedBitVec::from_words(&z_words);
        Ok(DensePauli::from_bits(x_bits, z_bits, exponent))
    }
}

impl<'py> IntoPyObject<'py> for SparsePauli {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let paulimer = PyModule::import(py, "paulimer")?;
        let cls = paulimer.getattr("SparsePauli")?;
        let instance = cls.call0()?;

        let x_bits: Vec<usize> = self.x_bits().clone().into_iter().collect();
        let z_bits: Vec<usize> = self.z_bits().clone().into_iter().collect();
        let exponent = self.xz_phase_exponent();

        instance.call_method1("__setstate__", ((x_bits, z_bits, exponent),))?;
        Ok(instance)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for SparsePauli {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let state: (Vec<usize>, Vec<usize>, u8) = ob.call_method0("__getstate__")?.extract()?;
        let (x_indices, z_indices, exponent) = state;
        let x_bits = IndexSet::from_iter(x_indices);
        let z_bits = IndexSet::from_iter(z_indices);
        Ok(SparsePauli::from_bits(x_bits, z_bits, exponent))
    }
}

impl<'py> IntoPyObject<'py> for CliffordUnitary {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let paulimer = PyModule::import(py, "paulimer")?;
        let cls = paulimer.getattr("CliffordUnitary")?;
        let instance = cls.call0()?;

        let (words, phases) = self.as_words();
        let words_vec = words.to_vec();
        let phases_vec = phases.to_vec();

        instance.call_method1("__setstate__", ((words_vec, phases_vec),))?;
        Ok(instance)
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for CliffordUnitary {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let state: (Vec<u64>, Vec<u8>) = ob.call_method0("__getstate__")?.extract()?;
        let (words, phases) = state;
        Ok(CliffordUnitary::from_words(&words, phases))
    }
}

impl<'py> IntoPyObject<'py> for PauliGroup {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        let paulimer = PyModule::import(py, "paulimer")?;
        let cls = paulimer.getattr("PauliGroup")?;

        let generators: Vec<Bound<'py, PyAny>> = self
            .generators
            .into_iter()
            .map(|g| g.into_pyobject(py))
            .collect::<Result<_, _>>()?;

        cls.call1((generators,))
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for PauliGroup {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        let generators: Vec<SparsePauli> = ob.getattr("generators")?.extract()?;
        Ok(PauliGroup::new(&generators))
    }
}
