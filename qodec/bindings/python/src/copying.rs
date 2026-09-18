use pyo3::prelude::*;
use pyo3::types::PyDict;

macro_rules! copy_protocol {
    ($($ty:ty),+ $(,)?) => {$ (
        #[pymethods]
        impl $ty {
            fn __copy__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                if slf.hasattr("_copy_shell")? {
                    Ok(slf.call_method0("_copy_shell")?.unbind())
                } else {
                    Ok(slf.clone().into_any().unbind())
                }
            }

            fn __deepcopy__(slf: &Bound<'_, Self>, memo: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
                Ok(slf.py().import("qodec._copying")?.getattr("_deepcopy")?.call1((slf, memo))?.unbind())
            }
        }
    )+};
}

macro_rules! replace_protocol {
    ($($ty:ty),+ $(,)?) => {
        copy_protocol!($($ty),+);
        $(
        #[pymethods]
        impl $ty {
            #[pyo3(signature = (**changes))]
            fn __replace__(slf: &Bound<'_, Self>, changes: Option<&Bound<'_, PyDict>>) -> PyResult<Py<PyAny>> {
                let empty = PyDict::new(slf.py());
                Ok(slf.py().import("qodec._copying")?.getattr("_replace")?
                    .call1((slf, changes.unwrap_or(&empty)))?.unbind())
            }
        }
        )+
    };
}

copy_protocol!(crate::PyReadout, crate::PyOutcome, crate::PyFlag,);

replace_protocol!(
    crate::PyQodec,
    crate::PyLayer,
    crate::PyCode,
    crate::PyInstructionSet,
    crate::PyInstruction,
    crate::PyGadget,
    crate::PyCircuit,
    crate::PyEncoding,
    crate::PyBlock,
    crate::PyBlockOperand,
    crate::PyParameter,
    crate::PyCondition,
    crate::PyStabilize,
    crate::PyClifford,
    crate::PyPauli,
    crate::PyObserve,
    crate::PyRotate,
    crate::PyReference,
    crate::PyInstructionCall,
);
