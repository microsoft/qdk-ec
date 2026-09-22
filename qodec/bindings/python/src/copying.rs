use pyo3::prelude::*;
use pyo3::types::PyDict;

macro_rules! copy_protocol {
    ($ownership:ident; $($ty:ty),+ $(,)?) => {$ (
        #[pymethods]
        impl $ty {
            fn __copy__(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
                copy_protocol!(@copy $ownership, slf)
            }

            fn __deepcopy__(slf: &Bound<'_, Self>, memo: &Bound<'_, PyDict>) -> PyResult<Py<PyAny>> {
                Ok(slf.py().import("qodec._copying")?.getattr("_deepcopy")?.call1((slf, memo))?.unbind())
            }
        }
    )+};
    (@copy mutable, $slf:ident) => { Ok($slf.call_method0("_copy_shell")?.unbind()) };
    (@copy immutable, $slf:ident) => { Ok($slf.clone().into_any().unbind()) };
}

macro_rules! replace_protocol {
    ($($ty:ty),+ $(,)?) => {
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

copy_protocol!(mutable;
    crate::PyQodec, crate::PyLayer, crate::PyCode, crate::PyInstructionSet,
    crate::PyInstruction, crate::PyGadget, crate::PyCircuit, crate::PyEncoding,
    crate::PyInstructionCall,
);

copy_protocol!(immutable;
    crate::PyReadout, crate::PyOutcome, crate::PyFlag, crate::PyBlock,
    crate::PyBlockOperand, crate::PyParameter, crate::PyCondition,
    crate::PyStabilize, crate::PyClifford, crate::PyPauli, crate::PyObserve,
    crate::PyRotate, crate::PyReference,
);

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
