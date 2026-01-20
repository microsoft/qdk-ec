use derive_more::{Display, FromStr, TryInto};
use paulimer::UnitaryOp;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyclass(eq, eq_int, str, name = "UnitaryOpcode", module = "paulimer")]
#[derive(PartialEq, Clone, FromStr, Display, TryInto)]
pub enum PyUnitaryOp {
    I = UnitaryOp::I as isize,
    X = UnitaryOp::X as isize,
    Y = UnitaryOp::Y as isize,
    Z = UnitaryOp::Z as isize,
    SqrtX = UnitaryOp::SqrtX as isize,
    SqrtXInv = UnitaryOp::SqrtXInv as isize,
    SqrtY = UnitaryOp::SqrtY as isize,
    SqrtYInv = UnitaryOp::SqrtYInv as isize,
    SqrtZ = UnitaryOp::SqrtZ as isize,
    SqrtZInv = UnitaryOp::SqrtZInv as isize,
    Hadamard = UnitaryOp::Hadamard as isize,
    Swap = UnitaryOp::Swap as isize,
    ControlledX = UnitaryOp::ControlledX as isize,
    ControlledZ = UnitaryOp::ControlledZ as isize,
    PrepareBell = UnitaryOp::PrepareBell as isize,
}

impl From<PyUnitaryOp> for UnitaryOp {
    fn from(value: PyUnitaryOp) -> Self {
        (value as isize).try_into().unwrap()
    }
}

#[pymethods]
impl PyUnitaryOp {
    #[staticmethod]
    fn from_string(s: &str) -> PyResult<Self> {
        s.parse::<Self>()
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Invalid UnitaryOp: {e}")))
    }

    #[must_use]
    pub fn __repr__(&self) -> String {
        self.to_string()
    }
}
