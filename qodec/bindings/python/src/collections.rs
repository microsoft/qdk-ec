use pyo3::prelude::*;

pub fn view<'py>(owner: &Bound<'py, PyAny>, field: &str, mapping: bool) -> PyResult<Bound<'py, PyAny>> {
    owner
        .py()
        .import("qodec._collections")?
        .getattr(if mapping { "_Mapping" } else { "_Sequence" })?
        .call1((owner, field))
}

pub fn mapping<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.py().get_type::<pyo3::types::PyDict>().call1((value,))
}
