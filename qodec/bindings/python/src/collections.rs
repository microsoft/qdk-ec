use pyo3::prelude::*;

pub fn view<'py>(owner: &Bound<'py, PyAny>, field: &str, mapping: bool) -> PyResult<Bound<'py, PyAny>> {
    view_at(owner, field, mapping, &[])
}

pub(crate) fn view_at<'py>(
    owner: &Bound<'py, PyAny>,
    field: &str,
    mapping: bool,
    path: &[Py<PyAny>],
) -> PyResult<Bound<'py, PyAny>> {
    owner
        .py()
        .import("qodec._collections")?
        .getattr(match (mapping, field, path.is_empty()) {
            (true, "frames", true) => "_FrameMapping",
            (true, _, _) => "_Mapping",
            (false, _, _) => "_Sequence",
        })?
        .call1((owner, field, pyo3::types::PyTuple::new(owner.py(), path)?))
}

pub fn mapping<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.py().get_type::<pyo3::types::PyDict>().call1((value,))
}
