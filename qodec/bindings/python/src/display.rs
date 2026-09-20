use pyo3::prelude::*;

pub(crate) fn yaml(value: &impl std::fmt::Display) -> String {
    value.to_string()
}

pub(crate) fn pretty(object: &Bound<'_, PyAny>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
    let text: String = if cycle { object.repr()? } else { object.str()? }.extract()?;
    for (index, line) in text.split('\n').enumerate() {
        if index > 0 {
            printer.call_method0("break_")?;
        }
        printer.call_method1("text", (line,))?;
    }
    Ok(())
}
