use pyo3::exceptions::PyValueError;
use pyo3::PyResult;

use paulimer::StringLayout;
use paulimer::StringNotation;

pub struct FormatSpec {
    pub layout: StringLayout,
    pub notation: StringNotation,
}

pub fn parse_format_spec(spec: &str) -> PyResult<FormatSpec> {
    let mut result = FormatSpec {
        layout: StringLayout::Dense,
        notation: StringNotation::Unicode,
    };
    if spec.is_empty() {
        return Ok(result);
    }
    for part in spec.split(',') {
        match part.trim() {
            "sparse" => result.layout = StringLayout::Sparse,
            "dense" => result.layout = StringLayout::Dense,
            "ascii" => result.notation = StringNotation::Ascii,
            "unicode" => result.notation = StringNotation::Unicode,
            "tex" => result.notation = StringNotation::Tex,
            other => {
                return Err(PyValueError::new_err(format!(
                    "Unknown format spec keyword: {other:?}. Expected: sparse, dense, ascii, unicode, tex"
                )));
            }
        }
    }
    Ok(result)
}
