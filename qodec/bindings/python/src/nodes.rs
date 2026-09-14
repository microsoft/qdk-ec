//! Live Python navigation with the core path grammar and borrowed value selection.

use pyo3::IntoPyObjectExt;
use pyo3::PyClass;
use pyo3::exceptions::{PyLookupError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use qodec as model;
use std::collections::BTreeMap;

#[path = "../../../src/node/path.rs"]
mod path;
#[allow(
    dead_code,
    reason = "The shared selector also contains root variants used only by the core."
)]
#[path = "../../../src/node/value.rs"]
mod value;

use crate::codes::PyCode;
use crate::container::{PyLayer, PyQodec};
use crate::gadgets::{PyCircuit, PyEncoding, PyGadget, PyReadout, PyReference};
use crate::types::{
    PyBlock, PyBlockOperand, PyCondition, PyInstruction, PyInstructionSet, PyParameter, action_step_to_py,
};
use path::{ModelPath, PathError, Segment};
use value::{Mapping, Sequence, Value};

fn path_error(error: PathError) -> PyErr {
    match error {
        PathError::Syntax(_) => PyValueError::new_err(error.to_string()),
        PathError::Missing(_) => PyLookupError::new_err(error.to_string()),
    }
}

#[pyclass(name = "_ModelPath", module = "qodec._native", frozen)]
pub(crate) struct PyModelPath {
    inner: ModelPath,
}

#[pymethods]
impl PyModelPath {
    #[new]
    fn new(text: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ModelPath::parse(text).map_err(path_error)?,
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn _resolve(&self, text: &str) -> PyResult<Self> {
        let prefix = self.inner.to_string();
        let separator = if prefix.is_empty() || text.is_empty() || text.starts_with('[') {
            ""
        } else {
            "."
        };
        Self::new(&format!("{prefix}{separator}{text}"))
    }

    fn _index(&self, index: usize) -> Self {
        Self {
            inner: self.inner.child(Segment::Index(index)),
        }
    }

    fn _key(&self, key: String) -> Self {
        Self {
            inner: self.inner.child(Segment::Key(key)),
        }
    }
}

struct Query<'path> {
    path: &'path ModelPath,
    request: &'path str,
}

impl Query<'_> {
    fn shared_sequence<T: PyClass>(&self, py: Python<'_>, values: &[Py<T>], start: usize) -> PyResult<Py<PyAny>> {
        if let Some(segment) = self.path.0.get(start) {
            let Segment::Index(index) = segment else {
                return Err(self.missing(start));
            };
            return self.live(
                values.get(*index).ok_or_else(|| self.missing(start))?.bind(py).as_any(),
                start + 1,
            );
        }
        match self.request {
            "exists" => true.into_py_any(py),
            "kind" => "list".into_py_any(py),
            "is_none" => false.into_py_any(py),
            "length" => values.len().into_py_any(py),
            "value" => values
                .iter()
                .map(|value| value.clone_ref(py))
                .collect::<Vec<_>>()
                .into_py_any(py),
            _ => Err(self.mismatch("list")),
        }
    }

    fn shared_mapping<T: PyClass>(
        &self,
        py: Python<'_>,
        values: &BTreeMap<String, Py<T>>,
        start: usize,
    ) -> PyResult<Py<PyAny>> {
        if let Some(segment) = self.path.0.get(start) {
            let Segment::Key(key) = segment else {
                return Err(self.missing(start));
            };
            return self.live(
                values.get(key).ok_or_else(|| self.missing(start))?.bind(py).as_any(),
                start + 1,
            );
        }
        match self.request {
            "exists" => true.into_py_any(py),
            "kind" => "dict".into_py_any(py),
            "is_none" => false.into_py_any(py),
            "keys" => values.keys().collect::<Vec<_>>().into_py_any(py),
            "value" => values
                .iter()
                .map(|(key, value)| (key, value.clone_ref(py)))
                .collect::<BTreeMap<_, _>>()
                .into_py_any(py),
            _ => Err(self.mismatch("dict")),
        }
    }

    fn missing(&self, position: usize) -> PyErr {
        path_error(PathError::Missing(
            ModelPath(self.path.0[..=position].to_vec()).to_string(),
        ))
    }

    fn mismatch(&self, kind: &str) -> PyErr {
        PyTypeError::new_err(format!(
            "{:?} contains {kind}, not a {}",
            self.path.to_string(),
            self.request
        ))
    }

    fn native(&self, py: Python<'_>, mut value: Value<'_>, start: usize) -> PyResult<Py<PyAny>> {
        for (position, segment) in self.path.0.iter().enumerate().skip(start) {
            value = value.child(segment).ok_or_else(|| self.missing(position))?;
        }
        match self.request {
            "exists" => true.into_py_any(py),
            "kind" => value.kind().into_py_any(py),
            "is_none" => matches!(value, Value::None).into_py_any(py),
            "length" => match value {
                Value::Sequence(values) => values.len().into_py_any(py),
                _ => Err(self.mismatch(value.kind())),
            },
            "keys" => match value {
                Value::Mapping(values) => values.entries().keys().copied().collect::<Vec<_>>().into_py_any(py),
                _ => Err(self.mismatch(value.kind())),
            },
            "value" => native_to_py(py, value),
            _ => Err(PyValueError::new_err("unknown node request")),
        }
    }

    fn live(&self, object: &Bound<'_, PyAny>, start: usize) -> PyResult<Py<PyAny>> {
        let py = object.py();
        if let Some(Segment::Field(field)) = self.path.0.get(start) {
            if let Ok(root) = object.extract::<PyRef<'_, PyQodec>>() {
                match field.as_str() {
                    "layers" => return self.shared_sequence(py, &root.layers, start + 1),
                    "metadata" => return self.native(py, Value::Mapping(Mapping::Metadata(&root.metadata)), start + 1),
                    _ => {}
                }
            }
            if let Ok(layer) = object.extract::<PyRef<'_, PyLayer>>()
                && field == "gadgets"
            {
                return self.shared_mapping(py, &layer.gadgets, start + 1);
            }
            if let Ok(gadget) = object.extract::<PyRef<'_, PyGadget>>() {
                match field.as_str() {
                    "inputs" => return self.shared_sequence(py, &gadget.inputs, start + 1),
                    "outputs" => return self.shared_sequence(py, &gadget.outputs, start + 1),
                    "readouts" => {
                        return self.native(
                            py,
                            Value::Sequence(Sequence::ReadoutSpecs(
                                &gadget.readouts,
                                gadget.implements.borrow(py).inner.observe_count(),
                            )),
                            start + 1,
                        );
                    }
                    "metadata" => {
                        return self.native(py, Value::Mapping(Mapping::Metadata(&gadget.metadata)), start + 1);
                    }
                    "checks" => return self.native(py, Value::Sequence(Sequence::Checks(&gadget.checks)), start + 1),
                    "frames" => return self.native(py, Value::Mapping(Mapping::Frames(&gadget.frames)), start + 1),
                    _ => {}
                }
            }
            if let Ok(encoding) = object.extract::<PyRef<'_, PyEncoding>>() {
                match field.as_str() {
                    "support" => {
                        return self.native(py, Value::Sequence(Sequence::Strings(&encoding.support)), start + 1);
                    }
                    "block_types" => {
                        return self.native(py, Value::Sequence(Sequence::Strings(&encoding.block_types)), start + 1);
                    }
                    _ => {}
                }
            }
        }
        if start == self.path.0.len() && matches!(self.request, "exists" | "value" | "is_none" | "kind") {
            return match self.request {
                "exists" => true.into_py_any(py),
                "value" => Ok(object.clone().unbind()),
                "is_none" => object.is_none().into_py_any(py),
                _ => object.get_type().name()?.into_py_any(py),
            };
        }
        if let Ok(value) = object.extract::<PyRef<'_, PyInstructionSet>>() {
            return self.native(py, Value::InstructionSet(&value.inner), start);
        }
        if let Ok(value) = object.extract::<PyRef<'_, PyInstruction>>() {
            return self.native(py, Value::Instruction(&value.inner), start);
        }
        if let Ok(value) = object.extract::<PyRef<'_, PyCode>>() {
            return self.native(py, Value::Code(&value.inner), start);
        }
        if let Ok(value) = object.extract::<PyRef<'_, PyReadout>>() {
            return self.native(py, Value::Readout(&value.inner), start);
        }
        if let Ok(value) = object.extract::<PyRef<'_, PyReference>>() {
            return self.native(py, Value::Reference(&value.inner), start);
        }
        if start == self.path.0.len() {
            return match self.request {
                "length" if object.is_instance_of::<PyList>() || object.is_instance_of::<PyTuple>() => {
                    object.len()?.into_py_any(py)
                }
                "keys" if object.is_instance_of::<PyDict>() => object.cast::<PyDict>()?.keys().into_py_any(py),
                _ => Err(self.mismatch(&object.get_type().name()?.extract::<String>()?)),
            };
        }
        let segment = &self.path.0[start];
        let child = match segment {
            Segment::Field(field) if live_field(object, field) => object.getattr(field.as_str())?,
            Segment::Key(key) if object.is_instance_of::<PyDict>() => object
                .cast::<PyDict>()?
                .get_item(key)?
                .ok_or_else(|| self.missing(start))?,
            Segment::Index(index) if object.is_instance_of::<PyList>() || object.is_instance_of::<PyTuple>() => {
                if *index >= object.len()? {
                    return Err(self.missing(start));
                }
                object.get_item(*index)?
            }
            _ => return Err(self.missing(start)),
        };
        self.live(&child, start + 1)
    }
}

fn live_field(object: &Bound<'_, PyAny>, field: &str) -> bool {
    if object.is_instance_of::<PyQodec>() {
        matches!(
            field,
            "name"
                | "description"
                | "schema_version"
                | "manifest_filename"
                | "metadata"
                | "layers"
                | "instruction_sets"
                | "codes"
        )
    } else if object.is_instance_of::<PyLayer>() {
        matches!(field, "instruction_set" | "codes" | "gadgets")
    } else if object.is_instance_of::<PyGadget>() {
        matches!(
            field,
            "implements"
                | "circuit"
                | "inputs"
                | "outputs"
                | "parameter_bindings"
                | "checks"
                | "readouts"
                | "frames"
                | "metadata"
        )
    } else if object.is_instance_of::<PyCircuit>() {
        matches!(field, "instruction_set" | "source" | "format")
    } else if object.is_instance_of::<PyEncoding>() {
        matches!(field, "code" | "support" | "block_types")
    } else {
        false
    }
}

fn native_to_py(py: Python<'_>, value: Value<'_>) -> PyResult<Py<PyAny>> {
    match value {
        Value::Str(value) => value.into_py_any(py),
        Value::Int(value) => value.into_py_any(py),
        Value::Float(value) => value.into_py_any(py),
        Value::Bool(value) => value.into_py_any(py),
        Value::None => Ok(py.None()),
        Value::Instruction(value) => PyInstruction { inner: value.clone() }.into_py_any(py),
        Value::Block(value) => PyBlock {
            name: value.name.clone(),
            encodes: value.encodes,
        }
        .into_py_any(py),
        Value::BlockOperand(value) => PyBlockOperand { inner: value.clone() }.into_py_any(py),
        Value::Parameter(value) => PyParameter { inner: value.clone() }.into_py_any(py),
        Value::ParameterKind(kind) => PyParameter {
            inner: model::Parameter {
                name: String::new(),
                kind,
            },
        }
        .kind(py)
        .map(Bound::unbind),
        Value::Condition(value) => PyCondition { inner: value.clone() }.into_py_any(py),
        Value::Action(value) => action_step_to_py(py, value),
        Value::Readout(value) => PyReadout { inner: value.clone() }.into_py_any(py),
        Value::ReadoutSpec {
            spec,
            position,
            observe_count,
        } => PyReadout {
            inner: model::Readout {
                position,
                is_flag: position >= observe_count,
                name: spec.name.clone(),
                equation: spec.equation.clone(),
            },
        }
        .into_py_any(py),
        Value::Reference(value) => PyReference { inner: value.clone() }.into_py_any(py),
        Value::Sequence(values) => {
            let items = values
                .iter()
                .map(|value| native_to_py(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            if matches!(
                values,
                value::Sequence::Terms(_)
                    | value::Sequence::Checks(_)
                    | value::Sequence::Readouts(_)
                    | value::Sequence::ReadoutSpecs(_, _)
            ) {
                PyTuple::new(py, items)?.into_py_any(py)
            } else {
                PyList::new(py, items)?.into_py_any(py)
            }
        }
        Value::Mapping(values) => {
            let mapping = PyDict::new(py);
            for (key, value) in values.entries() {
                mapping.set_item(key, native_to_py(py, value)?)?;
            }
            mapping.into_py_any(py)
        }
        _ => Err(PyTypeError::new_err("model object requires a live Python owner")),
    }
}

#[pyfunction]
pub(crate) fn _node_query(owner: &Bound<'_, PyAny>, path: &PyModelPath, request: &str) -> PyResult<Py<PyAny>> {
    Query {
        path: &path.inner,
        request,
    }
    .live(owner, 0)
}
