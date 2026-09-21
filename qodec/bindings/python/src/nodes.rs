//! Live Python navigation with the core path grammar and borrowed value selection.

use pyo3::IntoPyObjectExt;
use pyo3::PyClass;
use pyo3::exceptions::{PyLookupError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use qodec as model;
use std::collections::BTreeMap;

#[path = "../../../src/node/path.rs"]
pub(crate) mod path;
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

impl PyModelPath {
    fn normalized(path: ModelPath) -> PyResult<Self> {
        let mut result = ModelPath::default();
        for mut segment in path.0 {
            while let (Some(previous @ (Segment::Slice { .. } | Segment::Union(_))), Segment::Index(index)) =
                (result.0.last(), &segment)
            {
                let selected = path::indices(previous)
                    .nth(*index)
                    .ok_or_else(|| path_error(PathError::Missing(result.child(segment.clone()).to_string())))?;
                result.0.pop();
                segment = Segment::Index(selected);
            }
            result.0.push(segment);
        }
        Ok(Self { inner: result })
    }
}

#[pymethods]
impl PyModelPath {
    #[new]
    fn new(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(reference) = value.extract::<PyRef<'_, PyReference>>() {
            return Ok(Self {
                inner: ModelPath(reference.inner.segments().to_vec()),
            });
        }
        let text = value.extract::<String>()?;
        Ok(Self {
            inner: ModelPath::parse_reference(&text).map_err(|_| path_error(PathError::Syntax(text)))?,
        })
    }

    fn __str__(&self) -> String {
        self.inner.to_string()
    }

    fn _resolve(&self, value: &Bound<'_, PyAny>) -> PyResult<Self> {
        let mut path = self.inner.clone();
        path.0.extend(Self::new(value)?.inner.0);
        Ok(Self { inner: path })
    }

    fn _canonical(&self) -> PyResult<Self> {
        Self::normalized(self.inner.clone())
    }

    fn _index(&self, index: usize) -> PyResult<Self> {
        let mut prefix = self.inner.0.as_slice();
        let mut selected = index;
        while let Some((segment @ (Segment::Slice { .. } | Segment::Union(_)), remaining)) = prefix.split_last() {
            selected = path::indices(segment)
                .nth(selected)
                .ok_or_else(|| path_error(PathError::Missing(self.inner.child(Segment::Index(index)).to_string())))?;
            prefix = remaining;
        }
        Ok(Self {
            inner: ModelPath(prefix.to_vec()).child(Segment::Index(selected)),
        })
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
    fn shared_sequence<T: PyClass>(
        &self,
        owner: &Bound<'_, PyAny>,
        values: &[Py<T>],
        start: usize,
    ) -> PyResult<Py<PyAny>> {
        let py = owner.py();
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
            "value" => self.field_value(owner, start - 1),
            _ => Err(self.mismatch("list")),
        }
    }

    fn shared_mapping<T: PyClass>(
        &self,
        owner: &Bound<'_, PyAny>,
        values: &BTreeMap<String, Py<T>>,
        start: usize,
    ) -> PyResult<Py<PyAny>> {
        let py = owner.py();
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
            "value" => self.field_value(owner, start - 1),
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

    fn select_native<'model>(&self, mut value: Value<'model>, start: usize) -> PyResult<Value<'model>> {
        for (position, segment) in self.path.0.iter().enumerate().skip(start) {
            value = value.child(segment).ok_or_else(|| self.missing(position))?;
        }
        Ok(value)
    }

    fn field_value(&self, owner: &Bound<'_, PyAny>, position: usize) -> PyResult<Py<PyAny>> {
        let Segment::Field(field) = &self.path.0[position] else {
            return Err(self.missing(position));
        };
        let attribute = match field.as_str() {
            "in" => "inputs",
            "out" => "outputs",
            field => field,
        };
        owner.getattr(attribute).map(Bound::unbind)
    }

    fn native_object(&self, owner: &Bound<'_, PyAny>, value: Value<'_>, start: usize) -> PyResult<Py<PyAny>> {
        if let Some(segment @ Segment::Field(_)) = self.path.0.get(start) {
            let field = value.child(segment).ok_or_else(|| self.missing(start))?;
            return self.native_field(owner, field, start);
        }
        self.native(owner.py(), value, start)
    }

    fn native_field(&self, owner: &Bound<'_, PyAny>, value: Value<'_>, position: usize) -> PyResult<Py<PyAny>> {
        if self.request == "value" {
            if self.path.0.len() == position + 1 {
                return self.field_value(owner, position);
            }
            let selected = self.select_native(value, position + 1)?;
            if let Value::Sequence(Sequence::Json(_)) | Value::Mapping(Mapping::Metadata(_)) = selected {
                let path = self.path.0[position + 1..]
                    .iter()
                    .map(|segment| match segment {
                        Segment::Key(key) => key.into_py_any(owner.py()),
                        Segment::Index(index) => index.into_py_any(owner.py()),
                        _ => Err(self.missing(position)),
                    })
                    .collect::<PyResult<Vec<_>>>()?;
                return crate::collections::view_at(owner, "metadata", matches!(selected, Value::Mapping(_)), &path)
                    .map(Bound::unbind);
            }
            return native_to_py(owner.py(), selected);
        }
        self.native(owner.py(), value, position + 1)
    }

    fn native(&self, py: Python<'_>, value: Value<'_>, start: usize) -> PyResult<Py<PyAny>> {
        let value = self.select_native(value, start)?;
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
                    "layers" => return self.shared_sequence(object, &root.layers, start + 1),
                    "metadata" => {
                        return self.native_field(object, Value::Mapping(Mapping::Metadata(&root.metadata)), start);
                    }
                    "codes" | "instruction_sets" => {
                        if self.request == "value" && self.path.0.len() == start + 1 {
                            return self.field_value(object, start);
                        }
                        let values = object.call_method0(format!("_get_{field}"))?;
                        return self.live(&values, start + 1);
                    }
                    _ => {}
                }
            }
            if let Ok(layer) = object.extract::<PyRef<'_, PyLayer>>() {
                match field.as_str() {
                    "gadgets" => return self.shared_mapping(object, &layer.gadgets, start + 1),
                    "codes" => return self.shared_mapping(object, &layer.codes, start + 1),
                    _ => {}
                }
            }
            if let Ok(instruction_set) = object.extract::<PyRef<'_, PyInstructionSet>>() {
                return self.instruction_set_field(object, &instruction_set, field, start);
            }
            if let Ok(gadget) = object.extract::<PyRef<'_, PyGadget>>() {
                match field.as_str() {
                    "in" => return self.shared_sequence(object, &gadget.inputs, start + 1),
                    "out" => return self.shared_sequence(object, &gadget.outputs, start + 1),
                    "readouts" => {
                        return self.native_field(
                            object,
                            Value::Sequence(Sequence::ReadoutSpecs(
                                &gadget.readouts,
                                gadget.implements.borrow(py).inner.observe_count(),
                            )),
                            start,
                        );
                    }
                    "metadata" => {
                        return self.native_field(object, Value::Mapping(Mapping::Metadata(&gadget.metadata)), start);
                    }
                    "checks" => {
                        return self.native_field(object, Value::Sequence(Sequence::Checks(&gadget.checks)), start);
                    }
                    "frames" => {
                        return self.native_field(object, Value::Mapping(Mapping::Frames(&gadget.frames)), start);
                    }
                    "parameter_bindings" => {
                        return self.native_field(
                            object,
                            Value::Mapping(Mapping::ParameterBindings(&gadget.parameter_bindings)),
                            start,
                        );
                    }
                    _ => {}
                }
            }
            if let Ok(encoding) = object.extract::<PyRef<'_, PyEncoding>>() {
                match field.as_str() {
                    "stabilizers" | "x" | "z" => {
                        return self.live(encoding.code.bind(py).as_any(), start);
                    }
                    "support" => {
                        return self.native_field(object, Value::Sequence(Sequence::Strings(&encoding.support)), start);
                    }
                    "block_types" => {
                        return self.native_field(
                            object,
                            Value::Sequence(Sequence::Strings(&encoding.block_types)),
                            start,
                        );
                    }
                    _ => {}
                }
            }
        }
        if start == self.path.0.len() && matches!(self.request, "exists" | "value" | "is_none" | "kind") {
            return self.object_value(object);
        }
        if let Ok(value) = object.extract::<PyRef<'_, PyInstruction>>() {
            return self.native_object(object, Value::Instruction(&value.inner), start);
        }
        if let Ok(value) = object.extract::<PyRef<'_, PyCode>>() {
            return self.native_object(object, Value::Code(&value.inner), start);
        }
        if let Ok(value) = object.extract::<PyRef<'_, PyReadout>>() {
            return self.native_object(object, Value::Readout(&value.inner), start);
        }
        if let Ok(value) = object.extract::<PyRef<'_, PyReference>>() {
            return self.native_object(object, Value::Reference(&value.inner), start);
        }
        self.collection_child(object, start)
    }

    fn instruction_set_field(
        &self,
        owner: &Bound<'_, PyAny>,
        instruction_set: &PyInstructionSet,
        field: &str,
        start: usize,
    ) -> PyResult<Py<PyAny>> {
        let value = match field {
            "name" => Value::Str(&instruction_set.name),
            "description" => Value::Str(&instruction_set.description),
            "blocks" => Value::Sequence(Sequence::Blocks(&instruction_set.blocks)),
            "metadata" => Value::Mapping(Mapping::Metadata(&instruction_set.metadata)),
            "instructions" => {
                let py = owner.py();
                let instructions = instruction_set
                    .instructions
                    .iter()
                    .map(|instruction| (instruction.borrow(py).inner.mnemonic.clone(), instruction.clone_ref(py)))
                    .collect();
                return self.shared_mapping(owner, &instructions, start + 1);
            }
            _ => return Err(self.missing(start)),
        };
        self.native_field(owner, value, start)
    }

    fn object_value(&self, object: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let py = object.py();
        match self.request {
            "exists" => true.into_py_any(py),
            "value" => Ok(object.clone().unbind()),
            "is_none" => object.is_none().into_py_any(py),
            _ => object.get_type().name()?.into_py_any(py),
        }
    }

    fn collection_child(&self, object: &Bound<'_, PyAny>, start: usize) -> PyResult<Py<PyAny>> {
        let py = object.py();
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
            "implements" | "circuit" | "parameter_bindings" | "checks" | "readouts" | "frames" | "metadata"
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
        Value::Reference(value) => PyReference::from_inner(value.clone()).into_py_any(py),
        Value::Sequence(Sequence::Terms(terms)) => crate::gadgets::wrap_equation(py, terms)
            .map(Bound::into_any)
            .map(Bound::unbind),
        Value::Sequence(values @ (Sequence::Strings(_) | Sequence::Paulis(_) | Sequence::Observables(_))) => {
            let values = values
                .iter()
                .map(|value| native_to_py(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(py, values).map(Bound::into_any).map(Bound::unbind)
        }
        Value::Mapping(values @ Mapping::Generators(_)) => {
            let mapping = PyDict::new(py);
            for (key, value) in values.entries() {
                mapping.set_item(key, native_to_py(py, value)?)?;
            }
            py.import("types")?
                .getattr("MappingProxyType")?
                .call1((mapping,))
                .map(Bound::unbind)
        }
        _ => Err(PyTypeError::new_err("model object requires a live Python owner")),
    }
}

#[pyfunction]
pub(crate) fn _node_query(owner: &Bound<'_, PyAny>, path: &PyModelPath, request: &str) -> PyResult<Py<PyAny>> {
    query_node(owner, path, request)
}

fn query_node(owner: &Bound<'_, PyAny>, path: &PyModelPath, request: &str) -> PyResult<Py<PyAny>> {
    if let Some((position, selector)) = path.inner.0.iter().enumerate().find_map(|(position, segment)| {
        if matches!(segment, Segment::Slice { .. } | Segment::Union(_)) {
            Some((position, segment))
        } else {
            None
        }
    }) {
        let prefix = ModelPath(path.inner.0[..position].to_vec());
        let mut members = path::indices(selector)
            .map(|index| PyModelPath {
                inner: prefix.child(Segment::Index(index)),
            })
            .collect::<Vec<_>>();
        for member in &members {
            query_node(owner, member, "exists")?;
        }
        for (position, segment) in path.inner.0.iter().enumerate().skip(position + 1) {
            let missing = || {
                path_error(PathError::Missing(
                    ModelPath(path.inner.0[..=position].to_vec()).to_string(),
                ))
            };
            match segment {
                Segment::Index(index) => {
                    let mut inner = members.get(*index).ok_or_else(missing)?.inner.clone();
                    inner.0.extend_from_slice(&path.inner.0[position + 1..]);
                    return query_node(owner, &PyModelPath { inner }, request);
                }
                Segment::Slice { .. } | Segment::Union(_) => {
                    members = path::indices(segment)
                        .map(|index| {
                            members
                                .get(index)
                                .map(|member| PyModelPath {
                                    inner: member.inner.clone(),
                                })
                                .ok_or_else(missing)
                        })
                        .collect::<PyResult<_>>()?;
                }
                _ => return Err(missing()),
            }
        }
        return selected_query(owner, &members, request);
    }
    Query {
        path: &path.inner,
        request,
    }
    .live(owner, 0)
}

fn selected_query(owner: &Bound<'_, PyAny>, members: &[PyModelPath], request: &str) -> PyResult<Py<PyAny>> {
    let py = owner.py();
    match request {
        "exists" => true.into_py_any(py),
        "kind" => "list".into_py_any(py),
        "is_none" => false.into_py_any(py),
        "length" => members.len().into_py_any(py),
        "value" => {
            let values = members
                .iter()
                .map(|member| query_node(owner, member, "value"))
                .collect::<PyResult<Vec<_>>>()?;
            PyTuple::new(py, values).map(Bound::into_any).map(Bound::unbind)
        }
        _ => Err(PyTypeError::new_err("selection is a sequence, not a mapping")),
    }
}
