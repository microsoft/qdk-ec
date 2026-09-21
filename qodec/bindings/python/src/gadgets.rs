//! Python bindings for the gadget surface: gadgets, encodings,
//! checks, and readouts.

use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hash, Hasher};

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyInt, PyList, PyMapping, PyString, PyTuple};
use qodec::{ParityTerm, Reference, ReferenceSegment};

use crate::codes::PyCode;
use crate::types::{
    PyInstruction, PyInstructionCall, PyInstructionSet, instruction_call_to_py, validate_encoding_arity,
};

// ── Reference wrapping ───────────────────────────────────────────────────────────

/// An authored path and its cached parsed fields, shared with the Rust model.
#[pyclass(name = "Reference", module = "qodec", frozen)]
pub struct PyReference {
    pub(crate) inner: Reference,
}

impl PyReference {
    pub(crate) fn from_inner(inner: Reference) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyReference {
    #[new]
    fn new(value: ReferenceArg) -> Self {
        Self::from_inner(value.0)
    }

    #[getter]
    fn path(&self) -> &str {
        self.inner.path()
    }

    #[getter]
    fn segments<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let owner = py.get_type::<Self>();
        let values = self
            .inner
            .segments()
            .iter()
            .map(|segment| match segment {
                ReferenceSegment::Field(name) => owner.getattr("Field")?.call1((name,)),
                ReferenceSegment::Key(value) => owner.getattr("Key")?.call1((value,)),
                ReferenceSegment::Index(value) => owner.getattr("Index")?.call1((*value,)),
                ReferenceSegment::Slice { start, stop, step } => owner
                    .getattr("Slice")?
                    .call_method1("_from_validated", (*start, *stop, *step)),
                ReferenceSegment::Union(indices) => owner.getattr("Union")?.call1((PyTuple::new(py, indices)?,)),
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, values)
    }

    fn expand(slf: PyRef<'_, Self>) -> PyResult<Vec<Py<Self>>> {
        slf.inner
            .expand()
            .map(|inner| Py::new(slf.py(), Self::from_inner(inner)))
            .collect()
    }

    fn __str__(&self) -> &str {
        self.inner.path()
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Reference({})",
            PyString::new(py, self.inner.path()).repr()?.extract::<String>()?
        ))
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.inner.hash(&mut hasher);
        hasher.finish()
    }

    fn __richcmp__(&self, other: &Bound<'_, PyAny>, operation: pyo3::basic::CompareOp) -> PyResult<Py<PyAny>> {
        let py = other.py();
        let equal = if let Ok(reference) = other.extract::<PyRef<'_, Self>>() {
            self.inner == reference.inner
        } else {
            return Ok(py.NotImplemented());
        };
        let result = match operation {
            pyo3::basic::CompareOp::Eq => equal,
            pyo3::basic::CompareOp::Ne => !equal,
            _ => return Ok(py.NotImplemented()),
        };
        Ok(result.into_pyobject(py)?.to_owned().into_any().unbind())
    }
}

/// A property-path reference accepted from Python as either a plain `str`
/// or a `Reference` value object. Existing values retain their parsed fields.
struct ReferenceArg(Reference);

impl<'py> FromPyObject<'_, 'py> for ReferenceArg {
    type Error = PyErr;

    fn extract(object: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(reference) = object.extract::<PyRef<'_, PyReference>>() {
            return Ok(Self(reference.inner.clone()));
        }
        let text = object.extract::<String>()?;
        Reference::parse(&text)
            .map(Self)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

/// Retain validated references extracted from the Python inputs.
struct ParityTermArg(ParityTerm);

impl<'py> FromPyObject<'_, 'py> for ParityTermArg {
    type Error = PyErr;

    fn extract(object: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if object.is_instance_of::<PyBool>() {
            return Err(PyValueError::new_err(
                "parity constants must be integer 0 or 1, not booleans",
            ));
        }
        if object.is_instance_of::<PyInt>() {
            return match object.extract::<i64>()? {
                0 => Ok(Self(ParityTerm::Bit(false))),
                1 => Ok(Self(ParityTerm::Bit(true))),
                _ => Err(PyValueError::new_err("parity constants must be integer 0 or 1")),
            };
        }
        let reference = ReferenceArg::extract(object)?.0;
        parity_reference(reference).map(|reference| Self(reference.into()))
    }
}

fn parity_reference(reference: Reference) -> PyResult<Reference> {
    ParityTerm::Reference(reference.clone())
        .validate()
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    Ok(reference)
}

fn equations_from_py(rows: Vec<Vec<ParityTermArg>>) -> Vec<qodec::ParityEquation> {
    rows.into_iter()
        .map(|row| row.into_iter().map(|reference| reference.0).collect())
        .collect()
}

struct FramesArg(BTreeMap<Reference, Vec<ParityTermArg>>);

impl<'py> FromPyObject<'_, 'py> for FramesArg {
    type Error = PyErr;

    fn extract(object: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        let mapping = object.cast::<PyMapping>()?;
        let mut entries = BTreeMap::new();
        for item in mapping.items()?.iter() {
            let (target, terms): (ReferenceArg, Vec<ParityTermArg>) = item.extract()?;
            if entries.contains_key(&target.0) {
                return Err(PyValueError::new_err(format!(
                    "duplicate frame target '{}'",
                    target.0.path()
                )));
            }
            entries.insert(target.0, terms);
        }
        Ok(Self(entries))
    }
}

fn frames_from_py(
    frames: BTreeMap<Reference, Vec<ParityTermArg>>,
) -> PyResult<BTreeMap<qodec::Reference, qodec::ParityEquation>> {
    frames
        .into_iter()
        .map(|(target, terms)| {
            let target = parity_reference(target)?;
            Ok((target, terms.into_iter().map(|term| term.0).collect()))
        })
        .collect()
}

/// Wrap each reference of each parity equation (`checks` / `readouts`) in `Reference`.
pub(crate) fn wrap_equation<'py>(py: Python<'py>, equation: &[ParityTerm]) -> PyResult<Bound<'py, PyTuple>> {
    let terms = equation
        .iter()
        .map(|term| match term {
            ParityTerm::Reference(inner) => Py::new(py, PyReference::from_inner(inner.clone())).map(Py::into_any),
            ParityTerm::Bit(value) => Ok(u8::from(*value).into_pyobject(py)?.into_any().unbind()),
        })
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, terms)
}

fn wrap_equations<'py>(py: Python<'py>, equations: &[qodec::ParityEquation]) -> PyResult<Bound<'py, PyTuple>> {
    let rows = equations
        .iter()
        .map(|equation| wrap_equation(py, equation))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, rows)
}

/// Copy authored readout data, leaving positions and roles to the destination gadget.
fn readouts_from_py(value: &[Bound<'_, PyAny>]) -> PyResult<Vec<qodec::ReadoutSpec>> {
    let mut out = Vec::with_capacity(value.len());
    for item in value {
        if let Ok(readout) = item.extract::<PyRef<'_, PyReadout>>() {
            out.push(readout.inner.to_spec());
        } else if let Ok(dict) = item.cast::<PyMapping>() {
            if dict.len()? != 1 {
                return Err(PyValueError::new_err(
                    "a named readout must be a single-key {name: parity-equation} mapping",
                ));
            }
            let key = dict.keys()?.get_item(0)?;
            let val = dict.get_item(&key)?;
            let name: String = key.extract()?;
            let equation: Vec<ParityTermArg> = val.extract()?;
            out.push(qodec::ReadoutSpec {
                name: Some(name),
                equation: equation.into_iter().map(|reference| reference.0).collect(),
            });
        } else {
            let equation = item.extract::<Vec<ParityTermArg>>().map_err(|error| {
                if error.is_instance_of::<PyValueError>(item.py()) {
                    error
                } else {
                    PyValueError::new_err(
                        "each readout must be a Readout, a parity equation (a sequence of property-path references), \
                         or a single-key {name: parity-equation} mapping",
                    )
                }
            })?;
            out.push(qodec::ReadoutSpec {
                name: None,
                equation: equation.into_iter().map(|reference| reference.0).collect(),
            });
        }
    }
    Ok(out)
}

/// A gadget output equation with its position, optional name, and flag role.
///
/// Returned by ``Gadget.readouts`` and accepted by its constructor and setter.
/// Position and role are assigned by the destination gadget.
#[pyclass(name = "Readout", module = "qodec.gadgets", frozen)]
pub struct PyReadout {
    pub(crate) inner: qodec::Readout,
}

#[pymethods]
impl PyReadout {
    /// Index in ``Gadget.readouts``, addressed as ``readouts[i]``.
    #[getter]
    fn position(&self) -> usize {
        self.inner.position
    }

    /// The authored name, or `None` if the entry is anonymous.
    ///
    /// Independent of `is_flag`: an observable may be named too.
    #[getter]
    fn name(&self) -> Option<&str> {
        self.inner.name.as_deref()
    }

    /// Whether this position follows the implemented instruction's observe outcomes.
    #[getter]
    fn is_flag(&self) -> bool {
        self.inner.is_flag
    }

    /// The parity terms as an immutable tuple of references and integer bits.
    #[getter]
    fn equation<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        wrap_equation(py, &self.inner.equation)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|other| self.inner == other.inner)
    }

    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        let value = pythonize::pythonize(py, &self.inner.to_spec())?;
        let options = PyDict::new(py);
        options.set_item("ensure_ascii", false)?;
        py.import("json")?
            .call_method("dumps", (value,), Some(&options))?
            .extract()
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let is_flag = if self.inner.is_flag { "True" } else { "False" };
        let name = match &self.inner.name {
            Some(name) => PyString::new(py, name).repr()?.extract::<String>()?,
            None => "None".to_owned(),
        };
        let terms = self
            .inner
            .equation
            .iter()
            .map(|term| match term {
                ParityTerm::Reference(reference) => reference
                    .path()
                    .into_pyobject(py)
                    .map(|value| value.into_any().unbind()),
                ParityTerm::Bit(value) => u8::from(*value)
                    .into_pyobject(py)
                    .map(|value| value.into_any().unbind()),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let equation = PyList::new(py, terms)?.repr()?.extract::<String>()?;
        Ok(format!(
            "Readout(position={}, name={name}, is_flag={is_flag}, equation={equation})",
            self.inner.position
        ))
    }
}

fn readouts_to_py<'py>(
    py: Python<'py>,
    readouts: &[qodec::ReadoutSpec],
    observe_count: usize,
) -> PyResult<Bound<'py, PyTuple>> {
    let values = qodec::Readout::resolve_list(readouts, observe_count)
        .into_iter()
        .map(|inner| Py::new(py, PyReadout { inner }))
        .collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, values)
}

// ── Gadgets ─────────────────────────────────────────────────────────────────

/// Source text and a shared instruction set, with an optional format tag.
///
/// Construction stores the text without reading a file or parsing it.
/// ``calls``, ``blocks``, and ``readouts`` parse the current source
/// on access and raise ``ValueError`` on unsupported formats, parse failures,
/// or a call naming an instruction the instruction set does not declare.
#[pyclass(name = "Circuit", module = "qodec.gadgets")]
pub struct PyCircuit {
    /// The target instruction set the circuit source calls into.
    pub(crate) instruction_set: Py<PyInstructionSet>,
    /// The circuit source, verbatim.
    pub(crate) source: String,
    /// Optional explicit source format (`stim` / `openqasm`). `None`
    /// leaves the format to be inferred from the source on demand.
    pub(crate) format: Option<String>,
}

impl PyCircuit {
    /// The core circuit this wrapper mirrors, for the derived accessors.
    fn resolved(&self, py: Python<'_>) -> qodec::Circuit {
        qodec::Circuit {
            instruction_set: self.instruction_set.borrow(py).to_arc(py),
            source: self.source.clone(),
            format: self.format.clone(),
        }
    }
}

#[pymethods]
impl PyCircuit {
    #[new]
    #[pyo3(signature = (instruction_set, source, *, format = None))]
    fn new(instruction_set: Py<PyInstructionSet>, source: String, format: Option<String>) -> Self {
        Self {
            instruction_set,
            source,
            format,
        }
    }

    #[getter]
    fn instruction_set(&self, py: Python<'_>) -> Py<PyInstructionSet> {
        self.instruction_set.clone_ref(py)
    }

    fn _copy_shell(&self, py: Python<'_>) -> Self {
        Self {
            instruction_set: self.instruction_set.clone_ref(py),
            source: self.source.clone(),
            format: self.format.clone(),
        }
    }

    #[setter]
    fn set_instruction_set(&mut self, value: Py<PyInstructionSet>) {
        self.instruction_set = value;
    }

    #[getter]
    fn source(&self) -> &str {
        &self.source
    }

    #[setter]
    fn set_source(&mut self, value: String) {
        self.source = value;
    }

    #[getter]
    fn format(&self) -> Option<&str> {
        self.format.as_deref()
    }

    #[setter]
    fn set_format(&mut self, value: Option<String>) {
        self.format = value;
    }

    /// The format tag if set, otherwise the one inferred from the source.
    #[getter]
    fn effective_format(&self, py: Python<'_>) -> String {
        self.resolved(py).effective_format().to_owned()
    }

    #[pyo3(signature = (*, parser = None))]
    fn calls(&self, py: Python<'_>, parser: Option<Py<PyAny>>) -> PyResult<Vec<PyInstructionCall>> {
        let circuit = self.resolved(py);
        let calls = crate::parsers::with_errors(|| match parser {
            Some(parser) => circuit.calls_with(|source, target| {
                crate::parsers::invoke(py, parser.bind(py), source, target).map_err(crate::parsers::callback_error)
            }),
            None => circuit.calls(),
        })?;
        Ok(calls.iter().map(instruction_call_to_py).collect())
    }

    /// Distinct block labels in first-appearance order, as used by ``Encoding.support``.
    ///
    /// Multi-qubit blocks are not expanded into individual qubits.
    /// These are labels, not physical addresses or a simulator size.
    #[getter]
    fn blocks(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        crate::parsers::with_errors(|| self.resolved(py).blocks())
    }

    /// One entry per output bit this circuit produces, in record
    /// order, so the list index is the ``i`` of a ``circuit.readouts[i]``
    /// reference.
    ///
    /// Each call contributes its ``observe`` outcomes first, then its declared
    /// flags. Entries are `Outcome` or `Flag`; neither carries its own
    /// position, so ``enumerate`` it rather than filtering first.
    #[getter]
    fn readouts(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let readouts = crate::parsers::with_errors(|| self.resolved(py).readouts())?;
        readouts
            .into_iter()
            .map(|readout| match readout {
                qodec::CircuitReadout::Outcome {
                    instruction,
                    observable,
                } => Py::new(
                    py,
                    PyOutcome {
                        instruction,
                        observable: observable.0,
                    },
                )
                .map(|value: Py<PyOutcome>| value.into_any()),
                qodec::CircuitReadout::Flag { instruction, name } => {
                    Py::new(py, PyFlag { instruction, name }).map(|value: Py<PyFlag>| value.into_any())
                }
            })
            .collect()
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyCircuit>>()
            .is_ok_and(|other| circuit_struct_eq(self, &other, py))
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!("Circuit(instruction_set={:?})", self.instruction_set.borrow(py).name)
    }

    fn __str__(&self, py: Python<'_>) -> String {
        self.resolved(py).to_string()
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

/// One output bit from a circuit call's ``observe`` action.
#[pyclass(name = "Outcome", module = "qodec.gadgets", frozen)]
pub struct PyOutcome {
    instruction: usize,
    observable: String,
}

#[pymethods]
impl PyOutcome {
    /// Index into `Circuit.calls` of the call that produced this bit.
    #[getter]
    fn instruction(&self) -> usize {
        self.instruction
    }

    /// The observable as declared by the called instruction, not rewritten
    /// with this call's circuit labels.
    #[getter]
    fn observable(&self) -> &str {
        &self.observable
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|other| self.instruction == other.instruction && self.observable == other.observable)
    }

    fn __repr__(&self) -> String {
        format!(
            "Outcome(instruction={}, observable={:?})",
            self.instruction, self.observable
        )
    }
}

/// One output bit from a circuit call's declared flags.
#[pyclass(name = "Flag", module = "qodec.gadgets", frozen)]
pub struct PyFlag {
    instruction: usize,
    name: String,
}

#[pymethods]
impl PyFlag {
    /// Index into `Circuit.calls` of the call that produced this bit.
    #[getter]
    fn instruction(&self) -> usize {
        self.instruction
    }

    /// The flag's declared name on the called instruction.
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, Self>>()
            .is_ok_and(|other| self.instruction == other.instruction && self.name == other.name)
    }

    fn __repr__(&self) -> String {
        format!("Flag(instruction={}, name={:?})", self.instruction, self.name)
    }
}

#[pyclass(name = "Gadget", module = "qodec")]
pub struct PyGadget {
    pub(crate) implements: Py<PyInstruction>,
    /// The gadget circuit: the program source, its target instruction set, and format
    /// tag.
    pub(crate) circuit: Py<PyCircuit>,
    pub(crate) inputs: Vec<Py<PyEncoding>>,
    pub(crate) outputs: Vec<Py<PyEncoding>>,
    /// Bindings from implemented-instruction parameter names to the
    /// circuit-source parameters they forward into (`{"theta": "angle"}`).
    pub(crate) parameter_bindings: BTreeMap<String, String>,
    /// Deterministic syndrome checks, retaining parsed reference expressions.
    pub(crate) checks: Vec<qodec::ParityEquation>,
    /// Terminal readouts the gadget exposes, as one positional list: the
    /// implemented instruction's `observe` outcomes first (the observables),
    /// then its `flags:` flags (each a single parity). Each entry carries an
    /// optional name. At the Python boundary it is a
    /// `list[list[ReferenceLike] | dict[str, list[ReferenceLike]]]`.
    pub(crate) readouts: Vec<qodec::ReadoutSpec>,
    pub(crate) frames: BTreeMap<qodec::Reference, qodec::ParityEquation>,
    /// Free-form, qodec-opaque annotations (see [`qodec::Metadata`]).
    pub(crate) metadata: qodec::Metadata,
}

impl PyGadget {
    /// Materialize a resolved `Gadget` snapshot of the current state.
    pub fn to_resolved(&self, py: Python<'_>) -> qodec::Gadget {
        let checks: Vec<qodec::ParityEquation> = self.checks.clone();
        let circuit_ref = self.circuit.borrow(py);
        let circuit = qodec::Circuit {
            instruction_set: circuit_ref.instruction_set.borrow(py).to_arc(py),
            source: circuit_ref.source.clone(),
            format: circuit_ref.format.clone(),
        };
        let implements = self.implements.borrow(py).inner.clone();
        let readouts = qodec::Readout::resolve_list(&self.readouts, implements.observe_count());
        qodec::Gadget {
            implements,
            circuit,
            inputs: self
                .inputs
                .iter()
                .map(|encoding| encoding.borrow(py).to_resolved(py))
                .collect(),
            outputs: self
                .outputs
                .iter()
                .map(|encoding| encoding.borrow(py).to_resolved(py))
                .collect(),
            parameter_bindings: self.parameter_bindings.clone(),
            checks,
            readouts,
            frames: self.frames.clone(),
            metadata: self.metadata.clone(),
        }
    }

    /// Build a `PyGadget` from a resolved `Gadget`, sharing the supplied
    /// instruction, instruction set, and code cells for identity.
    pub fn from_resolved(
        gadget: &qodec::Gadget,
        implements: Py<PyInstruction>,
        instruction_set: Py<PyInstructionSet>,
        codes_by_name: &BTreeMap<String, Py<PyCode>>,
        py: Python<'_>,
    ) -> PyResult<Self> {
        let build_encodings = |encodings: &[qodec::Encoding]| -> PyResult<Vec<Py<PyEncoding>>> {
            encodings
                .iter()
                .map(|enc| {
                    let code = match codes_by_name.get(&enc.code.name) {
                        Some(code) => code.clone_ref(py),
                        None => Py::new(py, PyCode::from_inner((*enc.code).clone()))?,
                    };
                    Py::new(py, PyEncoding::from_resolved(enc, code))
                })
                .collect()
        };
        let circuit = Py::new(
            py,
            PyCircuit {
                instruction_set,
                source: gadget.circuit.source.clone(),
                format: gadget.circuit.format.clone(),
            },
        )?;
        Ok(Self {
            implements,
            circuit,
            inputs: build_encodings(&gadget.inputs)?,
            outputs: build_encodings(&gadget.outputs)?,
            parameter_bindings: gadget.parameter_bindings.clone(),
            checks: gadget.checks.clone(),
            readouts: gadget.readouts.iter().map(qodec::Readout::to_spec).collect(),
            frames: gadget.frames.clone(),
            metadata: gadget.metadata.clone(),
        })
    }
}

/// Structural equality of two circuits: same source and format, and an instruction set
/// whose underlying definition matches (by value, not cell identity).
fn circuit_struct_eq(a: &PyCircuit, b: &PyCircuit, py: Python<'_>) -> bool {
    a.source == b.source
        && a.format == b.format
        && a.instruction_set.borrow(py).to_inner(py) == b.instruction_set.borrow(py).to_inner(py)
}

/// Structural equality of two encodings: same support, block types, and a code whose
/// underlying definition matches (by value, not cell identity).
fn encoding_struct_eq(a: &PyEncoding, b: &PyEncoding, py: Python<'_>) -> bool {
    a.support == b.support && a.block_types == b.block_types && a.code.borrow(py).inner == b.code.borrow(py).inner
}

impl PyGadget {
    /// Structural (value) equality over every field, dereferencing shared
    /// `Py<...>` cells so that two gadgets compare equal when their
    /// contents match regardless of cell identity. Reused by `PyLayer`.
    pub(crate) fn struct_eq(&self, other: &PyGadget, py: Python<'_>) -> bool {
        self.implements.borrow(py).inner == other.implements.borrow(py).inner
            && circuit_struct_eq(&self.circuit.borrow(py), &other.circuit.borrow(py), py)
            && self.inputs.len() == other.inputs.len()
            && self
                .inputs
                .iter()
                .zip(&other.inputs)
                .all(|(a, b)| encoding_struct_eq(&a.borrow(py), &b.borrow(py), py))
            && self.outputs.len() == other.outputs.len()
            && self
                .outputs
                .iter()
                .zip(&other.outputs)
                .all(|(a, b)| encoding_struct_eq(&a.borrow(py), &b.borrow(py), py))
            && self.parameter_bindings == other.parameter_bindings
            && self.checks == other.checks
            && self.readouts == other.readouts
            && self.frames == other.frames
            && self.metadata == other.metadata
    }
}

#[pymethods]
impl PyGadget {
    /// Resolve an address relative to this gadget without interpreting circuit source.
    fn resolve(slf: &Bound<'_, Self>, path: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        let node = slf.py().import("qodec._nodes")?.getattr("Node")?;
        Ok(node.call_method1("_create", (slf, path))?.unbind())
    }

    #[new]
    #[pyo3(signature = (
        implements,
        circuit,
        *,
        inputs = Vec::new(),
        outputs = Vec::new(),
        checks = Vec::new(),
        readouts = None,
        frames = None,
        parameter_bindings = None,
        metadata = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        implements: Py<PyInstruction>,
        circuit: Py<PyCircuit>,
        inputs: Vec<Py<PyEncoding>>,
        outputs: Vec<Py<PyEncoding>>,
        checks: Vec<Vec<ParityTermArg>>,
        readouts: Option<Vec<Bound<'_, PyAny>>>,
        frames: Option<FramesArg>,
        parameter_bindings: Option<BTreeMap<String, String>>,
        metadata: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        {
            let implemented_ref = implements.borrow(py);
            validate_encoding_arity(&implemented_ref, &inputs, &outputs)?;
        }
        let readouts = match readouts {
            Some(list) => readouts_from_py(&list)?,
            None => Vec::new(),
        };
        Ok(Self {
            implements,
            circuit,
            inputs,
            outputs,
            parameter_bindings: parameter_bindings.unwrap_or_default(),
            checks: equations_from_py(checks),
            readouts,
            frames: frames
                .map(|value| frames_from_py(value.0))
                .transpose()?
                .unwrap_or_default(),
            metadata: crate::metadata_from_py(metadata.as_ref())?,
        })
    }

    #[getter]
    fn implements(&self, py: Python<'_>) -> Py<PyInstruction> {
        self.implements.clone_ref(py)
    }

    #[setter]
    fn set_implements(&mut self, py: Python<'_>, value: Py<PyInstruction>) -> PyResult<()> {
        if self.implements.borrow(py).inner.mnemonic != value.borrow(py).inner.mnemonic {
            return Err(PyValueError::new_err(
                "replacing implements cannot change the gadget mnemonic",
            ));
        }
        self.implements = value;
        Ok(())
    }

    #[getter]
    fn circuit(&self, py: Python<'_>) -> Py<PyCircuit> {
        self.circuit.clone_ref(py)
    }

    #[setter]
    fn set_circuit(&mut self, value: Py<PyCircuit>) {
        self.circuit = value;
    }

    #[getter]
    fn inputs<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "inputs", false)
    }

    fn _get_inputs(&self, py: Python<'_>) -> Vec<Py<PyEncoding>> {
        self.inputs.iter().map(|encoding| encoding.clone_ref(py)).collect()
    }

    #[setter]
    fn set_inputs(&mut self, value: Vec<Py<PyEncoding>>) {
        self.inputs = value;
    }

    #[getter]
    fn outputs<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "outputs", false)
    }

    fn _get_outputs(&self, py: Python<'_>) -> Vec<Py<PyEncoding>> {
        self.outputs.iter().map(|encoding| encoding.clone_ref(py)).collect()
    }

    #[setter]
    fn set_outputs(&mut self, value: Vec<Py<PyEncoding>>) {
        self.outputs = value;
    }

    #[getter]
    fn parameter_bindings<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "parameter_bindings", true)
    }

    fn _get_parameter_bindings(&self) -> BTreeMap<String, String> {
        self.parameter_bindings.clone()
    }

    #[setter]
    fn set_parameter_bindings(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let bindings = crate::collections::mapping(value)?.extract()?;
        slf.borrow_mut().parameter_bindings = bindings;
        Ok(())
    }

    #[getter]
    fn metadata<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "metadata", true)
    }

    fn _get_metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::metadata_to_py(py, &self.metadata)
    }

    #[setter]
    fn set_metadata(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let metadata = crate::metadata_from_py(Some(value))?;
        slf.borrow_mut().metadata = metadata;
        Ok(())
    }

    #[getter]
    fn checks<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "checks", false)
    }

    fn _get_checks<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        wrap_equations(py, &self.checks)
    }

    #[setter]
    fn set_checks(&mut self, value: Vec<Vec<ParityTermArg>>) {
        self.checks = equations_from_py(value);
    }

    #[getter]
    fn frames<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "frames", true)
    }

    fn _get_frames<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        for (target, equation) in &self.frames {
            result.set_item(target.path(), wrap_equation(py, equation)?)?;
        }
        Ok(result)
    }

    #[staticmethod]
    fn _validate_frame_key(value: ReferenceArg) -> PyResult<PyReference> {
        parity_reference(value.0).map(PyReference::from_inner)
    }

    #[setter]
    fn set_frames(&mut self, value: FramesArg) -> PyResult<()> {
        self.frames = frames_from_py(value.0)?;
        Ok(())
    }

    /// Terminal readouts this gadget exposes, resolved against the
    /// instruction it implements: the instruction's ``observe`` outcomes
    /// first, then its declared flags.
    ///
    /// Returns a live sequence of immutable descriptors. The setter copies equations and names from
    /// ``Readout`` values, parity sequences, or single-key named dictionaries.
    #[getter]
    fn readouts<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "readouts", false)
    }

    fn _get_readouts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let observe = self.implements.borrow(py).inner.observe_count();
        readouts_to_py(py, &self.readouts, observe)
    }

    #[setter]
    fn set_readouts(&mut self, value: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        self.readouts = readouts_from_py(&value)?;
        Ok(())
    }

    fn _copy_shell(&self, py: Python<'_>) -> Self {
        Self {
            implements: self.implements.clone_ref(py),
            circuit: self.circuit.clone_ref(py),
            inputs: self.inputs.iter().map(|value| value.clone_ref(py)).collect(),
            outputs: self.outputs.iter().map(|value| value.clone_ref(py)).collect(),
            checks: self.checks.clone(),
            readouts: self.readouts.clone(),
            frames: self.frames.clone(),
            parameter_bindings: self.parameter_bindings.clone(),
            metadata: self.metadata.clone(),
        }
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyGadget>>()
            .is_ok_and(|other| self.struct_eq(&other, py))
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        format!("Gadget({:?})", self.implements.borrow(py).inner.mnemonic)
    }

    fn __str__(&self, py: Python<'_>) -> String {
        self.to_resolved(py).to_string()
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

#[pyclass(name = "Encoding", module = "qodec.gadgets")]
pub struct PyEncoding {
    pub(crate) code: Py<PyCode>,
    pub(crate) support: Vec<String>,
    pub(crate) block_types: Vec<String>,
}

impl PyEncoding {
    /// Materialize a resolved `Encoding` snapshot of the current state.
    pub fn to_resolved(&self, py: Python<'_>) -> qodec::Encoding {
        qodec::Encoding {
            code: self.code.borrow(py).to_arc(),
            support: self.support.clone(),
            block_types: self.block_types.clone(),
        }
    }

    /// Build a `PyEncoding` from a resolved `Encoding`, sharing the supplied
    /// `Py<PyCode>` cell for code identity.
    pub fn from_resolved(encoding: &qodec::Encoding, code: Py<PyCode>) -> Self {
        Self {
            code,
            support: encoding.support.clone(),
            block_types: encoding.block_types.clone(),
        }
    }
}

#[pymethods]
impl PyEncoding {
    #[new]
    #[pyo3(signature = (code, *, support = Vec::new(), block_types = Vec::new()))]
    fn new(code: Py<PyCode>, support: Vec<String>, block_types: Vec<String>) -> Self {
        Self {
            code,
            support,
            block_types,
        }
    }

    #[getter]
    fn code(&self, py: Python<'_>) -> Py<PyCode> {
        self.code.clone_ref(py)
    }

    #[setter]
    fn set_code(&mut self, value: Py<PyCode>) {
        self.code = value;
    }

    #[getter]
    fn support<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "support", false)
    }

    fn _get_support(&self) -> Vec<String> {
        self.support.clone()
    }

    #[setter]
    fn set_support(&mut self, value: Vec<String>) {
        self.support = value;
    }

    #[getter]
    fn block_types<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "block_types", false)
    }

    fn _get_block_types(&self) -> Vec<String> {
        self.block_types.clone()
    }

    #[setter]
    fn set_block_types(&mut self, value: Vec<String>) {
        self.block_types = value;
    }

    fn _copy_shell(&self, py: Python<'_>) -> Self {
        Self {
            code: self.code.clone_ref(py),
            support: self.support.clone(),
            block_types: self.block_types.clone(),
        }
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyEncoding>>()
            .is_ok_and(|other| encoding_struct_eq(self, &other, py))
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let code_name = self.code.borrow(py).inner.name.clone();
        format!("Encoding(code={code_name:?})")
    }
}
