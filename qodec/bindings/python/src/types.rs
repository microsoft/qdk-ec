use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyTuple;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use crate::gadgets::PyEncoding;

/// Extract an `instructions` argument: accept either a list of `Instruction`
/// or a mapping whose keys match the instruction mnemonics.
fn extract_instructions(value: &Bound<'_, PyAny>) -> PyResult<Vec<Py<PyInstruction>>> {
    let py = value.py();
    let instructions = if value.hasattr("keys")? {
        let map = crate::collections::mapping(value)?;
        let mut instructions = Vec::new();
        for (key, value) in map.cast::<pyo3::types::PyDict>()?.iter() {
            let key: String = key.extract()?;
            let instruction: Py<PyInstruction> = value.extract()?;
            if key != instruction.borrow(py).inner.mnemonic {
                return Err(PyValueError::new_err("instruction key must match its mnemonic"));
            }
            instructions.push(instruction);
        }
        instructions
    } else {
        value.extract::<Vec<Py<PyInstruction>>>().map_err(|_| {
            PyValueError::new_err("InstructionSet instructions must be a list[Instruction] or dict[str, Instruction]")
        })?
    };
    let mut seen = std::collections::BTreeSet::new();
    for instruction in &instructions {
        let instruction = instruction.borrow(py);
        if !seen.insert(instruction.inner.mnemonic.clone()) {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "InstructionSet has duplicate instruction {:?}",
                instruction.inner.mnemonic
            )));
        }
    }
    Ok(instructions)
}

// ── Instruction Sets ────────────────────────────────────────────────────────

#[pyclass(name = "InstructionSet", module = "qodec")]
pub struct PyInstructionSet {
    pub(crate) name: String,
    description: String,
    pub(crate) blocks: Vec<qodec::Block>,
    pub(crate) instructions: Vec<Py<PyInstruction>>,
    metadata: qodec::Metadata,
}

impl PyInstructionSet {
    pub fn to_inner(&self, py: Python<'_>) -> qodec::InstructionSet {
        qodec::InstructionSet {
            name: self.name.clone(),
            description: self.description.clone(),
            blocks: self.blocks.clone(),
            instructions: self
                .instructions
                .iter()
                .map(|value| value.borrow(py).inner.clone())
                .collect(),
            metadata: self.metadata.clone(),
        }
    }

    /// Materialize a fresh `Arc<InstructionSet>` snapshot of the current state.
    pub fn to_arc(&self, py: Python<'_>) -> Arc<qodec::InstructionSet> {
        Arc::new(self.to_inner(py))
    }

    /// Build a `PyInstructionSet` cell from a Rust `InstructionSet` snapshot.
    pub fn from_inner(py: Python<'_>, inner: qodec::InstructionSet) -> PyResult<Self> {
        Ok(Self {
            name: inner.name,
            description: inner.description,
            blocks: inner.blocks,
            instructions: inner
                .instructions
                .into_iter()
                .map(|inner| Py::new(py, PyInstruction { inner }))
                .collect::<PyResult<_>>()?,
            metadata: inner.metadata,
        })
    }
}

#[pymethods]
impl PyInstructionSet {
    #[new]
    #[pyo3(signature = (
        name,
        *,
        description = String::new(),
        blocks = Vec::new(),
        instructions = None,
        metadata = None,
    ))]
    fn new(
        py: Python<'_>,
        name: String,
        description: String,
        blocks: Vec<PyRef<'_, PyBlock>>,
        instructions: Option<Bound<'_, PyAny>>,
        metadata: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let instructions = match instructions {
            None => Vec::new(),
            Some(value) => extract_instructions(&value)?,
        };
        let built = Self {
            name,
            description,
            blocks: blocks
                .into_iter()
                .map(|block| qodec::Block {
                    name: block.name.clone(),
                    encodes: block.encodes,
                })
                .collect(),
            instructions,
            metadata: crate::metadata_from_py(metadata.as_ref())?,
        };
        built
            .to_inner(py)
            .validate()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(built)
    }

    /// Load and validate a standalone instruction-set YAML file.
    ///
    /// Raises ``QodecLoadError`` if reading, parsing, or validation fails.
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = qodec::InstructionSet::load(&path)
            .map_err(|error| crate::QodecLoadError::new_err(format!("{}: {error}", path.display())))?;
        Self::from_inner(py, inner)
    }

    /// Validate and write a standalone instruction-set YAML file, creating parent directories.
    ///
    /// Raises ``QodecSaveError`` if validation, serialization, or writing fails.
    fn save(&self, py: Python<'_>, path: PathBuf) -> PyResult<()> {
        self.to_inner(py)
            .save(&path)
            .map_err(|error| crate::QodecSaveError::new_err(format!("{}: {error}", path.display())))
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[setter]
    fn set_name(&mut self, value: String) {
        self.name = value;
    }

    #[getter]
    fn description(&self) -> &str {
        &self.description
    }

    #[setter]
    fn set_description(&mut self, value: String) {
        self.description = value;
    }

    #[getter]
    fn blocks<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "blocks", false)
    }

    fn _get_blocks(&self) -> Vec<PyBlock> {
        self.blocks
            .iter()
            .map(|block| PyBlock {
                name: block.name.clone(),
                encodes: block.encodes,
            })
            .collect()
    }

    #[setter]
    fn set_blocks(&mut self, value: Vec<PyRef<'_, PyBlock>>) {
        self.blocks = value
            .into_iter()
            .map(|block| qodec::Block {
                name: block.name.clone(),
                encodes: block.encodes,
            })
            .collect();
    }

    #[getter]
    fn instructions<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "instructions", true)
    }

    fn _get_instructions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let result = pyo3::types::PyDict::new(py);
        for instruction in &self.instructions {
            result.set_item(&instruction.borrow(py).inner.mnemonic, instruction)?;
        }
        Ok(result)
    }

    #[setter]
    fn set_instructions(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let instructions = extract_instructions(value)?;
        slf.borrow_mut().instructions = instructions;
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

    fn _copy_shell(&self, py: Python<'_>) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            blocks: self.blocks.clone(),
            metadata: self.metadata.clone(),
            instructions: self.instructions.iter().map(|value| value.clone_ref(py)).collect(),
        }
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyInstructionSet>>()
            .is_ok_and(|other| self.to_inner(py) == other.to_inner(py))
    }

    fn __repr__(&self) -> String {
        format!("InstructionSet({:?})", self.name)
    }

    fn __str__(&self, py: Python<'_>) -> String {
        crate::display::yaml(&self.to_inner(py))
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

#[pyclass(name = "Block", module = "qodec.instructions")]
pub struct PyBlock {
    pub(crate) name: String,
    pub(crate) encodes: usize,
}

#[pymethods]
impl PyBlock {
    #[new]
    fn new(name: String, encodes: usize) -> Self {
        Self { name, encodes }
    }

    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    #[getter]
    fn encodes(&self) -> usize {
        self.encodes
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyBlock>>()
            .is_ok_and(|other| self.name == other.name && self.encodes == other.encodes)
    }

    fn __repr__(&self) -> String {
        format!("Block({:?}, encodes={})", self.name, self.encodes)
    }

    fn __str__(&self) -> String {
        crate::display::yaml(&qodec::Block {
            name: self.name.clone(),
            encodes: self.encodes,
        })
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

#[pyclass(name = "Instruction", module = "qodec")]
pub struct PyInstruction {
    pub(crate) inner: qodec::Instruction,
}

#[pymethods]
impl PyInstruction {
    #[new]
    #[pyo3(signature = (
        mnemonic,
        *,
        description = String::new(),
        inputs = Vec::new(),
        outputs = Vec::new(),
        flags = Vec::new(),
        parameters = Vec::new(),
        action = Vec::new(),
        metadata = None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        mnemonic: String,
        description: String,
        inputs: Vec<PyRef<'_, PyBlockOperand>>,
        outputs: Vec<PyRef<'_, PyBlockOperand>>,
        flags: Vec<String>,
        parameters: Vec<PyRef<'_, PyParameter>>,
        action: Vec<Bound<'_, PyAny>>,
        metadata: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let action_steps = action.iter().map(py_to_action_step).collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: qodec::Instruction {
                mnemonic,
                description,
                inputs: inputs.iter().map(|operand| operand.inner.clone()).collect(),
                outputs: outputs.iter().map(|operand| operand.inner.clone()).collect(),
                flags,
                parameters: parameters.iter().map(|parameter| parameter.inner.clone()).collect(),
                action: action_steps,
                metadata: crate::metadata_from_py(metadata.as_ref())?,
            },
        })
    }

    #[getter]
    fn mnemonic(&self) -> &str {
        &self.inner.mnemonic
    }

    #[getter]
    fn description(&self) -> &str {
        &self.inner.description
    }

    #[setter]
    fn set_description(&mut self, value: String) {
        self.inner.description = value;
    }

    #[getter]
    fn inputs<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "inputs", false)
    }

    fn _get_inputs(&self) -> Vec<PyBlockOperand> {
        self.inner
            .inputs
            .iter()
            .map(|op| PyBlockOperand { inner: op.clone() })
            .collect()
    }

    #[getter]
    fn outputs<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "outputs", false)
    }

    fn _get_outputs(&self) -> Vec<PyBlockOperand> {
        self.inner
            .outputs
            .iter()
            .map(|op| PyBlockOperand { inner: op.clone() })
            .collect()
    }

    #[getter]
    fn flags<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "flags", false)
    }

    fn _get_flags(&self) -> Vec<String> {
        self.inner.flags.clone()
    }

    #[setter]
    fn set_inputs(&mut self, values: Vec<PyRef<'_, PyBlockOperand>>) {
        self.inner.inputs = values.iter().map(|value| value.inner.clone()).collect();
    }

    #[setter]
    fn set_outputs(&mut self, values: Vec<PyRef<'_, PyBlockOperand>>) {
        self.inner.outputs = values.iter().map(|value| value.inner.clone()).collect();
    }

    #[setter]
    fn set_flags(&mut self, values: Vec<String>) -> PyResult<()> {
        let unique: std::collections::BTreeSet<_> = values.iter().collect();
        if unique.len() != values.len() {
            return Err(PyValueError::new_err("duplicate instruction flag"));
        }
        self.inner.flags = values;
        Ok(())
    }

    /// How many outcome bits this instruction's ``observe:`` actions produce.
    #[getter]
    fn observe_count(&self) -> usize {
        self.inner.observe_count()
    }

    #[getter]
    fn parameters<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "parameters", false)
    }

    fn _get_parameters(&self) -> Vec<PyParameter> {
        self.inner
            .parameters
            .iter()
            .map(|parameter| PyParameter {
                inner: parameter.clone(),
            })
            .collect()
    }

    #[getter]
    fn action<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.borrow().get_action(slf.py())?;
        crate::collections::view(slf.as_any(), "action", false)
    }

    #[pyo3(name = "_get_action")]
    fn get_action(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner
            .action
            .iter()
            .map(|step| action_step_to_py(py, step))
            .collect()
    }

    #[getter]
    fn metadata<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "metadata", true)
    }

    fn _get_metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        crate::metadata_to_py(py, &self.inner.metadata)
    }

    #[setter]
    fn set_parameters(&mut self, values: Vec<PyRef<'_, PyParameter>>) -> PyResult<()> {
        let parameters: Vec<_> = values.iter().map(|value| value.inner.clone()).collect();
        let unique: std::collections::BTreeSet<_> = parameters.iter().map(|value| &value.name).collect();
        if unique.len() != parameters.len() {
            return Err(PyValueError::new_err("duplicate instruction parameter"));
        }
        self.inner.parameters = parameters;
        Ok(())
    }

    #[setter]
    fn set_action(&mut self, values: Vec<Bound<'_, PyAny>>) -> PyResult<()> {
        self.inner.action = values.iter().map(py_to_action_step).collect::<PyResult<_>>()?;
        Ok(())
    }

    #[setter]
    fn set_metadata(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let metadata = crate::metadata_from_py(Some(value))?;
        slf.borrow_mut().inner.metadata = metadata;
        Ok(())
    }

    fn _copy_shell(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyInstruction>>()
            .is_ok_and(|other| self.inner == other.inner)
    }

    fn __repr__(&self) -> String {
        format!("Instruction({:?})", self.inner.mnemonic)
    }

    fn __str__(&self) -> String {
        crate::display::yaml(&self.inner)
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

#[pyclass(name = "BlockOperand", module = "qodec.instructions")]
pub struct PyBlockOperand {
    pub(crate) inner: qodec::BlockOperand,
}

#[pymethods]
impl PyBlockOperand {
    #[new]
    #[pyo3(signature = (block, *, is_variadic = false))]
    fn new(block: String, is_variadic: bool) -> Self {
        let (block, is_variadic) = match block.strip_suffix("...") {
            Some(stripped) => (stripped.to_owned(), true),
            None => (block, is_variadic),
        };
        Self {
            inner: qodec::BlockOperand { block, is_variadic },
        }
    }

    #[getter]
    fn block(&self) -> &str {
        &self.inner.block
    }

    #[getter]
    fn is_variadic(&self) -> bool {
        self.inner.is_variadic
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyBlockOperand>>()
            .is_ok_and(|other| self.inner == other.inner)
    }

    fn __repr__(&self) -> String {
        format!(
            "BlockOperand({:?}, is_variadic={})",
            self.inner.block,
            if self.inner.is_variadic { "True" } else { "False" },
        )
    }

    fn __str__(&self) -> String {
        crate::display::yaml(&self.inner)
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

pub(crate) fn validate_encoding_arity(
    implements: &PyInstruction,
    inputs: &[Py<PyEncoding>],
    outputs: &[Py<PyEncoding>],
) -> PyResult<()> {
    let input_count = implements.inner.inputs.len();
    let output_count = implements.inner.outputs.len();

    if inputs.len() != input_count {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "implemented instruction {:?} declares {} input operand(s) but the gadget \
             provides {} input encoding(s); encodings align positionally",
            implements.inner.mnemonic,
            input_count,
            inputs.len(),
        )));
    }
    if outputs.len() != output_count {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "implemented instruction {:?} declares {} output operand(s) but the gadget \
             provides {} output encoding(s); encodings align positionally",
            implements.inner.mnemonic,
            output_count,
            outputs.len(),
        )));
    }
    Ok(())
}

#[pyclass(name = "Parameter", module = "qodec.instructions")]
pub struct PyParameter {
    pub(crate) inner: qodec::Parameter,
}

impl PyParameter {
    /// The parameter's declared type as its lowercase property-path token
    /// (`bit`, `number`, …). Backs both `__repr__` and the wrapped
    /// `Parameter.Kind` returned by the `kind` getter.
    fn kind_token(&self) -> &'static str {
        match self.inner.kind {
            qodec::ParameterKind::Bit => "bit",
            qodec::ParameterKind::Number => "number",
            qodec::ParameterKind::Integer => "integer",
            qodec::ParameterKind::Boolean => "boolean",
            qodec::ParameterKind::String => "string",
            qodec::ParameterKind::Pauli => "pauli",
        }
    }
}

#[pymethods]
impl PyParameter {
    #[new]
    fn new(name: String, kind: &Bound<'_, PyAny>) -> PyResult<Self> {
        // Accept either the lowercase token string or a `Parameter.Kind`
        // enum member (whose `.value` is that token).
        let kind_str = match kind.extract::<String>() {
            Ok(token) => token,
            Err(_) => kind
                .getattr("value")?
                .extract::<String>()
                .map_err(|_| PyValueError::new_err("kind must be a str or a qodec.instructions.Parameter.Kind"))?,
        };
        let kind = match kind_str.as_str() {
            "bit" => qodec::ParameterKind::Bit,
            "number" => qodec::ParameterKind::Number,
            "integer" => qodec::ParameterKind::Integer,
            "boolean" => qodec::ParameterKind::Boolean,
            "string" => qodec::ParameterKind::String,
            "pauli" => qodec::ParameterKind::Pauli,
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown kind {other:?}; expected one of: bit, number, integer, boolean, string, pauli"
                )));
            }
        };
        Ok(Self {
            inner: qodec::Parameter { name, kind },
        })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }

    /// The parameter's declared type as a `qodec.instructions.Parameter.Kind`.
    #[getter]
    pub(crate) fn kind<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // Resolved once: the enum lives in a Python module that imports this extension.
        static KIND: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
        KIND.get_or_try_init(py, || {
            Ok::<_, PyErr>(
                py.import("qodec.instructions")?
                    .getattr("Parameter")?
                    .getattr("Kind")?
                    .unbind(),
            )
        })?
        .bind(py)
        .call1((self.kind_token(),))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyParameter>>()
            .is_ok_and(|other| self.inner == other.inner)
    }

    fn __repr__(&self) -> String {
        format!("Parameter({:?}, {:?})", self.inner.name, self.kind_token())
    }

    fn __str__(&self) -> String {
        crate::display::yaml(&self.inner)
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

/// An action guard over bit parameters or ``outcomes[i]`` within the instruction.
///
/// ``invert=False`` runs on XOR parity 1; ``invert=True`` runs on parity 0.
/// Construction stores strings without resolving names. instruction set validation rejects
/// flag names and ``readouts[i]`` as action guards.
#[pyclass(name = "Condition", module = "qodec.actions")]
#[derive(PartialEq)]
pub struct PyCondition {
    pub(crate) inner: qodec::Condition,
}

#[pymethods]
impl PyCondition {
    #[new]
    #[pyo3(signature = (predicates, *, invert = false))]
    fn new(predicates: Vec<String>, invert: bool) -> Self {
        Self {
            inner: qodec::Condition { predicates, invert },
        }
    }

    #[getter]
    fn predicates<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.predicates)
    }

    #[getter]
    fn invert(&self) -> bool {
        self.inner.invert
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyCondition>>()
            .is_ok_and(|other| *self == *other)
    }

    fn __repr__(&self) -> String {
        format!("Condition({:?})", self.inner.predicates)
    }

    fn __str__(&self) -> String {
        crate::display::yaml(&self.inner)
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

// ── Actions ─────────────────────────────────────────────────────────────────

#[pyclass(name = "Stabilize", module = "qodec.actions")]
#[derive(PartialEq)]
pub struct PyStabilize {
    operators: Vec<String>,
    condition: Option<qodec::Condition>,
}

#[pymethods]
impl PyStabilize {
    #[new]
    #[pyo3(signature = (operators, *, condition = None))]
    fn new(operators: Vec<Bound<'_, PyAny>>, condition: Option<PyRef<'_, PyCondition>>) -> PyResult<Self> {
        let operators = operators.iter().map(crate::pauli_text).collect::<PyResult<_>>()?;
        Ok(Self {
            operators,
            condition: condition.map(|condition| condition.inner.clone()),
        })
    }

    #[getter]
    fn operators<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.operators)
    }

    #[getter]
    fn condition(&self) -> Option<PyCondition> {
        condition_to_py(self.condition.as_ref())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyStabilize>>()
            .is_ok_and(|other| *self == *other)
    }

    fn __repr__(&self) -> String {
        format!("Stabilize({:?})", self.operators)
    }

    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(crate::display::yaml(&py_to_action_step(slf.as_any())?))
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

#[pyclass(name = "Clifford", module = "qodec.actions")]
#[derive(PartialEq)]
pub struct PyClifford {
    generators: std::collections::BTreeMap<String, String>,
    condition: Option<qodec::Condition>,
}

#[pymethods]
impl PyClifford {
    #[new]
    #[pyo3(signature = (generators, *, condition = None))]
    fn new(generators: Bound<'_, PyAny>, condition: Option<PyRef<'_, PyCondition>>) -> PyResult<Self> {
        if !generators.hasattr("keys")? {
            return Err(PyValueError::new_err("Clifford: generators must be a dict or mapping"));
        }
        let generators = crate::collections::mapping(&generators)?;
        let dict = generators.cast::<pyo3::types::PyDict>()?;
        let mut out = std::collections::BTreeMap::new();
        for (lhs_obj, rhs_obj) in dict.iter() {
            let lhs = crate::pauli_text(&lhs_obj)
                .map_err(|_| PyValueError::new_err("Clifford: dict keys must be Pauli generator strings"))?;
            let rhs = crate::pauli_text(&rhs_obj)
                .map_err(|_| PyValueError::new_err("Clifford: dict values must be Pauli image strings"))?;
            out.insert(lhs, rhs);
        }
        Ok(Self {
            generators: out,
            condition: condition.map(|condition| condition.inner.clone()),
        })
    }

    #[getter]
    fn generators<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("types")?
            .getattr("MappingProxyType")?
            .call1((&self.generators,))
    }

    #[getter]
    fn condition(&self) -> Option<PyCondition> {
        condition_to_py(self.condition.as_ref())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyClifford>>()
            .is_ok_and(|other| *self == *other)
    }

    fn __repr__(&self) -> String {
        format!("Clifford({:?})", self.generators)
    }

    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(crate::display::yaml(&py_to_action_step(slf.as_any())?))
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

#[pyclass(name = "Pauli", module = "qodec.actions")]
#[derive(PartialEq)]
pub struct PyPauli {
    operator: String,
    condition: Option<qodec::Condition>,
}

#[pymethods]
impl PyPauli {
    #[new]
    #[pyo3(signature = (operator, *, condition = None))]
    fn new(operator: &Bound<'_, PyAny>, condition: Option<PyRef<'_, PyCondition>>) -> PyResult<Self> {
        Ok(Self {
            operator: crate::pauli_text(operator)?,
            condition: condition.map(|condition| condition.inner.clone()),
        })
    }

    #[getter]
    fn operator(&self) -> String {
        self.operator.clone()
    }

    #[getter]
    fn condition(&self) -> Option<PyCondition> {
        condition_to_py(self.condition.as_ref())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.extract::<PyRef<'_, PyPauli>>().is_ok_and(|other| *self == *other)
    }

    fn __repr__(&self) -> String {
        format!("Pauli({:?})", self.operator)
    }

    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(crate::display::yaml(&py_to_action_step(slf.as_any())?))
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

/// Measure Pauli observables, producing one outcome per entry.
///
/// Outcomes are numbered across the instruction's action list and referenced
/// in action guards as ``outcomes[i]``. This action takes no ``condition``.
#[pyclass(name = "Observe", module = "qodec.actions")]
#[derive(PartialEq)]
pub struct PyObserve {
    observables: Vec<String>,
}

#[pymethods]
impl PyObserve {
    #[new]
    #[pyo3(signature = (observables))]
    fn new(observables: Vec<Bound<'_, PyAny>>) -> PyResult<Self> {
        let observables = observables.iter().map(crate::pauli_text).collect::<PyResult<_>>()?;
        Ok(Self { observables })
    }

    #[getter]
    fn observables<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.observables)
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyObserve>>()
            .is_ok_and(|other| *self == *other)
    }

    fn __repr__(&self) -> String {
        format!("Observe({:?})", self.observables)
    }

    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(crate::display::yaml(&py_to_action_step(slf.as_any())?))
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

#[pyclass(name = "Rotate", module = "qodec.actions")]
#[derive(PartialEq)]
pub struct PyRotate {
    pauli: String,
    angle: qodec::Scalar,
    condition: Option<qodec::Condition>,
}

#[pymethods]
impl PyRotate {
    #[new]
    #[pyo3(signature = (pauli, angle, *, condition = None))]
    fn new(
        pauli: &Bound<'_, PyAny>,
        angle: &Bound<'_, PyAny>,
        condition: Option<PyRef<'_, PyCondition>>,
    ) -> PyResult<Self> {
        let pauli = crate::pauli_text(pauli)?;
        let angle = if let Ok(name) = angle.extract::<String>() {
            qodec::Scalar::Parameter(name)
        } else {
            qodec::Scalar::Literal(angle.extract::<f64>()?)
        };
        Ok(Self {
            pauli,
            angle,
            condition: condition.map(|condition| condition.inner.clone()),
        })
    }

    #[getter]
    fn pauli(&self) -> &str {
        &self.pauli
    }

    #[getter]
    fn angle(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match &self.angle {
            qodec::Scalar::Literal(value) => value.into_py_any(py),
            qodec::Scalar::Parameter(name) => name.into_py_any(py),
        }
    }

    #[getter]
    fn condition(&self) -> Option<PyCondition> {
        condition_to_py(self.condition.as_ref())
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyRotate>>()
            .is_ok_and(|other| *self == *other)
    }

    fn __repr__(&self) -> String {
        match &self.angle {
            qodec::Scalar::Literal(value) => format!("Rotate({:?}, angle={})", self.pauli, value),
            qodec::Scalar::Parameter(name) => format!("Rotate({:?}, angle={:?})", self.pauli, name),
        }
    }

    fn __str__(slf: &Bound<'_, Self>) -> PyResult<String> {
        Ok(crate::display::yaml(&py_to_action_step(slf.as_any())?))
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

fn condition_to_py(condition: Option<&qodec::Condition>) -> Option<PyCondition> {
    condition.map(|condition| PyCondition {
        inner: condition.clone(),
    })
}

pub(crate) fn action_step_to_py(py: Python<'_>, step: &qodec::ActionStep) -> PyResult<Py<PyAny>> {
    let condition = step.condition.clone();
    match &step.action {
        qodec::Action::Stabilize(operators) => {
            let obj = PyStabilize {
                operators: operators.iter().map(|pauli| pauli.0.clone()).collect(),
                condition,
            };
            Ok(obj.into_pyobject(py)?.into_any().unbind())
        }
        qodec::Action::Clifford(generators) => {
            let obj = PyClifford {
                generators: generators
                    .iter()
                    .map(|(from, to)| (from.0.clone(), to.0.clone()))
                    .collect(),
                condition,
            };
            Ok(obj.into_pyobject(py)?.into_any().unbind())
        }
        qodec::Action::Pauli(pauli) => {
            let obj = PyPauli {
                operator: pauli.0.clone(),
                condition,
            };
            Ok(obj.into_pyobject(py)?.into_any().unbind())
        }
        qodec::Action::Observe(observables) => {
            if condition.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "conditional observe cannot be represented by the Python Observe action",
                ));
            }
            let obj = PyObserve {
                observables: observables
                    .iter()
                    .map(|observable| observable.pauli.0.clone())
                    .collect(),
            };
            Ok(obj.into_pyobject(py)?.into_any().unbind())
        }
        qodec::Action::Rotate { pauli, angle } => {
            let obj = PyRotate {
                pauli: pauli.0.clone(),
                angle: angle.clone(),
                condition,
            };
            Ok(obj.into_pyobject(py)?.into_any().unbind())
        }
    }
}

fn py_to_action_step(obj: &Bound<'_, PyAny>) -> PyResult<qodec::ActionStep> {
    use qodec::PauliString;
    use qodec::{Action, ActionStep};

    if let Ok(stab) = obj.extract::<PyRef<'_, PyStabilize>>() {
        let action = Action::Stabilize(
            stab.operators
                .iter()
                .map(|operator| PauliString(operator.clone()))
                .collect(),
        );
        return Ok(ActionStep {
            action,
            condition: stab.condition.clone(),
        });
    }
    if let Ok(cliff) = obj.extract::<PyRef<'_, PyClifford>>() {
        let action = Action::Clifford(
            cliff
                .generators
                .iter()
                .map(|(from, to)| {
                    (
                        qodec::PauliString::from(from.clone()),
                        qodec::PauliString::from(to.clone()),
                    )
                })
                .collect(),
        );
        return Ok(ActionStep {
            action,
            condition: cliff.condition.clone(),
        });
    }
    if let Ok(pauli) = obj.extract::<PyRef<'_, PyPauli>>() {
        let action = Action::Pauli(PauliString(pauli.operator.clone()));
        return Ok(ActionStep {
            action,
            condition: pauli.condition.clone(),
        });
    }
    if let Ok(observe) = obj.extract::<PyRef<'_, PyObserve>>() {
        let observables = observe
            .observables
            .iter()
            .map(|observable| qodec::Observable {
                pauli: PauliString(observable.clone()),
            })
            .collect();
        return Ok(ActionStep {
            action: Action::Observe(observables),
            condition: None,
        });
    }
    if let Ok(rotate) = obj.extract::<PyRef<'_, PyRotate>>() {
        let action = Action::Rotate {
            pauli: PauliString(rotate.pauli.clone()),
            angle: rotate.angle.clone(),
        };
        return Ok(ActionStep {
            action,
            condition: rotate.condition.clone(),
        });
    }
    Err(PyValueError::new_err(format!(
        "action step must be a Stabilize, Clifford, Pauli, Observe, or Rotate; got {}",
        obj.get_type()
            .name()
            .map_or_else(|_| "<unknown>".to_owned(), |name| name.to_string())
    )))
}

// ── Circuit IR ──────────────────────────────────────────────────────────────

fn py_literal_kind(value: &Bound<'_, PyAny>) -> Option<qodec::ParameterKind> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyString};
    use qodec::ParameterKind;

    if value.is_instance_of::<PyBool>() {
        Some(ParameterKind::Boolean)
    } else if value.is_instance_of::<PyInt>() {
        Some(ParameterKind::Integer)
    } else if value.is_instance_of::<PyFloat>() {
        Some(ParameterKind::Number)
    } else if value.is_instance_of::<PyString>() {
        Some(ParameterKind::String)
    } else {
        None
    }
}

fn py_value_eq(left: &Bound<'_, PyAny>, right: &Bound<'_, PyAny>) -> PyResult<bool> {
    let mut pending = vec![(left.clone(), right.clone())];
    let mut compared_lists = std::collections::BTreeSet::new();
    while let Some((left, right)) = pending.pop() {
        if let (Some(left_kind), Some(right_kind)) = (py_literal_kind(&left), py_literal_kind(&right))
            && left_kind != right_kind
        {
            return Ok(false);
        }
        if let (Ok(left), Ok(right)) = (left.cast::<pyo3::types::PyList>(), right.cast::<pyo3::types::PyList>()) {
            if left.len() != right.len() {
                return Ok(false);
            }
            if compared_lists.insert((left.as_ptr(), right.as_ptr())) {
                pending.extend(left.iter().zip(right.iter()));
            }
        } else if !left.eq(&right)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn py_value_map_eq(
    left: &BTreeMap<String, Py<PyAny>>,
    right: &BTreeMap<String, Py<PyAny>>,
    py: Python<'_>,
) -> PyResult<bool> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (key, value) in left {
        let Some(other) = right.get(key) else {
            return Ok(false);
        };
        if !py_value_eq(value.bind(py), other.bind(py))? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn py_value_list_eq(left: &[Py<PyAny>], right: &[Py<PyAny>], py: Python<'_>) -> PyResult<bool> {
    if left.len() != right.len() {
        return Ok(false);
    }
    for (value, other) in left.iter().zip(right) {
        if !py_value_eq(value.bind(py), other.bind(py))? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[pyclass(name = "InstructionCall", module = "qodec.instructions")]
pub struct PyInstructionCall {
    mnemonic: String,
    operands: Vec<Py<PyAny>>,
    arguments: BTreeMap<String, Py<PyAny>>,
    select: Vec<qodec::SelectPattern>,
}

impl PyInstructionCall {
    pub(crate) fn to_call(&self, py: Python<'_>) -> PyResult<qodec::InstructionCall> {
        let mut call = self.record_call(py)?;
        for (name, value) in &self.arguments {
            call.arguments.insert(name.clone(), argument_from_py(value.bind(py))?);
        }
        Ok(call)
    }

    fn record_call(&self, py: Python<'_>) -> PyResult<qodec::InstructionCall> {
        let operands = self
            .operands
            .iter()
            .map(|value| {
                if value.bind(py).is_instance_of::<pyo3::types::PyBool>() {
                    return Err(PyValueError::new_err(
                        "operands must be non-negative integers or strings",
                    ));
                }
                if let Ok(index) = value.extract::<usize>(py) {
                    Ok(qodec::Operand::Index(index))
                } else {
                    value.extract::<String>(py).map(qodec::Operand::Name)
                }
            })
            .collect::<PyResult<_>>()?;
        Ok(qodec::InstructionCall {
            mnemonic: self.mnemonic.clone(),
            operands,
            arguments: BTreeMap::new(),
            select: self.select.clone(),
        })
    }
}

fn argument_from_py(value: &Bound<'_, PyAny>) -> PyResult<qodec::Argument> {
    use pyo3::types::{PyBool, PyFloat, PyInt, PyList, PyString};
    use qodec::Argument;

    if value.is_instance_of::<PyBool>() {
        return value.extract().map(Argument::Boolean);
    }
    if value.is_instance_of::<PyInt>() {
        return value.extract::<i64>().map(Argument::Integer);
    }
    if value.is_instance_of::<PyFloat>() {
        return value.extract().map(Argument::Number);
    }
    if value.is_instance_of::<PyString>() {
        let text = value.extract::<String>()?;
        return Argument::parse_text(&text).map_err(PyValueError::new_err);
    }
    if let Ok(values) = value.cast::<PyList>() {
        if values
            .iter()
            .all(|item| item.is_instance_of::<PyInt>() && !item.is_instance_of::<PyBool>())
        {
            return values.extract().map(Argument::QubitList);
        }
        if values.iter().all(|item| item.is_instance_of::<PyString>()) {
            return values.extract().map(Argument::StringList);
        }
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "unsupported call argument value shape",
    ))
}

pub(crate) fn instruction_call_to_py(call: &qodec::InstructionCall) -> PyInstructionCall {
    Python::attach(|py| {
        let operands = call.operands.iter().map(|op| operand_to_py(py, op)).collect();
        let arguments = call
            .arguments
            .iter()
            .map(|(name, argument)| (name.clone(), argument_to_py(py, argument)))
            .collect();
        PyInstructionCall {
            mnemonic: call.mnemonic.clone(),
            operands,
            arguments,
            select: call.select.clone(),
        }
    })
}

/// Convert a bound block into the most natural Python value: an index stays
/// an `int`, a block name stays a `str`.
fn operand_to_py(py: Python<'_>, operand: &qodec::Operand) -> Py<PyAny> {
    match operand {
        qodec::Operand::Index(index) => (*index).into_py_any(py).expect("usize into Python should not fail"),
        qodec::Operand::Name(name) => name
            .clone()
            .into_py_any(py)
            .expect("String into Python should not fail"),
    }
}

/// Convert a Rust [`qodec::Argument`] into the most natural Python value.
///
/// `Qubit(usize)` and `Integer(i64)` both become Python `int`; `QubitList`
/// becomes `list[int]`; `Number(f64)` becomes `float`; `Boolean(bool)`
/// becomes `bool`; `Text(String)`
/// becomes `str`; `StringList(Vec<String>)` becomes `list[str]`.
/// Callback scalar integers become `Integer`; numeric block operands become
/// `Operand::Index` independently of arguments.
fn argument_to_py(py: Python<'_>, argument: &qodec::Argument) -> Py<PyAny> {
    match argument {
        qodec::Argument::Qubit(q) => (*q).into_py_any(py).expect("usize into Python should not fail"),
        qodec::Argument::QubitList(qs) => qs
            .clone()
            .into_py_any(py)
            .expect("Vec<usize> into Python should not fail"),
        qodec::Argument::Integer(i) => (*i).into_py_any(py).expect("i64 into Python should not fail"),
        qodec::Argument::Number(n) => (*n).into_py_any(py).expect("f64 into Python should not fail"),
        qodec::Argument::Boolean(value) => (*value).into_py_any(py).expect("bool into Python should not fail"),
        qodec::Argument::Text(t) => t.clone().into_py_any(py).expect("String into Python should not fail"),
        qodec::Argument::StringList(ss) => ss
            .clone()
            .into_py_any(py)
            .expect("Vec<String> into Python should not fail"),
        qodec::Argument::Readout(i) => format!("circuit.readouts[{i}]")
            .into_py_any(py)
            .expect("String into Python should not fail"),
    }
}

fn select_from_py(patterns: Vec<BTreeMap<String, Bound<'_, PyAny>>>) -> PyResult<Vec<qodec::SelectPattern>> {
    patterns
        .into_iter()
        .map(|pattern| {
            pattern
                .into_iter()
                .map(|(flag, value)| {
                    if value.is_instance_of::<pyo3::types::PyBool>() {
                        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                            "select: atom {flag:?} bit must be an integer 0 or 1, not bool"
                        )));
                    }
                    let bit = value.extract::<u8>()?;
                    if bit > 1 {
                        return Err(PyValueError::new_err(format!(
                            "select: atom {flag:?} bit must be 0 or 1, got {bit}"
                        )));
                    }
                    Ok((flag, bit))
                })
                .collect()
        })
        .collect()
}

#[pymethods]
impl PyInstructionCall {
    #[new]
    #[pyo3(signature = (
        mnemonic,
        *,
        operands = None,
        arguments = None,
        select = None,
    ))]
    fn new(
        mnemonic: String,
        operands: Option<Vec<Py<PyAny>>>,
        arguments: Option<BTreeMap<String, Py<PyAny>>>,
        select: Option<Vec<BTreeMap<String, Bound<'_, PyAny>>>>,
    ) -> PyResult<Self> {
        let select = select_from_py(select.unwrap_or_default())?;
        Ok(Self {
            mnemonic,
            operands: operands.unwrap_or_default(),
            arguments: arguments.unwrap_or_default(),
            select,
        })
    }

    #[getter]
    fn mnemonic(&self) -> &str {
        &self.mnemonic
    }

    #[getter]
    fn operands<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "operands", false)
    }

    fn _get_operands(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.operands.iter().map(|value| value.clone_ref(py)).collect()
    }

    #[getter]
    fn arguments<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "arguments", true)
    }

    fn _get_arguments(&self, py: Python<'_>) -> BTreeMap<String, Py<PyAny>> {
        self.arguments
            .iter()
            .map(|(k, v)| (k.clone(), v.clone_ref(py)))
            .collect()
    }

    #[getter]
    fn select<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "select", false)
    }

    fn _get_select(&self) -> Vec<BTreeMap<String, u8>> {
        self.select.clone()
    }

    #[setter]
    fn set_operands(&mut self, values: Vec<Py<PyAny>>) {
        self.operands = values;
    }

    #[setter]
    fn set_arguments(slf: &Bound<'_, Self>, values: &Bound<'_, PyAny>) -> PyResult<()> {
        let arguments = crate::collections::mapping(values)?.extract()?;
        slf.borrow_mut().arguments = arguments;
        Ok(())
    }

    #[setter]
    fn set_select(&mut self, values: Vec<BTreeMap<String, Bound<'_, PyAny>>>) -> PyResult<()> {
        self.select = select_from_py(values)?;
        Ok(())
    }

    fn _copy_shell(&self, py: Python<'_>) -> Self {
        Self {
            mnemonic: self.mnemonic.clone(),
            select: self.select.clone(),
            operands: self.operands.iter().map(|value| value.clone_ref(py)).collect(),
            arguments: self
                .arguments
                .iter()
                .map(|(key, value)| (key.clone(), value.clone_ref(py)))
                .collect(),
        }
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<bool> {
        let Ok(other) = other.extract::<PyRef<'_, PyInstructionCall>>() else {
            return Ok(false);
        };
        Ok(self.mnemonic == other.mnemonic
            && self.select == other.select
            && py_value_list_eq(&self.operands, &other.operands, py)?
            && py_value_map_eq(&self.arguments, &other.arguments, py)?)
    }

    fn __repr__(&self) -> String {
        Python::attach(|py| {
            let repr = |value: &Py<PyAny>| {
                value
                    .bind(py)
                    .repr()
                    .map_or_else(|_| "?".to_owned(), |rendered| rendered.to_string())
            };
            let mut parts: Vec<String> = self.operands.iter().map(&repr).collect();
            parts.extend(
                self.arguments
                    .iter()
                    .map(|(name, value)| format!("{name}={}", repr(value))),
            );
            if parts.is_empty() {
                format!("InstructionCall({:?})", self.mnemonic)
            } else {
                format!("InstructionCall({:?}, {})", self.mnemonic, parts.join(", "))
            }
        })
    }
}
