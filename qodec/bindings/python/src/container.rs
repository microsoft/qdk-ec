//! Python bindings for the top-level container types `PyQodec` and
//! `PyLayer`.

use std::collections::BTreeMap;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::codes::PyCode;
use crate::gadgets::PyGadget;
use crate::types::PyInstructionSet;

#[cfg(feature = "test-support")]
thread_local! {
    static NODE_SNAPSHOTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(feature = "test-support")]
#[pyfunction]
pub(crate) fn _test_node_snapshots() -> usize {
    NODE_SNAPSHOTS.with(|count| count.replace(0))
}

/// The loader checks `schema_version` for exact equality, so any other value
/// would save a qodec nothing can read back.
fn check_schema_version(value: Option<u32>) -> PyResult<Option<u32>> {
    match value {
        Some(version) if version != qodec::CURRENT_SCHEMA_VERSION => Err(PyValueError::new_err(format!(
            "schema_version must be {} or None (got {version}); the loader accepts no other value",
            qodec::CURRENT_SCHEMA_VERSION
        ))),
        other => Ok(other),
    }
}

// ── Package ─────────────────────────────────────────────────────────────────

#[pyclass(name = "Qodec", module = "qodec")]
pub struct PyQodec {
    name: Option<String>,
    description: Option<String>,
    /// The declared schema version; omission is preserved on save.
    schema_version: Option<u32>,
    pub(crate) layers: Vec<Py<PyLayer>>,
    /// The filename the manifest was loaded from, so save round-trips it.
    /// `"qodec.yaml"` for qodecs built through the Python constructor.
    manifest_filename: String,
    /// Free-form, qodec-opaque annotations on the manifest (see
    /// [`qodec::Metadata`]).
    pub(crate) metadata: qodec::Metadata,
    loaded: Option<qodec::Qodec>,
}

impl PyQodec {
    fn current_layers(&self, py: Python<'_>) -> Vec<qodec::Layer> {
        self.layers
            .iter()
            .map(|layer| layer.borrow(py).to_resolved(py))
            .collect()
    }

    fn current_model(&self, py: Python<'_>) -> qodec::Qodec {
        #[cfg(feature = "test-support")]
        NODE_SNAPSHOTS.with(|count| count.set(count.get() + 1));
        let mut model = qodec::Qodec::new(self.name.clone(), self.description.clone(), self.current_layers(py));
        self.copy_manifest_fields(&mut model);
        model
    }

    fn to_qodec(&self, py: Python<'_>) -> qodec::Qodec {
        let mut qodec = self
            .loaded
            .clone()
            .unwrap_or_else(|| qodec::Qodec::new(None, None, Vec::new()));
        *qodec.layers_mut() = self.current_layers(py);
        self.copy_manifest_fields(&mut qodec);
        qodec
    }

    fn copy_manifest_fields(&self, model: &mut qodec::Qodec) {
        model.set_name(self.name.clone());
        model.set_description(self.description.clone());
        model.set_schema_version(self.schema_version);
        model.set_manifest_filename(self.manifest_filename.clone());
        model.metadata_mut().clone_from(&self.metadata);
    }

    fn from_loaded(py: Python<'_>, inner: qodec::Qodec) -> PyResult<Self> {
        let definitions = PythonDefinitions::new(py, &inner)?;
        let layers = inner
            .layers()
            .iter()
            .map(|layer| definitions.layer(py, layer))
            .collect::<PyResult<_>>()?;
        Ok(Self {
            name: inner.name().map(str::to_owned),
            description: inner.description().map(str::to_owned),
            schema_version: inner.schema_version(),
            layers,
            manifest_filename: inner.manifest_filename().to_owned(),
            metadata: inner.metadata().clone(),
            loaded: Some(inner),
        })
    }
}

struct PythonDefinitions {
    instruction_sets: BTreeMap<String, Py<PyInstructionSet>>,
    codes: BTreeMap<String, Py<PyCode>>,
}

impl PythonDefinitions {
    fn new(py: Python<'_>, model: &qodec::Qodec) -> PyResult<Self> {
        let instruction_sets = model
            .instruction_sets()
            .iter()
            .map(|(name, instruction_set)| {
                let py_isa = Py::new(py, PyInstructionSet::from_inner(py, (**instruction_set).clone())?)?;
                Ok::<_, PyErr>((name.clone(), py_isa))
            })
            .collect::<PyResult<_>>()?;
        let codes = model
            .codes()
            .iter()
            .map(|(name, code)| {
                let py_code = Py::new(py, PyCode::from_inner((**code).clone()))?;
                Ok::<_, PyErr>((name.clone(), py_code))
            })
            .collect::<PyResult<_>>()?;
        Ok(Self {
            instruction_sets,
            codes,
        })
    }

    fn instruction_set(&self, py: Python<'_>, name: &str, role: &str) -> PyResult<Py<PyInstructionSet>> {
        self.instruction_sets
            .get(name)
            .map(|cell| cell.clone_ref(py))
            .ok_or_else(|| PyValueError::new_err(format!("internal: {role} '{name}' not found")))
    }

    fn layer(&self, py: Python<'_>, layer: &qodec::Layer) -> PyResult<Py<PyLayer>> {
        let instruction_set = self.instruction_set(py, &layer.instruction_set.name, "layer instruction set")?;
        let instructions: BTreeMap<_, _> = instruction_set
            .borrow(py)
            .instructions
            .iter()
            .map(|instruction| (instruction.borrow(py).inner.mnemonic.clone(), instruction.clone_ref(py)))
            .collect();
        let gadgets = layer
            .gadgets
            .iter()
            .map(|(mnemonic, gadget)| {
                let definition = instructions
                    .get(mnemonic)
                    .ok_or_else(|| PyValueError::new_err("loaded gadget has no instruction"))?;
                let target =
                    self.instruction_set(py, &gadget.circuit.instruction_set.name, "gadget instruction_set")?;
                let value = Py::new(
                    py,
                    PyGadget::from_resolved(gadget, definition.clone_ref(py), target, &self.codes, py)?,
                )?;
                Ok((mnemonic.clone(), value))
            })
            .collect::<PyResult<_>>()?;
        Py::new(
            py,
            PyLayer {
                instruction_set,
                codes: layer
                    .codes
                    .iter()
                    .map(|(block, code)| {
                        let code = self.codes.get(&code.name).ok_or_else(|| {
                            PyValueError::new_err(format!("internal: code '{}' not found", code.name))
                        })?;
                        Ok((block.clone(), code.clone_ref(py)))
                    })
                    .collect::<PyResult<_>>()?,
                gadgets,
            },
        )
    }
}

#[pymethods]
impl PyQodec {
    #[new]
    #[pyo3(signature = (
        layers,
        *,
        name = None,
        description = None,
        schema_version = None,
        metadata = None,
    ))]
    fn new(
        layers: Vec<Py<PyLayer>>,
        name: Option<String>,
        description: Option<String>,
        schema_version: Option<u32>,
        metadata: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        Ok(Self {
            name,
            description,
            schema_version: check_schema_version(schema_version)?,
            layers,
            manifest_filename: "qodec.yaml".to_owned(),
            metadata: crate::metadata_from_py(metadata.as_ref())?,
            loaded: None,
        })
    }

    fn _copy_shell(&self, py: Python<'_>) -> Self {
        Self {
            name: self.name.clone(),
            description: self.description.clone(),
            schema_version: self.schema_version,
            layers: self.layers.iter().map(|value| value.clone_ref(py)).collect(),
            manifest_filename: self.manifest_filename.clone(),
            metadata: self.metadata.clone(),
            loaded: self.loaded.clone(),
        }
    }

    fn _copy_history_to(&self, mut target: PyRefMut<'_, Self>) {
        target.loaded.clone_from(&self.loaded);
        target.manifest_filename.clone_from(&self.manifest_filename);
    }

    fn _replace_fields<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let fields = pyo3::types::PyDict::new(py);
        fields.set_item("layers", self.get_layers(py))?;
        fields.set_item("name", &self.name)?;
        fields.set_item("description", &self.description)?;
        fields.set_item("schema_version", self.schema_version)?;
        fields.set_item("metadata", crate::metadata_to_py(py, &self.metadata)?)?;
        Ok(fields)
    }

    /// Load from an explicit manifest or multi-document YAML bundle file path.
    ///
    /// The file may have any name. Directory paths are rejected, even when they
    /// contain a valid manifest. Raises ``QodecLoadError`` on I/O, parsing,
    /// resolution, or validation failure.
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = qodec::Qodec::load(path).map_err(|error| crate::QodecLoadError::new_err(error.to_string()))?;
        Self::from_loaded(py, inner)
    }

    /// Load a multi-document YAML bundle from a string.
    ///
    /// A self-contained bundle, as returned by ``dumps``, needs no filesystem
    /// access. External paths resolve relative to the current working directory.
    /// Raises ``QodecLoadError`` on parsing, resolution, or validation failure.
    #[staticmethod]
    fn loads(py: Python<'_>, text: &str) -> PyResult<Self> {
        let inner =
            qodec::Qodec::from_bundle_str(text).map_err(|error| crate::QodecLoadError::new_err(error.to_string()))?;
        Self::from_loaded(py, inner)
    }

    /// Check the current layers and components without reading or writing files.
    /// Raises ``ValueError`` on the first inconsistency.
    fn validate(&self, py: Python<'_>) -> PyResult<()> {
        self.current_model(py).validate().map_err(PyValueError::new_err)
    }

    /// Resolve an exact model path into a live node. Empty selects this qodec.
    fn resolve(slf: &Bound<'_, Self>, path: &str) -> PyResult<Py<PyAny>> {
        let node = slf.py().import("qodec._nodes")?.getattr("Node")?;
        Ok(node.call_method1("_create", (slf, path))?.unbind())
    }

    fn _node_source_location(&self, py: Python<'_>, path: &str) -> Option<(PathBuf, usize)> {
        let loaded = self.loaded.as_ref()?;
        let current = self.current_model(py);
        if *loaded != current {
            return None;
        }
        loaded
            .resolve(path)
            .ok()?
            .source_location()
            .map(|location| (location.path().to_owned(), location.line()))
    }

    /// Write the current objects to a destination directory, creating it if needed.
    ///
    /// ``single_file=False`` writes separate artifact files. ``single_file=True``
    /// writes a multi-document YAML bundle to ``destination/manifest_filename``,
    /// inlining circuit sources where possible and writing other sources alongside.
    /// Returns the written manifest as a ``pathlib.Path`` for use with ``Qodec.load``.
    /// Relative destinations remain relative, and ``..`` components are preserved.
    /// Compatible loaded artifact paths and unused layer code declarations are
    /// retained. The manifest filename is preserved; YAML formatting may change.
    /// Directory saves reuse unchanged external files after checking their loaded
    /// text; edits are copied locally. Bundles include all current artifact values.
    /// Raises ``QodecSaveError`` on validation, serialization, or
    /// write failure.
    #[pyo3(signature = (destination, *, single_file = false))]
    fn save(&self, py: Python<'_>, destination: PathBuf, single_file: bool) -> PyResult<PathBuf> {
        let inner = self.to_qodec(py);
        let result = if single_file {
            inner.save_bundle(destination)
        } else {
            inner.save(destination)
        };
        result.map_err(|error| crate::QodecSaveError::new_err(error.to_string()))
    }

    /// Serialize this qodec to a self-contained multi-document YAML bundle.
    ///
    /// Raises ``QodecSaveError`` if validation or serialization fails, or a circuit
    /// source cannot be inlined. Unlike ``save``, this cannot write separate source files.
    fn dumps(&self, py: Python<'_>) -> PyResult<String> {
        let inner = self.to_qodec(py);
        inner
            .to_bundle_string()
            .map_err(|e: std::io::Error| crate::QodecSaveError::new_err(e.to_string()))
    }

    #[getter]
    fn name(&self) -> String {
        self.name.clone().unwrap_or_default()
    }

    #[setter]
    fn set_name(&mut self, value: String) {
        self.name = Some(value);
    }

    #[getter]
    fn description(&self) -> String {
        self.description.clone().unwrap_or_default()
    }

    #[setter]
    fn set_description(&mut self, value: String) {
        self.description = Some(value);
    }

    /// Declared schema version; ``None`` uses the current version on save.
    ///
    /// Assigning an unsupported version raises ``ValueError``.
    #[getter]
    fn schema_version(&self) -> Option<u32> {
        self.schema_version
    }

    #[setter]
    fn set_schema_version(&mut self, value: Option<u32>) -> PyResult<()> {
        self.schema_version = check_schema_version(value)?;
        Ok(())
    }

    /// The filename the manifest is written to by ``save``. For a loaded bundle,
    /// this is the internal manifest filename, not necessarily the bundle's outer
    /// filename. ``"qodec.yaml"`` for qodecs built via the Python constructor.
    #[getter]
    fn manifest_filename(&self) -> &str {
        &self.manifest_filename
    }

    #[setter]
    fn set_manifest_filename(&mut self, value: String) {
        self.manifest_filename = value;
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
    fn layers<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "layers", false)
    }

    #[pyo3(name = "_get_layers")]
    fn get_layers(&self, py: Python<'_>) -> Vec<Py<PyLayer>> {
        self.layers.iter().map(|layer| layer.clone_ref(py)).collect()
    }

    #[setter]
    fn set_layers(&mut self, value: Vec<Py<PyLayer>>) {
        self.layers = value;
    }

    #[getter]
    fn instruction_sets<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py()
            .import("qodec._collections")?
            .getattr("_ReadOnlyMapping")?
            .call1((slf, "instruction_sets"))
    }

    fn _get_instruction_sets(&self, py: Python<'_>) -> BTreeMap<String, Py<PyInstructionSet>> {
        self.layers
            .iter()
            .map(|layer| {
                let instruction_set = layer.borrow(py).instruction_set.clone_ref(py);
                let name = instruction_set.borrow(py).name.clone();
                (name, instruction_set)
            })
            .collect()
    }

    #[getter]
    fn codes<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        slf.py()
            .import("qodec._collections")?
            .getattr("_ReadOnlyMapping")?
            .call1((slf, "codes"))
    }

    fn _get_codes(&self, py: Python<'_>) -> BTreeMap<String, Py<PyCode>> {
        let mut codes: BTreeMap<String, Py<PyCode>> = BTreeMap::new();
        for layer in &self.layers {
            for code in layer.borrow(py).code_bindings(py).into_values() {
                let name = code.borrow(py).inner.name.clone();
                codes.entry(name).or_insert(code);
            }
        }
        codes
    }

    /// Build a qodec from layer indices ``start <= index < stop``.
    ///
    /// Bounds must satisfy ``0 <= start <= stop <= len(layers)``; they are not
    /// clipped. Reversed or oversized bounds raise ``ValueError``; negative
    /// bounds raise ``OverflowError``. Equal bounds give no layers.
    ///
    /// Retained layers are shared, except the new bottom layer, which has no
    /// gadgets and shares its instruction set. Changes to shared layers, gadgets, instruction sets,
    /// and codes are visible in both qodecs. Manifest fields are copied.
    fn slice(&self, py: Python<'_>, start: usize, stop: usize) -> PyResult<Self> {
        if start > stop {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "slice: start index {start} must be <= stop index {stop}"
            )));
        }
        if stop > self.layers.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "slice: stop index {stop} out of range (qodec has {} layers)",
                self.layers.len()
            )));
        }
        let mut layers: Vec<Py<PyLayer>> = self.layers[start..stop]
            .iter()
            .map(|layer| layer.clone_ref(py))
            .collect();
        if let Some(last) = layers.last_mut() {
            let instruction_set = last.borrow(py).instruction_set.clone_ref(py);
            let codes = last.borrow(py).code_bindings(py);
            *last = Py::new(
                py,
                PyLayer {
                    instruction_set,
                    codes,
                    gadgets: BTreeMap::new(),
                },
            )?;
        }
        Ok(Self {
            name: self.name.clone(),
            description: self.description.clone(),
            schema_version: self.schema_version,
            layers,
            manifest_filename: self.manifest_filename.clone(),
            metadata: self.metadata.clone(),
            loaded: None,
        })
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        let Ok(other) = other.extract::<PyRef<'_, PyQodec>>() else {
            return false;
        };
        self.name == other.name
            && self.description == other.description
            && self.schema_version == other.schema_version
            && self.metadata == other.metadata
            && self.layers.len() == other.layers.len()
            && self
                .layers
                .iter()
                .zip(&other.layers)
                .all(|(a, b)| a.borrow(py).struct_eq(&b.borrow(py), py))
    }

    fn __repr__(&self) -> String {
        let name = self.name.as_deref().unwrap_or("unnamed");
        format!("Qodec({name:?})")
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }

    /// A summary of the layers, gadget counts, and referenced codes.
    fn __str__(&self, py: Python<'_>) -> String {
        use std::fmt::Write;

        let mut out = String::new();
        let name = self.name.as_deref().unwrap_or("unnamed");
        let _ = writeln!(out, "Qodec {name:?}");

        if let Some(description) = self.description.as_deref()
            && !description.is_empty()
        {
            let _ = writeln!(out, "  {description}");
        }

        let layer_names: Vec<String> = self
            .layers
            .iter()
            .map(|layer| layer.borrow(py).instruction_set.borrow(py).name.clone())
            .collect();
        if !layer_names.is_empty() {
            let _ = writeln!(out, "  Layers: {}", layer_names.join(" -> "));
        }

        let has_gadgets = self.layers.iter().any(|layer| !layer.borrow(py).gadgets.is_empty());
        if has_gadgets {
            let _ = writeln!(out, "  Lowering:");
            for (index, layer) in self.layers.iter().enumerate() {
                let layer_ref = layer.borrow(py);
                if layer_ref.gadgets.is_empty() {
                    continue;
                }
                let source = layer_names.get(index).map_or("?", String::as_str);
                let target = layer_names.get(index + 1).map_or("?", String::as_str);
                let mnemonics: Vec<String> = layer_ref.gadgets.keys().cloned().collect();
                let count = mnemonics.len();
                let preview = if mnemonics.len() > 5 {
                    let mut head: Vec<String> = mnemonics.iter().take(5).cloned().collect();
                    head.push(format!("...+{}", mnemonics.len() - 5));
                    head.join(", ")
                } else {
                    mnemonics.join(", ")
                };
                let suffix = if preview.is_empty() {
                    String::new()
                } else {
                    format!(" ({preview})")
                };
                let plural = if count == 1 { "gadget" } else { "gadgets" };
                let _ = writeln!(out, "    {source} -> {target}: {count} {plural}{suffix}");
            }
        }

        let mut code_names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for layer in &self.layers {
            let layer_ref = layer.borrow(py);
            for gadget in layer_ref.gadgets.values() {
                let gadget_ref = gadget.borrow(py);
                for encoding in gadget_ref.inputs.iter().chain(gadget_ref.outputs.iter()) {
                    code_names.insert(encoding.borrow(py).code.borrow(py).inner.name.clone());
                }
            }
        }
        if !code_names.is_empty() {
            let _ = writeln!(
                out,
                "  Codes: {}",
                code_names.into_iter().collect::<Vec<_>>().join(", "),
            );
        }

        // Drop trailing newline
        out.trim_end().to_string()
    }
}

#[pyclass(name = "Layer", module = "qodec")]
pub struct PyLayer {
    instruction_set: Py<PyInstructionSet>,
    pub(crate) codes: BTreeMap<String, Py<PyCode>>,
    pub(crate) gadgets: BTreeMap<String, Py<PyGadget>>,
}

impl PyLayer {
    fn code_bindings(&self, py: Python<'_>) -> BTreeMap<String, Py<PyCode>> {
        let instruction_set = self.instruction_set.borrow(py);
        let mut codes: BTreeMap<_, _> = self
            .codes
            .iter()
            .filter(|(block, _)| instruction_set.blocks.iter().any(|declared| &declared.name == *block))
            .map(|(block, code)| (block.clone(), code.clone_ref(py)))
            .collect();
        for gadget in self.gadgets.values() {
            let gadget = gadget.borrow(py);
            let instruction = gadget.implements.borrow(py);
            for (encoding, operand) in gadget
                .inputs
                .iter()
                .chain(&gadget.outputs)
                .zip(instruction.inner.inputs.iter().chain(&instruction.inner.outputs))
            {
                codes
                    .entry(operand.block.clone())
                    .or_insert_with(|| encoding.borrow(py).code.clone_ref(py));
            }
        }
        codes
    }

    /// Materialize a `qodec::Layer` snapshot.
    pub fn to_resolved(&self, py: Python<'_>) -> qodec::Layer {
        let instruction_set = self.instruction_set.borrow(py).to_arc(py);
        let mut gadgets = BTreeMap::new();
        for (name, gadget) in &self.gadgets {
            gadgets.insert(name.clone(), gadget.borrow(py).to_resolved(py));
        }
        qodec::Layer {
            instruction_set,
            codes: self
                .codes
                .iter()
                .map(|(block, code)| (block.clone(), code.borrow(py).to_arc()))
                .collect(),
            gadgets,
        }
    }

    /// Structural (value) equality: matching instruction set definition and a gadget
    /// map with the same keys whose gadgets are structurally equal. Reused
    /// by `PyQodec`.
    pub(crate) fn struct_eq(&self, other: &PyLayer, py: Python<'_>) -> bool {
        self.instruction_set.borrow(py).to_inner(py) == other.instruction_set.borrow(py).to_inner(py)
            && {
                let codes = self.code_bindings(py);
                let other_codes = other.code_bindings(py);
                codes.len() == other_codes.len()
                    && codes.iter().all(|(block, code)| {
                        other_codes
                            .get(block)
                            .is_some_and(|other_code| code.borrow(py).inner == other_code.borrow(py).inner)
                    })
            }
            && self.gadgets.len() == other.gadgets.len()
            && self.gadgets.iter().all(|(name, gadget)| {
                other
                    .gadgets
                    .get(name)
                    .is_some_and(|other_gadget| gadget.borrow(py).struct_eq(&other_gadget.borrow(py), py))
            })
    }
}

#[pymethods]
impl PyLayer {
    #[new]
    #[pyo3(signature = (instruction_set, *, gadgets = None, codes = None))]
    fn new(
        py: Python<'_>,
        instruction_set: Py<PyInstructionSet>,
        gadgets: Option<Bound<'_, PyAny>>,
        codes: Option<BTreeMap<String, Py<PyCode>>>,
    ) -> PyResult<Self> {
        let map = match gadgets {
            None => BTreeMap::new(),
            Some(value) => extract_gadgets(py, &value)?,
        };
        Ok(Self {
            instruction_set,
            codes: codes.unwrap_or_default(),
            gadgets: map,
        })
    }

    #[getter]
    fn instruction_set(&self, py: Python<'_>) -> Py<PyInstructionSet> {
        self.instruction_set.clone_ref(py)
    }

    #[setter]
    fn set_instruction_set(&mut self, value: Py<PyInstructionSet>) {
        self.instruction_set = value;
    }

    #[getter]
    fn codes<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "codes", true)
    }

    fn _get_codes(&self, py: Python<'_>) -> BTreeMap<String, Py<PyCode>> {
        self.codes
            .iter()
            .map(|(block, code)| (block.clone(), code.clone_ref(py)))
            .collect()
    }

    #[setter]
    fn set_codes(slf: &Bound<'_, Self>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let codes = crate::collections::mapping(value)?.extract()?;
        slf.borrow_mut().codes = codes;
        Ok(())
    }

    #[getter]
    fn gadgets<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::collections::view(slf.as_any(), "gadgets", true)
    }

    fn _get_gadgets(&self, py: Python<'_>) -> BTreeMap<String, Py<PyGadget>> {
        self.gadgets
            .iter()
            .map(|(name, gadget)| (name.clone(), gadget.clone_ref(py)))
            .collect()
    }

    #[setter]
    fn set_gadgets(slf: &Bound<'_, Self>, value: Bound<'_, PyAny>) -> PyResult<()> {
        let gadgets = extract_gadgets(slf.py(), &value)?;
        slf.borrow_mut().gadgets = gadgets;
        Ok(())
    }

    fn _copy_shell(&self, py: Python<'_>) -> Self {
        Self {
            instruction_set: self.instruction_set.clone_ref(py),
            gadgets: self
                .gadgets
                .iter()
                .map(|(key, value)| (key.clone(), value.clone_ref(py)))
                .collect(),
            codes: self
                .codes
                .iter()
                .map(|(key, value)| (key.clone(), value.clone_ref(py)))
                .collect(),
        }
    }

    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        other
            .extract::<PyRef<'_, PyLayer>>()
            .is_ok_and(|other| self.struct_eq(&other, py))
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let instruction_set_name = self.instruction_set.borrow(py).name.clone();
        format!("Layer({instruction_set_name:?}, {} gadgets)", self.gadgets.len())
    }

    fn __str__(&self, py: Python<'_>) -> String {
        let instruction_set = self.instruction_set.borrow(py);
        let names: Vec<&str> = self.gadgets.keys().map(String::as_str).collect();
        format!(
            "Layer {:?}\n  Instructions: {}\n  Gadgets: {} ({})",
            instruction_set.name,
            instruction_set.instructions.len(),
            names.len(),
            names.join(", "),
        )
    }

    fn _repr_pretty_(slf: &Bound<'_, Self>, printer: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        crate::display::pretty(slf.as_any(), printer, cycle)
    }
}

/// Extract a `gadgets` argument: accept either a list of `Gadget` or a dict
/// mapping mnemonic to gadget. Either way the key comes from the gadget's
/// `implements.mnemonic`, so a supplied key that disagrees is rejected rather
/// than stored and later contradicted by the getter.
fn extract_gadgets(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<BTreeMap<String, Py<PyGadget>>> {
    let value = if value.hasattr("keys")? {
        crate::collections::mapping(value)?
    } else {
        value.clone()
    };
    let gadgets: Vec<(Option<String>, Py<PyGadget>)> =
        if let Ok(map) = value.extract::<BTreeMap<String, Py<PyGadget>>>() {
            map.into_iter().map(|(key, gadget)| (Some(key), gadget)).collect()
        } else {
            value
                .extract::<Vec<Py<PyGadget>>>()
                .map_err(|_| PyValueError::new_err("Layer gadgets must be a list[Gadget] or dict[str, Gadget]"))?
                .into_iter()
                .map(|gadget| (None, gadget))
                .collect()
        };
    let mut out: BTreeMap<String, Py<PyGadget>> = BTreeMap::new();
    for (supplied, gadget) in gadgets {
        let mnemonic = {
            let gadget_ref = gadget.borrow(py);
            let obj_ref = gadget_ref.implements.borrow(py);
            obj_ref.inner.mnemonic.clone()
        };
        if let Some(supplied) = supplied
            && supplied != mnemonic
        {
            return Err(PyValueError::new_err(format!(
                "Layer gadget key {supplied:?} does not match the implemented instruction {mnemonic:?}"
            )));
        }
        if out.contains_key(&mnemonic) {
            return Err(PyValueError::new_err(format!(
                "Layer has duplicate gadget for implemented instruction {mnemonic:?}"
            )));
        }
        out.insert(mnemonic, gadget);
    }
    Ok(out)
}
