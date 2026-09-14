//! The [`Qodec`] container and its public surface.
//!
//! This module is the facade: the top-level object, and [`Qodec::load`] /
//! [`Qodec::save`] over both on-disk shapes — a directory of YAML artifacts and
//! a single-file bundle. The work behind them lives in submodules:
//!
//! - `loader` reads a manifest and its artifacts off disk.
//! - `resolver` ties the raw artifact maps into the resolved [`Layer`] chain.
//! - `builder` synthesizes raw artifacts back out for saving.
//! - `parsers` holds the [`ParserRegistry`] of circuit-source languages.
//! - `error` holds [`LoadError`].

use crate::Code;
use crate::GadgetSpec;
use crate::InstructionSet;
use crate::Layer;
use crate::Manifest;
use crate::SliceError;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

mod builder;
mod error;
pub(crate) mod loader;
mod parsers;
pub(crate) mod resolver;

pub use error::{LoadError, ParseError};
use loader::{
    RawArtifacts, SidecarLists, check_schema_version, ingest_artifacts, load_manifest, parse_bundle_str,
    resolve_codes_by_path, resolve_instruction_sets_by_path, resolve_sidecars, split_bundle_documents,
};
pub use parsers::ParserRegistry;
pub use parsers::register;
use resolver::{LoaderArtifacts, build_layers};

/// A protocol's current model and its loaded layout information.
///
/// [`Qodec::load`] follows the manifest's artifact references. Use
/// [`Qodec::new`] to build from resolved layers instead.
///
/// Equality compares the current layers, name, description, schema version, and
/// metadata. It ignores loading history, filenames, and inline-versus-file layout.
/// Component equality is structural, not equivalence of quantum operations.
#[derive(Debug, Clone)]
pub struct Qodec {
    pub(super) manifest: Manifest,
    /// Filename the manifest was loaded from (relative to the qodec root),
    /// preserved so that lossless save writes back to the same name.
    /// Defaults to `"qodec.yaml"` for qodecs constructed via
    /// [`Qodec::new`].
    pub(super) manifest_filename: String,
    pub(super) loaded_manifest_filename: Option<String>,
    pub(super) loaded_root: Option<PathBuf>,
    external_files: BTreeMap<PathBuf, loader::ExternalFile>,
    pub(super) instruction_sets: BTreeMap<PathBuf, InstructionSet>,
    pub(super) code_artifacts: BTreeMap<PathBuf, Code>,
    pub(super) gadgets: BTreeMap<PathBuf, GadgetSpec>,
    /// Loaded check, readout, and circuit-source paths reserved when saving.
    pub(super) sidecar_paths: BTreeSet<PathBuf>,
    pub(super) layers: Vec<Layer>,
    pub(crate) locations: BTreeMap<String, crate::SourceLocation>,
}

impl PartialEq for Qodec {
    fn eq(&self, other: &Self) -> bool {
        self.manifest.name == other.manifest.name
            && self.manifest.description == other.manifest.description
            && self.manifest.schema_version == other.manifest.schema_version
            && self.manifest.metadata == other.manifest.metadata
            && self.layers == other.layers
    }
}

impl Qodec {
    /// Build a `Qodec` from already-resolved layers.
    ///
    /// `layers` is the ordered chain from logical (top) to physical
    /// (bottom). Each [`Layer`] carries an instruction set and the gadget set that
    /// lowers it to the layer below; the most concrete (bottom) layer has
    /// an empty gadget set. This constructor does not validate the layers.
    /// Use [`Self::validate`] to check them. Saving constructs on-disk artifacts
    /// from these layers.
    #[must_use]
    pub fn new(name: Option<String>, description: Option<String>, layers: Vec<Layer>) -> Self {
        let manifest = Manifest {
            schema_version: None,
            name,
            description,
            layers: Vec::new(),
            metadata: crate::Metadata::default(),
        };
        Self {
            manifest,
            manifest_filename: "qodec.yaml".to_owned(),
            loaded_manifest_filename: None,
            loaded_root: None,
            external_files: BTreeMap::new(),
            instruction_sets: BTreeMap::new(),
            code_artifacts: BTreeMap::new(),
            gadgets: BTreeMap::new(),
            sidecar_paths: BTreeSet::new(),
            layers,
            locations: BTreeMap::new(),
        }
    }

    /// Write this qodec as a directory of YAML artifacts and circuit sources.
    ///
    /// Creates `destination` (and parents) if needed and emits one file
    /// per local artifact, preserving compatible loaded paths. The
    /// manifest is written to its original filename (defaults to
    /// `qodec.yaml`).
    ///
    /// Returns `destination.join(self.manifest_filename())` after a successful
    /// write, without making the path absolute or normalizing its components.
    /// Pass the returned path directly to [`Qodec::load`].
    ///
    /// Validates and writes the current layers, including changes made through
    /// [`Qodec::layers_mut`]. Loaded artifact paths and inline-versus-file choices
    /// are reused where compatible with the current model. Files loaded from outside
    /// the original manifest's directory remain external references when unchanged.
    /// Modified external artifacts are copied locally, including documents whose
    /// references must change. External input files are never overwritten.
    /// Reused files are checked against their loaded text before any writes; a missing
    /// or changed file is an error. The saved directory therefore depends on those
    /// external files. Use a bundle to copy all values without reusing external files.
    /// New files are written inside `destination`. The exception is a manifest filename that itself points
    /// above `destination`, such as `../entry`: that raises the root for the whole
    /// qodec, so `destination` is where the manifest path is resolved rather than a
    /// boundary. These path checks do not guard against symlinks or concurrent filesystem changes.
    /// Shared files whose
    /// contents diverged are separated; newly added components get unique paths.
    /// Referenced codes unused by gadgets are retained while their layer and block
    /// declaration remain. Existing destination files are not deleted.
    /// Use [`Qodec::save_bundle`] to group the artifacts into a YAML bundle.
    ///
    /// ```
    /// use qodec::Qodec;
    ///
    /// let qodec = Qodec::load("examples/repetition3/repetition3.qodec.yaml")?;
    /// let scratch = std::env::temp_dir().join("qodec_save_doctest");
    /// # let _ = std::fs::remove_dir_all(&scratch);
    ///
    /// let reloaded = Qodec::load(qodec.save(&scratch)?)?;
    ///
    /// assert_eq!(reloaded.name(), qodec.name());
    /// assert_eq!(reloaded.layers().len(), qodec.layers().len());
    ///
    /// // The same qodec also round-trips through the single-file layout.
    /// qodec.save_bundle(&scratch)?;
    /// # std::fs::remove_dir_all(&scratch)?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if validation, writing, serialization, or construction of
    /// on-disk artifacts fails.
    pub fn save(&self, destination: impl AsRef<Path>) -> std::io::Result<PathBuf> {
        builder::save_qodec(self, destination.as_ref(), false)
    }

    /// Write this qodec as a YAML bundle into a destination directory.
    ///
    /// Creates `destination` (and parents) if needed and writes a multi-document
    /// YAML bundle at `<destination>/<manifest_filename>`. Each artifact is a
    /// `{relative/path.yaml: body}` envelope; the manifest is the first document,
    /// keyed by the manifest filename. `destination` is a directory, not the
    /// bundle filename. Use [`Qodec::save`] for separate artifact files.
    ///
    /// Returns `destination.join(self.manifest_filename())` after a successful
    /// write, without making the path absolute or normalizing its components.
    /// Pass the returned path directly to [`Qodec::load`].
    ///
    /// Validates and writes the current model using the same preparation as
    /// [`Qodec::save`], independent of how the qodec was constructed. Circuit sources with known
    /// formats are inlined; other sources are written as sidecar files. This
    /// method therefore need not produce exactly one file. External artifacts are
    /// copied from the current model without rereading or reusing their original files.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` if validation, writing, serialization, or construction of
    /// on-disk artifacts fails.
    pub fn save_bundle(&self, destination: impl AsRef<Path>) -> std::io::Result<PathBuf> {
        builder::save_qodec(self, destination.as_ref(), true)
    }

    /// This qodec as a single-file bundle string, the inverse of
    /// [`Qodec::from_bundle_str`].
    ///
    /// Each document is a `{relative/path.yaml: body}` envelope, with the
    /// manifest first. Unlike saving to disk, every circuit source must be
    /// inlined because there is nowhere to write a sidecar file.
    ///
    /// # Errors
    ///
    /// Returns an `io::Error` on validation, YAML serialization, or synthesis failure, and
    /// when the qodec cannot be made self-contained: saving to a directory
    /// puts a circuit source whose format cannot be determined in a sidecar
    /// file, and a string has nowhere to put one.
    pub fn to_bundle_string(&self) -> std::io::Result<String> {
        builder::bundle_string(self)
    }

    /// Load a manifest file or a single-file bundle.
    ///
    /// `path` must name a file, not a directory. Its filename is unrestricted.
    /// The manifest's directory is the root for its relative artifact paths.
    ///
    /// ```
    /// use qodec::Qodec;
    ///
    /// let qodec = Qodec::load("examples/repetition3/repetition3.qodec.yaml")?;
    ///
    /// // Layers run from the most abstract instruction set down to the physical one.
    /// assert_eq!(qodec.layers().len(), 2);
    /// # Ok::<(), qodec::LoadError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::NotAManifestFile`] for a directory path,
    /// [`LoadError::Io`] for a file-read failure, and
    /// other [`LoadError`] variants for invalid YAML or referenced artifacts.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let manifest_path = path.as_ref();
        if manifest_path.is_dir() {
            return Err(LoadError::NotAManifestFile {
                path: manifest_path.to_path_buf(),
            });
        }
        let root = manifest_path.parent().map_or_else(PathBuf::new, Path::to_path_buf);

        let (manifest, manifest_filename, bundle_artifacts, origins) = load_manifest(manifest_path, &mut |_, _, _| {})?;
        check_schema_version(&manifest, manifest_path)?;
        Self::assemble(manifest, manifest_filename, bundle_artifacts, &root, origins)
    }

    /// Load a qodec from a single-file bundle held in memory.
    ///
    /// `text` uses the envelope format produced by [`Qodec::to_bundle_string`].
    /// A self-contained bundle needs no filesystem access. External references
    /// are resolved relative to the current working directory.
    ///
    /// ```
    /// use qodec::Qodec;
    ///
    /// let qodec = Qodec::load("examples/repetition3/repetition3.qodec.yaml")?;
    /// let reloaded = Qodec::from_bundle_str(&qodec.to_bundle_string()?)?;
    ///
    /// assert_eq!(reloaded.name(), qodec.name());
    /// assert_eq!(reloaded.layers().len(), qodec.layers().len());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`LoadError::MalformedBundle`] when `text` is not a bundle,
    /// and otherwise the same errors as [`Qodec::load`].
    pub fn from_bundle_str(text: &str) -> Result<Self, LoadError> {
        let origin = Path::new("<bundle>");
        let documents = parse_bundle_str(text, origin)?.ok_or_else(|| LoadError::MalformedBundle {
            manifest: origin.to_path_buf(),
            reason: "expected a single-file bundle: a multi-document YAML stream of \
                     `{path: body}` envelopes"
                .to_owned(),
        })?;
        let (manifest, manifest_filename, bundle_artifacts) =
            split_bundle_documents(documents, origin, &mut |_, _, _| {})?;
        check_schema_version(&manifest, origin)?;
        Self::assemble(
            manifest,
            manifest_filename,
            bundle_artifacts,
            Path::new(""),
            BTreeMap::new(),
        )
    }

    /// Shared tail of [`Qodec::load`] and [`Qodec::from_bundle_str`]: turn a
    /// manifest plus its artifacts into a validated, resolved qodec.
    ///
    /// No artifact file is read for a self-contained bundle, which may pass an
    /// empty `root`. The working directory is still resolved once, to give a
    /// relative and an absolute reference to the same file one identity.
    fn assemble(
        manifest: Manifest,
        manifest_filename: String,
        bundle_artifacts: Option<Vec<(PathBuf, serde_yaml::Value)>>,
        root: &Path,
        origins: BTreeMap<PathBuf, crate::node::source::Document>,
    ) -> Result<Self, LoadError> {
        let (resolved_manifest, raw) = ingest_artifacts(
            root,
            &manifest,
            &manifest_filename,
            bundle_artifacts,
            origins,
            &mut |_, _, _| {},
        )?;
        let RawArtifacts {
            instruction_sets,
            codes,
            gadgets,
            sidecar_documents,
            source_files,
            origins,
            root,
            external_files,
        } = raw;

        let SidecarLists {
            check_lists,
            readout_lists,
        } = resolve_sidecars(&gadgets, &sidecar_documents)?;

        let qodec = Self {
            manifest,
            loaded_manifest_filename: Some(manifest_filename.clone()),
            loaded_root: Some(root),
            external_files,
            manifest_filename,
            instruction_sets,
            code_artifacts: codes,
            gadgets,
            sidecar_paths: check_lists
                .keys()
                .chain(readout_lists.keys())
                .chain(source_files.keys())
                .cloned()
                .collect(),
            layers: Vec::new(),
            locations: BTreeMap::new(),
        };
        crate::validation::validate_bottom_layer(
            resolved_manifest
                .layers
                .last()
                .is_some_and(|layer| !layer.gadgets.is_empty()),
        )
        .map_err(|issue| LoadError::from_validation(issue, &qodec, &resolved_manifest))?;

        let instruction_sets_by_path = resolve_instruction_sets_by_path(&qodec.instruction_sets)?;
        let codes_by_path = resolve_codes_by_path(&qodec.code_artifacts)?;

        let layers = build_layers(
            &resolved_manifest,
            Path::new(&qodec.manifest_filename),
            &LoaderArtifacts {
                instruction_sets_by_path: &instruction_sets_by_path,
                gadgets: &qodec.gadgets,
                check_lists: &check_lists,
                readout_lists: &readout_lists,
                codes_by_path: &codes_by_path,
                source_files: &source_files,
            },
        )?;

        let mut qodec = Self { layers, ..qodec };
        qodec.locations = crate::node::source::locations(&qodec, &resolved_manifest, &origins);
        qodec
            .validate_model()
            .map_err(|issue| LoadError::from_validation(issue, &qodec, &resolved_manifest))?;
        let used_codes = qodec.codes();
        for (path, code) in &qodec.code_artifacts {
            if !used_codes.contains_key(&code.name) {
                code.validate().map_err(|error| LoadError::InvalidCode {
                    code: path.clone(),
                    error,
                })?;
            }
        }
        Ok(qodec)
    }

    /// The qodec's name, if the manifest gives one.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.manifest.name.as_deref()
    }

    /// Sets the qodec's name.
    pub fn set_name(&mut self, name: Option<String>) {
        self.locations.clear();
        self.manifest.name = name;
    }

    /// The qodec's description, if the manifest gives one.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.manifest.description.as_deref()
    }

    /// Sets the qodec's description.
    pub fn set_description(&mut self, description: Option<String>) {
        self.locations.clear();
        self.manifest.description = description;
    }

    /// The on-disk format version the manifest declares, if any.
    ///
    /// When present, the value must equal
    /// [`CURRENT_SCHEMA_VERSION`](crate::CURRENT_SCHEMA_VERSION) for validation
    /// and saving. Omission is preserved.
    #[must_use]
    pub fn schema_version(&self) -> Option<u32> {
        self.manifest.schema_version
    }

    /// Sets the declared on-disk format version.
    ///
    /// Stores the value for editing. Validation and saving reject an explicit
    /// value other than [`CURRENT_SCHEMA_VERSION`](crate::CURRENT_SCHEMA_VERSION).
    pub fn set_schema_version(&mut self, version: Option<u32>) {
        self.locations.clear();
        self.manifest.schema_version = version;
    }

    /// Free-form annotations qodec itself does not interpret.
    #[must_use]
    pub fn metadata(&self) -> &crate::Metadata {
        &self.manifest.metadata
    }

    /// Mutable access to the annotations.
    pub fn metadata_mut(&mut self) -> &mut crate::Metadata {
        self.locations.clear();
        &mut self.manifest.metadata
    }

    /// The lowering chain: ordered layers from logical (top) to physical
    /// (bottom). Each [`Layer`] holds an instruction set and the gadgets that lower it
    /// to the layer below; the bottom layer has an empty gadget set.
    #[must_use]
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }

    /// All instruction sets keyed by instruction set name.
    ///
    /// Derived from `layers`: every instruction set in the lowering chain appears
    /// once. To add or remove an instruction set, mutate `layers` directly.
    #[must_use]
    pub fn instruction_sets(&self) -> BTreeMap<String, Arc<InstructionSet>> {
        self.layers
            .iter()
            .map(|layer| (layer.instruction_set.name.clone(), layer.instruction_set.clone()))
            .collect()
    }

    /// All code definitions keyed by code name.
    ///
    /// Includes explicit layer bindings and gadget input/output encodings.
    /// Codes remain available after removing gadgets when the layer binds them.
    #[must_use]
    pub fn codes(&self) -> BTreeMap<String, Arc<Code>> {
        let mut codes = BTreeMap::new();
        for layer in &self.layers {
            for code in layer.code_bindings().into_values() {
                codes.entry(code.name.clone()).or_insert(code);
            }
        }
        codes
    }

    /// The filename [`Qodec::save`] writes the manifest to.
    ///
    /// For a directory qodec this is the file name it was loaded from. For a
    /// bundle it is the first document's envelope key, which is a qodec-root-relative
    /// path rather than the name of the file the bundle itself came from.
    #[must_use]
    pub fn manifest_filename(&self) -> &str {
        &self.manifest_filename
    }

    /// Sets the filename [`Qodec::save`] writes the manifest to.
    /// Moving it above the destination relocates relative artifacts alongside it.
    ///
    /// Source locations survive a rename because they record the absolute files
    /// the current model was loaded from, which renaming the output does not change.
    pub fn set_manifest_filename(&mut self, filename: String) {
        self.manifest_filename = filename;
    }

    /// Mutable access to the current model's layers.
    ///
    /// Changes are used by validation, equality, and every save method.
    pub fn layers_mut(&mut self) -> &mut Vec<Layer> {
        self.locations.clear();
        &mut self.layers
    }

    /// Build a new `Qodec` covering layers `start..stop` (half-open).
    ///
    /// `start` and `stop` are zero-based layer indices; `stop` is exclusive.
    /// `slice(start, start + 1)`
    /// yields a single-layer qodec and `slice(start, start)` yields an
    /// empty one. The slice shares `Arc<InstructionSet>` and `Arc<Code>`
    /// allocations with the parent, not mutable cells. The
    /// new bottom layer's gadget set is cleared, since the layer it used
    /// to lower to lies outside the slice.
    ///
    /// Retained layers keep their code bindings, including codes without gadgets.
    /// Name, description, metadata, explicit schema version, and manifest filename
    /// are copied. Stored artifact maps and source locations are not copied.
    /// A short slice is a saveable draft if no bottom-layer gadget needs an absent target.
    ///
    /// ```
    /// use qodec::Qodec;
    ///
    /// let qodec = Qodec::load("examples/repetition3/repetition3.qodec.yaml")?;
    /// let top = qodec.slice(0, 1)?;
    ///
    /// assert_eq!(top.layers().len(), 1);
    /// // The bottom layer of a slice has nothing left to lower to, so its
    /// // gadgets are dropped.
    /// assert!(top.layers()[0].gadgets.is_empty());
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`SliceError`] if `start > stop` or if `stop` is out of
    /// range.
    pub fn slice(&self, start: usize, stop: usize) -> Result<Self, SliceError> {
        if start > stop {
            return Err(SliceError::InvertedRange { start, stop });
        }
        if stop > self.layers.len() {
            return Err(SliceError::StopOutOfRange {
                stop,
                layer_count: self.layers.len(),
            });
        }
        let mut layers = self.layers[start..stop].to_vec();
        if let Some(last) = layers.last_mut() {
            last.codes = last.code_bindings();
            last.gadgets = BTreeMap::new();
        }
        let mut sliced = Self::new(self.manifest.name.clone(), self.manifest.description.clone(), layers);
        sliced.manifest.metadata.clone_from(&self.manifest.metadata);
        sliced.manifest.schema_version = self.manifest.schema_version;
        sliced.manifest_filename.clone_from(&self.manifest_filename);
        Ok(sliced)
    }
}

#[cfg(test)]
mod tests {
    use super::{LoadError, Qodec};
    use crate::SliceError;
    use std::path::Path;

    #[test]
    fn slice_preserves_manifest_values() {
        let mut original = Qodec::load("examples/repetition3/repetition3.qodec.yaml").unwrap();
        original.set_manifest_filename("nested/protocol.yaml".to_owned());
        original
            .metadata_mut()
            .insert("author".to_owned(), serde_json::json!({"name": "Ada"}));
        for version in [None, Some(crate::CURRENT_SCHEMA_VERSION)] {
            original.set_schema_version(version);
            for (start, stop) in [(0, 0), (0, 1), (1, 2), (0, 2)] {
                let sliced = original.slice(start, stop).unwrap();
                assert_eq!(sliced.name(), original.name());
                assert_eq!(sliced.description(), original.description());
                assert_eq!(sliced.metadata(), original.metadata());
                assert_eq!(sliced.schema_version(), version);
                assert_eq!(sliced.manifest_filename(), "nested/protocol.yaml");
                let reloaded = Qodec::from_bundle_str(&sliced.to_bundle_string().unwrap()).unwrap();
                assert_eq!(reloaded, sliced);
                assert!(sliced.locations.is_empty());
            }
        }
    }

    #[test]
    fn slice_metadata_is_independent_of_its_parent() {
        let mut original = Qodec::new(None, None, Vec::new());
        original
            .metadata_mut()
            .insert("author".to_owned(), serde_json::json!({"name": "Ada"}));
        let mut sliced = original.slice(0, 0).unwrap();
        sliced.metadata_mut()["author"]["name"] = serde_json::json!("Grace");
        assert_eq!(original.metadata()["author"]["name"], "Ada");
        assert_eq!(sliced.metadata()["author"]["name"], "Grace");
    }

    #[test]
    fn load_repetition3_example() {
        let qodec = Qodec::load("examples/repetition3/repetition3.qodec.yaml").expect("should load repetition3 qodec");

        assert_eq!(qodec.manifest.layers.len(), 2);

        assert!(qodec.instruction_sets().contains_key("repetition3"));
        assert!(qodec.instruction_sets().contains_key("stim+rz"));

        assert!(qodec.codes().contains_key("repetition3"));
    }

    #[test]
    fn resolved_layers_and_gadgets() {
        let qodec = Qodec::load("examples/repetition3/repetition3.qodec.yaml").expect("should load");

        let layers = qodec.layers();
        assert_eq!(layers.len(), 2);
        assert_eq!(layers[0].instruction_set.name, "repetition3");
        assert_eq!(layers[1].instruction_set.name, "stim+rz");

        let gadgets = &layers[0].gadgets;
        assert!(!gadgets.is_empty());
        assert!(layers[1].gadgets.is_empty());

        if let Some(idle) = gadgets.get("idle") {
            assert_eq!(idle.implements.mnemonic, "idle");
            assert_eq!(idle.inputs.len(), 1);
            assert_eq!(idle.inputs[0].code.name, "repetition3");
            assert!(!idle.checks.is_empty());
            assert_eq!(idle.inputs[0].block_types, vec!["qubit".to_owned(); 3]);
            assert_eq!(idle.outputs[0].block_types, vec!["qubit".to_owned(); 3]);
        }
    }

    #[test]
    fn validate_uses_current_layers() {
        let mut loaded = Qodec::load("examples/repetition3/repetition3.qodec.yaml").expect("load example");
        let constructed = Qodec::new(None, None, loaded.layers().to_vec());
        constructed.validate().expect("constructed layers are consistent");
        loaded.validate().expect("loaded layers are consistent");

        let instruction_set = std::sync::Arc::make_mut(&mut loaded.layers_mut()[1].instruction_set);
        instruction_set.blocks.push(instruction_set.blocks[0].clone());
        let error = loaded
            .validate()
            .expect_err("validate the edited layers, not the stored artifacts");
        assert!(error.contains("duplicate block declaration"), "{error}");
        constructed.validate().expect("the separate value is unchanged");
    }

    type LayerEdit = fn(&mut [crate::Layer]);

    fn inconsistent_layer_edits() -> &'static [(&'static str, LayerEdit)] {
        &[
            ("bottom layer", |layers| {
                let gadget = layers[0].gadgets["idle"].clone();
                layers[1].gadgets.insert("idle".to_owned(), gadget);
            }),
            ("conflicting instruction sets", |layers| {
                let name = layers[0].instruction_set.name.clone();
                std::sync::Arc::make_mut(&mut layers[1].instruction_set).name = name;
            }),
            ("is not declared by the layer", |layers| {
                let gadget = layers[0].gadgets.remove("idle").unwrap();
                layers[0].gadgets.insert("missing".to_owned(), gadget);
            }),
            ("implements differs", |layers| {
                layers[0]
                    .gadgets
                    .get_mut("idle")
                    .unwrap()
                    .implements
                    .description
                    .push_str(" changed");
            }),
            ("next layer's instruction set", |layers| {
                let circuit = &mut layers[0].gadgets.get_mut("idle").unwrap().circuit;
                std::sync::Arc::make_mut(&mut circuit.instruction_set).name = "other".to_owned();
            }),
            ("conflicting codes", |layers| {
                let code = &mut layers[0].gadgets.get_mut("idle").unwrap().outputs[0].code;
                std::sync::Arc::make_mut(code).description.push_str(" changed");
            }),
            ("bound to different codes", |layers| {
                let code = &mut layers[0].gadgets.get_mut("idle").unwrap().outputs[0].code;
                std::sync::Arc::make_mut(code).name = "other".to_owned();
            }),
            ("input encodings for", |layers| {
                layers[0].gadgets.get_mut("idle").unwrap().inputs.clear();
            }),
            ("different support and block-type lengths", |layers| {
                layers[0].gadgets.get_mut("idle").unwrap().inputs[0].block_types.pop();
            }),
            ("undeclared circuit block type", |layers| {
                layers[0].gadgets.get_mut("idle").unwrap().inputs[0].block_types[0] = "missing".to_owned();
            }),
            ("inconsistent position or flag role", |layers| {
                layers[0].gadgets.get_mut("measure_z").unwrap().readouts[0].position = 1;
            }),
            ("inconsistent position or flag role", |layers| {
                layers[0].gadgets.get_mut("measure_z").unwrap().readouts[0].is_flag = true;
            }),
        ]
    }

    #[test]
    fn validate_rejects_inconsistent_model_objects() {
        for (expected, edit) in inconsistent_layer_edits() {
            let loaded = Qodec::load("examples/repetition3/repetition3.qodec.yaml").expect("load example");
            let mut constructed = Qodec::new(None, None, loaded.layers().to_vec());
            edit(constructed.layers_mut());
            let error = constructed.validate().expect_err("reject inconsistent current state");
            assert!(error.contains(expected), "expected {expected:?}, got {error}");
        }
    }

    #[test]
    fn algebraically_invalid_code_can_validate_and_round_trip() {
        let mut qodec = Qodec::load("examples/repetition3/repetition3.qodec.yaml").expect("load example");
        let mut code = qodec.layers()[0].gadgets["idle"].inputs[0].code.as_ref().clone();
        code.z.clone_from(&code.x);
        let code = std::sync::Arc::new(code);
        qodec.layers_mut()[0]
            .codes
            .insert("repetition3".to_owned(), code.clone());
        for gadget in qodec.layers_mut()[0].gadgets.values_mut() {
            for encoding in gadget.inputs.iter_mut().chain(&mut gadget.outputs) {
                encoding.code = code.clone();
            }
        }
        qodec.validate().expect("code algebra is an analysis concern");
        let bundle = qodec.to_bundle_string().expect("save invalid algebra");
        let loaded = Qodec::from_bundle_str(&bundle).expect("load invalid algebra");
        assert_eq!(loaded.layers(), qodec.layers());
    }

    #[test]
    fn unresolved_parity_references_are_preserved_for_audit() {
        for reference in [
            "in[9].x[0]",
            "out[0].stabilizers[9]",
            "readouts[0]",
            "circuit.readouts[0:9]",
        ] {
            let mut protocol = Qodec::load("examples/repetition3/repetition3.qodec.yaml").unwrap();
            protocol.layers_mut()[0].gadgets.get_mut("idle").unwrap().checks =
                vec![vec![crate::Reference::parse(reference).unwrap().into()]];
            protocol.validate().unwrap();
            let reloaded = Qodec::from_bundle_str(&protocol.to_bundle_string().unwrap()).unwrap();
            assert_eq!(
                reloaded.layers()[0].gadgets["idle"].checks,
                protocol.layers()[0].gadgets["idle"].checks
            );
        }
    }

    #[test]
    fn incomplete_layer_chains_can_round_trip() {
        let qodec = Qodec::load("examples/repetition3/repetition3.qodec.yaml").expect("load example");
        for stop in [0, 1] {
            let mut sliced = qodec.slice(0, stop).expect("slicing remains unrestricted");
            for layer in sliced.layers_mut() {
                layer.gadgets.clear();
            }
            sliced.validate().unwrap();
            assert_eq!(
                Qodec::from_bundle_str(&sliced.to_bundle_string().unwrap())
                    .unwrap()
                    .layers(),
                sliced.layers()
            );
        }
    }

    #[test]
    fn save_returns_loadable_manifest_path() {
        let mut protocol = Qodec::load("examples/repetition3/repetition3.qodec.yaml").unwrap();
        for single_file in [false, true] {
            for relative in [false, true] {
                for filename in ["qodec.yaml", "nested/protocol.yaml", "../entry"] {
                    let directory = TempDir::new_in(".").unwrap();
                    let destination = if relative {
                        Path::new(".").join(directory.path().file_name().unwrap())
                    } else {
                        directory.path().to_path_buf()
                    }
                    .join("saved");
                    protocol.set_manifest_filename(filename.to_owned());
                    let manifest_path = if single_file {
                        protocol.save_bundle(&destination)
                    } else {
                        protocol.save(&destination)
                    }
                    .unwrap();
                    assert_eq!(manifest_path, destination.join(filename));
                    assert_eq!(manifest_path.is_relative(), relative);
                    assert_eq!(Qodec::load(manifest_path).unwrap(), protocol);
                }
            }
        }
    }

    #[test]
    fn save_propagates_write_errors() {
        let protocol = Qodec::new(None, None, Vec::new());
        for single_file in [false, true] {
            let directory = TempDir::new().unwrap();
            fs::create_dir(directory.path().join(protocol.manifest_filename())).unwrap();
            let result = if single_file {
                protocol.save_bundle(directory.path())
            } else {
                protocol.save(directory.path())
            };
            result.expect_err("writing a manifest over a directory must fail");
        }
    }

    #[test]
    fn validate_does_not_read_files() {
        let original = Qodec::load("examples/repetition3/repetition3.qodec.yaml").expect("load example");
        let directory = TempDir::new().expect("temporary directory");
        original.save(directory.path()).expect("save fixture");
        let path = directory.path().join(original.manifest_filename());
        let loaded = Qodec::load(&path).expect("load saved fixture");
        drop(directory);
        assert!(!path.exists());
        loaded.validate().expect("only current objects are needed");
    }

    #[test]
    fn slice_full_range_is_equivalent() {
        let qodec = Qodec::load("examples/repetition3/repetition3.qodec.yaml").expect("should load");
        let sub = qodec.slice(0, qodec.layers().len()).expect("full range");
        assert_eq!(sub.layers().len(), qodec.layers().len());
        assert_eq!(
            sub.layers()[0].instruction_set.name,
            qodec.layers()[0].instruction_set.name
        );
    }

    #[test]
    fn slice_single_layer_has_no_gadgets() {
        let qodec = Qodec::load("examples/repetition3/repetition3.qodec.yaml").expect("should load");
        let sub = qodec.slice(0, 1).expect("single layer");
        assert_eq!(sub.layers().len(), 1);
        assert!(sub.layers()[0].gadgets.is_empty());
    }

    #[test]
    fn slice_invalid_range_errors() {
        let qodec = Qodec::load("examples/repetition3/repetition3.qodec.yaml").expect("should load");

        let inverted = qodec.slice(1, 0).expect_err("start > stop");
        assert!(
            matches!(inverted, SliceError::InvertedRange { start: 1, stop: 0 }),
            "got: {inverted:?}"
        );

        let out_of_range = qodec.slice(0, 99).expect_err("stop past the last layer");
        assert!(
            matches!(out_of_range, SliceError::StopOutOfRange { stop: 99, .. }),
            "got: {out_of_range:?}"
        );
    }

    use std::fs;
    use tempfile::TempDir;

    fn write(dir: &Path, relative: &str, body: &str) {
        let full = dir.join(relative);
        if let Some(parent) = full.parent() {
            fs::create_dir_all(parent).expect("create parent");
        }
        fs::write(full, body).expect("write file");
    }

    fn minimal_instruction_set(name: &str, mnemonic: &str) -> String {
        format!(
            "name: {name}\ndescription: test\nblocks: {{q: 1}}\ninstructions:\n  - mnemonic: {mnemonic}\n    description: test\n    in: [q]\n    out: [q]\n    action:\n      - pauli: \"X_0\"\n"
        )
    }

    fn make_minimal_qodec(stack_layers: &[&str]) -> TempDir {
        let dir = TempDir::new().expect("temp dir");
        let manifest = format!(
            "name: test\nlayers:\n{}",
            stack_layers
                .iter()
                .map(|layer| format!("  - instruction_set: {layer}"))
                .collect::<Vec<_>>()
                .join("\n")
        );
        write(dir.path(), "qodec.yaml", &manifest);
        dir
    }

    #[test]
    fn manifest_path_rejects_directories_regardless_of_contents() {
        for filenames in [
            vec![],
            vec!["qodec.yaml"],
            vec!["only.qodec.yaml"],
            vec!["first.qodec.yaml", "second.qodec.yaml"],
        ] {
            let dir = TempDir::new().expect("temp dir");
            for filename in filenames {
                write(dir.path(), filename, "name: test\nlayers: []\n");
            }
            let error = Qodec::load(dir.path()).expect_err("a directory is not a manifest path");
            let LoadError::NotAManifestFile { path } = &error else {
                panic!("expected NotAManifestFile, got {error:?}");
            };
            assert_eq!(path, dir.path());
            assert!(error.to_string().contains("expected a manifest file path"));
        }
    }

    #[test]
    fn manifest_path_accepts_an_explicit_arbitrary_filename() {
        let dir = make_minimal_qodec(&["a.isa.yaml", "b.isa.yaml"]);
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("A", "op"));
        write(dir.path(), "b.isa.yaml", &minimal_instruction_set("B", "op"));
        fs::rename(dir.path().join("qodec.yaml"), dir.path().join("protocol.txt")).expect("rename manifest");
        write(dir.path(), "qodec.yaml", "name: not-selected\nlayers: []\n");

        let qodec = Qodec::load(dir.path().join("protocol.txt")).expect("load explicit file");
        assert_eq!(qodec.name(), Some("test"));
        assert_eq!(qodec.manifest_filename(), "protocol.txt");
        assert_eq!(qodec.layers().len(), 2);
    }

    #[test]
    fn stack_with_one_layer_is_preserved_as_a_draft() {
        let dir = make_minimal_qodec(&["a.isa.yaml"]);
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("A", "x"));
        let protocol = Qodec::load(dir.path().join("qodec.yaml")).unwrap();
        assert_eq!(
            Qodec::from_bundle_str(&protocol.to_bundle_string().unwrap())
                .unwrap()
                .layers(),
            protocol.layers()
        );
    }

    #[test]
    fn current_schema_version_is_accepted() {
        let protocol = Qodec::from_bundle_str("qodec.yaml:\n  schema_version: 1\n  layers: []\n").unwrap();
        assert_eq!(crate::CURRENT_SCHEMA_VERSION, 1);
        assert_eq!(protocol.schema_version(), Some(1));
    }

    #[test]
    fn unsupported_schema_version_is_rejected() {
        let dir = TempDir::new().expect("temp dir");
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("A", "op"));
        write(dir.path(), "b.isa.yaml", &minimal_instruction_set("B", "op"));
        for version in [0, crate::CURRENT_SCHEMA_VERSION + 1, u32::MAX] {
            let manifest = format!(
                "schema_version: {version}\nlayers:\n  - instruction_set: a.isa.yaml\n  - instruction_set: b.isa.yaml\n"
            );
            write(dir.path(), "qodec.yaml", &manifest);
            let error =
                Qodec::load(dir.path().join("qodec.yaml")).expect_err("should reject unsupported schema version");
            assert!(
                matches!(error, LoadError::UnsupportedSchemaVersion { declared, supported, .. }
                    if declared == version && supported == crate::CURRENT_SCHEMA_VERSION),
                "got {error:?}"
            );
        }
    }

    #[test]
    fn non_integer_schema_version_is_rejected_by_serde() {
        let dir = TempDir::new().expect("temp dir");
        let manifest =
            "schema_version: \"0.1\"\nlayers:\n  - instruction_set: a.isa.yaml\n  - instruction_set: b.isa.yaml\n";
        write(dir.path(), "qodec.yaml", manifest);
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("A", "op"));
        write(dir.path(), "b.isa.yaml", &minimal_instruction_set("B", "op"));
        let error = Qodec::load(dir.path().join("qodec.yaml")).expect_err("serde should reject non-integer");
        assert!(matches!(error, LoadError::Yaml { .. }), "got {error:?}");
    }

    #[test]
    fn missing_schema_version_is_accepted() {
        let dir = make_minimal_qodec(&["a.isa.yaml", "b.isa.yaml"]);
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("A", "op"));
        write(dir.path(), "b.isa.yaml", &minimal_instruction_set("B", "op"));
        Qodec::load(dir.path().join("qodec.yaml")).expect("missing schema version is accepted");
    }

    #[test]
    fn checks_file_without_gadget_is_ignored() {
        let dir = make_minimal_qodec(&["a.isa.yaml", "b.isa.yaml"]);
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("A", "op"));
        write(dir.path(), "b.isa.yaml", &minimal_instruction_set("B", "op"));
        // A *.checks.yaml that no gadget references is not auto-detected; it
        // is silently ignored rather than rejected as an orphan.
        write(dir.path(), "ghost.checks.yaml", "[]\n");
        Qodec::load(dir.path().join("qodec.yaml")).expect("unreferenced checks file is ignored");
    }

    #[test]
    fn manifest_path_selects_the_requested_file_among_multiple_manifests() {
        let dir = TempDir::new().expect("temp dir");
        let manifest = "layers:\n  - instruction_set: a.isa.yaml\n  - instruction_set: b.isa.yaml\n";
        write(dir.path(), "first.qodec.yaml", &format!("name: first\n{manifest}"));
        write(dir.path(), "second.qodec.yaml", &format!("name: second\n{manifest}"));
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("A", "op"));
        write(dir.path(), "b.isa.yaml", &minimal_instruction_set("B", "op"));
        for name in ["first", "second"] {
            let qodec = Qodec::load(dir.path().join(format!("{name}.qodec.yaml"))).expect("load selected file");
            assert_eq!(qodec.name(), Some(name));
        }
    }

    #[test]
    fn an_explicit_named_manifest_is_loaded() {
        let dir = TempDir::new().expect("temp dir");
        let manifest = "name: test\nlayers:\n  - instruction_set: a.isa.yaml\n  - instruction_set: b.isa.yaml\n";
        write(dir.path(), "only.qodec.yaml", manifest);
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("A", "op"));
        write(dir.path(), "b.isa.yaml", &minimal_instruction_set("B", "op"));
        let qodec = Qodec::load(dir.path().join("only.qodec.yaml")).expect("explicit named manifest should load");
        assert_eq!(qodec.layers().len(), 2);
    }

    #[test]
    fn duplicate_instruction_set_names_are_rejected() {
        // Two distinct instruction set files declaring the same `name` would silently
        // shadow each other in the name-keyed lookup; the loader must reject.
        let dir = make_minimal_qodec(&["a.isa.yaml", "b.isa.yaml"]);
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("Dup", "op"));
        write(dir.path(), "b.isa.yaml", &minimal_instruction_set("Dup", "op"));
        let error =
            Qodec::load(dir.path().join("qodec.yaml")).expect_err("should reject duplicate instruction set names");
        match error {
            LoadError::DuplicateInstructionSetName { name, .. } => assert_eq!(name, "Dup"),
            other => panic!("expected DuplicateInstructionSetName, got {other:?}"),
        }
    }

    #[test]
    fn duplicate_code_names_are_rejected() {
        let dir = make_minimal_qodec(&["a.isa.yaml", "b.isa.yaml"]);
        write(
            dir.path(),
            "qodec.yaml",
            "layers:\n  - instruction_set: a.isa.yaml\n    codes: {q: first.code.yaml}\n  - instruction_set: b.isa.yaml\n    codes: {q: second.code.yaml}\n",
        );
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("A", "op"));
        write(dir.path(), "b.isa.yaml", &minimal_instruction_set("B", "op"));
        let code = "name: Dup\ndescription: test\nstabilizers: [Z_0 Z_1]\nx: [X_0 X_1]\nz: [Z_0]\n";
        write(dir.path(), "first.code.yaml", code);
        write(dir.path(), "second.code.yaml", code);
        let error = Qodec::load(dir.path().join("qodec.yaml")).expect_err("should reject duplicate code names");
        match error {
            LoadError::DuplicateCodeName { name, .. } => assert_eq!(name, "Dup"),
            other => panic!("expected DuplicateCodeName, got {other:?}"),
        }
    }

    #[test]
    fn unreferenced_artifacts_do_not_change_the_loaded_qodec() {
        let dir = make_minimal_qodec(&["a.isa.yaml", "b.isa.yaml"]);
        write(
            dir.path(),
            "qodec.yaml",
            "layers:\n  - instruction_set: a.isa.yaml\n    codes: {q: first.code.yaml}\n  - instruction_set: b.isa.yaml\n",
        );
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("A", "op"));
        write(dir.path(), "b.isa.yaml", &minimal_instruction_set("B", "op"));
        let code = "name: Dup\ndescription: test\nstabilizers: [Z_0 Z_1]\nx: [X_0 X_1]\nz: [Z_0]\n";
        write(dir.path(), "first.code.yaml", code);
        let original = Qodec::load(dir.path().join("qodec.yaml")).expect("load referenced artifacts");

        write(dir.path(), "second.code.yaml", code);
        write(dir.path(), "orphan.gadget.yaml", "circuit: ./missing.stim\n");
        for filename in [
            "bad.isa.yaml",
            "bad.code.yaml",
            "bad.gadget.yaml",
            "bad.checks.yaml",
            "bad.readouts.yaml",
        ] {
            write(dir.path(), filename, "[invalid YAML\n");
        }

        let reloaded = Qodec::load(dir.path().join("qodec.yaml")).expect("unreferenced artifacts are ignored");
        assert_eq!(original, reloaded);
        assert_eq!(reloaded.codes().keys().map(String::as_str).collect::<Vec<_>>(), ["Dup"]);
    }

    #[test]
    fn manifest_path_reports_a_missing_file() {
        let dir = TempDir::new().expect("temp dir");
        let manifest = dir.path().join("missing.yaml");
        let error = Qodec::load(&manifest).expect_err("no such file");
        let LoadError::Io(source) = error else {
            panic!("expected I/O error, got {error:?}");
        };
        assert_eq!(source.kind(), std::io::ErrorKind::NotFound);
        assert!(source.to_string().contains(&manifest.display().to_string()));
    }

    #[test]
    fn an_explicit_canonical_manifest_is_loaded() {
        let dir = make_minimal_qodec(&["a.isa.yaml", "b.isa.yaml"]);
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("A", "op"));
        write(dir.path(), "b.isa.yaml", &minimal_instruction_set("B", "op"));
        write(dir.path(), "other.qodec.yaml", "name: ignored\nlayers: []\n");
        let qodec = Qodec::load(dir.path().join("qodec.yaml")).expect("load the requested manifest");
        assert_eq!(qodec.name(), Some("test"));
    }

    /// Every bundle document must be a single-key `{path: body}` envelope.
    #[test]
    fn malformed_bundle_documents_are_rejected() {
        let manifest =
            "qodec.yaml:\n  name: t\n  layers:\n    - instruction_set: a.isa.yaml\n    - instruction_set: b.isa.yaml\n";
        for (label, second) in [
            ("a sequence", "- not an envelope\n"),
            ("a two-key mapping", "a.isa.yaml: {}\nb.isa.yaml: {}\n"),
            ("a non-string key", "1: {}\n"),
        ] {
            let dir = TempDir::new().expect("temp dir");
            write(dir.path(), "qodec.yaml", &format!("{manifest}---\n{second}"));
            let error = Qodec::load(dir.path().join("qodec.yaml")).expect_err("malformed bundle");
            assert!(
                matches!(error, LoadError::MalformedBundle { .. }),
                "{label} should be rejected, got: {error:?}"
            );
        }
    }

    #[test]
    fn duplicate_bundle_document_keys_are_rejected() {
        let dir = TempDir::new().expect("temp dir");
        let bundle = format!(
            "qodec.yaml:\n  name: t\n  layers:\n    - instruction_set: a.isa.yaml\n    - instruction_set: b.isa.yaml\n---\na.isa.yaml:\n{}---\na.isa.yaml:\n{}",
            indent(&minimal_instruction_set("A", "op")),
            indent(&minimal_instruction_set("B", "op")),
        );
        write(dir.path(), "qodec.yaml", &bundle);
        let error = Qodec::load(dir.path().join("qodec.yaml")).expect_err("duplicate document key");
        match error {
            LoadError::MalformedBundle { reason, .. } => {
                assert!(reason.contains("duplicate document key"), "got: {reason}");
            }
            other => panic!("expected MalformedBundle, got {other:?}"),
        }
    }

    fn indent(body: &str) -> String {
        body.lines().fold(String::new(), |mut indented, line| {
            indented.push_str("  ");
            indented.push_str(line);
            indented.push('\n');
            indented
        })
    }

    #[test]
    fn unreferenced_duplicate_instruction_sets_are_ignored_in_any_directory() {
        let dir = make_minimal_qodec(&["a.isa.yaml", "b.isa.yaml"]);
        write(dir.path(), "a.isa.yaml", &minimal_instruction_set("A", "op"));
        write(dir.path(), "b.isa.yaml", &minimal_instruction_set("B", "op"));
        write(dir.path(), "duplicate.isa.yaml", &minimal_instruction_set("A", "op"));
        write(
            dir.path(),
            "nested/duplicate.isa.yaml",
            &minimal_instruction_set("A", "op"),
        );
        write(dir.path(), ".hidden/c.isa.yaml", &minimal_instruction_set("A", "op"));
        write(dir.path(), "target/d.isa.yaml", &minimal_instruction_set("A", "op"));
        let qodec = Qodec::load(dir.path().join("qodec.yaml")).expect("unreferenced instruction sets are ignored");
        assert_eq!(qodec.layers().len(), 2);
        assert_eq!(qodec.instruction_sets().len(), 2);
    }
}
