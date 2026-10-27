//! Prepare on-disk documents from the current model, then write a directory or bundle.
//!
//! Loaded documents provide paths and representation choices, never replacements
//! for current model values. All save entry points share preparation and validation.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::Qodec;
use super::loader::same_document;
use crate::Code;
use crate::InstructionSet;
use crate::Sourced;
use crate::error::SynthesisError;
use crate::resolved;
use crate::{EncodingSpec, GadgetSpec, Implements};
use crate::{LayerSpec, Manifest};
use crate::{ParityEquation, ReadoutsList};

/// The current model's prepared documents and source files, keyed by output-root-relative path.
#[derive(Debug, Default)]
pub(crate) struct SynthesizedArtifacts {
    pub manifest_layers: Vec<LayerSpec>,
    pub instruction_sets: BTreeMap<PathBuf, InstructionSet>,
    pub codes: BTreeMap<PathBuf, Code>,
    pub gadgets: BTreeMap<PathBuf, GadgetSpec>,
    pub check_lists: BTreeMap<PathBuf, Vec<ParityEquation>>,
    pub readout_lists: BTreeMap<PathBuf, ReadoutsList>,
    pub source_files: BTreeMap<PathBuf, String>,
}

struct ArtifactPaths<'a> {
    manifest_parent: &'a Path,
    /// Normalized `manifest_parent`: every artifact must be written inside it.
    root: PathBuf,
    relocate_loaded_paths: bool,
    loaded_root: Option<&'a Path>,
    external_files: &'a BTreeMap<PathBuf, super::loader::ExternalFile>,
    destination: Option<&'a Path>,
    reused: BTreeSet<PathBuf>,
    /// Every path that must not be handed out again, including loaded paths not
    /// yet claimed. `allocate` skips these when choosing a fresh name.
    reserved: BTreeSet<PathBuf>,
    /// Paths this save has already claimed for a value. A gadget may reclaim one
    /// it wrote itself, because a gadget document is rewritten in place rather
    /// than appended to.
    claimed: BTreeSet<PathBuf>,
}

impl ArtifactPaths<'_> {
    /// `None` when the loaded path would land outside the output root, so the
    /// caller allocates a fresh one beside the manifest instead.
    fn preferred_path(&self, path: &Path) -> Option<PathBuf> {
        let path = if path.is_absolute() {
            path.strip_prefix(self.loaded_root?).ok()?
        } else {
            path
        };
        let candidate = if self.relocate_loaded_paths {
            normalized_artifact_path(&self.manifest_parent.join(path))
        } else {
            normalized_artifact_path(path)
        };
        (!escapes(&candidate, &self.root)).then_some(candidate)
    }

    fn allocate(&mut self, stem: &str, suffix: &str) -> PathBuf {
        let stem = safe_filename(stem);
        let path_for = |stem: &str| normalized_artifact_path(&self.manifest_parent.join(format!("{stem}.{suffix}")));
        let mut path = path_for(&stem);
        let mut index = 1;
        while !self.reserved.insert(path.clone()) {
            path = path_for(&format!("{stem}.{index}"));
            index += 1;
        }
        self.claimed.insert(path.clone());
        path
    }

    fn store<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq>(
        &mut self,
        map: &mut BTreeMap<PathBuf, T>,
        preferred: Option<&Path>,
        stem: &str,
        suffix: &str,
        value: T,
    ) -> PathBuf {
        if let Some(original) = preferred
            && self.destination.is_some()
            && let Some(file) = self.external_files.get(original)
            && file.matches(&value)
        {
            self.reused.insert(original.to_owned());
            return file.path.clone();
        }
        if let Some(path) = preferred.and_then(|path| self.preferred_path(path)) {
            if map.get(&path).is_some_and(|previous| same_document(previous, &value)) {
                return path;
            }
            if !self.claimed.contains(&path) {
                self.claimed.insert(path.clone());
                self.reserved.insert(path.clone());
                map.insert(path.clone(), value);
                return path;
            }
        }
        let path = self.allocate(stem, suffix);
        map.insert(path.clone(), value);
        path
    }
}

fn safe_filename(stem: &str) -> String {
    let name: String = stem
        .chars()
        .map(|character| {
            if character.is_control() || matches!(character, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|') {
                '_'
            } else {
                character
            }
        })
        .collect();
    let name = name.trim_end_matches(['.', ' ']);
    if name.is_empty() {
        "artifact".to_owned()
    } else {
        name.to_owned()
    }
}

/// Prepare the current model and write it in the requested form.
pub(super) fn save_qodec(qodec: &Qodec, destination: &Path, single_file: bool) -> std::io::Result<PathBuf> {
    let absolute = normalized_artifact_path(&std::path::absolute(destination)?);
    let (artifacts, manifest) = prepare(qodec, (!single_file).then_some(absolute.as_path()))?;
    check_output_paths(&artifacts, &qodec.manifest_filename, &absolute)?;
    check_write_targets(qodec, &artifacts, &absolute, single_file)?;
    std::fs::create_dir_all(destination)?;
    let manifest_path = destination.join(&qodec.manifest_filename);
    std::fs::create_dir_all(manifest_path.parent().unwrap_or(destination))?;
    if single_file {
        write_bundle(destination, &artifacts, &manifest, &qodec.manifest_filename)
    } else {
        write_directory(destination, &artifacts, &manifest, &qodec.manifest_filename)
    }?;
    Ok(manifest_path)
}

fn check_write_targets(
    qodec: &Qodec,
    artifacts: &SynthesizedArtifacts,
    destination: &Path,
    single_file: bool,
) -> std::io::Result<()> {
    let manifest = destination.join(&qodec.manifest_filename);
    let mut targets = vec![manifest.clone()];
    if single_file {
        let (_, inlined) = inline_gadget_sources(&artifacts.gadgets, &artifacts.source_files);
        let sidecar_root = manifest.parent().unwrap_or(destination);
        targets.extend(
            artifacts
                .source_files
                .keys()
                .filter(|path| !inlined.contains(*path))
                .map(|path| sidecar_root.join(path)),
        );
    } else {
        targets.extend(artifacts.paths().map(|path| destination.join(path)));
    }
    let mut seen = BTreeSet::new();
    for target in targets {
        let target = normalized_artifact_path(&target);
        if !seen.insert(target.clone()) {
            return Err(std::io::Error::other(format!(
                "conflicting output artifact path {}",
                target.display()
            )));
        }
        if qodec.external_files.values().any(|file| file.path == target) {
            return Err(std::io::Error::other(format!(
                "output would overwrite external artifact {}",
                target.display()
            )));
        }
    }
    Ok(())
}

fn prepare(qodec: &Qodec, destination: Option<&Path>) -> std::io::Result<(SynthesizedArtifacts, Manifest)> {
    qodec.validate().map_err(std::io::Error::other)?;
    let artifacts = synthesize_raw_artifacts(qodec, destination)?;
    // Checked against the empty root first so a conflict between two model paths
    // is reported before any destination is touched; `save_qodec` checks again
    // against the absolute destination, where a relative and an absolute
    // artifact path can collide for the first time.
    check_output_paths(&artifacts, &qodec.manifest_filename, Path::new(""))?;
    let mut manifest = qodec.manifest.clone();
    manifest.layers.clone_from(&artifacts.manifest_layers);
    Ok((artifacts, manifest))
}

fn normalized_artifact_path(path: &Path) -> PathBuf {
    PathBuf::from(super::resolver::normalize_relative(&path.to_string_lossy()))
}

/// The shallowest directory every artifact must be written inside.
///
/// Normally the output root itself. A manifest filename that deliberately points
/// above the destination, such as `../entry`, raises the root to the manifest's
/// own directory so its siblings stay addressable.
fn output_root(manifest_parent: &Path) -> PathBuf {
    let normalized = normalized_artifact_path(manifest_parent);
    if normalized.is_absolute() || normalized.starts_with("..") {
        normalized
    } else {
        PathBuf::new()
    }
}

/// Whether `path` would be written outside `root`. Both are output-root-relative
/// and normalized, so only a leading `..` after the shared prefix can escape.
fn escapes(path: &Path, root: &Path) -> bool {
    if path.has_root() != root.has_root() {
        return true;
    }
    match path.strip_prefix(root) {
        Ok(rest) => {
            rest.starts_with("..")
                || rest
                    .components()
                    .any(|part| matches!(part, std::path::Component::Prefix(_) | std::path::Component::RootDir))
        }
        Err(_) => true,
    }
}

fn check_output_paths(artifacts: &SynthesizedArtifacts, manifest_filename: &str, root: &Path) -> std::io::Result<()> {
    let manifest = normalized_artifact_path(Path::new(manifest_filename));
    let contained = output_root(manifest.parent().unwrap_or(Path::new("")));
    let mut seen = BTreeSet::from([normalized_artifact_path(&root.join(manifest_filename))]);
    for path in artifacts.paths() {
        if !seen.insert(normalized_artifact_path(&root.join(path))) {
            return Err(std::io::Error::other(format!(
                "conflicting output artifact path {}",
                path.display()
            )));
        }
        if escapes(&normalized_artifact_path(path), &contained) {
            return Err(std::io::Error::other(format!(
                "output artifact path {} would be written outside the destination",
                path.display()
            )));
        }
    }
    Ok(())
}

impl SynthesizedArtifacts {
    fn paths(&self) -> impl Iterator<Item = &Path> {
        self.instruction_sets
            .keys()
            .chain(self.codes.keys())
            .chain(self.gadgets.keys())
            .chain(self.check_lists.keys())
            .chain(self.readout_lists.keys())
            .chain(self.source_files.keys())
            .map(PathBuf::as_path)
    }
}

/// Prepare current values, reusing compatible loaded paths and section layouts.
fn synthesize_raw_artifacts(qodec: &Qodec, destination: Option<&Path>) -> std::io::Result<SynthesizedArtifacts> {
    let mut builder = ArtifactBuilder::new(qodec, destination);
    builder.build_instruction_sets();
    builder.build_codes();
    builder.build_gadgets().map_err(std::io::Error::other)?;
    for path in &builder.paths.reused {
        builder.paths.external_files[path].check_unchanged()?;
    }
    Ok(builder.artifacts)
}

struct ArtifactBuilder<'model> {
    model: &'model Qodec,
    original_layers: Vec<Option<&'model LayerSpec>>,
    paths: ArtifactPaths<'model>,
    artifacts: SynthesizedArtifacts,
    instruction_set_paths: Vec<PathBuf>,
    code_paths: BTreeMap<String, PathBuf>,
}

impl<'model> ArtifactBuilder<'model> {
    fn new(model: &'model Qodec, destination: Option<&'model Path>) -> Self {
        let manifest = Path::new(&model.manifest_filename);
        let manifest_parent = manifest.parent().unwrap_or_else(|| Path::new(""));
        let moved = model.loaded_manifest_filename.as_deref().is_some_and(|original| {
            normalized_artifact_path(Path::new(original)) != normalized_artifact_path(manifest)
        });
        let mut paths = ArtifactPaths {
            manifest_parent,
            root: output_root(manifest_parent),
            loaded_root: model.loaded_root.as_deref(),
            external_files: &model.external_files,
            destination,
            reused: BTreeSet::new(),
            relocate_loaded_paths: moved
                && (manifest_parent.is_absolute() || normalized_artifact_path(manifest_parent).starts_with("..")),
            reserved: BTreeSet::from([normalized_artifact_path(manifest)]),
            claimed: BTreeSet::from([normalized_artifact_path(manifest)]),
        };
        for path in model
            .instruction_sets
            .keys()
            .chain(model.code_artifacts.keys())
            .chain(model.gadgets.keys())
            .chain(&model.sidecar_paths)
        {
            if let Some(path) = paths.preferred_path(path) {
                paths.reserved.insert(path);
            }
        }
        Self {
            model,
            original_layers: (0..model.layers.len())
                .map(|index| matching_loaded_layer(model, index))
                .collect(),
            paths,
            artifacts: SynthesizedArtifacts::default(),
            instruction_set_paths: Vec::new(),
            code_paths: BTreeMap::new(),
        }
    }

    fn loaded_path(&self, reference: &str) -> Option<PathBuf> {
        let manifest = Path::new(self.model.loaded_manifest_filename.as_deref()?);
        Some(super::resolver::resolve_relative(manifest, reference))
    }

    fn build_instruction_sets(&mut self) {
        let mut by_name = BTreeMap::new();
        for (index, layer) in self.model.layers.iter().enumerate() {
            let instruction_set = &layer.instruction_set;
            let path = by_name.entry(instruction_set.name.clone()).or_insert_with(|| {
                let preferred = self.original_layers[index].and_then(|layer| self.loaded_path(&layer.instruction_set));
                self.paths.store(
                    &mut self.artifacts.instruction_sets,
                    preferred.as_deref(),
                    &instruction_set.name,
                    "isa.yaml",
                    (**instruction_set).clone(),
                )
            });
            self.instruction_set_paths.push(path.clone());
            self.artifacts.manifest_layers.push(LayerSpec {
                instruction_set: relative_reference(Path::new(&self.model.manifest_filename), path),
                codes: BTreeMap::new(),
                gadgets: BTreeMap::new(),
            });
        }
    }

    fn build_codes(&mut self) {
        for (index, layer) in self.model.layers.iter().enumerate() {
            for (block, code) in layer.code_bindings() {
                let preferred = self
                    .model
                    .code_artifacts
                    .iter()
                    .find(|(_, original)| original.name == code.name)
                    .map(|(path, _)| path.clone())
                    .or_else(|| {
                        let reference = self.original_layers[index]?.codes.get(&block)?;
                        self.loaded_path(reference)
                    });
                let path = self.store_code(&code, preferred.as_deref());
                self.artifacts.manifest_layers[index].codes.insert(
                    block,
                    relative_reference(Path::new(&self.model.manifest_filename), &path),
                );
            }
        }
    }

    fn store_code(&mut self, code: &Code, preferred: Option<&Path>) -> PathBuf {
        self.code_paths
            .entry(code.name.clone())
            .or_insert_with(|| {
                self.paths.store(
                    &mut self.artifacts.codes,
                    preferred,
                    &code.name,
                    "code.yaml",
                    code.clone(),
                )
            })
            .clone()
    }

    fn build_gadgets(&mut self) -> Result<(), SynthesisError> {
        for (index, layer) in self
            .model
            .layers
            .iter()
            .enumerate()
            .take(self.model.layers.len().saturating_sub(1))
        {
            for (mnemonic, gadget) in &layer.gadgets {
                self.build_gadget(index, mnemonic, gadget)?;
            }
        }
        Ok(())
    }

    fn build_gadget(&mut self, index: usize, mnemonic: &str, gadget: &resolved::Gadget) -> Result<(), SynthesisError> {
        let stem = if self.model.layers.len() > 2 {
            format!("t{index}_{mnemonic}")
        } else {
            mnemonic.to_owned()
        };
        let preferred = self.original_layers[index]
            .and_then(|layer| layer.gadgets.get(mnemonic))
            .and_then(|reference| self.loaded_path(reference));
        let original = preferred.as_deref().and_then(|path| self.model.gadgets.get(path));
        let mut path = self.gadget_path(preferred.as_deref(), &stem);
        let mut document = self.gadget_document(index, gadget, &stem, &path, preferred.as_deref().zip(original))?;
        if let Some(original_path) = preferred.as_deref()
            && let Some(destination) = self.paths.destination
            && let Some(file) = self.paths.external_files.get(original_path)
            && let Some(mut original) = file.gadget()
        {
            let mut current = document.clone();
            elide_inferred_types(
                &mut original.circuit.inputs,
                &mut original.circuit.outputs,
                &gadget.circuit.instruction_set,
            );
            rebase_gadget(&mut original, &file.path, Path::new(""));
            rebase_gadget(&mut current, &destination.join(&path), Path::new(""));
            if same_document(&current, &original) {
                self.paths.reused.insert(original_path.to_owned());
                self.artifacts.manifest_layers[index]
                    .gadgets
                    .insert(mnemonic.to_owned(), file.path.to_string_lossy().into_owned());
                return Ok(());
            }
        }
        if self
            .artifacts
            .gadgets
            .get(&path)
            .is_some_and(|previous| !same_document(previous, &document))
        {
            let separate = self.paths.allocate(&stem, "gadget.yaml");
            rebase_gadget(&mut document, &path, &separate);
            path = separate;
        }
        self.artifacts.gadgets.insert(path.clone(), document);
        self.artifacts.manifest_layers[index].gadgets.insert(
            mnemonic.to_owned(),
            relative_reference(Path::new(&self.model.manifest_filename), &path),
        );
        Ok(())
    }

    fn gadget_path(&mut self, preferred: Option<&Path>, stem: &str) -> PathBuf {
        if let Some(path) = preferred.and_then(|path| self.paths.preferred_path(path))
            && (!self.paths.claimed.contains(&path) || self.artifacts.gadgets.contains_key(&path))
        {
            self.paths.claimed.insert(path.clone());
            return path;
        }
        self.paths.allocate(stem, "gadget.yaml")
    }

    fn gadget_document(
        &mut self,
        index: usize,
        gadget: &resolved::Gadget,
        stem: &str,
        path: &Path,
        original: Option<(&Path, &GadgetSpec)>,
    ) -> Result<GadgetSpec, SynthesisError> {
        let (mut inputs, mut outputs, circuit_in, circuit_out) = build_raw_realization(gadget, &self.code_paths)?;
        let original_spec = original.map(|(_, spec)| spec);
        if original_spec.is_some_and(|spec| spec.inputs.is_empty()) && default_support(&gadget.inputs) {
            inputs.clear();
        }
        if original_spec.is_some_and(|spec| spec.outputs.is_empty()) && default_support(&gadget.outputs) {
            outputs.clear();
        }
        let (source, format) = prepare_source(
            &gadget.circuit,
            path,
            original.map(|(path, spec)| (path, &spec.circuit)),
            stem,
            &mut self.artifacts.source_files,
            &mut self.paths,
        );
        let checks = preserve_section(
            &gadget.checks,
            original.map(|(path, spec)| (path, &spec.checks)),
            path,
            stem,
            "checks.yaml",
            &mut self.artifacts.check_lists,
            &mut self.paths,
        );
        let readout_specs: ReadoutsList = gadget.readouts.iter().map(crate::Readout::to_spec).collect();
        let readouts = preserve_section(
            &readout_specs,
            original.map(|(path, spec)| (path, &spec.readouts)),
            path,
            stem,
            "readouts.yaml",
            &mut self.artifacts.readout_lists,
            &mut self.paths,
        );
        Ok(GadgetSpec {
            implements: original_spec
                .is_none_or(|spec| spec.implements.is_some())
                .then(|| Implements {
                    instruction_set: relative_reference(path, &self.instruction_set_paths[index]),
                    mnemonic: gadget.implements.mnemonic.clone(),
                }),
            circuit: crate::CircuitSpec {
                instruction_set: original_spec
                    .is_none_or(|spec| spec.circuit.instruction_set.is_some())
                    .then(|| relative_reference(path, &self.instruction_set_paths[index + 1])),
                source,
                format,
                inputs: circuit_in,
                outputs: circuit_out,
            },
            inputs,
            outputs,
            checks,
            readouts,
            frames: gadget.frames.clone(),
            parameter_bindings: gadget.parameter_bindings.clone(),
            metadata: gadget.metadata.clone(),
        })
    }
}

fn rebase_gadget(gadget: &mut GadgetSpec, from: &Path, to: &Path) {
    if let Some(implements) = &mut gadget.implements {
        implements.instruction_set = relative_reference(
            to,
            &super::resolver::resolve_relative(from, &implements.instruction_set),
        );
    }
    if let Some(instruction_set) = &mut gadget.circuit.instruction_set {
        *instruction_set = relative_reference(to, &super::resolver::resolve_relative(from, instruction_set));
    }
    rebase_section(&mut gadget.circuit.source, from, to);
    rebase_section(&mut gadget.checks, from, to);
    rebase_section(&mut gadget.readouts, from, to);
}

fn loaded_instruction_set<'model>(qodec: &'model Qodec, layer: &LayerSpec) -> Option<&'model InstructionSet> {
    let manifest = Path::new(qodec.loaded_manifest_filename.as_deref()?);
    qodec
        .instruction_sets
        .get(&super::resolver::resolve_relative(manifest, &layer.instruction_set))
}

/// Reuse the same position when its instruction-set name matches, then follow
/// that name across reordering. Treat an unmatched name as a rename in place only
/// when the original name is absent from the current layers, so a moved layer
/// does not donate its artifact paths to another layer.
/// The loaded layer whose paths and layout the layer at `index` should reuse.
///
/// Tried in order: the loaded layer at the same position naming the same
/// instruction set, then that instruction set wherever it moved to, then the
/// loaded layer at the same position when its instruction set was renamed
/// rather than removed.
fn matching_loaded_layer(qodec: &Qodec, index: usize) -> Option<&LayerSpec> {
    let names_the_same_instruction_set = |layer: &&LayerSpec| {
        loaded_instruction_set(qodec, layer)
            .is_some_and(|original| original.name == qodec.layers[index].instruction_set.name)
    };
    let at_same_position = qodec.manifest.layers.get(index);
    if let Some(layer) = at_same_position.filter(names_the_same_instruction_set) {
        return Some(layer);
    }
    if let Some(moved) = qodec.manifest.layers.iter().find(names_the_same_instruction_set) {
        return Some(moved);
    }
    let original = loaded_instruction_set(qodec, at_same_position?)?;
    let kept_elsewhere = qodec
        .layers
        .iter()
        .any(|layer| layer.instruction_set.name == original.name);
    at_same_position.filter(|_| !kept_elsewhere)
}

fn relative_reference(from: &Path, to: &Path) -> String {
    let from = normalized_artifact_path(from);
    let to = normalized_artifact_path(to);
    if to.is_absolute() {
        return to.to_string_lossy().into_owned();
    }
    let parent = from.parent().unwrap_or_else(|| Path::new(""));
    let from_components: Vec<_> = parent.components().collect();
    let to_components: Vec<_> = to.components().collect();
    let common = from_components
        .iter()
        .zip(&to_components)
        .take_while(|(left, right)| left == right)
        .count();
    let mut path = PathBuf::new();
    for _ in common..from_components.len() {
        path.push("..");
    }
    for component in &to_components[common..] {
        path.push(component.as_os_str());
    }
    path.to_string_lossy().into_owned()
}

fn rebase_section<T>(section: &mut Sourced<T>, from: &Path, to: &Path) {
    if let Sourced::File { path } = section {
        *path = relative_reference(to, &super::resolver::resolve_relative(from, path));
    }
}

fn default_support(encodings: &[resolved::Encoding]) -> bool {
    let mut next = 0;
    encodings.iter().all(|encoding| {
        let width = encoding.code.physical_qubit_count();
        let matches = encoding.support.len() == width
            && encoding
                .support
                .iter()
                .enumerate()
                .all(|(offset, label)| *label == (next + offset).to_string());
        if !matches {
            return false;
        }
        let Some(end) = next.checked_add(width) else {
            return false;
        };
        next = end;
        true
    })
}

fn preserve_section<T: Clone + serde::Serialize + serde::de::DeserializeOwned + PartialEq>(
    value: &T,
    original: Option<(&Path, &Sourced<T>)>,
    gadget_path: &Path,
    stem: &str,
    suffix: &str,
    files: &mut BTreeMap<PathBuf, T>,
    paths: &mut ArtifactPaths<'_>,
) -> Sourced<T> {
    if let Some((from, Sourced::File { path })) = original {
        let preferred = super::resolver::resolve_relative(from, path);
        let assigned = paths.store(files, Some(&preferred), stem, suffix, value.clone());
        Sourced::file(relative_reference(gadget_path, &assigned))
    } else {
        Sourced::inline(value.clone())
    }
}

fn prepare_source(
    circuit: &resolved::Circuit,
    gadget_path: &Path,
    original: Option<(&Path, &crate::CircuitSpec)>,
    stem: &str,
    files: &mut BTreeMap<PathBuf, String>,
    paths: &mut ArtifactPaths<'_>,
) -> (Sourced<String>, Option<String>) {
    if let Some((from, spec)) = original {
        if let Sourced::File { path } = &spec.source {
            let format = super::parsers::ParserRegistry::format_for_path(path);
            if format == Some(circuit.effective_format()) || (format.is_none() && circuit.format.is_none()) {
                let preferred = super::resolver::resolve_relative(from, path);
                let suffix = Path::new(path)
                    .extension()
                    .and_then(|extension| extension.to_str())
                    .unwrap_or("source");
                let assigned = paths.store(files, Some(&preferred), stem, suffix, circuit.source.clone());
                return (Sourced::file(relative_reference(gadget_path, &assigned)), None);
            }
        } else {
            let format = spec
                .format
                .as_ref()
                .filter(|format| format.as_str() == circuit.effective_format())
                .cloned()
                .or_else(|| source_format(circuit));
            return (Sourced::inline(circuit.source.clone()), format);
        }
    }
    if circuit.effective_format() == "stim" {
        let sidecar_path = paths.allocate(stem, "stim");
        files.insert(sidecar_path.clone(), circuit.source.clone());
        (Sourced::file(relative_reference(gadget_path, &sidecar_path)), None)
    } else {
        (Sourced::inline(circuit.source.clone()), source_format(circuit))
    }
}

fn source_format(circuit: &resolved::Circuit) -> Option<String> {
    let format = circuit.effective_format();
    (format != "yaml" || serde_yaml::from_str::<serde_yaml::Sequence>(&circuit.source).is_err())
        .then(|| format.to_owned())
}

fn write_yaml_map<T: serde::Serialize>(destination: &Path, map: &BTreeMap<PathBuf, T>) -> std::io::Result<()> {
    for (relative, value) in map {
        let target = destination.join(relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let yaml = crate::yaml_output::to_string(value)?;
        std::fs::write(target, yaml)?;
    }
    Ok(())
}

fn write_directory(
    destination: &Path,
    artifacts: &SynthesizedArtifacts,
    manifest: &Manifest,
    manifest_filename: &str,
) -> std::io::Result<()> {
    write_yaml_map(destination, &artifacts.instruction_sets)?;
    write_yaml_map(destination, &artifacts.codes)?;
    write_yaml_map(destination, &artifacts.gadgets)?;
    write_yaml_map(destination, &artifacts.check_lists)?;
    write_yaml_map(destination, &artifacts.readout_lists)?;
    std::fs::write(
        destination.join(manifest_filename),
        crate::yaml_output::to_string(manifest)?,
    )?;
    write_source_sidecars(destination, &artifacts.source_files, &BTreeSet::new())
}

fn write_bundle(
    destination: &Path,
    artifacts: &SynthesizedArtifacts,
    manifest: &Manifest,
    manifest_filename: &str,
) -> std::io::Result<()> {
    let (gadgets, inlined) = inline_gadget_sources(&artifacts.gadgets, &artifacts.source_files);
    let stream = build_single_file_bundle(artifacts, &gadgets, manifest, manifest_filename)?;
    let manifest_path = destination.join(manifest_filename);
    let sidecar_root = manifest_path.parent().unwrap_or(destination);
    std::fs::write(&manifest_path, stream)?;
    write_source_sidecars(sidecar_root, &artifacts.source_files, &inlined)
}

/// Rewrite each gadget's external circuit-source reference to
/// an inline verbatim source drawn from `source_files`, so a single-file bundle
/// carries its sources inline instead of in sidecars. Returns the rewritten
/// gadgets and the set of `source_files` keys that were inlined (and so must
/// not also be written as sidecars). A source is inlined only when its circuit
/// `format` is known (already set, or inferable from the file extension);
/// otherwise it is left as a file reference so the sidecar is preserved,
/// even if another gadget inlines the same source.
fn inline_gadget_sources(
    gadgets: &BTreeMap<PathBuf, GadgetSpec>,
    source_files: &BTreeMap<PathBuf, String>,
) -> (BTreeMap<PathBuf, GadgetSpec>, BTreeSet<PathBuf>) {
    let mut inlined_gadgets = BTreeMap::new();
    let mut inlined = BTreeSet::new();
    for (gadget_path, gadget) in gadgets {
        let mut rewritten = gadget.clone();
        if let Sourced::File { path } = &rewritten.circuit.source {
            let key = super::resolver::resolve_relative(gadget_path, path);
            if let Some(text) = source_files.get(&key) {
                let format = rewritten
                    .circuit
                    .format
                    .clone()
                    .or_else(|| super::parsers::ParserRegistry::format_for_path(path).map(str::to_owned));
                if let Some(format) = format {
                    rewritten.circuit.format = Some(format);
                    rewritten.circuit.source = Sourced::inline(text.clone());
                    inlined.insert(key);
                }
            }
        }
        inlined_gadgets.insert(gadget_path.clone(), rewritten);
    }
    for (gadget_path, gadget) in &inlined_gadgets {
        if let Sourced::File { path } = &gadget.circuit.source {
            inlined.remove(&super::resolver::resolve_relative(gadget_path, path));
        }
    }
    (inlined_gadgets, inlined)
}

/// Write source-circuit sidecar files, skipping any keys in `inlined`
/// (already carried inline by a single-file bundle).
fn write_source_sidecars(
    destination: &Path,
    source_files: &BTreeMap<PathBuf, String>,
    inlined: &BTreeSet<PathBuf>,
) -> std::io::Result<()> {
    for (relative, contents) in source_files {
        if inlined.contains(relative) {
            continue;
        }
        let target = destination.join(relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(target, contents)?;
    }
    Ok(())
}

/// Append `value` to `stream` as one `{key: value}` document of a
/// multi-document YAML bundle.
fn push_envelope<T: serde::Serialize>(stream: &mut String, key: &str, value: &T) -> std::io::Result<()> {
    let body = serde_yaml::to_value(value).map_err(std::io::Error::other)?;
    let mut envelope = serde_yaml::Mapping::new();
    envelope.insert(serde_yaml::Value::String(key.to_owned()), body);
    let document = crate::yaml_output::to_string(&serde_yaml::Value::Mapping(envelope))?;
    stream.push_str("---\n");
    stream.push_str(&document);
    Ok(())
}

/// Append every entry of `map` (keyed by qodec-root-relative path) to
/// `stream` as a single-key envelope document.
fn push_envelope_map<T: serde::Serialize>(stream: &mut String, map: &BTreeMap<PathBuf, T>) -> std::io::Result<()> {
    for (relative, value) in map {
        push_envelope(stream, &relative.to_string_lossy(), value)?;
    }
    Ok(())
}

/// Build the qodec as a single multi-document YAML bundle. The manifest is
/// the first document (keyed by `manifest_filename`); every other artifact
/// follows as its own `{relative/path.yaml: body}` envelope, in the same kind
/// order the directory writer uses.
fn build_single_file_bundle(
    artifacts: &SynthesizedArtifacts,
    gadgets: &BTreeMap<PathBuf, GadgetSpec>,
    manifest: &Manifest,
    manifest_filename: &str,
) -> std::io::Result<String> {
    let mut stream = String::new();
    push_envelope(&mut stream, manifest_filename, manifest)?;
    push_envelope_map(&mut stream, &artifacts.instruction_sets)?;
    push_envelope_map(&mut stream, &artifacts.codes)?;
    push_envelope_map(&mut stream, gadgets)?;
    push_envelope_map(&mut stream, &artifacts.check_lists)?;
    push_envelope_map(&mut stream, &artifacts.readout_lists)?;
    Ok(stream)
}

/// Body of [`Qodec::to_bundle_string`]: the single-file bundle as a string.
///
/// Unlike [`save_qodec`] there is nowhere to put a sidecar, so a source
/// circuit that cannot be inlined is an error rather than a second file.
pub(super) fn bundle_string(qodec: &Qodec) -> std::io::Result<String> {
    let (artifacts, manifest) = prepare(qodec, None)?;
    let (inlined_gadgets, inlined) = inline_gadget_sources(&artifacts.gadgets, &artifacts.source_files);

    if let Some(stranded) = artifacts.source_files.keys().find(|key| !inlined.contains(*key)) {
        return Err(std::io::Error::other(format!(
            "cannot represent this qodec as a string: the source circuit {} would have to be written \
             as a separate file. Save it with `save` instead, or give its circuit a known `format` \
             so it can be inlined.",
            stranded.display(),
        )));
    }

    build_single_file_bundle(&artifacts, &inlined_gadgets, &manifest, &qodec.manifest_filename)
}

/// Boundary encoding declarations and circuit operand-type maps.
type RealizationFields = (
    Vec<EncodingSpec>,
    Vec<EncodingSpec>,
    BTreeMap<String, String>,
    BTreeMap<String, String>,
);

fn elide_inferred_types(
    inputs: &mut BTreeMap<String, String>,
    outputs: &mut BTreeMap<String, String>,
    target: &InstructionSet,
) {
    if let [block] = target.blocks.as_slice()
        && inputs.values().chain(outputs.values()).all(|kind| kind == &block.name)
    {
        inputs.clear();
        outputs.clear();
    }
}

fn build_raw_realization(
    resolved_gadget: &resolved::Gadget,
    code_paths: &BTreeMap<String, PathBuf>,
) -> Result<RealizationFields, SynthesisError> {
    let mut circuit_in: BTreeMap<String, String> = BTreeMap::new();
    let mut circuit_out: BTreeMap<String, String> = BTreeMap::new();
    let inputs = build_encoding_specs(
        &resolved_gadget.inputs,
        &resolved_gadget.implements.inputs,
        code_paths,
        &resolved_gadget.circuit.instruction_set,
        &mut circuit_in,
    )?;
    let outputs = build_encoding_specs(
        &resolved_gadget.outputs,
        &resolved_gadget.implements.outputs,
        code_paths,
        &resolved_gadget.circuit.instruction_set,
        &mut circuit_out,
    )?;
    elide_inferred_types(
        &mut circuit_in,
        &mut circuit_out,
        &resolved_gadget.circuit.instruction_set,
    );
    Ok((inputs, outputs, circuit_in, circuit_out))
}

fn build_encoding_specs(
    encodings: &[resolved::Encoding],
    operands: &[crate::BlockOperand],
    code_paths: &BTreeMap<String, PathBuf>,
    source_instruction_set: &Arc<InstructionSet>,
    circuit_side_map: &mut BTreeMap<String, String>,
) -> Result<Vec<EncodingSpec>, SynthesisError> {
    encodings
        .iter()
        .enumerate()
        .map(|(entry, enc)| {
            // The caller needs this code to emit the layer's binding.
            if !code_paths.contains_key(&enc.code.name) {
                return Err(SynthesisError::CodeMissing {
                    name: enc.code.name.clone(),
                });
            }
            enc.record_block_types(source_instruction_set, circuit_side_map)
                .map_err(|error| SynthesisError::InvalidEncoding { entry, error })?;
            let circuit_support = enc.support.iter().cloned().map(crate::BlockName::new).collect();
            // The entry is keyed on disk by the block type the instruction
            // declares at this position. Fall back to the realization instruction set's
            // sole block type only when the operand is untyped.
            let block_type = match operands.get(entry) {
                Some(operand) if !operand.block.is_empty() => operand.block.clone(),
                _ => match source_instruction_set.blocks.as_slice() {
                    [single] => single.name.clone(),
                    _ => String::new(),
                },
            };
            Ok(EncodingSpec {
                block_type,
                support: circuit_support,
            })
        })
        .collect()
}
