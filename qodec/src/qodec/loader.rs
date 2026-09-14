//! Load the manifest and follow its artifact references.
//!
//! Reference fields select artifact types. Bundle entries supply content in
//! place of files; unreferenced content is ignored.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;

use super::LoadError;
use super::resolver::{normalize_relative, resolve_relative};
use crate::Code;
use crate::GadgetSpec;
use crate::InstructionSet;
use crate::Manifest;
use crate::{ParityEquation, ReadoutsList};

#[cfg(test)]
#[path = "../../tests/unit/schemas_test.rs"]
mod schemas_test;

pub(crate) fn read_yaml<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, LoadError> {
    let content = std::fs::read_to_string(path)?;
    serde_yaml::from_str(&content).map_err(|source| LoadError::Yaml {
        path: path.to_path_buf(),
        source: crate::ParseError::new(source),
    })
}

/// Write one artifact to its own YAML file, checking it first.
///
/// Backs the standalone `save` on [`crate::Code`] and
/// [`crate::InstructionSet`], whose fields are public and so can be left
/// illegal between construction and save.
pub(crate) fn save_artifact<T: serde::Serialize>(
    artifact: &T,
    path: &Path,
    check: impl FnOnce(&T) -> Result<(), String>,
) -> std::io::Result<()> {
    check(artifact).map_err(std::io::Error::other)?;
    let yaml = crate::yaml_output::to_string(artifact)?;
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, yaml)
}

/// Deserialize an already-parsed YAML value into a concrete artifact type,
/// attributing any error to `relative` (the artifact's qodec-root-relative
/// path). Used when ingesting documents that were split from a single-file
/// bundle, where there is no per-artifact file to read.
pub(super) fn from_value<T: serde::de::DeserializeOwned>(
    relative: &Path,
    value: serde_yaml::Value,
) -> Result<T, LoadError> {
    serde_yaml::from_value(value).map_err(|source| LoadError::Yaml {
        path: relative.to_path_buf(),
        source: crate::ParseError::new(source),
    })
}

/// If `value` is a single-key mapping `{key: body}` whose key is a string,
/// return the key and body. This is the envelope shape used by every
/// document in a single-file qodec bundle.
fn into_envelope(value: serde_yaml::Value) -> Option<(String, serde_yaml::Value)> {
    let serde_yaml::Value::Mapping(map) = value else {
        return None;
    };
    if map.len() != 1 {
        return None;
    }
    match map.into_iter().next()? {
        (serde_yaml::Value::String(key), body) => Some((key, body)),
        _ => None,
    }
}

/// Parse the located manifest file as a possible single-file qodec bundle: a
/// multi-document YAML stream where each document is a single-key envelope
/// `{relative/path.yaml: body}`.
///
/// Returns `Ok(Some(documents))` with one `(relative-path, body)` pair per
/// document when the file is a bundle, or `Ok(None)` when it is a
/// conventional single-document manifest (directory mode). A file is treated
/// as a bundle when it holds more than one document, or when its lone
/// document is a single-key envelope containing a manifest mapping.
/// `origin` names the source in
/// error messages only; it is never read.
pub(super) fn parse_bundle_str(
    content: &str,
    origin: &Path,
) -> Result<Option<Vec<(PathBuf, serde_yaml::Value)>>, LoadError> {
    let manifest_path = origin;
    let mut documents: Vec<serde_yaml::Value> = Vec::new();
    for document in serde_yaml::Deserializer::from_str(content) {
        let value = serde_yaml::Value::deserialize(document).map_err(|source| LoadError::Yaml {
            path: manifest_path.to_path_buf(),
            source: crate::ParseError::new(source),
        })?;
        if value.is_null() {
            continue;
        }
        documents.push(value);
    }

    let is_bundle = documents.len() > 1
        || documents
            .first()
            .and_then(serde_yaml::Value::as_mapping)
            .filter(|mapping| mapping.len() == 1)
            .and_then(|mapping| mapping.values().next())
            .and_then(serde_yaml::Value::as_mapping)
            .is_some_and(|mapping| mapping.contains_key("layers"));
    if !is_bundle {
        return Ok(None);
    }

    let mut out: Vec<(PathBuf, serde_yaml::Value)> = Vec::new();
    let mut seen: BTreeSet<PathBuf> = BTreeSet::new();
    for value in documents {
        let (key, body) = into_envelope(value).ok_or_else(|| LoadError::MalformedBundle {
            manifest: manifest_path.to_path_buf(),
            reason: "every document must be a single-key `{path: body}` mapping".to_owned(),
        })?;
        let relative = PathBuf::from(normalize_relative(&key));
        if !seen.insert(relative.clone()) {
            return Err(LoadError::MalformedBundle {
                manifest: manifest_path.to_path_buf(),
                reason: format!("duplicate document key '{key}'"),
            });
        }
        out.push((relative, body));
    }
    Ok(Some(out))
}

/// Verify the manifest's `schema_version`, if declared, equals this
/// loader's [`crate::CURRENT_SCHEMA_VERSION`]. Absent values
/// are accepted as the loader's current version.
pub(super) fn check_schema_version(manifest: &Manifest, manifest_path: &Path) -> Result<(), LoadError> {
    let Some(declared) = manifest.schema_version else {
        return Ok(());
    };
    let supported = crate::CURRENT_SCHEMA_VERSION;
    if declared != supported {
        return Err(LoadError::UnsupportedSchemaVersion {
            manifest: manifest_path.to_path_buf(),
            declared,
            supported,
        });
    }
    Ok(())
}

/// Fetch a referenced check or readout document from the loaded artifact map.
pub(super) fn take_sidecar(
    sidecars: &BTreeMap<PathBuf, serde_yaml::Value>,
    gadget_path: &Path,
    resolved: &Path,
) -> Result<serde_yaml::Value, LoadError> {
    sidecars
        .get(resolved)
        .cloned()
        .ok_or_else(|| LoadError::MissingArtifact {
            referenced_from: gadget_path.to_path_buf(),
            path: resolved.to_path_buf(),
        })
}

/// The parsed manifest, the filename it came from (preserved for lossless
/// save), and the bundle's artifact documents (`Some`) or `None` for a plain
/// directory. Returned by [`load_manifest`].
pub(super) type LoadedManifest = (Manifest, String, Option<Vec<(PathBuf, serde_yaml::Value)>>);
type LocatedManifest = (
    Manifest,
    String,
    Option<Vec<(PathBuf, serde_yaml::Value)>>,
    BTreeMap<PathBuf, crate::node::source::Document>,
);

/// Read the manifest and, for a single-file bundle, split out its artifact
/// documents.
///
/// Returns the parsed manifest, the filename it came from (preserved for
/// lossless save), and `Some(artifacts)` for a bundle or `None` for a plain
/// manifest (whose referenced artifacts are read by [`ingest_artifacts`]).
pub(super) fn load_manifest(manifest_path: &Path, visit: DocumentVisitor<'_>) -> Result<LocatedManifest, LoadError> {
    // A qodec is either a directory of individual artifact files or a
    // single-file bundle: a multi-document YAML stream whose documents are
    // `{relative/path.yaml: body}` envelopes. `parse_bundle` returns the split
    // documents for a bundle, or `None` for a plain directory.
    let content = std::fs::read_to_string(manifest_path).map_err(|source| {
        LoadError::Io(std::io::Error::new(
            source.kind(),
            format!("failed to read manifest {}: {source}", manifest_path.display()),
        ))
    })?;
    let origins = crate::node::source::parse(&content, manifest_path);
    let Some(documents) = parse_bundle_str(&content, manifest_path)? else {
        let manifest: Manifest = serde_yaml::from_str(&content).map_err(|source| LoadError::Yaml {
            path: manifest_path.to_path_buf(),
            source: crate::ParseError::new(source),
        })?;
        let filename = manifest_path
            .file_name()
            .map_or_else(|| "qodec.yaml".to_owned(), |name| name.to_string_lossy().into_owned());
        visit(Path::new(&filename), "manifest", &Document::Text(content));
        let origins = origins
            .into_iter()
            .next()
            .map(|document| (PathBuf::from(&filename), document))
            .into_iter()
            .collect();
        return Ok((manifest, filename, None, origins));
    };
    let origins = origins
        .into_iter()
        .filter_map(crate::node::source::Document::envelope)
        .collect();
    let (manifest, filename, documents) = split_bundle_documents(documents, manifest_path, visit)?;
    Ok((manifest, filename, documents, origins))
}

/// Split a bundle's documents into its manifest and its artifacts.
///
/// The first document is the manifest. Remaining entries are resolved by path
/// when the manifest or a gadget references them; keys do not encode types.
/// `origin` names the source in error messages only; it is never read.
pub(super) fn split_bundle_documents(
    documents: Vec<(PathBuf, serde_yaml::Value)>,
    origin: &Path,
    visit: DocumentVisitor<'_>,
) -> Result<LoadedManifest, LoadError> {
    let mut documents = documents.into_iter();
    let (path, value) = documents.next().ok_or_else(|| LoadError::MalformedBundle {
        manifest: origin.to_path_buf(),
        reason: "bundle has no manifest document".to_owned(),
    })?;
    let document = Document::Parsed(value);
    visit(&path, "manifest", &document);
    let manifest: Manifest = document.typed(&path).map_err(|error| LoadError::MalformedBundle {
        manifest: origin.to_path_buf(),
        reason: format!("first bundle document must be the manifest: {error}"),
    })?;
    Ok((manifest, path.to_string_lossy().into_owned(), Some(documents.collect())))
}

/// The raw, per-file artifacts deserialized from a qodec, keyed by
/// qodec-root-relative path. External check / readout / flag documents stay
/// raw in `sidecar_documents` until a gadget references them.
pub(super) struct RawArtifacts {
    pub(super) instruction_sets: BTreeMap<PathBuf, InstructionSet>,
    pub(super) codes: BTreeMap<PathBuf, Code>,
    pub(super) gadgets: BTreeMap<PathBuf, GadgetSpec>,
    pub(super) sidecar_documents: BTreeMap<PathBuf, serde_yaml::Value>,
    pub(super) source_files: BTreeMap<PathBuf, String>,
    pub(super) origins: BTreeMap<PathBuf, crate::node::source::Document>,
    pub(super) root: PathBuf,
    pub(super) external_files: BTreeMap<PathBuf, ExternalFile>,
}

#[derive(Debug, Clone)]
pub(super) struct ExternalFile {
    pub(super) path: PathBuf,
    text: String,
    is_source: bool,
}

impl ExternalFile {
    pub(super) fn matches<T: serde::Serialize + serde::de::DeserializeOwned + PartialEq>(&self, value: &T) -> bool {
        if self.is_source {
            serde_yaml::to_value(value).is_ok_and(|value| value.as_str() == Some(self.text.as_str()))
        } else {
            serde_yaml::from_str::<T>(&self.text).is_ok_and(|original| &original == value)
        }
    }

    pub(super) fn gadget(&self) -> Option<GadgetSpec> {
        serde_yaml::from_str(&self.text).ok()
    }

    pub(super) fn check_unchanged(&self) -> std::io::Result<()> {
        let current = std::fs::read_to_string(&self.path).map_err(|error| {
            std::io::Error::new(
                error.kind(),
                format!("cannot reuse external artifact {}: {error}", self.path.display()),
            )
        })?;
        if current != self.text {
            return Err(std::io::Error::other(format!(
                "external artifact {} changed since loading",
                self.path.display()
            )));
        }
        Ok(())
    }
}

/// One artifact document, either already parsed (a bundle splits a single file
/// into values) or still as text (a directory reads one file per artifact).
///
/// The distinction is worth keeping: `serde_yaml` reports the field path and
/// line only when it deserializes from text. Going via a `Value` first costs
/// both, turning "instructions[1]: missing field `description` at line 10" into
/// a bare "missing field `description`".
pub(super) enum Document {
    Parsed(serde_yaml::Value),
    Text(String),
}

type DocumentVisitor<'a> = &'a mut dyn FnMut(&Path, &'static str, &Document);

struct ArtifactFiles<'a> {
    root: &'a Path,
    documents: BTreeMap<PathBuf, serde_yaml::Value>,
    kinds: BTreeMap<PathBuf, &'static str>,
    origins: BTreeMap<PathBuf, crate::node::source::Document>,
    external_files: BTreeMap<PathBuf, ExternalFile>,
    visit: DocumentVisitor<'a>,
}

impl ArtifactFiles<'_> {
    fn require_kind(&mut self, path: &Path, referrer: &Path, kind: &'static str) -> Result<(), LoadError> {
        let absolute = self.root.join(path);
        let identity = PathBuf::from(normalize_relative(&absolute.to_string_lossy()));
        let previous = self.kinds.entry(identity).or_insert(kind);
        if *previous != kind {
            return Err(LoadError::ConflictingArtifactKind {
                referenced_from: referrer.to_path_buf(),
                path: path.to_path_buf(),
                kind,
                previous,
            });
        }
        Ok(())
    }

    fn read(&mut self, relative: &Path, referrer: &Path, kind: &'static str) -> Result<Document, LoadError> {
        self.require_kind(relative, referrer, kind)?;
        if let Some(value) = self.documents.get(relative) {
            let document = Document::Parsed(value.clone());
            (self.visit)(relative, kind, &document);
            return Ok(document);
        }
        let absolute = std::path::absolute(self.root.join(relative))?;
        let absolute = PathBuf::from(normalize_relative(&absolute.to_string_lossy()));
        let text = std::fs::read_to_string(&absolute).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                LoadError::MissingArtifact {
                    referenced_from: referrer.to_path_buf(),
                    path: relative.to_path_buf(),
                }
            } else {
                LoadError::Io(std::io::Error::new(
                    error.kind(),
                    format!("failed to read {}: {error}", absolute.display()),
                ))
            }
        })?;
        if !absolute.starts_with(self.root) {
            self.external_files.insert(
                relative.to_owned(),
                ExternalFile {
                    path: absolute.clone(),
                    text: text.clone(),
                    is_source: kind == "circuit source",
                },
            );
        }
        let origin = if kind == "circuit source" {
            Some(crate::node::source::Document::text(&absolute))
        } else {
            crate::node::source::parse(&text, &absolute).into_iter().next()
        };
        if let Some(origin) = origin {
            self.origins.insert(relative.to_owned(), origin);
        }
        let document = Document::Text(text);
        (self.visit)(relative, kind, &document);
        Ok(document)
    }
}

impl Document {
    fn typed<T: serde::de::DeserializeOwned>(self, relative: &Path) -> Result<T, LoadError> {
        match self {
            Self::Text(text) => serde_yaml::from_str(&text).map_err(|source| LoadError::Yaml {
                path: relative.to_path_buf(),
                source: crate::ParseError::new(source),
            }),
            Self::Parsed(value) => from_value(relative, value),
        }
    }

    fn into_value(self, relative: &Path) -> Result<serde_yaml::Value, LoadError> {
        match self {
            Self::Text(text) => serde_yaml::from_str(&text).map_err(|source| LoadError::Yaml {
                path: relative.to_path_buf(),
                source: crate::ParseError::new(source),
            }),
            Self::Parsed(value) => Ok(value),
        }
    }

    fn into_source(self, relative: &Path) -> Result<String, LoadError> {
        match self {
            Self::Text(text) | Self::Parsed(serde_yaml::Value::String(text)) => Ok(text),
            Self::Parsed(value) => serde_yaml::to_string(&value).map_err(|source| LoadError::Yaml {
                path: relative.to_path_buf(),
                source: crate::ParseError::new(source),
            }),
        }
    }
}

/// Load artifacts named by the manifest and its gadgets, using the referring
/// field to select the type. Unreferenced files and bundle entries are not read.
pub(super) fn ingest_artifacts(
    root: &Path,
    manifest: &Manifest,
    manifest_filename: &str,
    bundle_artifacts: Option<Vec<(PathBuf, serde_yaml::Value)>>,
    origins: BTreeMap<PathBuf, crate::node::source::Document>,
    visit: DocumentVisitor<'_>,
) -> Result<(Manifest, RawArtifacts), LoadError> {
    let mut manifest = manifest.clone();
    for layer in &mut manifest.layers {
        for path in std::iter::once(&mut layer.instruction_set)
            .chain(layer.codes.values_mut())
            .chain(layer.gadgets.values_mut())
        {
            *path = resolve_relative(Path::new(manifest_filename), path)
                .to_string_lossy()
                .into_owned();
        }
    }
    let mut raw = RawArtifacts {
        instruction_sets: BTreeMap::new(),
        codes: BTreeMap::new(),
        gadgets: BTreeMap::new(),
        sidecar_documents: BTreeMap::new(),
        source_files: BTreeMap::new(),
        origins: BTreeMap::new(),
        root: PathBuf::new(),
        external_files: BTreeMap::new(),
    };

    let referrer = Path::new(manifest_filename);
    let file_root = if root.is_absolute() {
        root.to_path_buf()
    } else {
        std::env::current_dir()?.join(root)
    };
    let file_root = PathBuf::from(normalize_relative(&file_root.to_string_lossy()));
    let mut files = ArtifactFiles {
        root: &file_root,
        documents: bundle_artifacts.unwrap_or_default().into_iter().collect(),
        kinds: BTreeMap::new(),
        origins,
        external_files: BTreeMap::new(),
        visit,
    };
    files.require_kind(referrer, referrer, "manifest")?;
    for layer in &manifest.layers {
        let relative = PathBuf::from(normalize_relative(&layer.instruction_set));
        if let std::collections::btree_map::Entry::Vacant(entry) = raw.instruction_sets.entry(relative) {
            let instruction_set = files
                .read(entry.key(), referrer, "instruction set")?
                .typed(entry.key())?;
            entry.insert(instruction_set);
        }
        for path in layer.codes.values() {
            let relative = PathBuf::from(normalize_relative(path));
            if let std::collections::btree_map::Entry::Vacant(entry) = raw.codes.entry(relative) {
                let code = files.read(entry.key(), referrer, "code")?.typed(entry.key())?;
                entry.insert(code);
            }
        }
        for path in layer.gadgets.values() {
            let relative = PathBuf::from(normalize_relative(path));
            if let std::collections::btree_map::Entry::Vacant(entry) = raw.gadgets.entry(relative) {
                let gadget = files.read(entry.key(), referrer, "gadget")?.typed(entry.key())?;
                entry.insert(gadget);
            }
        }
    }
    for (path, gadget) in &raw.gadgets {
        for (kind, reference) in [
            ("check list", gadget.checks.path()),
            ("readout list", gadget.readouts.path()),
        ] {
            let Some(reference) = reference else { continue };
            let relative = resolve_relative(path, reference);
            files.require_kind(&relative, path, kind)?;
            if let std::collections::btree_map::Entry::Vacant(entry) = raw.sidecar_documents.entry(relative) {
                let value = files.read(entry.key(), path, kind)?.into_value(entry.key())?;
                entry.insert(value);
            }
        }
        if let Some(reference) = gadget.circuit.source.path() {
            let relative = resolve_relative(path, reference);
            if let std::collections::btree_map::Entry::Vacant(entry) = raw.source_files.entry(relative) {
                let source = files
                    .read(entry.key(), path, "circuit source")?
                    .into_source(entry.key())?;
                entry.insert(source);
            }
        }
    }

    raw.origins = files.origins;
    raw.external_files = files.external_files;
    raw.root = file_root;
    Ok((manifest, raw))
}

/// The external check / readout lists a gadget names by path, resolved
/// into typed maps keyed by qodec-root-relative path.
pub(super) struct SidecarLists {
    pub(super) check_lists: BTreeMap<PathBuf, Vec<ParityEquation>>,
    pub(super) readout_lists: BTreeMap<PathBuf, ReadoutsList>,
}

/// Resolve every external check / readout file that a gadget names
/// explicitly. Sidecars are identified by the gadget's `Sourced::File` path —
/// there is no auto-detection by extension. A referenced sidecar that is
/// absent is a load error; unreferenced sidecars are ignored.
pub(super) fn resolve_sidecars(
    gadgets: &BTreeMap<PathBuf, GadgetSpec>,
    sidecar_documents: &BTreeMap<PathBuf, serde_yaml::Value>,
) -> Result<SidecarLists, LoadError> {
    let mut lists = SidecarLists {
        check_lists: BTreeMap::new(),
        readout_lists: BTreeMap::new(),
    };
    for (gadget_path, gadget) in gadgets {
        if let Some(rel) = gadget.checks.path() {
            let resolved = resolve_relative(gadget_path, rel);
            let value = take_sidecar(sidecar_documents, gadget_path, &resolved)?;
            lists
                .check_lists
                .insert(resolved, from_value::<Vec<ParityEquation>>(gadget_path, value)?);
        }
        if let Some(rel) = gadget.readouts.path() {
            let resolved = resolve_relative(gadget_path, rel);
            let value = take_sidecar(sidecar_documents, gadget_path, &resolved)?;
            lists
                .readout_lists
                .insert(resolved, from_value::<ReadoutsList>(gadget_path, value)?);
        }
    }
    Ok(lists)
}

/// Resolve instruction set artifacts into shared `Arc` cells keyed by file path, after
/// verifying that no two files declare the same instruction set `name` (a collision would
/// silently shadow one file in the name-keyed lookups downstream).
pub(super) fn resolve_instruction_sets_by_path(
    instruction_sets: &BTreeMap<PathBuf, InstructionSet>,
) -> Result<BTreeMap<PathBuf, Arc<InstructionSet>>, LoadError> {
    let mut origin: BTreeMap<&str, &Path> = BTreeMap::new();
    let mut resolved = BTreeMap::new();
    for (path, instruction_set) in instruction_sets {
        if let Some(first) = origin.insert(instruction_set.name.as_str(), path.as_path()) {
            return Err(LoadError::DuplicateInstructionSetName {
                name: instruction_set.name.clone(),
                first: first.to_path_buf(),
                second: path.clone(),
            });
        }
        resolved.insert(path.clone(), Arc::new(instruction_set.clone()));
    }
    Ok(resolved)
}

/// Resolve code artifacts by file path after rejecting duplicate code names.
pub(super) fn resolve_codes_by_path(
    codes: &BTreeMap<PathBuf, Code>,
) -> Result<BTreeMap<PathBuf, Arc<Code>>, LoadError> {
    let mut origin: BTreeMap<&str, &Path> = BTreeMap::new();
    let mut resolved = BTreeMap::new();
    for (path, code) in codes {
        if let Some(first) = origin.insert(code.name.as_str(), path.as_path()) {
            return Err(LoadError::DuplicateCodeName {
                name: code.name.clone(),
                first: first.to_path_buf(),
                second: path.clone(),
            });
        }
        resolved.insert(path.clone(), Arc::new(code.clone()));
    }
    Ok(resolved)
}

#[cfg(test)]
mod location_tests {
    /// A directory qodec must report *where* a bad artifact went wrong.
    /// Deserializing through a `serde_yaml::Value` drops the field path and the
    /// line, which is what this guards against.
    #[test]
    fn artifact_error_names_the_field_and_line() {
        let directory = tempfile::tempdir().expect("create fixture directory");
        let manifest = directory.path().join("q.qodec.yaml");
        std::fs::write(&manifest, "layers: [{instruction_set: ./unused/../bad.isa.yaml}]\n").unwrap();
        let instruction_set = [
            "name: T",
            "description: d",
            "blocks: {q: 1}",
            "instructions:",
            "  - mnemonic: A",
            "    description: fine",
            "    in: [q]",
            "    out: [q]",
            "    action: []",
            "  - mnemonic: B",
            "    in: [q]",
            "    out: [q]",
            "    action: []",
            "",
        ]
        .join("\n");
        std::fs::write(directory.path().join("bad.isa.yaml"), instruction_set).unwrap();

        let error = crate::Qodec::load(manifest).expect_err("instruction B has no description");

        let crate::LoadError::Yaml { path, source } = error else {
            panic!("expected YAML error, got {error:?}");
        };
        assert_eq!(path, std::path::Path::new("bad.isa.yaml"));
        let error = source.to_string();
        assert!(error.contains("missing field `description`"), "{error}");
        assert!(error.contains("instructions[1]"), "{error}");
        assert!(error.contains("line 10"), "{error}");
    }

    #[test]
    fn missing_artifact_paths_are_normalized_for_plain_and_bundled_manifests() {
        for content in [
            "layers:\n  - instruction_set: ./unused/../absent\n  - instruction_set: physical\n",
            "./draft/../entry:\n  layers:\n    - instruction_set: ./unused/../absent\n    - instruction_set: physical\n",
        ] {
            let directory = tempfile::tempdir().expect("temp dir");
            let manifest = directory.path().join("entry");
            std::fs::write(&manifest, content).expect("write manifest");

            let error = crate::Qodec::load(&manifest).expect_err("referenced instruction set is missing");
            let crate::LoadError::MissingArtifact { referenced_from, path } = error else {
                panic!("expected missing artifact, got {error:?}");
            };
            assert_eq!(referenced_from, std::path::Path::new("entry"));
            assert_eq!(path, std::path::Path::new("absent"));
        }
    }

    #[test]
    fn a_bundle_reports_missing_fields_in_its_first_document() {
        let error =
            crate::Qodec::from_bundle_str("./draft/../entry: {name: missing-layers}\n---\nqodec.yaml: {layers: []}\n")
                .expect_err("the first document must be a manifest");
        let crate::LoadError::MalformedBundle { manifest, reason } = error else {
            panic!("expected malformed bundle, got {error:?}");
        };
        assert_eq!(manifest, std::path::Path::new("<bundle>"));
        assert!(
            reason.contains("first bundle document must be the manifest"),
            "{reason}"
        );
        assert!(reason.contains("YAML error in entry:"), "{reason}");
        assert!(reason.contains("missing field `layers`"), "{reason}");
    }

    /// `.yml` is accepted wherever `.yaml` is, for both the manifest and the
    /// artifacts it references.
    #[test]
    fn a_directory_spelled_yml_loads() {
        let directory = tempfile::tempdir().expect("create fixture directory");
        let files: &[(&str, &[&str])] = &[
            (
                "qodec.yml",
                &[
                    "name: p",
                    "description: d",
                    "layers:",
                    "  - instruction_set: top.isa.yml",
                    "  - instruction_set: phys.isa.yml",
                ],
            ),
            (
                "top.isa.yml",
                &[
                    "name: top",
                    "description: d",
                    "blocks: {b: 1}",
                    "instructions:",
                    "  - mnemonic: idle",
                    "    description: d",
                    "    in: [b]",
                    "    action: []",
                ],
            ),
            (
                "phys.isa.yml",
                &[
                    "name: phys",
                    "description: d",
                    "blocks: {q: 1}",
                    "instructions:",
                    "  - mnemonic: R",
                    "    description: d",
                    "    in: [q]",
                    "    action: []",
                ],
            ),
        ];
        for (filename, lines) in files {
            std::fs::write(directory.path().join(filename), lines.join("\n")).expect("write fixture");
        }

        let qodec = crate::Qodec::load(directory.path().join("qodec.yml")).expect("a .yml qodec should load");

        assert_eq!(
            qodec
                .layers()
                .iter()
                .map(|layer| layer.instruction_set.name.as_str())
                .collect::<Vec<_>>(),
            ["top", "phys"]
        );
    }
}
