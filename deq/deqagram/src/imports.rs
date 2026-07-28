//! IMPORT resolution: an opt-in transform that flattens a file's `IMPORT`
//! statements by parsing the referenced files and merging their definitions.
//!
//! Parsing itself is I/O-free: `str::parse::<DeqFile>()` retains `IMPORT` paths
//! as data in [`DeqFile::imports`](crate::ast::DeqFile::imports) and never
//! touches the filesystem. This module is the separate, opt-in step that reads
//! and inlines them.
//!
//! # Semantics
//!
//! Resolution matches deq's reference behaviour:
//!
//! * **Relative:** each import path is resolved relative to the file that
//!   contains the `IMPORT`.
//! * **Include-guard by canonical id:** every file is loaded at most once, keyed
//!   by the loader's canonical id. A re-visited file (including an import cycle)
//!   is silently skipped — this is **not** an error.
//! * **Depth-first, imports first:** a file's imported definitions appear in the
//!   merged output before its own, in import order.
//!
//! # Provenance
//!
//! The merged definitions come from different files, and each definition's
//! [`Span`](crate::Span) is a byte offset into *its own* file's text. To locate
//! a definition you therefore need both its span and which file it came from, so
//! [`ResolvedDeqFile`] pairs the merged definitions with a parallel `sources`
//! table ([`FileId`] per definition) and a `files` table mapping each
//! [`FileId`] to the loader id it was read from. The parsed AST is left
//! unmodified — provenance is a side-table, not a field on the nodes.
//!
//! # I/O abstraction
//!
//! Reading files is delegated to an [`ImportLoader`], so the core is testable
//! without a filesystem and usable over virtual/in-memory sources. [`FsLoader`]
//! is the [`std::fs`] implementation, and [`parse_file`] is the one-shot
//! convenience over it.

use std::fmt;
use std::path::{Path, PathBuf};

use crate::Spanned;
use crate::ast::{Definition, DeqFile};

/// Index of a source file within a [`ResolvedDeqFile`]'s `files` table.
pub type FileId = usize;

/// A `.deq` file with all `IMPORT`s recursively resolved and inlined.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ResolvedDeqFile {
    /// Merged definitions: each imported file's definitions (depth-first, imports
    /// first) followed by the importing file's own.
    pub definitions: Vec<Spanned<Definition>>,
    /// Parallel to `definitions`: the file each definition was parsed from.
    pub sources: Vec<FileId>,
    /// Maps a [`FileId`] to the loader id (e.g. canonical path) it was read from,
    /// in the order files were first loaded (the root is `FileId` 0).
    pub files: Vec<String>,
}

impl ResolvedDeqFile {
    /// The loader id (e.g. canonical path) definition `index` came from.
    #[must_use]
    pub fn source_of(&self, index: usize) -> &str {
        &self.files[self.sources[index]]
    }

    /// Discards provenance, returning a flat [`DeqFile`] with no `imports`.
    #[must_use]
    pub fn into_deq_file(self) -> DeqFile {
        DeqFile {
            imports: Vec::new(),
            definitions: self.definitions,
        }
    }
}

/// Supplies file contents to the resolver, abstracting away the filesystem.
///
/// Implement this to resolve imports over an in-memory map, an archive, a
/// sandbox, etc. [`FsLoader`] is the [`std::fs`]-backed implementation.
pub trait ImportLoader {
    /// Resolves the import `path` (as written in an `IMPORT`, e.g. `"code.deq"`)
    /// relative to `base` — the id of the file containing the import, or `None`
    /// for the root file — into a stable, canonical id.
    ///
    /// The returned id is used both as the include-guard key and as the argument
    /// to [`load`](ImportLoader::load), so two paths denoting the same file must
    /// return equal ids.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError`] if the path cannot be resolved (e.g. it does not
    /// exist).
    fn resolve(&mut self, base: Option<&str>, path: &str) -> Result<String, LoadError>;

    /// Reads the full text of the file with the given canonical `id`.
    ///
    /// # Errors
    ///
    /// Returns [`LoadError`] if the file cannot be read.
    fn load(&mut self, id: &str) -> Result<String, LoadError>;
}

/// An error from an [`ImportLoader`] (a path that cannot be resolved or read).
#[derive(Debug, Clone)]
pub struct LoadError {
    pub path: String,
    pub message: String,
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "cannot load {:?}: {}", self.path, self.message)
    }
}

impl std::error::Error for LoadError {}

/// An error while resolving imports: a load failure or a parse failure (tagged
/// with the offending file).
#[derive(Debug)]
pub enum ResolveError {
    Load(LoadError),
    Parse {
        file: String,
        // Boxed: `ParseError` wraps pest's large `Error<Rule>`, and boxing keeps
        // the `Result<_, ResolveError>` returned throughout this module small.
        error: Box<crate::ParseError>,
    },
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResolveError::Load(e) => write!(f, "{e}"),
            ResolveError::Parse { file, error } => write!(f, "{file}: {error}"),
        }
    }
}

impl std::error::Error for ResolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ResolveError::Load(e) => Some(e),
            ResolveError::Parse { error, .. } => Some(error),
        }
    }
}

impl From<LoadError> for ResolveError {
    fn from(e: LoadError) -> Self {
        ResolveError::Load(e)
    }
}

/// Resolves and inlines the imports of the file identified by `root_id`, reading
/// every file through `loader`.
///
/// ```
/// use deqagram::imports::{resolve, ImportLoader, LoadError};
///
/// // A tiny in-memory loader (no filesystem): ids are already canonical.
/// struct Memory;
/// impl ImportLoader for Memory {
///     fn resolve(&mut self, _base: Option<&str>, path: &str) -> Result<String, LoadError> {
///         Ok(path.to_string())
///     }
///     fn load(&mut self, id: &str) -> Result<String, LoadError> {
///         Ok(match id {
///             "root" => "IMPORT \"lib\"\nGADGET Main {\n    R 0\n}\n",
///             "lib" => "CODE Lib [[1,1]] {\n    STABILIZER X0\n}\n",
///             other => return Err(LoadError { path: other.to_string(), message: "not found".into() }),
///         }
///         .to_string())
///     }
/// }
///
/// let resolved = resolve("root", &mut Memory).unwrap();
/// // Imported definitions come first, then the importer's: [Lib, Main].
/// assert_eq!(resolved.definitions.len(), 2);
/// assert_eq!(resolved.source_of(0), "lib");
/// ```
///
/// # Errors
///
/// Returns [`ResolveError`] if any file fails to load or parse. Import cycles
/// are *not* an error (a re-visited file is skipped).
pub fn resolve(root_id: &str, loader: &mut impl ImportLoader) -> Result<ResolvedDeqFile, ResolveError> {
    let mut resolved = ResolvedDeqFile::default();
    resolve_into(root_id, loader, &mut resolved)?;
    Ok(resolved)
}

fn resolve_into(id: &str, loader: &mut impl ImportLoader, resolved: &mut ResolvedDeqFile) -> Result<(), ResolveError> {
    // Include-guard: a file already seen (or a cycle) is silently skipped.
    // Linear scan — import trees are tiny; switch to a map if needed.
    if resolved.files.iter().any(|f| f == id) {
        return Ok(());
    }
    let file_id: FileId = resolved.files.len();
    resolved.files.push(id.to_string());

    let text = loader.load(id)?;
    let parsed: DeqFile = text.parse().map_err(|error| ResolveError::Parse {
        file: id.to_string(),
        error: Box::new(error),
    })?;

    // Depth-first: resolve imports before adding this file's own definitions.
    for import in &parsed.imports {
        let child = loader.resolve(Some(id), import)?;
        resolve_into(&child, loader, resolved)?;
    }

    for definition in parsed.definitions {
        resolved.definitions.push(definition);
        resolved.sources.push(file_id);
    }
    Ok(())
}

/// An [`ImportLoader`] backed by the local filesystem.
///
/// Import paths are resolved relative to the importing file's directory and
/// canonicalized (via [`std::fs::canonicalize`]) so the include-guard dedups
/// files reached by different paths.
#[derive(Debug, Default, Clone, Copy)]
pub struct FsLoader;

impl ImportLoader for FsLoader {
    fn resolve(&mut self, base: Option<&str>, path: &str) -> Result<String, LoadError> {
        let target: PathBuf = match base {
            Some(base) => Path::new(base).parent().unwrap_or_else(|| Path::new(".")).join(path),
            None => PathBuf::from(path),
        };
        target
            .canonicalize()
            .map(|p| p.to_string_lossy().into_owned())
            .map_err(|e| LoadError {
                path: target.to_string_lossy().into_owned(),
                message: e.to_string(),
            })
    }

    fn load(&mut self, id: &str) -> Result<String, LoadError> {
        std::fs::read_to_string(id).map_err(|e| LoadError {
            path: id.to_string(),
            message: e.to_string(),
        })
    }
}

/// Parses `path` and resolves its imports from the filesystem in one call.
///
/// Convenience over [`resolve`] with a [`FsLoader`].
///
/// # Errors
///
/// Returns [`ResolveError`] if any file fails to load or parse.
pub fn parse_file(path: &Path) -> Result<ResolvedDeqFile, ResolveError> {
    let mut loader = FsLoader;
    let root = loader.resolve(None, &path.to_string_lossy())?;
    resolve(&root, &mut loader)
}

#[cfg(test)]
mod tests;
