use std::collections::HashMap;

use super::*;
use crate::ast::Definition;

/// An in-memory loader: file id == the map key, imports resolved by exact key.
/// (Ids are already "canonical" here, so `resolve` just returns the path.)
struct MapLoader {
    files: HashMap<String, String>,
}

impl MapLoader {
    fn new(files: &[(&str, &str)]) -> Self {
        Self {
            files: files
                .iter()
                .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
                .collect(),
        }
    }
}

impl ImportLoader for MapLoader {
    fn resolve(&mut self, _base: Option<&str>, path: &str) -> Result<String, LoadError> {
        if self.files.contains_key(path) {
            Ok(path.to_string())
        } else {
            Err(LoadError {
                path: path.to_string(),
                message: "not found".to_string(),
            })
        }
    }

    fn load(&mut self, id: &str) -> Result<String, LoadError> {
        self.files.get(id).cloned().ok_or_else(|| LoadError {
            path: id.to_string(),
            message: "not found".to_string(),
        })
    }
}

fn def_name(def: &Spanned<Definition>) -> &str {
    match &def.node {
        Definition::Code(c) => &c.name,
        Definition::Gadget(g) => &g.name,
        Definition::Compose(c) => &c.name,
        Definition::Program(p) => &p.name,
    }
}

#[test]
fn no_imports_returns_the_single_file() {
    let mut loader = MapLoader::new(&[("root", "CODE C [[1,1]] {\n    STABILIZER X0\n}\n")]);
    let r = resolve("root", &mut loader).unwrap();
    assert_eq!(r.definitions.len(), 1);
    assert_eq!(def_name(&r.definitions[0]), "C");
    assert_eq!(r.files, ["root"]);
    assert_eq!(r.sources, [0]);
}

#[test]
fn imported_definitions_come_before_the_importers() {
    let mut loader = MapLoader::new(&[
        ("root", "IMPORT \"lib\"\nGADGET Main {\n    R 0\n}\n"),
        ("lib", "CODE Lib [[1,1]] {\n    STABILIZER X0\n}\n"),
    ]);
    let r = resolve("root", &mut loader).unwrap();

    let names: Vec<_> = r.definitions.iter().map(def_name).collect();
    assert_eq!(names, ["Lib", "Main"]); // imports first, depth-first
    assert!(r.into_deq_file().imports.is_empty());
}

#[test]
fn provenance_tracks_each_definitions_file() {
    let mut loader = MapLoader::new(&[
        ("root", "IMPORT \"lib\"\nGADGET Main {\n    R 0\n}\n"),
        ("lib", "CODE Lib [[1,1]] {\n    STABILIZER X0\n}\n"),
    ]);
    let r = resolve("root", &mut loader).unwrap();
    // definitions[0] = Lib from "lib"; definitions[1] = Main from "root".
    assert_eq!(r.source_of(0), "lib");
    assert_eq!(r.source_of(1), "root");
}

#[test]
fn diamond_import_loads_shared_file_once() {
    // root -> a, b ; a -> shared ; b -> shared
    let mut loader = MapLoader::new(&[
        ("root", "IMPORT \"a\"\nIMPORT \"b\"\n"),
        ("a", "IMPORT \"shared\"\nGADGET A {\n    R 0\n}\n"),
        ("b", "IMPORT \"shared\"\nGADGET B {\n    R 0\n}\n"),
        ("shared", "CODE S [[1,1]] {\n    STABILIZER X0\n}\n"),
    ]);
    let r = resolve("root", &mut loader).unwrap();

    // Shared appears once; each file loaded once.
    let names: Vec<_> = r.definitions.iter().map(def_name).collect();
    assert_eq!(names, ["S", "A", "B"]);
    assert_eq!(r.files, ["root", "a", "shared", "b"]);
}

#[test]
fn import_cycle_is_not_an_error() {
    // a <-> b: each imports the other.
    let mut loader = MapLoader::new(&[
        ("a", "IMPORT \"b\"\nGADGET A {\n    R 0\n}\n"),
        ("b", "IMPORT \"a\"\nGADGET B {\n    R 0\n}\n"),
    ]);
    let r = resolve("a", &mut loader).unwrap();
    // Guard breaks the cycle; both definitions present, each file once.
    let names: Vec<_> = r.definitions.iter().map(def_name).collect();
    assert_eq!(names, ["B", "A"]);
    assert_eq!(r.files, ["a", "b"]);
}

#[test]
fn missing_import_is_a_load_error() {
    let mut loader = MapLoader::new(&[("root", "IMPORT \"gone\"\n")]);
    let err = resolve("root", &mut loader).unwrap_err();
    assert!(matches!(err, ResolveError::Load(_)));
    assert!(err.to_string().contains("gone"));
}

#[test]
fn syntax_error_names_the_offending_file() {
    let mut loader = MapLoader::new(&[
        ("root", "IMPORT \"bad\"\n"),
        ("bad", "CODE oops {\n"), // malformed
    ]);
    let err = resolve("root", &mut loader).unwrap_err();
    match err {
        ResolveError::Parse { file, .. } => assert_eq!(file, "bad"),
        ResolveError::Load(_) => panic!("expected a parse error, got a load error"),
    }
}
