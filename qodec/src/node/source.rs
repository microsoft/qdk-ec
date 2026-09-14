//! Record where each loaded value sits in its YAML file.
//!
//! Locations are best effort and exist only to point a reader at a line. Any
//! failure to scan a document — malformed YAML, an unreadable working directory,
//! an event shape this scanner does not model — yields no locations for that
//! file rather than an error, because a qodec that already parsed must still load.

use crate::{Manifest, Qodec, SourceLocation};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use yaml_rust2::parser::{Event, MarkedEventReceiver, Parser};
use yaml_rust2::scanner::Marker;

#[derive(Debug, Clone)]
struct Mark {
    line: usize,
    scalar: Option<String>,
    sequence: Vec<Self>,
    mapping: Vec<(Self, Self)>,
}

impl Mark {
    fn field(&self, name: &str) -> Option<&Self> {
        self.mapping
            .iter()
            .find(|(key, _)| key.scalar.as_deref() == Some(name))
            .map(|(_, value)| value)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Document {
    file: PathBuf,
    root: Mark,
}

impl Document {
    pub(crate) fn text(file: &Path) -> Self {
        Self {
            file: file.to_owned(),
            root: Mark {
                line: 1,
                scalar: None,
                sequence: vec![],
                mapping: vec![],
            },
        }
    }
    pub(crate) fn envelope(self) -> Option<(PathBuf, Self)> {
        if self.root.mapping.len() != 1 {
            return None;
        }
        let (key, root) = self.root.mapping.into_iter().next()?;
        Some((
            PathBuf::from(crate::qodec::resolver::normalize_relative(&key.scalar?)),
            Self { file: self.file, root },
        ))
    }
}

#[derive(Default)]
struct Events(Vec<(Event, Marker)>);
impl MarkedEventReceiver for Events {
    fn on_event(&mut self, event: Event, marker: Marker) {
        self.0.push((event, marker));
    }
}

fn marked(events: &mut std::vec::IntoIter<(Event, Marker)>) -> Option<Mark> {
    let (event, marker) = events.next()?;
    let mut mark = Mark {
        line: marker.line(),
        scalar: None,
        sequence: vec![],
        mapping: vec![],
    };
    match event {
        Event::Scalar(value, ..) => mark.scalar = Some(value),
        Event::SequenceStart(..) => {
            while !matches!(events.as_slice().first()?.0, Event::SequenceEnd) {
                mark.sequence.push(marked(events)?);
            }
            events.next();
        }
        Event::MappingStart(..) => {
            while !matches!(events.as_slice().first()?.0, Event::MappingEnd) {
                mark.mapping.push((marked(events)?, marked(events)?));
            }
            events.next();
        }
        // An alias, and any event this scanner does not model, yields no location.
        _ => return None,
    }
    Some(mark)
}

pub(crate) fn parse(text: &str, file: &Path) -> Vec<Document> {
    let mut receiver = Events::default();
    if Parser::new_from_str(text).load(&mut receiver, true).is_err() {
        return vec![];
    }
    let file = if file.is_absolute() {
        file.to_owned()
    } else if let Ok(root) = std::env::current_dir() {
        root.join(file)
    } else {
        return vec![];
    };
    let mut events = receiver.0.into_iter();
    let mut documents = vec![];
    while let Some((event, _)) = events.next() {
        if event == Event::DocumentStart {
            let Some(root) = marked(&mut events) else { return vec![] };
            documents.push(Document {
                file: file.clone(),
                root,
            });
        }
    }
    documents
}

fn key_path(path: &str, key: &str) -> String {
    format!("{path}[{}]", serde_json::Value::String(key.to_owned()))
}

struct Collector<'model> {
    model: &'model Qodec,
    locations: BTreeMap<String, SourceLocation>,
}

impl Collector<'_> {
    fn put(&mut self, path: &str, file: &Path, mark: &Mark) {
        if self.model.resolve(path).is_ok() {
            self.locations.insert(
                path.to_owned(),
                SourceLocation {
                    path: file.to_owned(),
                    line: mark.line,
                },
            );
        }
    }
    fn json(&mut self, path: &str, file: &Path, mark: &Mark) {
        self.put(path, file, mark);
        for (index, child) in mark.sequence.iter().enumerate() {
            self.json(&format!("{path}[{index}]"), file, child);
        }
        for (key, child) in &mark.mapping {
            if let Some(key) = &key.scalar {
                self.json(&key_path(path, key), file, child);
            }
        }
    }
    fn record(&mut self, path: &str, file: &Path, mark: &Mark) {
        self.put(path, file, mark);
        for (key, child) in &mark.mapping {
            let Some(field) = key.scalar.as_deref() else { continue };
            let field = match field {
                "in" => "inputs",
                "out" => "outputs",
                other => other,
            };
            let child_path = if path.is_empty() {
                field.to_owned()
            } else {
                format!("{path}.{field}")
            };
            let Ok(node) = self.model.resolve(&child_path) else {
                continue;
            };
            if matches!(node.value, super::Value::Mapping(_)) {
                self.json(&child_path, file, child);
            } else if matches!(node.value, super::Value::Sequence(_)) {
                self.put(&child_path, file, child);
                for (index, entry) in child.sequence.iter().enumerate() {
                    self.record(&format!("{child_path}[{index}]"), file, entry);
                }
            } else {
                self.record(&child_path, file, child);
            }
        }
        for (index, child) in mark.sequence.iter().enumerate() {
            self.record(&format!("{path}[{index}]"), file, child);
        }
    }
    fn instruction_set(&mut self, path: &str, document: &Document) {
        self.record(path, &document.file, &document.root);
        if let Some(instructions) = document.root.field("instructions") {
            for instruction in &instructions.sequence {
                if let Some(name) = instruction.field("mnemonic").and_then(|name| name.scalar.as_deref()) {
                    self.instruction(
                        &key_path(&format!("{path}.instructions"), name),
                        &document.file,
                        instruction,
                    );
                }
            }
        }
        if let Some(blocks) = document.root.field("blocks") {
            for (index, (key, value)) in blocks.mapping.iter().enumerate() {
                let block = format!("{path}.blocks[{index}]");
                self.put(&block, &document.file, key);
                self.put(&format!("{block}.name"), &document.file, key);
                self.put(&format!("{block}.encodes"), &document.file, value);
            }
        }
    }
    fn instruction(&mut self, path: &str, file: &Path, mark: &Mark) {
        self.record(path, file, mark);
        self.instruction_operands(path, file, mark);
        self.instruction_parameters(path, file, mark);
        self.instruction_actions(path, file, mark);
    }

    fn instruction_operands(&mut self, path: &str, file: &Path, mark: &Mark) {
        for (authored, field) in [("in", "inputs"), ("out", "outputs")] {
            let Some(operands) = mark.field(authored) else { continue };
            for (index, operand) in operands.sequence.iter().enumerate() {
                let operand_path = format!("{path}.{field}[{index}]");
                self.put(
                    &format!("{operand_path}.block"),
                    file,
                    operand.sequence.first().unwrap_or(operand),
                );
                self.put(&format!("{operand_path}.is_variadic"), file, operand);
            }
        }
    }

    fn instruction_parameters(&mut self, path: &str, file: &Path, mark: &Mark) {
        let Some(parameters) = mark.field("parameters") else {
            return;
        };
        for (index, (key, value)) in parameters.mapping.iter().enumerate() {
            let parameter = format!("{path}.parameters[{index}]");
            self.put(&parameter, file, key);
            self.put(&format!("{parameter}.name"), file, key);
            self.put(&format!("{parameter}.kind"), file, value);
        }
    }

    fn instruction_actions(&mut self, path: &str, file: &Path, mark: &Mark) {
        let Some(actions) = mark.field("action") else { return };
        for (index, action) in actions.sequence.iter().enumerate() {
            let action_path = format!("{path}.action[{index}]");
            self.action(&action_path, file, action);
            self.condition(&action_path, file, action);
        }
    }

    fn action(&mut self, path: &str, file: &Path, mark: &Mark) {
        for (authored, field) in [
            ("stabilize", "operators"),
            ("observe", "observables"),
            ("pauli", "operator"),
            ("clifford", "generators"),
        ] {
            let Some(value) = mark.field(authored) else { continue };
            let field_path = format!("{path}.{field}");
            self.json(&field_path, file, value);
            if value.scalar.is_some() {
                self.put(&format!("{field_path}[0]"), file, value);
            }
        }
        if let Some(value) = mark.field("rotate") {
            self.record(path, file, value);
        }
    }

    fn condition(&mut self, path: &str, file: &Path, mark: &Mark) {
        let Some(value) = mark.field("if").or_else(|| mark.field("unless")) else {
            return;
        };
        let condition_path = format!("{path}.condition");
        self.put(&condition_path, file, value);
        self.json(&format!("{condition_path}.predicates"), file, value);
        self.put(&format!("{condition_path}.invert"), file, value);
    }

    fn equations(&mut self, path: &str, file: &Path, mark: &Mark, readouts: bool) {
        self.put(path, file, mark);
        for (index, entry) in mark.sequence.iter().enumerate() {
            let entry_path = format!("{path}[{index}]");
            self.put(&entry_path, file, entry);
            let (equation_path, equation) = if readouts {
                if let Some((name, equation)) = entry.mapping.first() {
                    self.put(&format!("{entry_path}.name"), file, name);
                    (format!("{entry_path}.equation"), equation)
                } else {
                    (format!("{entry_path}.equation"), entry)
                }
            } else {
                (entry_path, entry)
            };
            self.put(&equation_path, file, equation);
            for (term, value) in equation.sequence.iter().enumerate() {
                let term_path = format!("{equation_path}[{term}]");
                self.put(&term_path, file, value);
                self.put(&format!("{term_path}.path"), file, value);
            }
        }
    }
}

/// Context for locating one gadget's parts across the files that declare them.
struct GadgetOrigins<'a> {
    layer: usize,
    mnemonic: &'a str,
    gadget_file: &'a Path,
    document: &'a Document,
    instruction_set: Option<&'a Document>,
}

pub(crate) fn locations(
    model: &Qodec,
    manifest: &Manifest,
    origins: &BTreeMap<PathBuf, Document>,
) -> BTreeMap<String, SourceLocation> {
    let mut collector = Collector {
        model,
        locations: BTreeMap::new(),
    };
    if let Some(document) = origins.get(Path::new(model.manifest_filename())) {
        collector.record("", &document.file, &document.root);
    }
    for (index, layer) in manifest.layers.iter().enumerate() {
        let layer_path = format!("layers[{index}]");
        let instruction_set = origins.get(Path::new(&layer.instruction_set));
        if let Some(document) = instruction_set {
            collector.instruction_set(&format!("{layer_path}.instruction_set"), document);
            collector.instruction_set(
                &key_path("instruction_sets", &model.layers()[index].instruction_set.name),
                document,
            );
        }
        for (mnemonic, gadget_file) in &layer.gadgets {
            let gadget_file = Path::new(gadget_file);
            let Some(document) = origins.get(gadget_file) else {
                continue;
            };
            collector.gadget(
                &key_path(&format!("{layer_path}.gadgets"), mnemonic),
                &GadgetOrigins {
                    layer: index,
                    mnemonic,
                    gadget_file,
                    document,
                    instruction_set,
                },
                manifest,
                origins,
            );
        }
    }
    collector.locations
}

impl Collector<'_> {
    /// Record every located part of one gadget: its own document, the instruction
    /// it implements, its equations, its boundary encodings, and its circuit source.
    fn gadget(
        &mut self,
        path: &str,
        gadget: &GadgetOrigins<'_>,
        manifest: &Manifest,
        origins: &BTreeMap<PathBuf, Document>,
    ) {
        let GadgetOrigins {
            layer: index,
            mnemonic,
            gadget_file,
            document,
            instruction_set,
        } = *gadget;
        let layer = &manifest.layers[index];
        self.record(path, &document.file, &document.root);
        if let Some((instruction, source)) = instruction_set
            .and_then(|document| {
                document
                    .root
                    .field("instructions")
                    .map(|instructions| (instructions, document))
            })
            .and_then(|(instructions, document)| {
                instructions
                    .sequence
                    .iter()
                    .find(|instruction| {
                        instruction.field("mnemonic").and_then(|name| name.scalar.as_deref()) == Some(mnemonic)
                    })
                    .map(|instruction| (instruction, document))
            })
        {
            self.instruction(&format!("{path}.implements"), &source.file, instruction);
        }
        let spec = &self.model.gadgets[gadget_file];
        for (field, reference, readouts) in [
            ("checks", spec.checks.path(), false),
            ("readouts", spec.readouts.path(), true),
        ] {
            let external = reference
                .and_then(|reference| origins.get(&crate::qodec::resolver::resolve_relative(gadget_file, reference)));
            if let Some(external) = external {
                self.equations(&format!("{path}.{field}"), &external.file, &external.root, readouts);
            } else if let Some(mark) = document.root.field(field) {
                self.equations(&format!("{path}.{field}"), &document.file, mark, readouts);
            }
        }
        let resolved = &self.model.layers()[index].gadgets[mnemonic];
        for (authored, side, operands) in [
            ("in", "inputs", &resolved.implements.inputs),
            ("out", "outputs", &resolved.implements.outputs),
        ] {
            for (entry, operand) in operands.iter().enumerate() {
                if let Some(support) = document
                    .root
                    .field(authored)
                    .and_then(|boundary| boundary.sequence.get(entry))
                    .and_then(|encoding| encoding.mapping.first())
                    .map(|(_, support)| support)
                {
                    self.json(&format!("{path}.{side}[{entry}].support"), &document.file, support);
                }
                if let Some(code) = layer
                    .codes
                    .get(&operand.block)
                    .and_then(|file| origins.get(Path::new(file)))
                {
                    self.record(&format!("{path}.{side}[{entry}].code"), &code.file, &code.root);
                    if let Some(name) = code.root.field("name").and_then(|name| name.scalar.as_deref()) {
                        self.record(&key_path("codes", name), &code.file, &code.root);
                    }
                }
            }
        }
        if let Some(next) = manifest
            .layers
            .get(index + 1)
            .and_then(|layer| origins.get(Path::new(&layer.instruction_set)))
        {
            self.instruction_set(&format!("{path}.circuit.instruction_set"), next);
        }
        if let Some(source) = spec
            .circuit
            .source
            .path()
            .and_then(|reference| origins.get(&crate::qodec::resolver::resolve_relative(gadget_file, reference)))
        {
            self.put(&format!("{path}.circuit.source"), &source.file, &source.root);
        } else if spec.circuit.source.path().is_none()
            && let Some(source) = document.root.field("circuit")
        {
            self.put(
                &format!("{path}.circuit.source"),
                &document.file,
                source.field("source").unwrap_or(source),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse;
    use std::path::Path;

    fn instruction_source_fixture() -> (tempfile::TempDir, crate::Qodec) {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(
            directory.path().join("qodec.yaml"),
            "layers: [{instruction_set: instruction.yaml}]\n",
        )
        .unwrap();
        let source = [
            "name: test",
            "blocks: {qubit: 1}",
            "instructions:",
            "  - mnemonic: probe",
            "    description: Source mapping fixture.",
            "    in: [qubit, [qubit]]",
            "    out: [qubit]",
            "    parameters: {enabled: bit, theta: number}",
            "    action:",
            "      - stabilize: Z_0",
            "      - observe: [Z_0]",
            "      - pauli: X_0",
            "        if: [enabled]",
            "      - clifford: {X_0: Z_0, Z_0: X_0}",
            "      - rotate: {pauli: Z_0, angle: theta}",
            "        unless: [enabled]",
        ]
        .join("\n");
        std::fs::write(directory.path().join("instruction.yaml"), source).unwrap();
        let model = crate::Qodec::load(directory.path().join("qodec.yaml")).unwrap();
        (directory, model)
    }

    #[test]
    fn instruction_components_keep_their_authored_locations() {
        let (directory, model) = instruction_source_fixture();
        let instruction = model
            .resolve("layers[0].instruction_set.instructions[\"probe\"]")
            .unwrap();
        for (field, line) in [
            ("inputs[0].block", 6),
            ("inputs[1].block", 6),
            ("inputs[1].is_variadic", 6),
            ("outputs[0].block", 7),
            ("parameters[0].name", 8),
            ("parameters[1].kind", 8),
            ("action[0].operators[0]", 10),
            ("action[1].observables[0]", 11),
            ("action[2].operator", 12),
            ("action[2].condition.predicates[0]", 13),
            ("action[3].generators[\"X_0\"]", 14),
            ("action[4].pauli", 15),
            ("action[4].angle", 15),
            ("action[4].condition", 16),
            ("action[4].condition.invert", 16),
        ] {
            let node = instruction.resolve(field).unwrap();
            let location = node
                .source_location()
                .unwrap_or_else(|| panic!("missing location: {field}"));
            assert_eq!(location.path, directory.path().join("instruction.yaml"), "{field}");
            assert_eq!(location.line, line, "{field}");
        }
    }

    #[test]
    fn marks_lines_and_bundle_envelopes() {
        let documents = parse(
            "---\na.yaml:\n  readouts:\n    - [\"circuit.readouts[0]\"]\n",
            Path::new("bundle.yaml"),
        );
        let (key, document) = documents.into_iter().next().unwrap().envelope().unwrap();
        assert_eq!(key, Path::new("a.yaml"));
        assert_eq!(document.root.field("readouts").unwrap().sequence[0].line, 4);
    }
    #[test]
    fn loaded_locations_disappear_before_mutation() {
        let mut model = crate::Qodec::load("examples/repetition3/repetition3.qodec.yaml").unwrap();
        assert!(model.resolve("").unwrap().source_location().is_some());
        let gadgets = model.resolve("layers[0].gadgets").unwrap().as_mapping().unwrap();
        for gadget in gadgets.values() {
            assert!(gadget.source_location().is_some());
        }
        model.layers_mut();
        assert!(model.resolve("").unwrap().source_location().is_none());
    }
}
