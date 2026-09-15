//! The registry of circuit-source languages a qodec circuit may be written in.

use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock, RwLock};

use crate::{InstructionCall, InstructionSet};

type RegisteredParser = Arc<dyn Fn(&str, &InstructionSet) -> Result<Vec<InstructionCall>, String> + Send + Sync>;
static REGISTERED: OnceLock<Result<ParserRegistry, String>> = OnceLock::new();

fn registry() -> Result<&'static ParserRegistry, String> {
    REGISTERED
        .get_or_init(|| {
            let registry = ParserRegistry::default();
            registry.register(parse_yaml, "yaml")?;
            Ok(registry)
        })
        .as_ref()
        .map_err(Clone::clone)
}

fn parse_yaml(source: &str, _instruction_set: &InstructionSet) -> Result<Vec<InstructionCall>, String> {
    crate::inline_yaml_parser::parse_inline_yaml(source)
}

/// Register a source parser for a format tag in this process.
///
/// The parser receives verbatim source and the declared target instruction set.
/// It must preserve execution and readout order or return an error. The latest
/// registration replaces the earlier parser, including the built-in YAML parser.
/// YAML is registered when the registry is first initialized. Calls already in
/// progress keep their selected parser. Callbacks run without holding the registry
/// lock. Registration does not affect persistence.
///
/// # Errors
///
/// Returns an error for an invalid format tag or an unavailable registry.
pub fn register<F>(parser: F, format: &str) -> Result<(), String>
where
    F: Fn(&str, &InstructionSet) -> Result<Vec<InstructionCall>, String> + Send + Sync + 'static,
{
    registry()?.register(parser, format)
}

/// Source-language parsers keyed by format tag.
#[derive(Default)]
pub struct ParserRegistry {
    parsers: RwLock<BTreeMap<String, RegisteredParser>>,
}

impl std::fmt::Debug for ParserRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParserRegistry")
            .field(
                "formats",
                &self
                    .parsers
                    .read()
                    .map(|parsers| parsers.keys().cloned().collect::<Vec<_>>()),
            )
            .finish()
    }
}

impl ParserRegistry {
    pub(crate) fn calls(
        format: &str,
        source: &str,
        instruction_set: &InstructionSet,
    ) -> Result<Vec<InstructionCall>, String> {
        let parser = registry()?
            .parsers
            .read()
            .map_err(|_| "parser registry is unavailable")?
            .get(format)
            .cloned()
            .ok_or_else(|| format!("No source parser registered for '.{format}'"))?;
        let calls = parser(source, instruction_set)?;
        crate::require_declared_mnemonics(&calls, instruction_set).map_err(|error| error.to_string())?;
        Ok(calls)
    }

    pub(super) fn format_for_path(path: &str) -> Option<&'static str> {
        match std::path::Path::new(path).extension()?.to_str()? {
            "stim" => Some("stim"),
            "qasm" => Some("openqasm"),
            "yaml" | "yml" => Some("yaml"),
            _ => None,
        }
    }

    fn register<F>(&self, parser: F, format: &str) -> Result<(), String>
    where
        F: Fn(&str, &InstructionSet) -> Result<Vec<InstructionCall>, String> + Send + Sync + 'static,
    {
        if format.is_empty() || format.trim() != format {
            return Err("parser format must be a nonempty tag without surrounding whitespace".to_owned());
        }
        let previous = self
            .parsers
            .write()
            .map_err(|_| "parser registry is unavailable")?
            .insert(format.to_owned(), Arc::new(parser));
        drop(previous);
        Ok(())
    }

    /// Infer the format of a circuit source that carries no explicit `format:`.
    ///
    /// Inline-YAML sources are a sequence, so they open with `-` (block) or `[`
    /// (flow); anything else is stim text.
    #[must_use]
    pub fn infer_format(source: &str) -> &'static str {
        let trimmed = source.trim_start();
        if trimmed.starts_with('-') || trimmed.starts_with('[') {
            "yaml"
        } else {
            "stim"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Weak;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct DropProbe {
        registry: Weak<ParserRegistry>,
        unlocked: Arc<AtomicBool>,
    }

    impl Drop for DropProbe {
        fn drop(&mut self) {
            let registry = self.registry.upgrade().unwrap();
            self.unlocked
                .store(registry.parsers.try_write().is_ok(), Ordering::SeqCst);
        }
    }

    fn circuit(format: &str) -> crate::Circuit {
        crate::Circuit {
            instruction_set: Arc::new(serde_yaml::from_str(
                "name: Test\nblocks: {qubit: 1}\ninstructions:\n- mnemonic: M\n  description: measure\n  in: [qubit]\n  action: [{observe: Z_0}]",
            ).unwrap()),
            source: "verbatim source".to_owned(),
            format: Some(format.to_owned()),
        }
    }

    #[allow(clippy::unnecessary_wraps)] // Must match the registered parser signature.
    fn measurement(source: &str, instruction_set: &InstructionSet) -> Result<Vec<InstructionCall>, String> {
        assert_eq!(source, "verbatim source");
        assert_eq!(instruction_set.name, "Test");
        Ok(vec![InstructionCall {
            mnemonic: "M".to_owned(),
            operands: vec![crate::Operand::Index(7)],
            arguments: BTreeMap::new(),
            select: Vec::new(),
        }])
    }

    #[test]
    fn registered_parser_drives_all_circuit_views() {
        let circuit = circuit("custom-registry-test");
        assert!(circuit.calls().unwrap_err().contains("No source parser"));
        register(measurement, "custom-registry-test").unwrap();
        assert_eq!(circuit.calls().unwrap().len(), 1);
        assert_eq!(circuit.blocks().unwrap(), ["7"]);
        assert_eq!(circuit.readouts().unwrap().len(), 1);
        register(|_, _| Ok(Vec::new()), "custom-registry-test").unwrap();
        assert!(circuit.calls().unwrap().is_empty());
        assert!(circuit.blocks().unwrap().is_empty());
        assert!(circuit.readouts().unwrap().is_empty());
    }

    #[test]
    fn explicit_parser_needs_no_registration() {
        let circuit = circuit("explicit-only-test");
        assert_eq!(circuit.calls_with(measurement).unwrap().len(), 1);
        assert!(circuit.calls().is_err());
        assert!(register(measurement, " ").is_err());
    }

    #[test]
    fn yaml_is_registered_normally() {
        assert!(registry().unwrap().parsers.read().unwrap().contains_key("yaml"));
        let mut source = circuit("yaml");
        assert_eq!(source.calls_with(measurement).unwrap().len(), 1);
        source.source = "- M: [7]".to_owned();
        assert_eq!(source.calls().unwrap()[0].operands, [crate::Operand::Index(7)]);
    }

    #[test]
    fn yaml_registration_can_be_replaced() {
        let registry = ParserRegistry::default();
        registry.register(parse_yaml, "yaml").unwrap();
        let previous = registry.parsers.read().unwrap()["yaml"].clone();
        registry.register(measurement, "yaml").unwrap();
        let source = circuit("yaml");
        let parser = registry.parsers.read().unwrap()["yaml"].clone();
        let current = parser(&source.source, &source.instruction_set).unwrap();
        assert_eq!(current[0].operands, [crate::Operand::Index(7)]);
        assert_eq!(
            previous("- M: [8]", &source.instruction_set).unwrap()[0].operands,
            [crate::Operand::Index(8)]
        );
    }

    #[test]
    fn replaced_parser_is_dropped_outside_the_registry_lock() {
        let registry = Arc::new(ParserRegistry::default());
        let unlocked = Arc::new(AtomicBool::new(false));
        let probe = DropProbe {
            registry: Arc::downgrade(&registry),
            unlocked: unlocked.clone(),
        };
        registry
            .register(
                move |_, _| {
                    let _probe = &probe;
                    Ok(Vec::new())
                },
                "drop-test",
            )
            .unwrap();
        registry.register(measurement, "drop-test").unwrap();
        assert!(unlocked.load(Ordering::SeqCst));
    }

    #[test]
    fn replacement_does_not_change_a_running_callback() {
        register(
            |source, target| {
                register(|_, _| Ok(Vec::new()), "replace-in-callback")?;
                measurement(source, target)
            },
            "replace-in-callback",
        )
        .unwrap();
        let source = circuit("replace-in-callback");
        assert_eq!(source.calls().unwrap()[0].operands, [crate::Operand::Index(7)]);
        assert!(source.calls().unwrap().is_empty());
    }

    #[test]
    fn callbacks_can_register_without_a_registry_lock() {
        register(
            |source, instruction_set| {
                register(measurement, "registered-inside-callback")?;
                measurement(source, instruction_set)
            },
            "reentrant-test",
        )
        .unwrap();
        assert_eq!(circuit("reentrant-test").calls().unwrap().len(), 1);
        assert_eq!(circuit("registered-inside-callback").calls().unwrap().len(), 1);
    }

    #[test]
    fn callback_calls_must_name_declared_instructions() {
        let error = circuit("explicit-only")
            .calls_with(|source, instruction_set| {
                let mut calls = measurement(source, instruction_set)?;
                calls[0].mnemonic = "missing".to_owned();
                Ok(calls)
            })
            .unwrap_err();
        assert!(error.contains("unknown instruction"), "{error}");
    }

    #[test]
    fn blocks_keep_first_appearance_order_without_expanding_encodes() {
        let circuit = crate::Circuit {
            instruction_set: Arc::new(serde_yaml::from_str(
                "name: Test\nblocks: {pair: 2}\ninstructions:\n- mnemonic: M\n  description: measure\n  in: [pair]\n  action: [{observe: Z_0}]",
            ).unwrap()),
            source: "- M: [named]\n- M: [7]\n- M: [named]\n- M: [3]".to_owned(),
            format: Some("yaml".to_owned()),
        };
        assert_eq!(circuit.blocks().unwrap(), ["named", "7", "3"]);
        let calls = circuit.calls().unwrap();
        assert_eq!(circuit.blocks_with(|_, _| Ok(calls)).unwrap(), ["named", "7", "3"]);
    }

    #[test]
    fn an_explicit_parser_also_drives_blocks_and_readouts() {
        let circuit = circuit("explicit-views-test");
        assert_eq!(circuit.blocks_with(measurement).unwrap(), ["7"]);
        assert_eq!(
            circuit.readouts_with(measurement).unwrap(),
            [crate::CircuitReadout::Outcome {
                instruction: 0,
                observable: crate::PauliString("Z_0".to_owned()),
            }]
        );
        assert!(circuit.blocks().unwrap_err().contains("No source parser"));
        assert!(circuit.readouts().unwrap_err().contains("No source parser"));
    }
}
