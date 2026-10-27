//! Resolved model objects — cross-references followed.
//!
//! Loading a qodec produces these: a [`Layer`] pairs an instruction set with the gadgets
//! that lower it, and a resolved gadget holds the actual [`Code`] and
//! [`Instruction`] values its raw form referenced by name or path. Shared
//! artifacts are held behind `Arc`, so slicing a qodec is cheap.

use crate::Code;
use crate::Instruction;
use crate::InstructionSet;
use crate::ParityEquation;
use crate::ResolveError;
use std::collections::BTreeMap;
use std::sync::Arc;

/// A layer in a `Qodec`'s lowering chain: an instruction set together with the gadget
/// set that lowers it to the next (more concrete) layer below it.
///
/// `gadgets` are keyed by the source instruction's mnemonic; each gadget
/// realizes that instruction in the instruction set of the layer below. The most
/// concrete (bottom) layer has no layer beneath it and therefore carries
/// an empty `gadgets` map.
#[derive(Debug, Clone)]
pub struct Layer {
    /// The instructions available at this layer, shared with every gadget that
    /// implements one of them.
    pub instruction_set: Arc<InstructionSet>,
    /// Explicit code bindings, keyed by this layer's block type, not code name.
    /// Sparse: gadget encodings supply omitted bindings. Retained independently
    /// of gadgets while the block remains declared by the instruction set.
    pub codes: BTreeMap<String, Arc<Code>>,
    /// The gadgets lowering this layer's instructions, keyed by the mnemonic
    /// each one implements. Empty on the bottom layer.
    pub gadgets: BTreeMap<String, Gadget>,
}

impl Layer {
    pub(crate) fn code_bindings(&self) -> BTreeMap<String, Arc<Code>> {
        let mut codes: BTreeMap<_, _> = self
            .codes
            .iter()
            .filter(|(block, _)| {
                self.instruction_set
                    .blocks
                    .iter()
                    .any(|declared| &declared.name == *block)
            })
            .map(|(block, code)| (block.clone(), code.clone()))
            .collect();
        for gadget in self.gadgets.values() {
            for (encoding, operand) in gadget
                .inputs
                .iter()
                .chain(&gadget.outputs)
                .zip(gadget.implements.inputs.iter().chain(&gadget.implements.outputs))
            {
                codes
                    .entry(operand.block.clone())
                    .or_insert_with(|| encoding.code.clone());
            }
        }
        codes
    }
}

impl PartialEq for Layer {
    fn eq(&self, other: &Self) -> bool {
        self.instruction_set == other.instruction_set
            && self.gadgets == other.gadgets
            && self.code_bindings() == other.code_bindings()
    }
}

/// Circuit source and the instruction set it calls into.
///
/// The on-disk operand-type maps are used to resolve [`Encoding::block_types`]
/// and are not retained in this value.
///
/// [`std::fmt::Display`] emits a layer-relative YAML snippet with verbatim source
/// and an explicit effective format. It does not parse or validate the circuit
/// and omits the parent-owned instruction-set reference.
#[derive(Debug, Clone, PartialEq)]
pub struct Circuit {
    /// The target instruction set the circuit source calls into.
    pub instruction_set: Arc<InstructionSet>,
    /// Circuit source text. Loading replaces source-file paths with their contents.
    pub source: String,
    /// Source format, such as `stim` or `openqasm`. If absent, methods infer it
    /// from the source text.
    pub format: Option<String>,
}

impl Circuit {
    /// The format the source is actually read as: the declared `format:` tag
    /// when there is one, otherwise the one inferred from the source itself.
    #[must_use]
    pub fn effective_format(&self) -> &str {
        self.format
            .as_deref()
            .unwrap_or_else(|| crate::ParserRegistry::infer_format(&self.source))
    }

    /// The calls this circuit's source makes, in program order.
    ///
    /// The source is parsed and checked against the target instruction set on each call, so
    /// a circuit in a format qodec has no parser for fails here rather than at load.
    ///
    /// # Errors
    ///
    /// Returns a description when the source cannot be parsed in its format,
    /// or a call names an instruction outside the target instruction set.
    pub fn calls(&self) -> Result<Vec<crate::InstructionCall>, String> {
        crate::ParserRegistry::calls(self.effective_format(), &self.source, &self.instruction_set)
    }

    /// Interpret this source with an explicit parser without changing registration.
    ///
    /// # Errors
    ///
    /// Returns the parser error or an error for calls outside the target instruction set.
    pub fn calls_with(
        &self,
        parser: impl FnOnce(&str, &InstructionSet) -> Result<Vec<crate::InstructionCall>, String>,
    ) -> Result<Vec<crate::InstructionCall>, String> {
        let calls = parser(&self.source, &self.instruction_set)?;
        crate::require_declared_mnemonics(&calls, &self.instruction_set).map_err(|error| error.to_string())?;
        Ok(calls)
    }

    /// Distinct block labels the calls name, in first-appearance order.
    ///
    /// The label space an [`Encoding`]'s support is written in, not a
    /// physical layout. Multi-qubit blocks are not expanded into individual qubits.
    ///
    /// # Errors
    ///
    /// As [`Circuit::calls`].
    pub fn blocks(&self) -> Result<Vec<String>, String> {
        Ok(crate::circuit_blocks(&self.calls()?))
    }

    /// Derive block labels using an explicit parser without registering it.
    ///
    /// # Errors
    ///
    /// As [`Circuit::calls_with`].
    pub fn blocks_with(
        &self,
        parser: impl FnOnce(&str, &InstructionSet) -> Result<Vec<crate::InstructionCall>, String>,
    ) -> Result<Vec<String>, String> {
        Ok(crate::circuit_blocks(&self.calls_with(parser)?))
    }

    /// One entry per measurement-record bit, in record order.
    ///
    /// The index is the `i` of a `circuit.readouts[i]` reference.
    ///
    /// # Errors
    ///
    /// As [`Circuit::calls`], plus conditional observations without a fixed record size.
    pub fn readouts(&self) -> Result<Vec<crate::CircuitReadout>, String> {
        crate::circuit_readouts(&self.calls()?, &self.instruction_set)
    }

    /// Derive the readout record using an explicit parser without registering it.
    ///
    /// # Errors
    ///
    /// As [`Circuit::calls_with`], plus conditional observations without a fixed record size.
    pub fn readouts_with(
        &self,
        parser: impl FnOnce(&str, &InstructionSet) -> Result<Vec<crate::InstructionCall>, String>,
    ) -> Result<Vec<crate::CircuitReadout>, String> {
        crate::circuit_readouts(&self.calls_with(parser)?, &self.instruction_set)
    }
}

/// An instruction implementation with resolved codes and instruction sets.
///
/// Parity equations retain parsed [`crate::Reference`] expressions and their
/// original spelling. Empty checks or readouts do not assert any missing relations
/// or imply that stabilizer signs pass through the circuit unchanged.
///
/// [`std::fmt::Display`] emits a layer-relative YAML snippet, not a bundle.
/// It includes the circuit, boundary mappings, and declared equations without
/// validation. The parent supplies instruction-set references and code bindings.
/// An unmatched draft encoding uses its code name as its block label.
#[derive(Debug, Clone, PartialEq)]
pub struct Gadget {
    /// The instruction this gadget realizes, resolved from the layer above.
    pub implements: Instruction,
    /// Circuit source, target instruction set, and format.
    pub circuit: Circuit,
    /// Input boundary encodings, aligned positionally with `implements.inputs`.
    pub inputs: Vec<Encoding>,
    /// Output boundary encodings, aligned positionally with `implements.outputs`.
    pub outputs: Vec<Encoding>,
    /// Bindings from implemented-instruction parameter names to the source
    /// parameters they forward into (plain source parameter names, with the
    /// on-disk `circuit.source.` prefix stripped).
    pub parameter_bindings: BTreeMap<String, String>,
    /// Deterministic syndrome bits, as property-path parity equations.
    pub checks: Vec<ParityEquation>,
    /// Terminal readouts the gadget exposes, as one positional list in
    /// declaration order: the implemented instruction's `observe` outcomes
    /// first (the observables), then its `flags:` flags (each a single
    /// parity). Each carries its own position and whether it is a flag.
    pub readouts: Vec<crate::Readout>,
    /// Sparse additional output logical-sign corrections, keyed by a parsed
    /// reference. Missing entries and
    /// empty equations apply no correction; incoming frame transport is unchanged.
    /// Values may use circuit readouts, literal bits, and readout aliases
    /// resolving only to those terms, never input or output encoding signs.
    pub frames: BTreeMap<crate::Reference, ParityEquation>,
    /// Annotations preserved without interpretation; see [`crate::Metadata`].
    pub metadata: crate::Metadata,
}

/// An encoding with its code definition resolved.
///
/// `support` lists circuit-operand labels in code-qubit order. One operand may
/// supply several code qubits, according to its block type's `encodes` count.
/// `block_types[i]` names the type of `support[i]` in the circuit instruction set. An empty
/// `block_types` list leaves the types unspecified. Occurrences of a circuit
/// label on the same gadget boundary must agree on its block type.
#[derive(Debug, Clone, PartialEq)]
pub struct Encoding {
    /// The code this boundary block carries, shared with the layer that binds it.
    pub code: Arc<Code>,
    /// Circuit labels in code-qubit order.
    pub support: Vec<String>,
    /// Circuit block-type names parallel to `support`, or empty when unspecified.
    pub block_types: Vec<String>,
}

impl Encoding {
    /// Add this encoding's support types to one circuit boundary's map.
    /// Missing types are inferred only for a single-block instruction set.
    pub(crate) fn record_block_types(
        &self,
        instruction_set: &InstructionSet,
        boundary_types: &mut BTreeMap<String, String>,
    ) -> Result<(), String> {
        if !self.block_types.is_empty() && self.block_types.len() != self.support.len() {
            return Err("different support and block-type lengths".to_owned());
        }
        for (index, label) in self.support.iter().enumerate() {
            let block_type = match self.block_types.get(index) {
                Some(block_type) => block_type,
                None => match instruction_set.blocks.as_slice() {
                    [single] => &single.name,
                    _ => {
                        return Err(format!(
                            "instruction set '{}' declares {} block types; cannot infer support type without explicit block_types",
                            instruction_set.name,
                            instruction_set.blocks.len()
                        ));
                    }
                },
            };
            if !instruction_set.blocks.iter().any(|block| &block.name == block_type) {
                return Err(format!("uses undeclared circuit block type '{block_type}'"));
            }
            if let Some(previous) = boundary_types.get(label) {
                if previous != block_type {
                    return Err(format!(
                        "circuit operand '{label}' has conflicting block types '{previous}' and '{block_type}' on the same side"
                    ));
                }
            } else {
                boundary_types.insert(label.clone(), block_type.clone());
            }
        }
        Ok(())
    }
}

impl InstructionSet {
    /// Look up the [`Instruction`] a mnemonic names, such as a gadget's
    /// `implements`.
    ///
    /// Returns a copy of the declaration, not a model-path node. A gadget's
    /// circuit and encodings are assembled by the caller.
    ///
    /// # Errors
    ///
    /// Returns [`ResolveError`] when this instruction set does not declare `mnemonic`.
    pub fn instruction(&self, mnemonic: &str) -> Result<Instruction, ResolveError> {
        self.instructions
            .iter()
            .find(|instr| instr.mnemonic == mnemonic)
            .cloned()
            .ok_or_else(|| ResolveError::InstructionNotFound {
                mnemonic: mnemonic.to_owned(),
                instruction_set: self.name.clone(),
            })
    }
}

#[cfg(test)]
mod tests {
    use super::ResolveError;
    use crate::Instruction;
    use crate::InstructionSet;

    fn instruction_set() -> InstructionSet {
        InstructionSet {
            name: "probe".to_owned(),
            description: String::new(),
            blocks: Vec::new(),
            instructions: vec![Instruction {
                mnemonic: "idle".to_owned(),
                description: String::new(),
                inputs: Vec::new(),
                outputs: Vec::new(),
                flags: Vec::new(),
                parameters: Vec::new(),
                action: Vec::new(),
                metadata: crate::Metadata::default(),
            }],
            metadata: crate::Metadata::default(),
        }
    }

    #[test]
    fn a_declared_mnemonic_resolves() {
        let instruction = instruction_set().instruction("idle").expect("idle is declared");
        assert_eq!(instruction.mnemonic, "idle");
    }

    #[test]
    fn support_type_inference_requires_one_block_type() {
        let encoding = super::Encoding {
            code: std::sync::Arc::new(
                serde_yaml::from_str::<crate::Code>("name: qubit\nstabilizers: []\nx: [X_0]\nz: [Z_0]\n").unwrap(),
            ),
            support: vec!["0".to_owned()],
            block_types: vec![],
        };
        let mut instruction_set = instruction_set();
        let mut types = std::collections::BTreeMap::new();
        let error = encoding.record_block_types(&instruction_set, &mut types).unwrap_err();
        assert!(error.contains("declares 0 block types"), "{error}");
        instruction_set.blocks.push(crate::Block {
            name: "qubit".to_owned(),
            encodes: 1,
        });
        encoding
            .record_block_types(&instruction_set, &mut types)
            .expect("infer the sole type");
        assert_eq!(types.get("0").map(String::as_str), Some("qubit"));
        instruction_set.blocks.push(crate::Block {
            name: "other".to_owned(),
            encodes: 1,
        });
        let error = encoding.record_block_types(&instruction_set, &mut types).unwrap_err();
        assert!(error.contains("declares 2 block types"), "{error}");
    }

    #[test]
    fn an_undeclared_mnemonic_names_the_mnemonic_and_instruction_set() {
        let error = instruction_set()
            .instruction("absent")
            .expect_err("absent is not declared");
        let ResolveError::InstructionNotFound {
            mnemonic,
            instruction_set,
        } = error;
        assert_eq!(mnemonic, "absent");
        assert_eq!(instruction_set, "probe");
    }
}
