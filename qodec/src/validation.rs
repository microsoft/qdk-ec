//! Preconditions for preserving resolved model values without ambiguity or loss.
//!
//! Protocol correctness belongs to audit. Circuit contents are interpreted only
//! by operations that need calls, not by loading or saving source text.

use crate::pauli::PauliToken;
use crate::{Code, Gadget, InstructionSet, Layer, Qodec};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug)]
pub(crate) enum ModelIssue {
    Model(String),
    InstructionSet {
        layer: usize,
        name: String,
        error: String,
    },
    Code {
        name: String,
        error: String,
    },
    Gadget {
        layer: usize,
        mnemonic: String,
        error: String,
    },
}

impl std::fmt::Display for ModelIssue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Model(error) => formatter.write_str(error),
            Self::InstructionSet { layer, name, error } => {
                write!(formatter, "layer {layer}, instruction set '{name}': {error}")
            }
            Self::Code { name, error } => write!(formatter, "code '{name}': {error}"),
            Self::Gadget { layer, mnemonic, error } => write!(formatter, "layer {layer}: gadget '{mnemonic}': {error}"),
        }
    }
}

pub(crate) fn validate_bottom_layer(bottom_has_gadgets: bool) -> Result<(), ModelIssue> {
    if bottom_has_gadgets {
        return Err(ModelIssue::Model(
            "the bottom layer must have no gadgets: no target layer can resolve them".to_owned(),
        ));
    }
    Ok(())
}

impl Qodec {
    /// Check that current resolved values can be preserved without ambiguity.
    ///
    /// Checks named definitions, gadget membership, layer code bindings,
    /// aligned encodings, and derived readout roles. Incomplete protocols and
    /// invalid uses inside circuit text, actions, and parity equations are
    /// preserved for audit. This does not guarantee readiness for analysis.
    ///
    /// # Errors
    ///
    /// Returns the first preservation failure with its component location.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_model().map_err(|issue| issue.to_string())
    }

    pub(crate) fn validate_model(&self) -> Result<(), ModelIssue> {
        if let Some(version) = self.schema_version()
            && version != crate::CURRENT_SCHEMA_VERSION
        {
            return Err(ModelIssue::Model(format!(
                "schema_version must be {} or omitted (got {version})",
                crate::CURRENT_SCHEMA_VERSION
            )));
        }
        let layers = self.layers();
        validate_bottom_layer(layers.last().is_some_and(|layer| !layer.gadgets.is_empty()))?;
        self.validate_instruction_sets()?;
        self.validate_codes()?;

        for (layer_index, (source, target)) in layers.iter().zip(layers.iter().skip(1)).enumerate() {
            source.validate(&target.instruction_set, layer_index)?;
        }
        Ok(())
    }

    fn validate_instruction_sets(&self) -> Result<(), ModelIssue> {
        let mut definitions = BTreeMap::new();
        for (layer_index, layer) in self.layers().iter().enumerate() {
            let instruction_set = layer.instruction_set.as_ref();
            let name = instruction_set.name.as_str();
            if let Some(&previous) = definitions.get(name) {
                if previous != instruction_set {
                    return Err(ModelIssue::Model(format!(
                        "conflicting instruction sets named '{name}'"
                    )));
                }
                continue;
            }

            instruction_set.validate().map_err(|error| ModelIssue::InstructionSet {
                layer: layer_index,
                name: name.to_owned(),
                error,
            })?;
            definitions.insert(name, instruction_set);
        }
        Ok(())
    }

    fn validate_codes(&self) -> Result<(), ModelIssue> {
        let codes = self.layers().iter().flat_map(|layer| {
            layer.codes.values().chain(layer.gadgets.values().flat_map(|gadget| {
                gadget
                    .inputs
                    .iter()
                    .chain(&gadget.outputs)
                    .map(|encoding| &encoding.code)
            }))
        });
        let mut definitions = BTreeMap::new();
        for code in codes {
            let code = code.as_ref();
            let name = code.name.as_str();
            if let Some(&previous) = definitions.get(name) {
                if previous != code {
                    return Err(ModelIssue::Model(format!("conflicting codes named '{name}'")));
                }
                continue;
            }

            code.validate().map_err(|error| ModelIssue::Code {
                name: name.to_owned(),
                error,
            })?;
            definitions.insert(name, code);
        }
        Ok(())
    }
}

impl Layer {
    fn validate(&self, target: &InstructionSet, layer_index: usize) -> Result<(), ModelIssue> {
        let mut code_bindings: BTreeMap<_, _> = self
            .codes
            .iter()
            .map(|(block, code)| (block.as_str(), code.as_ref()))
            .collect();
        for (mnemonic, gadget) in &self.gadgets {
            let located = |error| ModelIssue::Gadget {
                layer: layer_index,
                mnemonic: mnemonic.clone(),
                error,
            };

            self.validate_membership(mnemonic, gadget, target).map_err(&located)?;
            gadget.validate().map_err(&located)?;

            for (encodings, operands) in [
                (&gadget.inputs, &gadget.implements.inputs),
                (&gadget.outputs, &gadget.implements.outputs),
            ] {
                for (encoding, operand) in encodings.iter().zip(operands) {
                    let code = encoding.code.as_ref();
                    let previous = code_bindings.entry(operand.block.as_str()).or_insert(code);
                    if *previous != code {
                        return Err(located(format!(
                            "block '{}' is bound to different codes in the layer",
                            operand.block
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_membership(&self, mnemonic: &str, gadget: &Gadget, target: &InstructionSet) -> Result<(), String> {
        let instruction = self
            .instruction_set
            .instructions
            .iter()
            .find(|instruction| instruction.mnemonic == mnemonic)
            .ok_or_else(|| format!("instruction '{mnemonic}' is not declared by the layer"))?;
        if &gadget.implements != instruction {
            return Err("implements differs from the layer's instruction".to_owned());
        }
        if gadget.circuit.instruction_set.as_ref() != target {
            return Err("circuit instruction set differs from the next layer's instruction set".to_owned());
        }
        Ok(())
    }
}

impl InstructionSet {
    /// Check names that must remain unambiguous in maps and serialization.
    ///
    /// Does not check action indices, parameter uses, or protocol correctness.
    ///
    /// # Errors
    ///
    /// Returns the first duplicate block, instruction, or parameter name.
    pub fn validate(&self) -> Result<(), String> {
        check_unique(self.blocks.iter().map(|block| block.name.as_str()), "block declaration")?;
        check_unique(
            self.instructions
                .iter()
                .map(|instruction| instruction.mnemonic.as_str()),
            "instruction mnemonic",
        )?;
        for instruction in &self.instructions {
            check_unique(
                instruction.parameters.iter().map(|parameter| parameter.name.as_str()),
                "parameter",
            )?;
        }
        Ok(())
    }
}

impl Code {
    /// Check the Pauli token syntax used to infer dimensions and default support.
    ///
    /// Unequal X/Z list lengths and algebraic errors are preserved for audit.
    ///
    /// # Errors
    ///
    /// Returns the first malformed Pauli string.
    pub fn validate(&self) -> Result<(), String> {
        for (kind, operators) in [
            ("stabilizer", &self.stabilizers),
            ("logical x", &self.x),
            ("logical z", &self.z),
        ] {
            for (index, operator) in operators.iter().enumerate() {
                for token in operator.0.split_whitespace() {
                    PauliToken::parse(token).map_err(|error| format!("{kind} {index}: {error}"))?;
                }
            }
        }
        Ok(())
    }
}

impl Gadget {
    pub(crate) fn validate(&self) -> Result<(), String> {
        self.validate_encodings()?;
        for target in self.frames.keys() {
            target.require_parity().map_err(|error| error.to_string())?;
        }
        for term in self
            .checks
            .iter()
            .chain(self.readouts.iter().map(|readout| &readout.equation))
            .chain(self.frames.values())
            .flatten()
        {
            if let crate::ParityTerm::Reference(reference) = term {
                reference.require_parity().map_err(|error| error.to_string())?;
            }
        }
        self.validate_readout_roles()
    }

    fn validate_encodings(&self) -> Result<(), String> {
        let target = &self.circuit.instruction_set;
        let can_infer_block_type = target.blocks.len() == 1;
        for (side, encodings, operands) in [
            ("input", &self.inputs, &self.implements.inputs),
            ("output", &self.outputs, &self.implements.outputs),
        ] {
            if encodings.len() != operands.len() {
                return Err(format!(
                    "{} {side} encodings for {} operands",
                    encodings.len(),
                    operands.len()
                ));
            }
            let mut boundary_types = BTreeMap::new();
            for (entry, encoding) in encodings.iter().enumerate() {
                // An entry with no declared types contributes no type to compare
                // against, unless the circuit's single block type supplies one.
                if encoding.block_types.is_empty() && !can_infer_block_type {
                    continue;
                }
                encoding
                    .record_block_types(target, &mut boundary_types)
                    .map_err(|error| format!("{side} encoding {entry} {error}"))?;
            }
        }
        Ok(())
    }

    fn validate_readout_roles(&self) -> Result<(), String> {
        let observe_count = self.implements.observe_count();
        for (position, readout) in self.readouts.iter().enumerate() {
            if readout.position != position || readout.is_flag != (position >= observe_count) {
                return Err(format!(
                    "gadget readout {position} has an inconsistent position or flag role"
                ));
            }
        }
        Ok(())
    }
}

fn check_unique<'a>(items: impl Iterator<Item = &'a str>, kind: &str) -> Result<(), String> {
    let mut seen = BTreeSet::new();
    for item in items {
        if !seen.insert(item) {
            return Err(format!("duplicate {kind} '{item}'"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn general_addresses_cannot_be_saved_as_parity_declarations() {
        let model = Qodec::load("examples/repetition3/repetition3.qodec.yaml").unwrap();
        let reference = crate::Reference::parse("metadata[\"description\"]").unwrap();
        let mut gadget = model.layers()[0].gadgets["idle"].clone();
        gadget.checks.push(vec![reference.clone().into()]);
        assert!(gadget.validate().is_err());
        gadget.checks.pop();
        gadget.frames.insert(reference, vec![]);
        assert!(gadget.validate().is_err());
        let encoded = "circuit: []\nframes: { 'metadata[\"description\"]': [] }";
        assert!(serde_yaml::from_str::<crate::GadgetSpec>(encoded).is_err());
    }

    #[test]
    fn model_checks_report_the_first_failure_in_dependency_order() {
        let mut protocol = Qodec::load("examples/repetition3/repetition3.qodec.yaml").unwrap();
        let layers = protocol.layers_mut();
        let gadget = layers[0].gadgets["idle"].clone();
        layers[1].gadgets.insert("idle".to_owned(), gadget);
        let instruction_set = Arc::make_mut(&mut layers[0].instruction_set);
        instruction_set.blocks.push(instruction_set.blocks[0].clone());
        let gadget = layers[0].gadgets.get_mut("idle").unwrap();
        Arc::make_mut(&mut gadget.inputs[0].code).stabilizers[0] = "Q_0".into();
        gadget.implements.description.push_str(" changed");
        let invalid_code = gadget.inputs[0].code.clone();
        layers[0].codes.insert("repetition3".to_owned(), invalid_code);

        assert_eq!(
            protocol.validate().unwrap_err(),
            "the bottom layer must have no gadgets: no target layer can resolve them"
        );
        protocol.layers_mut()[1].gadgets.clear();
        assert_eq!(
            protocol.validate().unwrap_err(),
            "layer 0, instruction set 'repetition3': duplicate block declaration 'repetition3'"
        );
        Arc::make_mut(&mut protocol.layers_mut()[0].instruction_set)
            .blocks
            .pop();
        assert_eq!(
            protocol.validate().unwrap_err(),
            "code 'repetition3': stabilizer 0: invalid Pauli token 'Q_0': unknown basis 'Q'"
        );
        let gadget = protocol.layers_mut()[0].gadgets.get_mut("idle").unwrap();
        gadget.inputs[0].code = gadget.outputs[0].code.clone();
        let restored_code = gadget.inputs[0].code.clone();
        protocol.layers_mut()[0]
            .codes
            .insert("repetition3".to_owned(), restored_code);
        assert_eq!(
            protocol.validate().unwrap_err(),
            "layer 0: gadget 'idle': implements differs from the layer's instruction"
        );
    }

    #[test]
    fn layer_checks_validate_encodings_before_pairing_code_bindings() {
        let mut protocol = Qodec::load("examples/repetition3/repetition3.qodec.yaml").unwrap();
        let gadget = protocol.layers_mut()[0].gadgets.get_mut("idle").unwrap();
        let input = gadget.inputs.pop().unwrap();
        Arc::make_mut(&mut gadget.outputs[0].code).name = "other".to_owned();

        assert_eq!(
            protocol.validate().unwrap_err(),
            "layer 0: gadget 'idle': 0 input encodings for 1 operands"
        );
        protocol.layers_mut()[0]
            .gadgets
            .get_mut("idle")
            .unwrap()
            .inputs
            .push(input);
        assert_eq!(
            protocol.validate().unwrap_err(),
            "layer 0: gadget 'idle': block 'repetition3' is bound to different codes in the layer"
        );
    }
}
