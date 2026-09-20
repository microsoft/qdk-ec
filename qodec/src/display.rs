use std::collections::BTreeMap;
use std::fmt;

use crate::{
    BlockName, BlockOperand, Circuit, CircuitSpec, Encoding, EncodingSpec, Gadget, GadgetSpec, Readout, Sourced,
};

/// Render `value` as YAML, or an explanatory comment when it cannot be serialized.
/// A `Display` impl that returns `fmt::Error` makes `to_string()` and `format!` panic,
/// so a draft that cannot round-trip must still print something a reader can act on.
fn yaml_or_reason<T: serde::Serialize>(value: &T) -> String {
    match crate::yaml_output::to_string(value) {
        Ok(yaml) => yaml,
        Err(error) => format!("# this value cannot be written as YAML: {error}\n"),
    }
}

/// Names an encoding whose position has no operand on the implemented instruction,
/// which happens only in a draft. A code name here would read as a block type.
const UNBOUND_BLOCK_TYPE: &str = "?";

macro_rules! yaml_display {
    ($($type:ty),+ $(,)?) => {
        $(
            impl fmt::Display for $type {
                fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                    formatter.write_str(&yaml_or_reason(self))
                }
            }
        )+
    };
}

yaml_display!(
    crate::Code,
    crate::InstructionSet,
    crate::Instruction,
    crate::Block,
    crate::BlockOperand,
    crate::Parameter,
    crate::Condition,
    crate::ActionStep,
);

fn circuit_spec(circuit: &Circuit) -> CircuitSpec {
    CircuitSpec {
        source: Sourced::inline(circuit.source.clone()),
        format: Some(circuit.effective_format().to_owned()),
        ..CircuitSpec::default()
    }
}

fn encoding_specs(boundary: &[Encoding], operands: &[BlockOperand]) -> Vec<EncodingSpec> {
    boundary
        .iter()
        .enumerate()
        .map(|(index, encoding)| EncodingSpec {
            block_type: operands
                .get(index)
                .map_or_else(|| UNBOUND_BLOCK_TYPE.to_owned(), |operand| operand.block.clone()),
            support: encoding.support.iter().cloned().map(BlockName).collect(),
        })
        .collect()
}

fn boundary_types(boundary: &[Encoding]) -> BTreeMap<String, String> {
    boundary
        .iter()
        .flat_map(|encoding| {
            encoding
                .support
                .iter()
                .cloned()
                .zip(encoding.block_types.iter().cloned())
        })
        .collect()
}

/// A layer-relative YAML snippet with verbatim source, without parsing or validation.
impl fmt::Display for Circuit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&yaml_or_reason(&circuit_spec(self)))
    }
}

/// A layer-relative YAML snippet; parent instruction-set and code definitions are not expanded.
impl fmt::Display for Gadget {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut circuit = circuit_spec(&self.circuit);
        circuit.inputs = boundary_types(&self.inputs);
        circuit.outputs = boundary_types(&self.outputs);
        let spec = GadgetSpec {
            implements: None,
            circuit,
            inputs: encoding_specs(&self.inputs, &self.implements.inputs),
            outputs: encoding_specs(&self.outputs, &self.implements.outputs),
            checks: Sourced::inline(self.checks.clone()),
            readouts: Sourced::inline(self.readouts.iter().map(Readout::to_spec).collect()),
            frames: self.frames.clone(),
            parameter_bindings: self.parameter_bindings.clone(),
            metadata: self.metadata.clone(),
        };
        formatter.write_str(&yaml_or_reason(&spec))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gadget() -> Gadget {
        crate::Qodec::load(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/examples/repetition3/repetition3.qodec.yaml"
        ))
        .unwrap()
        .layers()[0]
            .gadgets["idle"]
            .clone()
    }

    #[test]
    fn circuit_display_retains_uninterpreted_source() {
        let mut circuit = gadget().circuit;
        for (source, format) in [("[broken", "yaml"), ("not stim\n", "stim"), ("", "unknown")] {
            circuit.source = source.to_owned();
            circuit.format = Some(format.to_owned());
            let displayed: CircuitSpec = serde_yaml::from_str(&circuit.to_string()).unwrap();
            assert_eq!(displayed.source, Sourced::inline(source.to_owned()));
            assert_eq!(displayed.format.as_deref(), Some(format));
            assert!(displayed.instruction_set.is_none());
        }
        circuit.format = None;
        let displayed: CircuitSpec = serde_yaml::from_str(&circuit.to_string()).unwrap();
        assert_eq!(displayed.format.as_deref(), Some(circuit.effective_format()));
    }

    #[test]
    fn gadget_display_retains_equations_and_boundaries() {
        let gadget = gadget();
        let displayed: GadgetSpec = serde_yaml::from_str(&gadget.to_string()).unwrap();
        assert!(displayed.implements.is_none());
        assert_eq!(displayed.checks, Sourced::inline(gadget.checks.clone()));
        assert_eq!(displayed.inputs.len(), gadget.inputs.len());
        assert_eq!(displayed.outputs.len(), gadget.outputs.len());
        assert_eq!(displayed.circuit.inputs, boundary_types(&gadget.inputs));
        assert_eq!(displayed.circuit.outputs, boundary_types(&gadget.outputs));
        assert_eq!(displayed.inputs[0].block_type, gadget.implements.inputs[0].block);
        assert_eq!(displayed.parameter_bindings, gadget.parameter_bindings);
        assert_eq!(displayed.metadata, gadget.metadata);
    }

    #[test]
    fn unmatched_draft_encoding_is_marked_rather_than_named_after_its_code() {
        let mut gadget = gadget();
        gadget.implements.inputs.clear();
        let displayed: GadgetSpec = serde_yaml::from_str(&gadget.to_string()).unwrap();
        assert_eq!(displayed.inputs.len(), gadget.inputs.len());
        assert_eq!(displayed.inputs[0].block_type, UNBOUND_BLOCK_TYPE);
        assert_ne!(displayed.inputs[0].block_type, gadget.inputs[0].code.name);
    }

    #[test]
    fn an_unserializable_draft_prints_the_reason_instead_of_panicking() {
        // An inline circuit source must be a YAML sequence; a bare word is not.
        let spec = CircuitSpec {
            source: Sourced::inline("not: a: sequence:".to_owned()),
            ..CircuitSpec::default()
        };
        let rendered = yaml_or_reason(&spec);
        assert!(
            rendered.starts_with("# this value cannot be written as YAML"),
            "{rendered}"
        );
    }
}
