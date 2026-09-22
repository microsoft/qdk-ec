//! Parsed circuit sources and concrete instruction calls.
//!
//! Parsers produce [`InstructionCall`] values. [`require_declared_mnemonics`]
//! checks them against the target instruction set without changing them.

use std::collections::{BTreeMap, BTreeSet};

/// A concrete invocation of an instruction set instruction.
///
/// Binds an [`Instruction`](crate::Instruction)'s operands to blocks and its
/// parameters to values. [`Self::select`] can request postselection on flags.
#[derive(Debug, Clone, PartialEq)]
pub struct InstructionCall {
    /// The instruction invoked, by mnemonic.
    pub mnemonic: String,
    /// Circuit block labels bound to the instruction's positional operands.
    pub operands: Vec<Operand>,
    /// Map from parameter names to argument values; contains only supplied bindings.
    pub arguments: BTreeMap<String, Argument>,
    /// Requested postselection on this call's flags, matching any listed pattern.
    /// References use declared flag names or `flags[i]`, where `i` is the
    /// zero-based index in the called instruction's flag list. An empty list
    /// imposes no selection constraint.
    pub select: Vec<SelectPattern>,
}

/// A sparse map from flag references to expected bits (`0` or `1`).
///
/// One pattern in [`InstructionCall::select`]. It matches when all its
/// constraints hold; omitted flags are unconstrained.
pub type SelectPattern = BTreeMap<String, u8>;

/// A block bound to one of an instruction's declared [`crate::BlockOperand`]
/// positions.
///
/// Both variants render as labels returned by [`crate::Circuit::blocks`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operand {
    /// A numeric block label, not a position in the list of circuit blocks.
    Index(usize),
    /// A text block label.
    Name(String),
}

impl Operand {
    /// The numeric index as decimal text, or the name unchanged.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::Index(index) => index.to_string(),
            Self::Name(name) => name.clone(),
        }
    }
}

/// A value supplied by a call: an argument for one of the instruction's declared
/// [`crate::Parameter`]s, or a positional operand.
///
/// [`Self::Qubit`] and [`Self::QubitList`] carry circuit-qubit identifiers and so
/// appear only in the operand role; no [`crate::ParameterKind`] names them.
#[derive(Debug, Clone, PartialEq)]
pub enum Argument {
    /// One circuit-qubit identifier.
    Qubit(usize),
    /// A list of circuit-qubit identifiers.
    QubitList(Vec<usize>),
    /// An integer literal.
    Integer(i64),
    /// A real literal.
    Number(f64),
    /// A boolean literal.
    Boolean(bool),
    /// A block label carried as text, or a string-valued argument.
    Text(String),
    /// A list of string literals.
    StringList(Vec<String>),
    /// A zero-based index into the circuit's readout record, written
    /// `circuit.readouts[i]` in inline YAML. Binds a prior call's readout to a
    /// `bit` parameter. The record includes both outcomes and flags and is
    /// cumulative across calls.
    Readout(usize),
}

impl Argument {
    /// Interpret a string as a circuit-readout reference or unchanged text.
    ///
    /// `circuit.readouts[i]` denotes one non-negative index. A slice selecting
    /// exactly one position denotes that same index. Other strings remain text.
    /// Record bounds are not checked.
    ///
    /// ```
    /// use qodec::Argument;
    /// assert_eq!(Argument::parse_text("circuit.readouts[3]")?, Argument::Readout(3));
    /// assert_eq!(Argument::parse_text("circuit.readouts[3:4]")?, Argument::Readout(3));
    /// assert_eq!(Argument::parse_text("label")?, Argument::Text("label".into()));
    /// # Ok::<(), String>(())
    /// ```
    ///
    /// # Errors
    ///
    /// Rejects malformed `circuit.readouts[...]` references and gadget-local
    /// `readouts[...]` references, which use a different index space.
    pub fn parse_text(text: &str) -> Result<Self, String> {
        if !text.starts_with("circuit.readouts[") {
            if text.starts_with("readouts[") {
                return Err(format!(
                    "argument {text:?} names the gadget's own readout space; \
                     to thread a record bit use `circuit.{text}`"
                ));
            }
            return Ok(Self::Text(text.to_owned()));
        }
        let reference =
            crate::Reference::parse(text).map_err(|error| format!("malformed readout reference {text:?}: {error}"))?;
        let [
            crate::ReferenceSegment::Field(circuit),
            crate::ReferenceSegment::Field(readouts),
            selector,
        ] = reference.segments()
        else {
            return Err(format!(
                "malformed readout reference {text:?}: expected `circuit.readouts[<i>]`"
            ));
        };
        if circuit != "circuit" || readouts != "readouts" {
            return Err(format!(
                "malformed readout reference {text:?}: expected `circuit.readouts[<i>]`"
            ));
        }
        let mut selected = crate::node::path::indices(selector);
        match (selected.next(), selected.next()) {
            (Some(index), None) => Ok(Self::Readout(index)),
            _ => Err(format!("readout reference {text:?} must select exactly one position")),
        }
    }
}

/// A call naming an instruction the target instruction set does not declare.
#[derive(Debug, PartialEq, Eq, derive_more::Display, derive_more::Error)]
#[display("call to unknown instruction {mnemonic:?} in instruction set {instruction_set:?}")]
pub struct UndeclaredMnemonic {
    pub mnemonic: String,
    pub instruction_set: String,
}

/// Check that every call names an instruction the target set declares.
///
/// Leaves the calls untouched: operands, arguments and selection
/// patterns are not inspected.
///
/// # Errors
///
/// Returns [`UndeclaredMnemonic`] for the first call the set does not declare.
pub fn require_declared_mnemonics(
    calls: &[InstructionCall],
    instruction_set: &crate::InstructionSet,
) -> Result<(), UndeclaredMnemonic> {
    for call in calls {
        declaration_of(instruction_set, &call.mnemonic)?;
    }
    Ok(())
}

/// The distinct block labels `calls` name, in first-appearance order.
///
/// Collects only [`InstructionCall::operands`], using [`Operand::label`].
/// Arguments are not included. These are block labels, not physical addresses.
pub fn circuit_blocks(calls: &[InstructionCall]) -> Vec<String> {
    let mut seen: BTreeSet<String> = BTreeSet::new();
    let mut blocks = Vec::new();
    for call in calls {
        for operand in &call.operands {
            let label = operand.label();
            if seen.insert(label.clone()) {
                blocks.push(label);
            }
        }
    }
    blocks
}

/// A readout declared by a circuit call.
///
/// Identifies the producing call and its outcome or flag. A gadget's
/// [`Readout`](crate::Readout) instead gives the parity equation defining a bit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitReadout {
    /// A bit produced by an `observe:` action.
    Outcome {
        /// Zero-based index into the calls returned by [`crate::Circuit::calls`].
        instruction: usize,
        /// Observable as declared, using the instruction's flat logical qubit
        /// indexes (`Z_0`), not the circuit's block labels.
        observable: crate::PauliString,
    },
    /// A bit that is one of the called instruction's declared flags.
    Flag {
        /// Zero-based index into the calls returned by [`crate::Circuit::calls`].
        instruction: usize,
        /// The flag's declared name.
        name: String,
    },
}

/// The readout record declared by `calls`, in record order.
///
/// Each call contributes all `Observe` outcomes in action order, then flags
/// in declaration order. Conditional observations are rejected because their
/// record size depends on execution. Position `i` is the zero-based index
/// used by `circuit.readouts[i]`, cumulative across calls.
///
/// # Errors
///
/// Returns an error if a call names an undeclared instruction or a declaration
/// contains a conditional observation.
pub fn circuit_readouts(
    calls: &[InstructionCall],
    instruction_set: &crate::InstructionSet,
) -> Result<Vec<CircuitReadout>, String> {
    let mut readouts = Vec::new();
    for (instruction, call) in calls.iter().enumerate() {
        let declaration = declaration_of(instruction_set, &call.mnemonic).map_err(|error| error.to_string())?;
        for step in &declaration.action {
            if let crate::Action::Observe(observables) = &step.action {
                if step.condition.is_some() {
                    return Err(format!(
                        "call '{}': conditional observe has no statically sized readout record",
                        call.mnemonic
                    ));
                }
                readouts.extend(observables.iter().map(|observable| CircuitReadout::Outcome {
                    instruction,
                    observable: observable.pauli.clone(),
                }));
            }
        }
        readouts.extend(declaration.flags.iter().map(|name| CircuitReadout::Flag {
            instruction,
            name: name.clone(),
        }));
    }
    Ok(readouts)
}

fn declaration_of<'a>(
    instruction_set: &'a crate::InstructionSet,
    mnemonic: &str,
) -> Result<&'a crate::Instruction, UndeclaredMnemonic> {
    instruction_set
        .instructions
        .iter()
        .find(|instruction| instruction.mnemonic == mnemonic)
        .ok_or_else(|| UndeclaredMnemonic {
            mnemonic: mnemonic.to_string(),
            instruction_set: instruction_set.name.clone(),
        })
}

#[cfg(test)]
mod tests {
    use super::{Argument, InstructionCall, Operand, UndeclaredMnemonic, require_declared_mnemonics};
    use crate::InstructionSet;
    use crate::Metadata;
    use crate::PauliString;
    use crate::{Action, ActionStep, Instruction, Parameter, ParameterKind};
    use std::collections::BTreeMap;

    fn declaring_instruction_set() -> InstructionSet {
        InstructionSet {
            name: "test".to_owned(),
            description: String::new(),
            blocks: Vec::new(),
            instructions: vec![
                Instruction {
                    mnemonic: "h".to_owned(),
                    description: String::new(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    flags: Vec::new(),
                    parameters: Vec::new(),
                    action: Vec::new(),
                    metadata: Metadata::default(),
                },
                Instruction {
                    mnemonic: "measure".to_owned(),
                    description: String::new(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    flags: Vec::new(),
                    parameters: Vec::new(),
                    action: Vec::new(),
                    metadata: Metadata::default(),
                },
                Instruction {
                    mnemonic: "z".to_owned(),
                    description: String::new(),
                    inputs: Vec::new(),
                    outputs: Vec::new(),
                    flags: Vec::new(),
                    parameters: vec![Parameter {
                        name: "flag".to_owned(),
                        kind: ParameterKind::Bit,
                    }],
                    action: vec![ActionStep {
                        action: Action::Pauli(PauliString("Z_0".to_owned())),
                        condition: None,
                    }],
                    metadata: Metadata::default(),
                },
            ],
            metadata: Metadata::default(),
        }
    }

    fn call(mnemonic: &str, operands: &[Operand]) -> InstructionCall {
        InstructionCall {
            mnemonic: mnemonic.to_owned(),
            operands: operands.to_vec(),
            arguments: BTreeMap::new(),
            select: Vec::new(),
        }
    }

    #[test]
    fn straight_line_passes_through() {
        let calls = vec![call("h", &[Operand::Index(4)]), call("measure", &[Operand::Index(4)])];
        assert_eq!(require_declared_mnemonics(&calls, &declaring_instruction_set()), Ok(()));
    }

    #[test]
    fn call_values_pass_through_without_binding_parameters() {
        for mnemonic in ["h", "z"] {
            let supplied = InstructionCall {
                arguments: BTreeMap::from([("flag".to_owned(), Argument::Boolean(true))]),
                select: vec![BTreeMap::from([("reject".to_owned(), 1)])],
                ..call(mnemonic, &[Operand::Name("data".to_owned())])
            };
            let calls = vec![call("measure", &[Operand::Index(4)]), supplied];
            assert_eq!(require_declared_mnemonics(&calls, &declaring_instruction_set()), Ok(()));
        }
    }

    #[test]
    fn call_outside_the_instruction_set_fails() {
        let calls = vec![call("not_an_instruction", &[Operand::Index(0)])];

        let result = require_declared_mnemonics(&calls, &declaring_instruction_set());
        assert_eq!(
            result,
            Err(UndeclaredMnemonic {
                mnemonic: "not_an_instruction".to_owned(),
                instruction_set: "test".to_owned(),
            })
        );
    }
}
