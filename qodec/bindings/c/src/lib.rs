//! C bindings for qodec: load a qodec and read its lowering chain.
//!
//! [`qodec_load`] builds a read-only projection and returns its [`Qodec`] root.
//! Fields expose the layers, instruction sets, actions, gadgets, parsed circuit
//! calls, boundary encodings and codes. Metadata is JSON object text in
//! `metadata_json` fields (`instruction_set_metadata_json` on layers), including `{}` when empty.
//! Circuit qubit and measurement-record lists are not projected.
//!
//! There are five C functions: [`qodec_load`], [`qodec_unload`],
//! [`qodec_find_gadget`], [`qodec_abi_version`] and [`qodec_last_error`].
//! Authoring and mutation use the Rust or Python API.
//!
//! # Ownership
//!
//! [`qodec_load`] produces a [`Qodec`] and [`qodec_unload`] releases it.
//! Treat the root and all reachable storage as read-only. Nested pointers,
//! strings and collections borrow from the root and become invalid on unload.
//! Pass the original root pointer to `qodec_unload` exactly once, not a copy of
//! the struct. Never free the root or its nested storage yourself.
//! Several threads may read it concurrently, but all readers must finish before
//! it is unloaded. Unloading must not race with a read.
//!
//! # Conventions
//!
//! - Ordinary collections are `{ count, items }`; iterate `0 .. count`.
//!   Empty collections have null storage pointers; do not dereference them.
//! - Parity equations, string lists and selection patterns use flat storage
//!   with `count + 1` offsets when `count > 0`. Do not read offsets when empty.
//! - Strings are borrowed, NUL-terminated UTF-8. A string containing NUL fails
//!   C projection with `QODEC_STATUS_ERROR`. An absent optional string is null.
//! - [`QodecAction`] and [`QodecArgumentValue`] are C tagged unions: switch on
//!   `tag` and read only the matching member.
//! - Operands are blocks; parameters are declared classical inputs; arguments
//!   are values supplied to those parameters. [`QodecInstructionCall`] separates
//!   `operands` from `arguments`; both contain [`QodecArgument`] entries.
//! - [`QodecParameter::kind`] describes the declared parameter type, while
//!   [`QodecArgumentValue`]'s tag selects the supplied value's representation.
//! - [`QodecReference`] is a flat struct. Its `tag` selects the reference kind;
//!   only encoding-property references use `boundary`, `property` and `entry`.
//!
//! # Stability
//!
//! Struct layouts are part of the ABI. Adding, removing or reordering fields
//! can break compatibility. Compare [`qodec_abi_version`] with the header's
//! `QODEC_ABI_VERSION` before reading structs; do not proceed on a mismatch.
//! The ABI revision is independent of the package and on-disk schema versions.
//!
//! # Errors
//!
//! Only [`qodec_load`] returns a status: `QODEC_STATUS_OK` on success or a
//! negative status on failure. [`qodec_find_gadget`] returns null on failure.
//! Both record errors through [`qodec_last_error`], including caught Rust
//! panics. Successful calls do not clear an older error.

use std::cell::RefCell;
use std::ffi::{CStr, CString, NulError, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use qodec::{Action, ActionStep, BlockOperand, Instruction, Parameter, ParameterKind, Scalar};
use qodec::{Argument, InstructionCall, Operand, SelectPattern};
use qodec::{Circuit, Encoding, Gadget};
use qodec::{ParityTerm, ReferenceSegment};

/// ABI revision. Bump for any breaking change to a signature, symbol, struct
/// layout or calling convention below.
pub const QODEC_ABI_VERSION: u32 = 1;

/// The call succeeded.
pub const QODEC_STATUS_OK: i32 = 0;
/// The qodec could not be loaded; see `qodec_last_error()`.
pub const QODEC_STATUS_ERROR: i32 = -1;
/// `qodec_load` received a null argument or a non-UTF-8 path.
pub const QODEC_STATUS_INVALID_ARG: i32 = -2;
/// A Rust panic was caught during `qodec_load`; see `qodec_last_error()`.
pub const QODEC_STATUS_PANIC: i32 = -3;

/// `QodecReference.tag`: `circuit.readouts[index]`, a measurement-record bit
/// at a zero-based position in measurement order.
pub const QODEC_REFERENCE_CIRCUIT_READOUT: u8 = 0;
/// `QodecReference.tag`: `readouts[index]`, a zero-based gadget readout position.
pub const QODEC_REFERENCE_READOUT: u8 = 1;
/// `QodecReference.tag`: `{in,out}[entry].{stabilizers,x,z}[index]`, an
/// encoding sign. Only this tag uses `boundary`, `property` and `entry`.
pub const QODEC_REFERENCE_ENCODING_PROPERTY: u8 = 2;
/// A literal parity bit. `index` is 0 or 1; all other fields are zero.
pub const QODEC_REFERENCE_CONSTANT: u8 = 3;

/// `QodecReference.boundary`: the gadget's `inputs` (`in:` on disk).
pub const QODEC_BOUNDARY_IN: u8 = 0;
/// `QodecReference.boundary`: the gadget's `outputs` (`out:` on disk).
pub const QODEC_BOUNDARY_OUT: u8 = 1;

/// `QodecReference.property` — a stabilizer-generator sign.
pub const QODEC_PROPERTY_STABILIZER: u8 = 0;
/// `QodecReference.property` — a logical-X operator sign.
pub const QODEC_PROPERTY_LOGICAL_X: u8 = 1;
/// `QodecReference.property` — a logical-Z operator sign.
pub const QODEC_PROPERTY_LOGICAL_Z: u8 = 2;

/// `QodecParameter.kind`: a runtime classical bit parameter, eligible in conditions.
pub const QODEC_PARAMETER_BIT: u8 = 0;
/// `QodecParameter.kind`: a parameter accepting a compile-time real literal.
pub const QODEC_PARAMETER_NUMBER: u8 = 1;
/// `QodecParameter.kind`: a parameter accepting a compile-time integer literal.
pub const QODEC_PARAMETER_INTEGER: u8 = 2;
/// `QodecParameter.kind`: a parameter accepting a compile-time boolean literal.
pub const QODEC_PARAMETER_BOOLEAN: u8 = 3;
/// `QodecParameter.kind`: a parameter accepting a compile-time string literal.
pub const QODEC_PARAMETER_STRING: u8 = 4;
/// `QodecParameter.kind`: a parameter accepting a compile-time Pauli literal.
pub const QODEC_PARAMETER_PAULI: u8 = 5;

/// One term of a parity equation, in fully expanded form.
///
/// `tag` selects the reference kind. `boundary`, `property` and `entry` are
/// meaningful only for `QODEC_REFERENCE_ENCODING_PROPERTY` and are zero
/// otherwise. Every index is relative to the containing gadget.
///
/// Selectors are already expanded: an authored `circuit.readouts[0,2,5]` is
/// three references.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QodecReference {
    /// Which reference shape this is; one of the `QODEC_REFERENCE_*` values.
    pub tag: u8,
    /// For an encoding property, one of the `QODEC_BOUNDARY_*` values.
    pub boundary: u8,
    /// For an encoding property, one of the `QODEC_PROPERTY_*` values.
    pub property: u8,
    /// For an encoding property, the zero-based encoding position in the
    /// gadget's `inputs` or `outputs`, selected by `boundary`.
    pub entry: u64,
    /// Zero-based circuit or gadget readout position; for an encoding property,
    /// the operator position in the encoding code's `stabilizers`, `x` or `z`.
    /// For `QODEC_REFERENCE_CONSTANT`, the literal bit 0 or 1.
    pub index: u64,
}

/// A list of parity equations, in compressed-sparse-row form.
///
/// Equation `equation_index` uses the borrowed range
/// `references[offsets[equation_index] .. offsets[equation_index + 1]]`.
/// This storage is shared by checks, readouts, and frames; only checks assert zero parity.
/// Do not read offsets when `count == 0` or references when `total == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecParity {
    /// How many equations there are.
    pub count: usize,
    /// `count + 1` offsets delimiting `references`; null when `count` is zero.
    pub offsets: *const usize,
    /// The equations' references, concatenated.
    pub references: *const QodecReference,
    /// Total number of references. Zero when `count == 0`; otherwise `offsets[count]`.
    pub total: usize,
}

/// A list of strings, in compressed-sparse-row form.
///
/// For `string_index < count`, `bytes + offsets[string_index]` is a borrowed,
/// NUL-terminated UTF-8 string. Offsets include the terminators. A string
/// containing NUL fails C projection. When `count == 0`, do not read either buffer.
///
/// ```c
/// const QodecStrings *stabilizers = &encoding->code.stabilizers;
/// for (size_t string_index = 0; string_index < stabilizers->count; ++string_index) {
///     puts(stabilizers->bytes + stabilizers->offsets[string_index]);
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecStrings {
    /// How many strings there are.
    pub count: usize,
    /// `count + 1` offsets into `bytes`; null when `count` is zero.
    pub offsets: *const usize,
    /// The strings, concatenated, each NUL-terminated; null when `count` is zero.
    pub bytes: *const c_char,
    /// Total length of `bytes` including every terminator.
    pub total: usize,
}

/// A borrowed run of numeric circuit-qubit identifiers, not array positions.
///
/// Identifiers are `uint64_t` wherever they appear, including
/// `QodecArgumentValue.qubit.index`; counts are `size_t`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecIndices {
    /// How many indices there are.
    pub count: usize,
    /// The indices; null when `count` is zero.
    pub items: *const u64,
}

/// A block type declared by an instruction set.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecBlock {
    /// The block type's name, as instruction operands reference it.
    pub name: *const c_char,
    /// How many logical qubits a block of this type encodes.
    pub encodes: usize,
}

/// A borrowed run of block declarations.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecBlocks {
    /// How many block declarations there are.
    pub count: usize,
    /// The declarations; null when `count` is zero.
    pub items: *const QodecBlock,
}

/// One positional block operand in an instruction's `in:` / `out:` list.
///
/// Operands are nameless: position fixes the contiguous range the entry
/// occupies in the flat index space the action addresses.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecBlockOperand {
    /// The block type this entry's qubits are encoded in.
    pub block: *const c_char,
    /// Whether the entry is variadic (`[block]` on disk).
    pub is_variadic: bool,
}

/// A borrowed run of block operands.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecBlockOperands {
    /// How many operands there are.
    pub count: usize,
    /// The operands; null when `count` is zero.
    pub items: *const QodecBlockOperand,
}

/// A declared classical input to an instruction.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecParameter {
    /// The parameter's name, as call sites and conditions reference it.
    pub name: *const c_char,
    /// Its declared parameter type; one of the `QODEC_PARAMETER_*` values.
    /// `BIT` is the runtime, condition-eligible type; the rest accept compile-time
    /// literals. This is not a `QodecArgumentValue` tag.
    pub kind: u8,
}

/// A borrowed run of parameter declarations.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecParameters {
    /// How many parameters there are.
    pub count: usize,
    /// The parameters; null when `count` is zero.
    pub items: *const QodecParameter,
}

/// One operation in an instruction's formal semantics.
///
/// In C, switch on `tag` and read only its matching union member. For example,
/// `QodecAction_Observe` selects `observe.observables`.
#[repr(C, u8)]
#[derive(Debug, Clone, Copy)]
pub enum QodecAction {
    /// Force the state into the +1 eigenspace of every operator listed.
    Stabilize { paulis: QodecStrings },
    /// A Clifford unitary, as the tableau mapping `from[i]` to `to[i]`.
    Clifford { from: QodecStrings, to: QodecStrings },
    /// Apply a Pauli unitary.
    Pauli { pauli: *const c_char },
    /// Measure each observable, one classical bit per entry, in order.
    Observe { observables: QodecStrings },
    /// Rotate about `axis` by a literal angle or a referenced parameter's value.
    Rotate {
        axis: *const c_char,
        /// Whether the angle is a literal rather than a parameter reference.
        angle_is_literal: bool,
        /// The angle, when `angle_is_literal`.
        angle_literal: f64,
        /// Parameter name when `angle_is_literal` is false; null for a literal.
        /// `angle_operand` is the ABI field name for this parameter reference.
        angle_operand: *const c_char,
    },
}

/// One step of an instruction's action, with its optional guard.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecActionStep {
    /// The operation this step performs.
    pub action: QodecAction,
    /// Whether the step is guarded.
    pub has_condition: bool,
    /// The bits XOR-ed to form the guard, when `has_condition`.
    pub condition_predicates: QodecStrings,
    /// Whether the guard is inverted (`unless` rather than `if`), when `has_condition`.
    pub condition_invert: bool,
}

/// A borrowed run of action steps.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecActionSteps {
    /// How many steps there are.
    pub count: usize,
    /// The steps; null when `count` is zero.
    pub items: *const QodecActionStep,
}

/// One instruction declared by an instruction set.
///
/// `QodecAction_Observe` steps declare measurement outcomes in action order.
/// `flags` declares additional named output bits.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecInstruction {
    /// The instruction's mnemonic.
    pub mnemonic: *const c_char,
    /// Free-text description, or the empty string.
    pub description: *const c_char,
    /// Input block operands (logical qubits consumed or passed through).
    pub inputs: QodecBlockOperands,
    /// Output block operands (logical qubits produced or passed through).
    pub outputs: QodecBlockOperands,
    /// Named classical outputs following the action's measurement outcomes.
    pub flags: QodecStrings,
    /// Declared classical inputs; calls supply arguments for these parameters.
    pub parameters: QodecParameters,
    /// The formal semantics: an ordered list of guarded operations.
    pub action: QodecActionSteps,
    /// Free-form annotations as JSON object text, including `{}` when empty.
    pub metadata_json: *const c_char,
}

/// A borrowed run of instructions, in declaration order.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecInstructions {
    /// How many instructions there are.
    pub count: usize,
    /// The instructions; null when `count` is zero.
    pub items: *const QodecInstruction,
}

/// A quantum error-correcting code.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecCode {
    /// The code's name.
    pub name: *const c_char,
    /// Free-text description, or the empty string.
    pub description: *const c_char,
    /// Stabilizer generators, as Pauli strings.
    pub stabilizers: QodecStrings,
    /// Logical X operators, one per logical qubit.
    pub x: QodecStrings,
    /// Logical Z operators, one per logical qubit, aligned with `x`.
    pub z: QodecStrings,
    /// Number of logical qubits, equal to `x.count`. A valid code declares as
    /// many logical Z operators, but a draft need not: bound `z` by `z.count`.
    pub logical_count: usize,
    /// One more than the highest qubit index across stabilizers and logical
    /// operators, or zero when none is used.
    pub physical_qubit_count: u64,
    /// Free-form annotations as JSON object text, including `{}` when empty.
    pub metadata_json: *const c_char,
}

/// One boundary encoding: the code a gadget operand is encoded in, and where
/// that code's blocks land in the circuit.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecEncoding {
    /// The code this encoding uses.
    pub code: QodecCode,
    /// Circuit-operand labels in code-block order, such as `"0"` or `"ancilla"`.
    /// These are labels, not positions in a projected array.
    pub support: QodecStrings,
    /// Block-type names parallel to `support`, or an empty list if unavailable.
    pub block_types: QodecStrings,
}

/// A borrowed run of encodings.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecEncodings {
    /// How many encodings there are.
    pub count: usize,
    /// The encodings; null when `count` is zero.
    pub items: *const QodecEncoding,
}

/// A block label or an argument value supplied at a call site.
///
/// Used by both `QodecInstructionCall.operands` and
/// `QodecInstructionCall.arguments`. The tag describes the supplied
/// representation, not the declared `QodecParameter.kind`.
///
/// In C, switch on `tag` and read only its matching union member. For example,
/// `QodecArgumentValue_Qubit` selects `qubit.index`, not `integer.value`.
#[repr(C, u8)]
#[derive(Debug, Clone, Copy)]
pub enum QodecArgumentValue {
    /// A numeric block label or qubit identifier, not a position in a projected array.
    Qubit { index: u64 },
    /// A list of numeric circuit-qubit identifiers.
    QubitList { qubits: QodecIndices },
    /// An integer literal.
    Integer { value: i64 },
    /// A real literal.
    Number { value: f64 },
    /// A block label carried as text, or a string-valued argument.
    Text { value: *const c_char },
    /// A list of string literals.
    StringList { strings: QodecStrings },
    /// A prior measurement-record bit at an absolute, zero-based position in
    /// measurement order across preceding calls.
    Readout { index: u64 },
    /// A Boolean literal, distinct from integer literals 0 and 1.
    Boolean { value: bool },
}

/// One block operand or parameter argument at a call site.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecArgument {
    /// The parameter name for a named argument; null for a positional block operand.
    pub name: *const c_char,
    /// The block label or supplied argument value.
    pub value: QodecArgumentValue,
}

/// A borrowed run of block operands or parameter arguments.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecArguments {
    /// How many entries there are.
    pub count: usize,
    /// The entries; null when `count` is zero.
    pub items: *const QodecArgument,
}

/// One constraint in a `select` pattern: a flag bit and its expected value.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecSelectConstraint {
    /// The call's flag, named or addressed by its zero-based `flags[<index>]` position.
    pub flag: *const c_char,
    /// The value it is expected to take, `0` or `1`.
    pub bit: u8,
}

/// Selection patterns over a call's own flags, in compressed-sparse-row form.
///
/// Pattern `pattern_index` uses the borrowed range
/// `constraints[offsets[pattern_index] .. offsets[pattern_index + 1]]`.
/// A pattern matches when every constraint matches; selection matches when any
/// pattern matches. `count == 0` imposes no constraint.
/// Do not read offsets when `count == 0` or constraints when `total == 0`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecSelect {
    /// How many patterns there are.
    pub count: usize,
    /// `count + 1` offsets delimiting `constraints`; null when `count` is zero.
    pub offsets: *const usize,
    /// The patterns' constraints, concatenated.
    pub constraints: *const QodecSelectConstraint,
    /// Total number of constraints. Zero when `count == 0`; otherwise `offsets[count]`.
    pub total: usize,
}

/// One invocation of a `QodecInstruction` in a circuit.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecInstructionCall {
    /// The mnemonic invoked, naming an instruction in the circuit's instruction set.
    pub mnemonic: *const c_char,
    /// Blocks in the instruction's declared operand order. Each `name` is null;
    /// its value is a numeric label (`Qubit`) or a named label (`Text`).
    pub operands: QodecArguments,
    /// Arguments supplied to the instruction's declared parameters, each carrying
    /// the parameter name in `name`.
    pub arguments: QodecArguments,
    /// Selection patterns over this call's flags; empty when no selection is specified.
    pub select: QodecSelect,
}

/// A borrowed run of instruction calls, in program order.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecInstructionCalls {
    /// How many calls there are.
    pub count: usize,
    /// The calls; null when `count` is zero.
    pub items: *const QodecInstructionCall,
}

/// A gadget's circuit: the program that runs, and the instruction set it calls into.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecCircuit {
    /// The name of the target instruction set the source calls into.
    pub instruction_set_name: *const c_char,
    /// Inlined program text, preserved verbatim. NUL bytes fail C projection.
    pub source: *const c_char,
    /// The resolved source-format tag, or null when inferred from the text.
    pub format: *const c_char,
    /// The format selected for parsing: `format` when non-null, otherwise
    /// inferred from the text. This does not guarantee a parser is available.
    pub effective_format: *const c_char,
    /// Calls parsed and checked against the target instruction set during `qodec_load`.
    /// Empty on a parse error, an undeclared instruction, or a program with no
    /// calls. Check `error` to distinguish failure from an empty program.
    pub calls: QodecInstructionCalls,
    /// Parse or instruction-check error, or null on success. This error leaves
    /// `calls` empty without failing `qodec_load` and is not recorded in
    /// `qodec_last_error`.
    pub error: *const c_char,
}

/// One gadget: the per-instruction rule lowering it to the layer below.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecGadget {
    /// The instruction this gadget declares it implements.
    pub implements: QodecInstruction,
    /// The circuit supplied for this implementation.
    pub circuit: QodecCircuit,
    /// Declared zero-parity checks. Loading does not verify them against the circuit.
    pub checks: QodecParity,
    /// Readout equations in output order: the instruction's `observe` outcomes
    /// first, then its flags. These are not all zero-parity checks.
    pub readouts: QodecParity,
    /// One name per readout, parallel to `readouts`; an empty string for an
    /// anonymous readout. Names are labels, not reference indices.
    pub readout_names: QodecStrings,
    /// Number of `observe` outcomes declared by the implemented instruction.
    /// For `readout_index < readouts.count`, `readout_index >= observe_count`
    /// identifies a flag. An incomplete draft may declare fewer readouts than
    /// outcomes, so this can exceed `readouts.count`.
    pub observe_count: usize,
    /// Instruction parameter names forwarded into the circuit source, parallel
    /// to `parameter_targets`.
    pub parameter_names: QodecStrings,
    /// Circuit-source parameter names, parallel to `parameter_names`, without
    /// the on-disk `circuit.source.` prefix.
    pub parameter_targets: QodecStrings,
    /// Input boundary encodings. An encoding-property reference selects an
    /// entry here when its `boundary` is `QODEC_BOUNDARY_IN`.
    pub inputs: QodecEncodings,
    /// Output boundary encodings, indexed likewise for `QODEC_BOUNDARY_OUT`.
    pub outputs: QodecEncodings,
    /// Free-form annotations as JSON object text, including `{}` when empty.
    pub metadata_json: *const c_char,
    /// Output logical-sign reference paths in sorted order, parallel to `frames`.
    pub frame_targets: QodecStrings,
    /// Additional output-sign corrections as XOR equations. Empty entries apply
    /// no correction. These are definitions, not zero-parity checks.
    pub frames: QodecParity,
}

/// A borrowed run of gadgets, in sorted mnemonic order.
/// Ordering uses UTF-8 bytes.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecGadgets {
    /// How many gadgets there are.
    pub count: usize,
    /// The gadgets; null when `count` is zero.
    pub items: *const QodecGadget,
}

/// One layer of the lowering chain: an instruction set plus the gadgets lowering it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecLayer {
    /// The name of this layer's instruction set.
    pub instruction_set_name: *const c_char,
    /// The instruction set's description, or the empty string.
    pub instruction_set_description: *const c_char,
    /// The block types the instruction set declares.
    pub blocks: QodecBlocks,
    /// The instruction set's instructions, in declaration order.
    pub instructions: QodecInstructions,
    /// The gadgets lowering this layer to the next. Empty on the bottom layer,
    /// which is the target of the layer above and lowers no further.
    pub gadgets: QodecGadgets,
    /// The instruction set's free-form annotations as JSON object text, including `{}` when empty.
    pub instruction_set_metadata_json: *const c_char,
}

/// A borrowed run of layers, ordered logical to physical.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct QodecLayers {
    /// How many layers there are.
    pub count: usize,
    /// The layers; null when `count` is zero.
    pub items: *const QodecLayer,
}

/// The root of a loaded qodec.
///
/// Produced by `qodec_load()` and released by `qodec_unload()`. Treat this struct
/// and all reachable storage as read-only. All nested pointers borrow from
/// this root and become invalid when it is unloaded.
///
/// Pass the original root pointer to `qodec_unload` exactly once, not a copy of
/// the struct. Never free the root or its nested storage yourself.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Qodec {
    /// The qodec's name, or the empty string when the manifest has none.
    pub name: *const c_char,
    /// The manifest description, or the empty string.
    pub description: *const c_char,
    /// Whether the manifest declared an on-disk schema version.
    pub has_schema_version: bool,
    /// The declared on-disk schema version; meaningful only when `has_schema_version`.
    pub schema_version: u32,
    /// The lowering chain, ordered logical to physical.
    pub layers: QodecLayers,
    /// The manifest's free-form annotations as JSON object text, including `{}` when empty.
    pub metadata_json: *const c_char,
}

impl QodecReference {
    fn from_head(head: &[ReferenceSegment]) -> Result<Self, &'static str> {
        Ok(match head {
            [ReferenceSegment::Field(circuit), ReferenceSegment::Field(readouts)]
                if circuit == "circuit" && readouts == "readouts" =>
            {
                Self {
                    tag: QODEC_REFERENCE_CIRCUIT_READOUT,
                    boundary: 0,
                    property: 0,
                    entry: 0,
                    index: 0,
                }
            }
            [ReferenceSegment::Field(readouts)] if readouts == "readouts" => Self {
                tag: QODEC_REFERENCE_READOUT,
                boundary: 0,
                property: 0,
                entry: 0,
                index: 0,
            },
            [
                ReferenceSegment::Field(boundary),
                ReferenceSegment::Index(entry),
                ReferenceSegment::Field(property),
            ] => Self::encoding_property(boundary, *entry, property)?,
            [
                ReferenceSegment::Field(boundary),
                ReferenceSegment::Index(entry),
                ReferenceSegment::Field(code),
                ReferenceSegment::Field(property),
            ] if code == "code" => Self::encoding_property(boundary, *entry, property)?,
            _ => return Err("model address is not a parity reference"),
        })
    }

    fn encoding_property(boundary: &str, entry: usize, property: &str) -> Result<Self, &'static str> {
        Ok(Self {
            tag: QODEC_REFERENCE_ENCODING_PROPERTY,
            boundary: match boundary {
                "in" => QODEC_BOUNDARY_IN,
                "out" => QODEC_BOUNDARY_OUT,
                _ => return Err("model address is not a parity reference"),
            },
            property: match property {
                "stabilizers" => QODEC_PROPERTY_STABILIZER,
                "x" => QODEC_PROPERTY_LOGICAL_X,
                "z" => QODEC_PROPERTY_LOGICAL_Z,
                _ => return Err("model address is not a parity reference"),
            },
            entry: entry as u64,
            index: 0,
        })
    }
}

// ── The C-ready projection ───────────────────────────────────────────────────
//
// qodec stores parity equations as parsed expressions, names as Rust `String`s and
// collections as `Vec`/`BTreeMap`, none of which C can traverse. Building a
// projection once at open is what lets the whole tree be plain structs the
// caller neither allocates nor frees.
//
// Everything is interned through `Arena`, which boxes each buffer before
// handing back a pointer into it. Boxed buffers and `Vec` heap storage do not
// move when the vectors holding them grow, so pointers taken during
// construction stay valid for the life of the allocation.

/// A parity section, flattened to CSR.
struct Parity {
    offsets: Vec<usize>,
    references: Vec<QodecReference>,
}

impl Parity {
    /// Expand cached selectors in equation order without parsing or omitting terms.
    fn build<'a>(
        equations: impl IntoIterator<Item = impl IntoIterator<Item = &'a ParityTerm>>,
    ) -> Result<Self, &'static str> {
        let mut offsets = vec![0_usize];
        let mut references = Vec::new();
        for equation in equations {
            for atom in equation {
                match atom {
                    ParityTerm::Reference(reference) => {
                        Self::append_reference(&mut references, reference)?;
                    }
                    ParityTerm::Bit(value) => references.push(QodecReference {
                        tag: QODEC_REFERENCE_CONSTANT,
                        boundary: 0,
                        property: 0,
                        entry: 0,
                        index: u64::from(*value),
                    }),
                }
            }
            offsets.push(references.len());
        }
        Ok(Self { offsets, references })
    }

    fn append_reference(
        references: &mut Vec<QodecReference>,
        reference: &qodec::Reference,
    ) -> Result<(), &'static str> {
        let Some((selector, head)) = reference.segments().split_last() else {
            return Err("model address is not a parity reference");
        };
        let target = QodecReference::from_head(head)?;
        let mut append = |index: usize| {
            references.push(QodecReference {
                index: index as u64,
                ..target
            });
        };
        match selector {
            ReferenceSegment::Index(index) => append(*index),
            ReferenceSegment::Union(indices) => indices.iter().copied().for_each(append),
            ReferenceSegment::Slice { start, stop, step } => (*start..*stop).step_by(*step).for_each(append),
            _ => return Err("model address is not a parity reference"),
        }
        Ok(())
    }

    fn view(&self) -> QodecParity {
        if self.offsets.len() <= 1 {
            return QodecParity {
                count: 0,
                offsets: std::ptr::null(),
                references: std::ptr::null(),
                total: 0,
            };
        }
        QodecParity {
            count: self.offsets.len() - 1,
            offsets: self.offsets.as_ptr(),
            references: self.references.as_ptr(),
            total: self.references.len(),
        }
    }
}

#[cfg(test)]
mod parity_tests {
    use super::*;
    use qodec::Reference;

    #[test]
    fn model_addresses_cannot_be_projected_as_parity() {
        for path in [
            "",
            "metadata",
            "metadata[0]",
            "circuit.readouts",
            "readouts.name",
            "out[0].unknown[0]",
            "other[0].z[0]",
            "out[0:1].z[0]",
        ] {
            let equations = [vec![Reference::parse(path).unwrap().into()]];
            assert_eq!(
                Parity::build(&equations).err(),
                Some("model address is not a parity reference"),
                "{path}",
            );
        }
    }

    #[test]
    fn projection_retains_all_encoding_targets_and_aliases() {
        for (boundary, expected_boundary) in [("in", QODEC_BOUNDARY_IN), ("out", QODEC_BOUNDARY_OUT)] {
            for (property, expected_property) in [
                ("stabilizers", QODEC_PROPERTY_STABILIZER),
                ("x", QODEC_PROPERTY_LOGICAL_X),
                ("z", QODEC_PROPERTY_LOGICAL_Z),
            ] {
                for prefix in ["", "code."] {
                    let reference = Reference::parse(&format!("{boundary}[2].{prefix}{property}[1:4:2]")).unwrap();
                    let parity = Parity::build(&[vec![reference.into()]]).unwrap();
                    assert_eq!(parity.offsets, [0, 2]);
                    for (term, index) in parity.references.iter().zip([1, 3]) {
                        assert_eq!(term.tag, QODEC_REFERENCE_ENCODING_PROPERTY);
                        assert_eq!(term.boundary, expected_boundary);
                        assert_eq!(term.entry, 2);
                        assert_eq!(term.property, expected_property);
                        assert_eq!(term.index, index);
                    }
                }
            }
        }
    }

    #[test]
    fn large_slice_streams_every_selected_index() {
        let reference = Reference::parse("circuit.readouts[0:1000000:3]").unwrap();
        let parity = Parity::build(&[vec![reference.into()]]).unwrap();
        assert_eq!(parity.offsets, [0, 333_334]);
        assert!(
            parity
                .references
                .iter()
                .enumerate()
                .all(|(position, term)| term.index == (position * 3) as u64
                    && term.tag == QODEC_REFERENCE_CIRCUIT_READOUT)
        );
    }

    #[test]
    fn cached_selectors_preserve_equation_boundaries_order_and_duplicates() {
        let equations = [
            vec![
                Reference::parse("out[01].z[3, 1,3]").unwrap().into(),
                Reference::parse("circuit.readouts[00:03:2]").unwrap().into(),
            ],
            vec![],
            vec![
                Reference::parse("readouts[02]").unwrap().into(),
                ParityTerm::Bit(false),
                ParityTerm::Bit(true),
            ],
        ];
        let parity = Parity::build(&equations).unwrap();
        assert_eq!(parity.offsets, [0, 5, 5, 8]);
        assert_eq!(
            parity.references.iter().map(|term| term.index).collect::<Vec<_>>(),
            [3, 1, 3, 0, 2, 2, 0, 1]
        );
        for term in &parity.references[..3] {
            assert_eq!(term.tag, QODEC_REFERENCE_ENCODING_PROPERTY);
            assert_eq!(term.boundary, QODEC_BOUNDARY_OUT);
            assert_eq!(term.entry, 1);
            assert_eq!(term.property, QODEC_PROPERTY_LOGICAL_Z);
        }
        assert_eq!(parity.references[3].tag, QODEC_REFERENCE_CIRCUIT_READOUT);
        assert_eq!(parity.references[5].tag, QODEC_REFERENCE_READOUT);
        assert_eq!(parity.references[6].tag, QODEC_REFERENCE_CONSTANT);
        assert_eq!(parity.references[7].tag, QODEC_REFERENCE_CONSTANT);
    }
}

/// A string list, flattened to CSR with each entry NUL-terminated in place.
struct Strings {
    offsets: Vec<usize>,
    bytes: Vec<u8>,
}

impl Strings {
    fn build<'a>(values: impl IntoIterator<Item = &'a str>) -> Result<Self, NulError> {
        let mut offsets = Vec::new();
        let mut bytes = Vec::new();
        for value in values {
            let value = CString::new(value)?;
            offsets.push(bytes.len());
            bytes.extend_from_slice(value.as_bytes_with_nul());
        }
        offsets.push(bytes.len());
        Ok(Self { offsets, bytes })
    }

    fn view(&self) -> QodecStrings {
        if self.offsets.len() <= 1 {
            return QodecStrings {
                count: 0,
                offsets: std::ptr::null(),
                bytes: std::ptr::null(),
                total: 0,
            };
        }
        QodecStrings {
            count: self.offsets.len() - 1,
            offsets: self.offsets.as_ptr(),
            bytes: self.bytes.as_ptr().cast::<c_char>(),
            total: self.bytes.len(),
        }
    }
}

/// One `select` expectation, flattened to CSR.
struct Select {
    offsets: Vec<usize>,
    constraints: Vec<QodecSelectConstraint>,
}

/// Owns every buffer the projected tree points into.
///
/// Each `intern_*` builds a buffer, stores it, and returns a view of it. Every
/// stored type keeps its data in its own heap allocation — `CString`, and the
/// `Vec`s inside `Strings`, `Parity` and `Select` — and the views point only at
/// those, so later interning can move the structs without invalidating
/// anything.
#[derive(Default)]
struct Arena {
    cstrings: Vec<CString>,
    strings: Vec<Strings>,
    parities: Vec<Parity>,
    selects: Vec<Select>,
    indices: Vec<Vec<u64>>,
    blocks: Vec<Vec<QodecBlock>>,
    operands: Vec<Vec<QodecBlockOperand>>,
    parameters: Vec<Vec<QodecParameter>>,
    steps: Vec<Vec<QodecActionStep>>,
    instructions: Vec<Vec<QodecInstruction>>,
    arguments: Vec<Vec<QodecArgument>>,
    calls: Vec<Vec<QodecInstructionCall>>,
    encodings: Vec<Vec<QodecEncoding>>,
    gadgets: Vec<Vec<QodecGadget>>,
    layers: Vec<QodecLayer>,
}

impl Arena {
    fn text(&mut self, value: &str) -> Result<*const c_char, NulError> {
        let owned = CString::new(value)?;
        let pointer = owned.as_ptr();
        self.cstrings.push(owned);
        Ok(pointer)
    }

    /// Free-form annotations as JSON object text, including `{}` when empty.
    fn metadata(&mut self, metadata: &qodec::Metadata) -> Result<*const c_char, NulError> {
        if metadata.is_empty() {
            return self.text("{}");
        }
        let json = serde_json::to_string(metadata).unwrap_or_default();
        self.text(&json)
    }

    fn text_list<'a>(&mut self, values: impl IntoIterator<Item = &'a str>) -> Result<QodecStrings, NulError> {
        let owned = Strings::build(values)?;
        let view = owned.view();
        self.strings.push(owned);
        Ok(view)
    }

    fn parity<'a>(
        &mut self,
        equations: impl IntoIterator<Item = impl IntoIterator<Item = &'a ParityTerm>>,
    ) -> Result<QodecParity, &'static str> {
        let owned = Parity::build(equations)?;
        let view = owned.view();
        self.parities.push(owned);
        Ok(view)
    }

    fn index_list(&mut self, values: &[usize]) -> QodecIndices {
        if values.is_empty() {
            return QodecIndices {
                count: 0,
                items: std::ptr::null(),
            };
        }
        self.indices.push(values.iter().map(|value| *value as u64).collect());
        let stored = self.indices.last().expect("just pushed");
        QodecIndices {
            count: stored.len(),
            items: stored.as_ptr(),
        }
    }
}

/// Build a `{count, items}` run from a completed vector of runs.
macro_rules! run {
    ($arena:expr, $field:ident, $built:expr, $ty:ident) => {{
        let built: Vec<_> = $built;
        if built.is_empty() {
            $ty {
                count: 0,
                items: std::ptr::null(),
            }
        } else {
            $arena.$field.push(built);
            let stored = $arena.$field.last().expect("just pushed");
            $ty {
                count: stored.len(),
                items: stored.as_ptr(),
            }
        }
    }};
}

fn parameter_kind(kind: ParameterKind) -> u8 {
    match kind {
        ParameterKind::Bit => QODEC_PARAMETER_BIT,
        ParameterKind::Number => QODEC_PARAMETER_NUMBER,
        ParameterKind::Integer => QODEC_PARAMETER_INTEGER,
        ParameterKind::Boolean => QODEC_PARAMETER_BOOLEAN,
        ParameterKind::String => QODEC_PARAMETER_STRING,
        ParameterKind::Pauli => QODEC_PARAMETER_PAULI,
    }
}

fn empty_strings() -> QodecStrings {
    QodecStrings {
        count: 0,
        offsets: std::ptr::null(),
        bytes: std::ptr::null(),
        total: 0,
    }
}

fn build_operand(arena: &mut Arena, operand: &Operand) -> Result<QodecArgument, NulError> {
    let value = match operand {
        Operand::Index(index) => QodecArgumentValue::Qubit { index: *index as u64 },
        Operand::Name(name) => QodecArgumentValue::Text {
            value: arena.text(name)?,
        },
    };
    Ok(QodecArgument {
        name: std::ptr::null(),
        value,
    })
}

fn build_argument(arena: &mut Arena, name: Option<&str>, argument: &Argument) -> Result<QodecArgument, NulError> {
    let value = match argument {
        Argument::Qubit(qubit) => QodecArgumentValue::Qubit { index: *qubit as u64 },
        Argument::QubitList(qubits) => QodecArgumentValue::QubitList {
            qubits: arena.index_list(qubits),
        },
        Argument::Integer(value) => QodecArgumentValue::Integer { value: *value },
        Argument::Number(value) => QodecArgumentValue::Number { value: *value },
        Argument::Boolean(value) => QodecArgumentValue::Boolean { value: *value },
        Argument::Text(value) => QodecArgumentValue::Text {
            value: arena.text(value)?,
        },
        Argument::StringList(values) => QodecArgumentValue::StringList {
            strings: arena.text_list(values.iter().map(String::as_str))?,
        },
        Argument::Readout(index) => QodecArgumentValue::Readout { index: *index as u64 },
    };
    Ok(QodecArgument {
        name: match name {
            Some(name) => arena.text(name)?,
            None => std::ptr::null(),
        },
        value,
    })
}

fn build_select(arena: &mut Arena, patterns: &[SelectPattern]) -> Result<QodecSelect, NulError> {
    if patterns.is_empty() {
        return Ok(QodecSelect {
            count: 0,
            offsets: std::ptr::null(),
            constraints: std::ptr::null(),
            total: 0,
        });
    }
    let mut offsets = Vec::with_capacity(patterns.len() + 1);
    let mut constraints = Vec::new();
    for pattern in patterns {
        offsets.push(constraints.len());
        for (flag, bit) in pattern {
            constraints.push(QodecSelectConstraint {
                flag: arena.text(flag)?,
                bit: *bit,
            });
        }
    }
    offsets.push(constraints.len());

    let owned = Select { offsets, constraints };
    let view = QodecSelect {
        count: owned.offsets.len() - 1,
        offsets: owned.offsets.as_ptr(),
        constraints: owned.constraints.as_ptr(),
        total: owned.constraints.len(),
    };
    arena.selects.push(owned);
    Ok(view)
}

fn build_call(arena: &mut Arena, call: &InstructionCall) -> Result<QodecInstructionCall, NulError> {
    let mnemonic = arena.text(&call.mnemonic)?;
    let operands_built: Vec<QodecArgument> = call
        .operands
        .iter()
        .map(|operand| build_operand(arena, operand))
        .collect::<Result<_, _>>()?;
    let operands = run!(arena, arguments, operands_built, QodecArguments);
    let arguments_built: Vec<QodecArgument> = call
        .arguments
        .iter()
        .map(|(name, argument)| build_argument(arena, Some(name), argument))
        .collect::<Result<_, _>>()?;
    let arguments = run!(arena, arguments, arguments_built, QodecArguments);

    Ok(QodecInstructionCall {
        mnemonic,
        operands,
        arguments,
        select: build_select(arena, &call.select)?,
    })
}

/// Parse a gadget circuit and check its mnemonics, mirroring Python's `Circuit.calls`.
///
/// Loading a qodec does not itself parse circuit sources, so this can fail on an
/// otherwise-valid qodec. Reporting the failure per circuit keeps the rest of
/// the tree readable, where failing the open would not.
fn build_circuit(arena: &mut Arena, circuit: &Circuit) -> Result<QodecCircuit, NulError> {
    let source = arena.text(&circuit.source)?;
    let parsed = circuit.calls();

    let (calls, error) = match parsed {
        Ok(calls) => (calls, std::ptr::null()),
        Err(message) => (Vec::new(), arena.text(&message)?),
    };
    let built: Vec<QodecInstructionCall> = calls
        .iter()
        .map(|call| build_call(arena, call))
        .collect::<Result<_, _>>()?;

    Ok(QodecCircuit {
        instruction_set_name: arena.text(&circuit.instruction_set.name)?,
        source,
        format: match &circuit.format {
            Some(format) => arena.text(format)?,
            None => std::ptr::null(),
        },
        effective_format: arena.text(circuit.effective_format())?,
        calls: run!(arena, calls, built, QodecInstructionCalls),
        error,
    })
}

fn build_action(arena: &mut Arena, action: &Action) -> Result<QodecAction, NulError> {
    Ok(match action {
        Action::Stabilize(paulis) => QodecAction::Stabilize {
            paulis: arena.text_list(paulis.iter().map(|pauli| pauli.0.as_str()))?,
        },
        Action::Clifford(tableau) => QodecAction::Clifford {
            from: arena.text_list(tableau.keys().map(|pauli| pauli.0.as_str()))?,
            to: arena.text_list(tableau.values().map(|pauli| pauli.0.as_str()))?,
        },
        Action::Pauli(pauli) => QodecAction::Pauli {
            pauli: arena.text(&pauli.0)?,
        },
        Action::Observe(observables) => QodecAction::Observe {
            observables: arena.text_list(observables.iter().map(|observable| observable.pauli.0.as_str()))?,
        },
        Action::Rotate { pauli, angle } => {
            let (angle_is_literal, angle_literal, angle_operand) = match angle {
                Scalar::Literal(value) => (true, *value, std::ptr::null()),
                Scalar::Parameter(name) => (false, 0.0, arena.text(name)?),
            };
            QodecAction::Rotate {
                axis: arena.text(&pauli.0)?,
                angle_is_literal,
                angle_literal,
                angle_operand,
            }
        }
    })
}

fn build_step(arena: &mut Arena, step: &ActionStep) -> Result<QodecActionStep, NulError> {
    let action = build_action(arena, &step.action)?;
    Ok(match &step.condition {
        Some(condition) => QodecActionStep {
            action,
            has_condition: true,
            condition_predicates: arena.text_list(condition.predicates.iter().map(String::as_str))?,
            condition_invert: condition.invert,
        },
        None => QodecActionStep {
            action,
            has_condition: false,
            condition_predicates: empty_strings(),
            condition_invert: false,
        },
    })
}

fn build_operands(arena: &mut Arena, operands: &[BlockOperand]) -> Result<QodecBlockOperands, NulError> {
    let built: Vec<QodecBlockOperand> = operands
        .iter()
        .map(|operand| {
            Ok(QodecBlockOperand {
                block: arena.text(&operand.block)?,
                is_variadic: operand.is_variadic,
            })
        })
        .collect::<Result<_, _>>()?;
    Ok(run!(arena, operands, built, QodecBlockOperands))
}

fn build_parameters(arena: &mut Arena, parameters: &[Parameter]) -> Result<QodecParameters, NulError> {
    let built: Vec<QodecParameter> = parameters
        .iter()
        .map(|parameter| {
            Ok(QodecParameter {
                name: arena.text(&parameter.name)?,
                kind: parameter_kind(parameter.kind),
            })
        })
        .collect::<Result<_, _>>()?;
    Ok(run!(arena, parameters, built, QodecParameters))
}

fn build_instruction(arena: &mut Arena, instruction: &Instruction) -> Result<QodecInstruction, NulError> {
    let mnemonic = arena.text(&instruction.mnemonic)?;
    let description = arena.text(&instruction.description)?;
    let inputs = build_operands(arena, &instruction.inputs)?;
    let outputs = build_operands(arena, &instruction.outputs)?;
    let flags = arena.text_list(instruction.flags.iter().map(String::as_str))?;
    let parameters = build_parameters(arena, &instruction.parameters)?;
    let steps: Vec<QodecActionStep> = instruction
        .action
        .iter()
        .map(|step| build_step(arena, step))
        .collect::<Result<_, _>>()?;
    let action = run!(arena, steps, steps, QodecActionSteps);
    let metadata_json = arena.metadata(&instruction.metadata)?;

    Ok(QodecInstruction {
        mnemonic,
        description,
        inputs,
        outputs,
        flags,
        parameters,
        action,
        metadata_json,
    })
}

fn build_encoding(arena: &mut Arena, encoding: &Encoding) -> Result<QodecEncoding, NulError> {
    let code = QodecCode {
        name: arena.text(&encoding.code.name)?,
        description: arena.text(&encoding.code.description)?,
        stabilizers: arena.text_list(encoding.code.stabilizers.iter().map(|pauli| pauli.0.as_str()))?,
        x: arena.text_list(encoding.code.x.iter().map(|pauli| pauli.0.as_str()))?,
        z: arena.text_list(encoding.code.z.iter().map(|pauli| pauli.0.as_str()))?,
        logical_count: encoding.code.logical_count(),
        physical_qubit_count: encoding.code.physical_qubit_count() as u64,
        metadata_json: arena.metadata(&encoding.code.metadata)?,
    };
    Ok(QodecEncoding {
        code,
        support: arena.text_list(encoding.support.iter().map(String::as_str))?,
        block_types: arena.text_list(encoding.block_types.iter().map(String::as_str))?,
    })
}

fn build_gadget(
    arena: &mut Arena,
    gadget: &Gadget,
    implements: QodecInstruction,
) -> Result<QodecGadget, Box<dyn std::error::Error>> {
    let circuit = build_circuit(arena, &gadget.circuit)?;
    let checks = arena.parity(gadget.checks.iter().map(|check| check.iter()))?;
    let readouts = arena.parity(gadget.readouts.iter().map(|readout| readout.equation.iter()))?;
    let readout_names = arena.text_list(
        gadget
            .readouts
            .iter()
            .map(|readout| readout.name.as_deref().unwrap_or_default()),
    )?;
    let parameter_names = arena.text_list(gadget.parameter_bindings.keys().map(String::as_str))?;
    let parameter_targets = arena.text_list(gadget.parameter_bindings.values().map(String::as_str))?;

    let inputs_built: Vec<QodecEncoding> = gadget
        .inputs
        .iter()
        .map(|encoding| build_encoding(arena, encoding))
        .collect::<Result<_, _>>()?;
    let inputs = run!(arena, encodings, inputs_built, QodecEncodings);
    let outputs_built: Vec<QodecEncoding> = gadget
        .outputs
        .iter()
        .map(|encoding| build_encoding(arena, encoding))
        .collect::<Result<_, _>>()?;
    let outputs = run!(arena, encodings, outputs_built, QodecEncodings);

    Ok(QodecGadget {
        implements,
        circuit,
        checks,
        readouts,
        readout_names,
        observe_count: gadget.implements.observe_count(),
        parameter_names,
        parameter_targets,
        inputs,
        outputs,
        metadata_json: arena.metadata(&gadget.metadata)?,
        frame_targets: arena.text_list(gadget.frames.keys().map(qodec::Reference::path))?,
        frames: arena.parity(gadget.frames.values().map(|terms| terms.iter()))?,
    })
}

/// Everything the C API can see, plus the buffers it points into.
///
/// `repr(C)` with `root` first is load-bearing: [`qodec_load`] hands out a
/// pointer to `root`, and [`qodec_unload`] casts it back to reclaim the box.
#[repr(C)]
struct Prepared {
    root: Qodec,
    _arena: Arena,
}

impl Prepared {
    fn new(qodec: &qodec::Qodec) -> Result<Box<Self>, Box<dyn std::error::Error>> {
        let mut arena = Arena::default();

        let layers: Vec<QodecLayer> = qodec
            .layers()
            .iter()
            .map(|layer| {
                let instruction_set_name = arena.text(&layer.instruction_set.name)?;
                let instruction_set_description = arena.text(&layer.instruction_set.description)?;

                let blocks_built: Vec<QodecBlock> = layer
                    .instruction_set
                    .blocks
                    .iter()
                    .map(|block| {
                        Ok(QodecBlock {
                            name: arena.text(&block.name)?,
                            encodes: block.encodes,
                        })
                    })
                    .collect::<Result<_, NulError>>()?;
                let blocks = run!(arena, blocks, blocks_built, QodecBlocks);

                let instructions_built: Vec<QodecInstruction> = layer
                    .instruction_set
                    .instructions
                    .iter()
                    .map(|instruction| build_instruction(&mut arena, instruction))
                    .collect::<Result<_, _>>()?;
                let instruction_views: std::collections::BTreeMap<_, _> = layer
                    .instruction_set
                    .instructions
                    .iter()
                    .zip(&instructions_built)
                    .map(|(instruction, view)| (instruction.mnemonic.as_str(), *view))
                    .collect();
                let instructions = run!(arena, instructions, instructions_built, QodecInstructions);

                let mut gadgets_built: Vec<QodecGadget> = layer
                    .gadgets
                    .iter()
                    .map(|(mnemonic, gadget)| build_gadget(&mut arena, gadget, instruction_views[mnemonic.as_str()]))
                    .collect::<Result<_, _>>()?;
                gadgets_built.sort_by(|left, right| unsafe {
                    CStr::from_ptr(left.implements.mnemonic).cmp(CStr::from_ptr(right.implements.mnemonic))
                });
                let gadgets = run!(arena, gadgets, gadgets_built, QodecGadgets);
                let instruction_set_metadata_json = arena.metadata(&layer.instruction_set.metadata)?;

                Ok(QodecLayer {
                    instruction_set_name,
                    instruction_set_description,
                    blocks,
                    instructions,
                    gadgets,
                    instruction_set_metadata_json,
                })
            })
            .collect::<Result<_, Box<dyn std::error::Error>>>()?;

        let name = arena.text(qodec.name().unwrap_or_default())?;
        let description = arena.text(qodec.description().unwrap_or_default())?;
        let schema_version = qodec.schema_version();
        let metadata_json = arena.metadata(qodec.metadata())?;

        arena.layers = layers;
        let root = Qodec {
            name,
            description,
            has_schema_version: schema_version.is_some(),
            schema_version: schema_version.unwrap_or(0),
            layers: QodecLayers {
                count: arena.layers.len(),
                items: if arena.layers.is_empty() {
                    std::ptr::null()
                } else {
                    arena.layers.as_ptr()
                },
            },
            metadata_json,
        };

        Ok(Box::new(Self { root, _arena: arena }))
    }
}

thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Record `message` as the calling thread's most recent error.
fn set_last_error(message: &str) {
    // `with` panics during thread-local teardown; a dropped message is better
    // than an abort out of `extern "C"`.
    let _ = LAST_ERROR.try_with(|cell| {
        let mut buffer = cell.borrow_mut();
        buffer.clear();
        buffer.extend_from_slice(message.replace('\0', "\\0").as_bytes());
        buffer.push(0);
    });
}

/// Run `body`, converting a panic into `fallback` and recording its payload.
fn guard<T>(context: &str, fallback: T, body: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(payload) => {
            let reason = payload
                .downcast_ref::<&str>()
                .map(|text| (*text).to_owned())
                .or_else(|| payload.downcast_ref::<String>().cloned());
            match reason {
                Some(reason) => set_last_error(&format!("{context}: panicked: {reason}")),
                None => set_last_error(&format!("{context}: panicked")),
            }
            fallback
        }
    }
}

/// The ABI revision this library implements.
///
/// Compare it with `QODEC_ABI_VERSION` from the header before reading projected
/// structs. Do not use the projection if the revisions differ.
#[unsafe(no_mangle)]
pub extern "C" fn qodec_abi_version() -> u32 {
    QODEC_ABI_VERSION
}

/// The calling thread's most recent error, or null if there is none.
///
/// The returned NUL-terminated string is read-only and borrows thread-local
/// storage. It remains valid until another error is recorded on this thread
/// or the thread exits. Copy it to retain the message; do not free it.
///
/// Successful calls do not clear an older error. Check the load status or
/// lookup result before consulting this message.
/// NUL bytes in diagnostic text are escaped as `\0`.
#[unsafe(no_mangle)]
pub extern "C" fn qodec_last_error() -> *const c_char {
    // This is the function a C caller reaches for on a cleanup path, where the
    // thread-local may already be going away.
    LAST_ERROR
        .try_with(|cell| {
            let buffer = cell.borrow();
            if buffer.is_empty() {
                std::ptr::null()
            } else {
                buffer.as_ptr().cast::<c_char>()
            }
        })
        .unwrap_or(std::ptr::null())
}

/// Load a qodec from a manifest file or single-file bundle at `path`.
/// The path must name a file, not a directory, and must be UTF-8.
///
/// Returns `QODEC_STATUS_OK` and writes the root to `*out_qodec` on success.
/// The root and all reachable storage are read-only; release the root with
/// `qodec_unload()`.
///
/// On failure, leaves `*out_qodec` unchanged and records `qodec_last_error()`.
/// Returns `QODEC_STATUS_INVALID_ARG` for a null argument or non-UTF-8 path,
/// `QODEC_STATUS_ERROR` for a loader error (including a directory path) or a
/// string containing NUL in the C projection, or
/// `QODEC_STATUS_PANIC` for a caught Rust panic. Initialize the output pointer
/// to null before calling.
///
/// Caller requirements:
///
/// `path` must point to a valid, readable NUL-terminated C string. `out_qodec`
/// must be non-null, aligned and writable for one root pointer. Both must
/// remain valid for the duration of the call.
// cbindgen copies this comment into qodec.h verbatim, so the requirements are
// written under a C heading rather than rustdoc's `# Safety`.
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qodec_load(path: *const c_char, out_qodec: *mut *mut Qodec) -> i32 {
    guard("qodec_load", QODEC_STATUS_PANIC, || {
        if out_qodec.is_null() {
            set_last_error("qodec_load: out_qodec is null");
            return QODEC_STATUS_INVALID_ARG;
        }
        if path.is_null() {
            set_last_error("qodec_load: path is null");
            return QODEC_STATUS_INVALID_ARG;
        }
        // SAFETY: caller guarantees `path` is a valid NUL-terminated string.
        let Ok(path) = unsafe { CStr::from_ptr(path) }.to_str() else {
            set_last_error("qodec_load: path is not valid UTF-8");
            return QODEC_STATUS_INVALID_ARG;
        };

        match qodec::Qodec::load(Path::new(path)) {
            Ok(qodec) => {
                let prepared = match Prepared::new(&qodec) {
                    Ok(prepared) => prepared,
                    Err(error) => {
                        let message = match error.downcast::<NulError>() {
                            Ok(error) => format!(
                                "qodec_load: C projection cannot represent a string containing NUL: {:?}",
                                String::from_utf8_lossy(&error.into_vec())
                            ),
                            Err(error) => format!("qodec_load: {error}"),
                        };
                        set_last_error(&message);
                        return QODEC_STATUS_ERROR;
                    }
                };
                // `root` is the first field of `Prepared`, so this is both the
                // root handed out and the allocation qodec_unload reclaims.
                let root = Box::into_raw(prepared).cast::<Qodec>();
                // SAFETY: out_qodec checked non-null above.
                unsafe { out_qodec.write(root) };
                QODEC_STATUS_OK
            }
            Err(error) => {
                set_last_error(&error.to_string());
                QODEC_STATUS_ERROR
            }
        }
    })
}

/// Release a qodec from `qodec_load()`, invalidating everything reachable from
/// it. Null is a no-op.
///
/// Caller requirements:
///
/// A non-null `qodec` must be exactly the pointer `qodec_load()` produced, not
/// previously unloaded and not a copy of the struct. The root and its reachable
/// storage must not have been modified or freed. All readers must have finished;
/// unloading must not race with any read. Unloading twice is undefined behavior.
// cbindgen copies this comment into qodec.h verbatim, so the requirements are
// written under a C heading rather than rustdoc's `# Safety`.
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qodec_unload(qodec: *mut Qodec) {
    const _: () = assert!(
        std::mem::offset_of!(Prepared, root) == 0,
        "this cast requires `root` to be the first field of `Prepared`"
    );

    if qodec.is_null() {
        return;
    }
    // SAFETY: `root` is the first field of `Prepared`, so this pointer is the
    // one Box::into_raw produced in qodec_load; single-shot per contract.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(qodec.cast::<Prepared>()) });
    }));
}

/// Find the first gadget in `layer` whose `implements.mnemonic` matches
/// `mnemonic`. The comparison is exact and case-sensitive.
///
/// Returns a borrowed pointer, valid until the owning qodec is unloaded. Do not
/// modify or free it. Returns null and records `qodec_last_error()` for no match,
/// a null argument, or a caught Rust panic.
///
/// Caller requirements:
///
/// `layer` must be null or point to a valid, unmodified layer in a live qodec.
/// `mnemonic` must point to a valid, readable NUL-terminated C string for the
/// duration of the call. Keep the owning qodec alive during the call and while
/// using the returned pointer; unloading must not race with either use.
// cbindgen copies this comment into qodec.h verbatim, so the requirements are
// written under a C heading rather than rustdoc's `# Safety`.
#[allow(clippy::missing_safety_doc)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn qodec_find_gadget(layer: *const QodecLayer, mnemonic: *const c_char) -> *const QodecGadget {
    guard("qodec_find_gadget", std::ptr::null(), || {
        if layer.is_null() {
            set_last_error("qodec_find_gadget: layer is null");
            return std::ptr::null();
        }
        if mnemonic.is_null() {
            set_last_error("qodec_find_gadget: mnemonic is null");
            return std::ptr::null();
        }
        // SAFETY: caller guarantees both pointers are valid.
        let (layer, wanted) = unsafe { (&*layer, CStr::from_ptr(mnemonic)) };
        if layer.gadgets.items.is_null() {
            set_last_error("qodec_find_gadget: this layer has no gadgets");
            return std::ptr::null();
        }
        // SAFETY: `items` is non-null and holds `count` initialized gadgets.
        let gadgets = unsafe { std::slice::from_raw_parts(layer.gadgets.items, layer.gadgets.count) };
        for gadget in gadgets {
            // SAFETY: every mnemonic is a NUL-terminated string built at open.
            if unsafe { CStr::from_ptr(gadget.implements.mnemonic) } == wanted {
                return std::ptr::from_ref(gadget);
            }
        }
        set_last_error(&format!(
            "no gadget implementing '{}' in this layer",
            wanted.to_string_lossy()
        ));
        std::ptr::null()
    })
}

#[cfg(test)]
mod tests {
    use super::{QODEC_STATUS_PANIC, guard, qodec_last_error};
    use std::ffi::CStr;

    #[test]
    fn a_panicking_body_returns_the_fallback_and_records_its_payload() {
        let status = guard("probe", QODEC_STATUS_PANIC, || panic!("deliberate {}", "failure"));
        assert_eq!(status, QODEC_STATUS_PANIC);

        let message = qodec_last_error();
        assert!(!message.is_null(), "a caught panic must record an error");
        // SAFETY: `qodec_last_error` returns a NUL-terminated thread-local buffer.
        let message = unsafe { CStr::from_ptr(message) }.to_string_lossy().into_owned();
        assert_eq!(message, "probe: panicked: deliberate failure");
    }
}
