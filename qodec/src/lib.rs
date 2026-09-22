#![warn(missing_docs)]
//! A model of quantum error-correction protocols.
//!
//! [`Qodec`] organizes a protocol into layers. Each layer has an instruction set
//! and gadgets that implement its instructions in the layer below. Codes define
//! the encodings used at each boundary.
//!
//! Build the model from Rust values or load it from YAML. [`Qodec::load`] and
//! [`Qodec::save`] support both directories of artifacts and single-file bundles.
//!
//! # Examples
//!
//! Load a qodec from disk and walk its lowering chain:
//!
//! ```
//! use qodec::Qodec;
//!
//! let qodec = Qodec::load("examples/repetition3/repetition3.qodec.yaml")?;
//! let [logical, physical] = qodec.layers() else {
//!     panic!("repetition3 has exactly two layers")
//! };
//!
//! assert_eq!(logical.instruction_set.name, "repetition3");
//! assert_eq!(physical.instruction_set.name, "stim+rz");
//!
//! // Each logical instruction is lowered by the gadget listed under it.
//! let measure_z = &logical.gadgets["measure_z"];
//! assert!(measure_z.circuit.source.contains('M'));
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Working with the model
//!
//! Public types are available at the crate root. [`Qodec`] contains [`Layer`]
//! values; each layer pairs an [`InstructionSet`] with its [`Gadget`]s. A gadget
//! holds a [`Circuit`], input and output [`Encoding`]s, and its checks and readouts.
//! Each parity equation stores parsed [`Reference`] values, one per authored
//! expression. Loading retains both the original path and its parsed
//! [`ReferenceSegment`] values. Use [`Reference::segments`] to inspect structure
//! or [`Reference::expand`] to expand the final index selector without reparsing;
//! saving preserves the original selector spelling.
//!
//! Use [`Qodec::new`] to assemble a qodec from Rust values.
//! [`Qodec::save`] writes separate artifact files; [`Qodec::save_bundle`] writes
//! a YAML bundle and any source sidecars. Both take a destination directory.
//! [`Qodec::to_bundle_string`] and [`Qodec::from_bundle_str`]
//! provide the text round trip without writing files.
//!
//! Artifact types such as [`Code`] and [`InstructionSet`] support serde. The
//! resolved [`Qodec`] and [`Gadget`] values instead hold linked model objects;
//! use the qodec load/save methods for the complete on-disk representation.
//! [`Code::validate`] and [`InstructionSet::validate`] check values built in memory.
//! [`Qodec::validate`] checks the current layers and their components together,
//! without file access, including values built with `new` or edited after loading.
//!
//! # Errors
//!
//! Loading returns [`LoadError`]; saving returns [`std::io::Error`]. Parsing a
//! reference returns [`ReferenceParseError`]. The documentation on each method
//! describes its failure conditions.
//!
//! # Stability
//!
//! qodec is in early development. The public API and on-disk format may change
//! between releases. The manifest's `schema_version` is independent of the
//! package version; when present, it must match [`CURRENT_SCHEMA_VERSION`].
//! See `CHANGELOG.md` for format changes and pin the crate version you depend on.

/// Annotations with string keys and JSON values.
///
/// Available on the manifest, instruction sets, codes, gadgets, and instructions.
/// qodec preserves these values without interpreting them. They participate in
/// structural equality.
///
/// Keys are sorted for deterministic serialization. No keys are reserved;
/// grouping them by tool can avoid collisions.
pub type Metadata = serde_json::Map<String, serde_json::Value>;

pub(crate) mod block;
pub(crate) mod code;
mod display;
pub(crate) mod error;
pub(crate) mod gadget;
pub(crate) mod inline_yaml_parser;
pub(crate) mod instruction;
pub(crate) mod instruction_set;
pub(crate) mod ir;
pub(crate) mod manifest;
pub(crate) mod node;
pub(crate) mod parity;
pub(crate) mod pauli;
pub(crate) mod qodec;
pub(crate) mod resolved;
pub(crate) mod sourced;
pub(crate) mod validation;
mod yaml_output;

pub(crate) use block::BlockName;
pub use code::Code;
pub use error::{ResolveError, SliceError};
pub(crate) use gadget::{CircuitSpec, EncodingSpec, GadgetSpec, Implements};
pub use instruction::{
    Action, ActionStep, BlockOperand, Condition, Instruction, Observable, Parameter, ParameterKind, Scalar,
};
pub use instruction_set::{Block, InstructionSet};
pub use ir::{Argument, CircuitReadout, InstructionCall, Operand, SelectPattern};
pub(crate) use ir::{circuit_blocks, circuit_readouts, require_declared_mnemonics};
pub use manifest::CURRENT_SCHEMA_VERSION;
pub(crate) use manifest::{LayerSpec, Manifest};
pub use node::{Node, PathError, SourceLocation};
pub(crate) use parity::ReadoutsList;
pub use parity::{
    MAX_SELECTED_POSITIONS, ParityEquation, ParityTerm, Readout, ReadoutSpec, Reference, ReferenceParseError,
    ReferenceSegment,
};
pub use pauli::PauliString;
pub(crate) use qodec::ParserRegistry;
pub use qodec::{LoadError, ParseError, Qodec, register};
pub use resolved::{Circuit, Encoding, Gadget, Layer};
pub(crate) use sourced::Sourced;
