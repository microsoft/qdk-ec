//! Python wrappers for target, Pauli, and operator AST types.
//!
//! Each `Py*` type mirrors a `deqagram::ast` type and converts from it via
//! `From`. Enums map to pyo3 complex enums; AST tuple variants become named
//! struct variants here for a cleaner Python API. Source spans are not exposed.

use ::deqagram::ast;
use pyo3::prelude::*;

/// A Pauli letter.
#[pyclass(name = "Pauli", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum Pauli {
    I,
    X,
    Y,
    Z,
}

impl From<ast::Pauli> for Pauli {
    fn from(p: ast::Pauli) -> Self {
        match p {
            ast::Pauli::I => Self::I,
            ast::Pauli::X => Self::X,
            ast::Pauli::Y => Self::Y,
            ast::Pauli::Z => Self::Z,
        }
    }
}

/// The `IN`/`OUT` side of a port.
#[pyclass(name = "PortKind", eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum PortKind {
    In,
    Out,
}

impl From<ast::PortKind> for PortKind {
    fn from(k: ast::PortKind) -> Self {
        match k {
            ast::PortKind::In => Self::In,
            ast::PortKind::Out => Self::Out,
        }
    }
}

/// The `IN`/`OUT` side and index of a port-scoped target.
#[pyclass(name = "Port", frozen, get_all, eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct Port {
    pub kind: PortKind,
    pub index: u64,
}

impl From<ast::Port> for Port {
    fn from(p: ast::Port) -> Self {
        Self {
            kind: p.kind.into(),
            index: p.index,
        }
    }
}

/// A logical Pauli operator: `LX0`, or port-scoped `IN0.LX0` / `OUT1.LZ2`.
#[pyclass(name = "LogicalPauliTarget", frozen, get_all, eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct LogicalPauliTarget {
    pub pauli: Pauli,
    pub index: u64,
    pub port: Option<Port>,
}

impl From<ast::LogicalPauliTarget> for LogicalPauliTarget {
    fn from(t: ast::LogicalPauliTarget) -> Self {
        Self {
            pauli: t.pauli.into(),
            index: t.index,
            port: t.port.map(Into::into),
        }
    }
}

/// A single Pauli term on a qubit, e.g. `Z0`.
#[pyclass(name = "PauliTerm", frozen, get_all, eq)]
#[derive(Clone, Copy, PartialEq)]
pub struct PauliTerm {
    pub pauli: Pauli,
    pub index: u64,
}

impl From<ast::PauliTerm> for PauliTerm {
    fn from(t: ast::PauliTerm) -> Self {
        Self {
            pauli: t.pauli.into(),
            index: t.index,
        }
    }
}

/// A product of Pauli terms (`Z0*Z1`) or the identity (`_`).
#[pyclass(name = "PauliProduct", eq)]
#[derive(Clone, PartialEq)]
pub enum PauliProduct {
    Identity {},
    Terms { terms: Vec<PauliTerm> },
}

impl From<&ast::PauliProduct> for PauliProduct {
    fn from(p: &ast::PauliProduct) -> Self {
        match p {
            ast::PauliProduct::Identity => Self::Identity {},
            ast::PauliProduct::Terms(terms) => Self::Terms {
                terms: terms.iter().map(|t| (*t).into()).collect(),
            },
        }
    }
}

/// A target of an embedded Stim instruction, `CHECK`, or `ASSERT_EQ`.
#[pyclass(name = "Target", eq)]
#[derive(Clone, PartialEq)]
pub enum Target {
    Qubit { inverted: bool, index: u64 },
    Pauli { inverted: bool, pauli: Pauli, index: u64 },
    MeasurementRecord { offset: u64 },
    PhysicalMeasurement { index: u64 },
    InputVirtual { port: u64, stabilizer: u64 },
    OutputVirtual { port: u64, stabilizer: u64 },
    SweepBit { index: u64 },
    Combiner {},
}

impl From<&ast::Target> for Target {
    fn from(t: &ast::Target) -> Self {
        match *t {
            ast::Target::Qubit { inverted, index } => Self::Qubit { inverted, index },
            ast::Target::Pauli { inverted, pauli, index } => Self::Pauli {
                inverted,
                pauli: pauli.into(),
                index,
            },
            ast::Target::MeasurementRecord { offset } => Self::MeasurementRecord { offset },
            ast::Target::PhysicalMeasurement { index } => Self::PhysicalMeasurement { index },
            ast::Target::InputVirtual { port, stabilizer } => Self::InputVirtual { port, stabilizer },
            ast::Target::OutputVirtual { port, stabilizer } => Self::OutputVirtual { port, stabilizer },
            ast::Target::SweepBit { index } => Self::SweepBit { index },
            ast::Target::Combiner => Self::Combiner {},
        }
    }
}

/// A measurement reference usable in `CONDITIONAL` / `PRESELECT` conditions.
#[pyclass(name = "MeasurementRef", eq)]
#[derive(Clone, Copy, PartialEq)]
pub enum MeasurementRef {
    Record { offset: u64 },
    Physical { index: u64 },
    InputVirtual { port: u64, stabilizer: u64 },
    OutputVirtual { port: u64, stabilizer: u64 },
}

impl From<ast::MeasurementRef> for MeasurementRef {
    fn from(m: ast::MeasurementRef) -> Self {
        match m {
            ast::MeasurementRef::Record { offset } => Self::Record { offset },
            ast::MeasurementRef::Physical { index } => Self::Physical { index },
            ast::MeasurementRef::InputVirtual { port, stabilizer } => Self::InputVirtual { port, stabilizer },
            ast::MeasurementRef::OutputVirtual { port, stabilizer } => Self::OutputVirtual { port, stabilizer },
        }
    }
}

/// An item of a `READOUT` statement.
#[pyclass(name = "ReadoutTargetItem", eq)]
#[derive(Clone, PartialEq)]
pub enum ReadoutTargetItem {
    Target { target: Target },
    Logical { logical: LogicalPauliTarget },
    Destabilizer { port: u64, stabilizer: u64 },
}

impl From<&ast::ReadoutTargetItem> for ReadoutTargetItem {
    fn from(item: &ast::ReadoutTargetItem) -> Self {
        match item {
            ast::ReadoutTargetItem::Target(t) => Self::Target { target: t.into() },
            ast::ReadoutTargetItem::Logical(l) => Self::Logical { logical: (*l).into() },
            ast::ReadoutTargetItem::Destabilizer { port, stabilizer } => Self::Destabilizer {
                port: *port,
                stabilizer: *stabilizer,
            },
        }
    }
}

/// A target of an `ERROR` statement.
#[pyclass(name = "ErrorTarget", eq)]
#[derive(Clone, PartialEq)]
pub enum ErrorTarget {
    Check { index: u64 },
    Readout { index: u64 },
    Logical { logical: LogicalPauliTarget },
}

impl From<&ast::ErrorTarget> for ErrorTarget {
    fn from(t: &ast::ErrorTarget) -> Self {
        match *t {
            ast::ErrorTarget::Check(index) => Self::Check { index },
            ast::ErrorTarget::Readout(index) => Self::Readout { index },
            ast::ErrorTarget::Logical(l) => Self::Logical { logical: l.into() },
        }
    }
}

/// A term after `FROM` in a `PROPAGATE` statement.
#[pyclass(name = "PropagateTerm", eq)]
#[derive(Clone, PartialEq)]
pub enum PropagateTerm {
    Logical { logical: LogicalPauliTarget },
    Destabilizer { port: u64, stabilizer: u64 },
    MeasurementRecord { offset: u64 },
    PhysicalMeasurement { index: u64 },
    Readout { index: u64 },
}

impl From<&ast::PropagateTerm> for PropagateTerm {
    fn from(t: &ast::PropagateTerm) -> Self {
        match *t {
            ast::PropagateTerm::Logical(l) => Self::Logical { logical: l.into() },
            ast::PropagateTerm::Destabilizer { port, stabilizer } => Self::Destabilizer { port, stabilizer },
            ast::PropagateTerm::MeasurementRecord { offset } => Self::MeasurementRecord { offset },
            ast::PropagateTerm::PhysicalMeasurement { index } => Self::PhysicalMeasurement { index },
            ast::PropagateTerm::Readout { index } => Self::Readout { index },
        }
    }
}

/// The condition of a `CONDITIONAL` statement.
#[pyclass(name = "Condition", eq)]
#[derive(Clone, Copy, PartialEq)]
pub enum Condition {
    Readout { index: u64 },
    Measurement { measurement: MeasurementRef },
}

impl From<ast::Condition> for Condition {
    fn from(c: ast::Condition) -> Self {
        match c {
            ast::Condition::Readout(index) => Self::Readout { index },
            ast::Condition::Measurement(m) => Self::Measurement { measurement: m.into() },
        }
    }
}
