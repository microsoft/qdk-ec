//! Python wrappers for statement AST types, including the recursive per-kind
//! body statement enums.

use ::deqagram::ast;
use pyo3::prelude::*;

use crate::decorators::Decorator;
use crate::targets::{
    Condition, ErrorTarget, LogicalPauliTarget, MeasurementRef, PauliTerm, PropagateTerm, ReadoutTargetItem, Target,
};

/// An embedded Stim instruction, e.g. `X_ERROR(0.03) 0 1 2`.
#[pyclass(name = "Instruction", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct Instruction {
    pub name: String,
    pub tag: Option<String>,
    pub arguments: Vec<f64>,
    pub targets: Vec<Target>,
}

impl From<&ast::Instruction> for Instruction {
    fn from(i: &ast::Instruction) -> Self {
        Self {
            name: i.name.clone(),
            tag: i.tag.clone(),
            arguments: i.arguments.clone(),
            targets: i.targets.iter().map(|t| (&t.node).into()).collect(),
        }
    }
}

/// An `INPUT`/`OUTPUT CodeName qubit_indices...` port declaration.
#[pyclass(name = "PortDeclaration", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct PortDeclaration {
    pub code_name: String,
    pub qubit_indices: Vec<u64>,
}

impl From<&ast::PortDeclaration> for PortDeclaration {
    fn from(p: &ast::PortDeclaration) -> Self {
        Self {
            code_name: p.code_name.node.clone(),
            qubit_indices: p.qubit_indices.clone(),
        }
    }
}

/// A gadget application with explicit port bindings, e.g. `Idle IN(0 1) OUT(0 1)`.
#[pyclass(name = "GadgetApplication", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct GadgetApplication {
    pub gadget_name: String,
    pub in_indices: Option<Vec<u64>>,
    pub out_indices: Option<Vec<u64>>,
}

impl From<&ast::GadgetApplication> for GadgetApplication {
    fn from(g: &ast::GadgetApplication) -> Self {
        Self {
            gadget_name: g.gadget_name.node.clone(),
            in_indices: g.in_indices.clone(),
            out_indices: g.out_indices.clone(),
        }
    }
}

/// A `READOUT`/`OBSERVABLE_INCLUDE targets... [FLIP]` statement.
#[pyclass(name = "ReadoutStatement", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct ReadoutStatement {
    pub targets: Vec<ReadoutTargetItem>,
    pub flip: bool,
}

impl From<&ast::ReadoutStatement> for ReadoutStatement {
    fn from(s: &ast::ReadoutStatement) -> Self {
        Self {
            targets: s.targets.iter().map(|t| (&t.node).into()).collect(),
            flip: s.flip,
        }
    }
}

/// A `CHECK`/`DETECTOR targets... [FLIP]` statement.
#[pyclass(name = "CheckStatement", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct CheckStatement {
    pub targets: Vec<Target>,
    pub flip: bool,
}

impl From<&ast::CheckStatement> for CheckStatement {
    fn from(s: &ast::CheckStatement) -> Self {
        Self {
            targets: s.targets.iter().map(|t| (&t.node).into()).collect(),
            flip: s.flip,
        }
    }
}

/// An `ERROR(p) targets...` statement.
#[pyclass(name = "ErrorStatement", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct ErrorStatement {
    pub probability: f64,
    pub targets: Vec<ErrorTarget>,
}

impl From<&ast::ErrorStatement> for ErrorStatement {
    fn from(s: &ast::ErrorStatement) -> Self {
        Self {
            probability: s.probability,
            targets: s.targets.iter().map(|t| (&t.node).into()).collect(),
        }
    }
}

/// A `CONDITIONAL condition L<P><i>...` statement.
#[pyclass(name = "ConditionalStatement", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct ConditionalStatement {
    pub condition: Condition,
    pub targets: Vec<LogicalPauliTarget>,
}

impl From<&ast::ConditionalStatement> for ConditionalStatement {
    fn from(s: &ast::ConditionalStatement) -> Self {
        Self {
            condition: s.condition.into(),
            targets: s.targets.iter().map(|t| t.node.into()).collect(),
        }
    }
}

/// A `VIRTUAL L<P><i>...` logical-correction statement (inside a GADGET).
#[pyclass(name = "VirtualLogicalStatement", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct VirtualLogicalStatement {
    pub targets: Vec<LogicalPauliTarget>,
}

impl From<&ast::VirtualLogicalStatement> for VirtualLogicalStatement {
    fn from(s: &ast::VirtualLogicalStatement) -> Self {
        Self {
            targets: s.targets.iter().map(|t| t.node.into()).collect(),
        }
    }
}

/// A `PROPAGATE L<P><i> FROM terms... [FLIP]` statement.
#[pyclass(name = "PropagateStatement", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct PropagateStatement {
    pub target: LogicalPauliTarget,
    pub terms: Vec<PropagateTerm>,
    pub flip: bool,
}

impl From<&ast::PropagateStatement> for PropagateStatement {
    fn from(s: &ast::PropagateStatement) -> Self {
        Self {
            target: s.target.node.into(),
            terms: s.terms.iter().map(|t| (&t.node).into()).collect(),
            flip: s.flip,
        }
    }
}

/// A `PRESELECT <target>+ [<bit>]` statement.
#[pyclass(name = "PreselectStatement", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct PreselectStatement {
    pub conditions: Vec<MeasurementRef>,
    pub expected_value: u64,
}

impl From<&ast::PreselectStatement> for PreselectStatement {
    fn from(s: &ast::PreselectStatement) -> Self {
        Self {
            conditions: s.conditions.iter().map(|c| (*c).into()).collect(),
            expected_value: s.expected_value,
        }
    }
}

/// An `ASSERT_EQ target value` statement.
#[pyclass(name = "AssertStatement", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct AssertStatement {
    pub target: Target,
    pub expected_value: u64,
}

impl From<&ast::AssertStatement> for AssertStatement {
    fn from(s: &ast::AssertStatement) -> Self {
        Self {
            target: (&s.target.node).into(),
            expected_value: s.expected_value,
        }
    }
}

/// A `VIRTUAL X0*Y1 wire` Pauli-correction statement (inside a PROGRAM).
#[pyclass(name = "VirtualCorrection", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct VirtualCorrection {
    pub paulis: Vec<PauliTerm>,
    pub wire: u64,
}

impl From<&ast::VirtualCorrection> for VirtualCorrection {
    fn from(s: &ast::VirtualCorrection) -> Self {
        Self {
            paulis: s
                .paulis
                .iter()
                .map(|&(pauli, index)| PauliTerm {
                    pauli: pauli.into(),
                    index,
                })
                .collect(),
            wire: s.wire,
        }
    }
}

/// A `CONDITIONAL rec[-k] X0*Y1 wire` conditional Pauli-correction statement.
#[pyclass(name = "ConditionalCorrection", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct ConditionalCorrection {
    pub readout_offset: u64,
    pub paulis: Vec<PauliTerm>,
    pub wire: u64,
}

impl From<&ast::ConditionalCorrection> for ConditionalCorrection {
    fn from(s: &ast::ConditionalCorrection) -> Self {
        Self {
            readout_offset: s.readout_offset,
            paulis: s
                .paulis
                .iter()
                .map(|&(pauli, index)| PauliTerm {
                    pauli: pauli.into(),
                    index,
                })
                .collect(),
            wire: s.wire,
        }
    }
}

/// A statement inside a `GADGET` body.
#[pyclass(name = "GadgetStatement", eq)]
#[derive(Clone, PartialEq)]
pub enum GadgetStatement {
    Instruction { instruction: Instruction },
    Repeat { count: u64, body: Vec<GadgetStatement> },
    InputPort { port: PortDeclaration },
    OutputPort { port: PortDeclaration },
    Readout { readout: ReadoutStatement },
    Check { check: CheckStatement },
    Error { error: ErrorStatement },
    Conditional { conditional: ConditionalStatement },
    VirtualLogical { statement: VirtualLogicalStatement },
    Propagate { propagate: PropagateStatement },
    Preselect { preselect: PreselectStatement },
    Decorator { decorator: Decorator },
}

impl From<&ast::GadgetStatement> for GadgetStatement {
    fn from(s: &ast::GadgetStatement) -> Self {
        match s {
            ast::GadgetStatement::Instruction(i) => Self::Instruction { instruction: i.into() },
            ast::GadgetStatement::Repeat { count, body } => Self::Repeat {
                count: *count,
                body: body.iter().map(|st| (&st.node).into()).collect(),
            },
            ast::GadgetStatement::InputPort(p) => Self::InputPort { port: p.into() },
            ast::GadgetStatement::OutputPort(p) => Self::OutputPort { port: p.into() },
            ast::GadgetStatement::Readout(r) => Self::Readout { readout: r.into() },
            ast::GadgetStatement::Check(c) => Self::Check { check: c.into() },
            ast::GadgetStatement::Error(e) => Self::Error { error: e.into() },
            ast::GadgetStatement::Conditional(c) => Self::Conditional { conditional: c.into() },
            ast::GadgetStatement::VirtualLogical(v) => Self::VirtualLogical { statement: v.into() },
            ast::GadgetStatement::Propagate(p) => Self::Propagate { propagate: p.into() },
            ast::GadgetStatement::Preselect(p) => Self::Preselect { preselect: p.into() },
            ast::GadgetStatement::Decorator(d) => Self::Decorator { decorator: d.into() },
        }
    }
}

/// A statement inside a `COMPOSE` body.
#[pyclass(name = "ComposeStatement", eq)]
#[derive(Clone, PartialEq)]
pub enum ComposeStatement {
    Instruction { instruction: Instruction },
    Repeat { count: u64, body: Vec<ComposeStatement> },
    InputPort { port: PortDeclaration },
    OutputPort { port: PortDeclaration },
    GadgetApplication { application: GadgetApplication },
    ConditionalCorrection { correction: ConditionalCorrection },
    Decorator { decorator: Decorator },
}

impl From<&ast::ComposeStatement> for ComposeStatement {
    fn from(s: &ast::ComposeStatement) -> Self {
        match s {
            ast::ComposeStatement::Instruction(i) => Self::Instruction { instruction: i.into() },
            ast::ComposeStatement::Repeat { count, body } => Self::Repeat {
                count: *count,
                body: body.iter().map(|st| (&st.node).into()).collect(),
            },
            ast::ComposeStatement::InputPort(p) => Self::InputPort { port: p.into() },
            ast::ComposeStatement::OutputPort(p) => Self::OutputPort { port: p.into() },
            ast::ComposeStatement::GadgetApplication(g) => Self::GadgetApplication { application: g.into() },
            ast::ComposeStatement::ConditionalCorrection(c) => Self::ConditionalCorrection { correction: c.into() },
            ast::ComposeStatement::Decorator(d) => Self::Decorator { decorator: d.into() },
        }
    }
}

/// A statement inside a `PROGRAM` body.
#[pyclass(name = "ProgramStatement", eq)]
#[derive(Clone, PartialEq)]
pub enum ProgramStatement {
    Instruction { instruction: Instruction },
    Repeat { count: u64, body: Vec<ProgramStatement> },
    InputPort { port: PortDeclaration },
    OutputPort { port: PortDeclaration },
    GadgetApplication { application: GadgetApplication },
    Assert { assertion: AssertStatement },
    VirtualCorrection { correction: VirtualCorrection },
    ConditionalCorrection { correction: ConditionalCorrection },
    Decorator { decorator: Decorator },
}

impl From<&ast::ProgramStatement> for ProgramStatement {
    fn from(s: &ast::ProgramStatement) -> Self {
        match s {
            ast::ProgramStatement::Instruction(i) => Self::Instruction { instruction: i.into() },
            ast::ProgramStatement::Repeat { count, body } => Self::Repeat {
                count: *count,
                body: body.iter().map(|st| (&st.node).into()).collect(),
            },
            ast::ProgramStatement::InputPort(p) => Self::InputPort { port: p.into() },
            ast::ProgramStatement::OutputPort(p) => Self::OutputPort { port: p.into() },
            ast::ProgramStatement::GadgetApplication(g) => Self::GadgetApplication { application: g.into() },
            ast::ProgramStatement::Assert(a) => Self::Assert { assertion: a.into() },
            ast::ProgramStatement::VirtualCorrection(v) => Self::VirtualCorrection { correction: v.into() },
            ast::ProgramStatement::ConditionalCorrection(c) => Self::ConditionalCorrection { correction: c.into() },
            ast::ProgramStatement::Decorator(d) => Self::Decorator { decorator: d.into() },
        }
    }
}
