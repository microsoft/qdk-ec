//! Python wrappers for decorator AST types.

use ::deqagram::ast;
use pyo3::prelude::*;

/// A decorator value: a string literal (without quotes), an integer, or a float.
#[pyclass(name = "DecoratorValue", eq)]
#[derive(Clone, PartialEq)]
pub enum DecoratorValue {
    String { value: String },
    Int { value: i64 },
    Float { value: f64 },
}

impl From<&ast::DecoratorValue> for DecoratorValue {
    fn from(v: &ast::DecoratorValue) -> Self {
        match v {
            ast::DecoratorValue::String(s) => Self::String { value: s.clone() },
            ast::DecoratorValue::Int(i) => Self::Int { value: *i },
            ast::DecoratorValue::Float(x) => Self::Float { value: *x },
        }
    }
}

/// A decorator argument: a positional value or a `key=value` pair.
#[pyclass(name = "DecoratorArg", eq)]
#[derive(Clone, PartialEq)]
pub enum DecoratorArg {
    Value { value: DecoratorValue },
    Keyword { key: String, value: DecoratorValue },
}

impl From<&ast::DecoratorArg> for DecoratorArg {
    fn from(a: &ast::DecoratorArg) -> Self {
        match a {
            ast::DecoratorArg::Value(v) => Self::Value { value: v.into() },
            ast::DecoratorArg::Keyword { key, value } => Self::Keyword {
                key: key.clone(),
                value: value.into(),
            },
        }
    }
}

/// A decorator like `@GTYPE(1)`. `name` excludes the leading `@`.
#[pyclass(name = "Decorator", frozen, get_all, eq)]
#[derive(Clone, PartialEq)]
pub struct Decorator {
    pub name: String,
    pub arguments: Vec<DecoratorArg>,
}

impl From<&ast::Decorator> for Decorator {
    fn from(d: &ast::Decorator) -> Self {
        Self {
            name: d.name.clone(),
            arguments: d.arguments.iter().map(Into::into).collect(),
        }
    }
}
