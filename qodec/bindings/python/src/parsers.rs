//! Python callbacks stored in the core parser registry.
//!
//! Native callers keep string errors. Scoped error slots retain the original
//! exception for Python entry points, including nested callback invocations.

use std::cell::RefCell;
use std::sync::{Arc, Mutex, Weak};

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;

use crate::types::{PyInstructionCall, PyInstructionSet};

thread_local! {
    static CALLBACK_ERRORS: RefCell<Vec<Option<PyErr>>> = const { RefCell::new(Vec::new()) };
}

struct ErrorScope;

impl Drop for ErrorScope {
    fn drop(&mut self) {
        CALLBACK_ERRORS.with(|errors| errors.borrow_mut().pop());
    }
}

struct PythonParser {
    callable: Mutex<Option<Py<PyAny>>>,
}

#[pyclass(name = "_ParserLifetime", module = "qodec._native")]
struct ParserLifetime {
    parser: Weak<PythonParser>,
}

#[pymethods]
impl ParserLifetime {
    fn close(&self) -> PyResult<()> {
        if let Some(parser) = self.parser.upgrade() {
            let callable = parser
                .callable
                .lock()
                .map_err(|_| PyValueError::new_err("Python parser is unavailable"))?
                .take();
            drop(callable);
        }
        Ok(())
    }
}

impl PythonParser {
    fn call(
        &self,
        py: Python<'_>,
        source: &str,
        target: &qodec::InstructionSet,
    ) -> PyResult<Vec<qodec::InstructionCall>> {
        let callable = self
            .callable
            .lock()
            .map_err(|_| PyValueError::new_err("Python parser is unavailable"))?
            .as_ref()
            .map(|callable| callable.clone_ref(py))
            .ok_or_else(|| PyValueError::new_err("Python parser's interpreter has shut down"))?;
        invoke(py, callable.bind(py), source, target)
    }
}

pub(crate) fn with_errors<T>(operation: impl FnOnce() -> Result<T, String>) -> PyResult<T> {
    CALLBACK_ERRORS.with(|errors| errors.borrow_mut().push(None));
    let _scope = ErrorScope;
    operation().map_err(|message| {
        CALLBACK_ERRORS
            .with(|errors| errors.borrow_mut().last_mut().and_then(Option::take))
            .unwrap_or_else(|| PyValueError::new_err(message))
    })
}

pub(crate) fn callback_error(error: PyErr) -> String {
    let message = error.to_string();
    CALLBACK_ERRORS.with(|errors| {
        if let Some(current) = errors.borrow_mut().last_mut() {
            *current = Some(error);
        }
    });
    message
}

pub(crate) fn invoke(
    py: Python<'_>,
    parser: &Bound<'_, PyAny>,
    source: &str,
    instruction_set: &qodec::InstructionSet,
) -> PyResult<Vec<qodec::InstructionCall>> {
    if !parser.is_callable() {
        return Err(PyTypeError::new_err("parser must be callable"));
    }
    let target = Py::new(py, PyInstructionSet::from_inner(py, instruction_set.clone())?)?;
    let result = parser.call1((source, target))?;
    let sequence = py.import("collections.abc")?.getattr("Sequence")?;
    if !result.is_instance(&sequence)? {
        return Err(PyTypeError::new_err(
            "parser must return a sequence of InstructionCall values",
        ));
    }
    result
        .try_iter()?
        .enumerate()
        .map(|(position, item)| {
            let item = item?;
            let call = item
                .extract::<PyRef<'_, PyInstructionCall>>()
                .map_err(|_| PyTypeError::new_err(format!("parser result[{position}] is not an InstructionCall")))?;
            call.to_call(py)
        })
        .collect()
}

#[pyfunction]
#[pyo3(signature = (parser, *, format))]
/// Register a Python source parser in the shared Rust registry. Latest registration wins.
pub(crate) fn register(parser: Py<PyAny>, format: &str, py: Python<'_>) -> PyResult<()> {
    if !parser.bind(py).is_callable() {
        return Err(PyTypeError::new_err("parser must be callable"));
    }
    let parser = Arc::new(PythonParser {
        callable: Mutex::new(Some(parser)),
    });
    let lifetime = Py::new(
        py,
        ParserLifetime {
            parser: Arc::downgrade(&parser),
        },
    )?;
    py.import("atexit")?
        .call_method1("register", (lifetime.bind(py).getattr("close")?,))?;
    qodec::register(
        move |source, target| {
            Python::try_attach(|py| parser.call(py, source, target).map_err(callback_error))
                .unwrap_or_else(|| Err("Python parser requires a running interpreter".to_owned()))
        },
        format,
    )
    .map_err(PyValueError::new_err)
}

#[cfg(feature = "test-support")]
pub(crate) mod test_support {
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use std::sync::Arc;

    use crate::types::{PyInstructionCall, instruction_call_to_py};

    #[pyfunction]
    #[pyo3(signature = (format, *, expected_integer = None))]
    pub(crate) fn _test_native_calls(
        py: Python<'_>,
        format: String,
        expected_integer: Option<i64>,
    ) -> PyResult<Vec<PyInstructionCall>> {
        let calls = py
            .detach(|| {
                std::thread::spawn(move || {
                    let instruction_set = serde_yaml::from_str(
                        "name: test\nblocks: {}\ninstructions: [{mnemonic: idle, description: ''}]",
                    )
                    .map_err(|error| error.to_string())?;
                    qodec::Circuit {
                        instruction_set: Arc::new(instruction_set),
                        source: "authored source".to_owned(),
                        format: Some(format),
                    }
                    .calls()
                })
                .join()
            })
            .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("native parser thread panicked"))?
            .map_err(PyValueError::new_err)?;
        if let Some(expected) = expected_integer
            && !matches!(
                calls.as_slice(),
                [call] if call.arguments.get("value") == Some(&qodec::Argument::Integer(expected))
            )
        {
            return Err(pyo3::exceptions::PyAssertionError::new_err(format!(
                "expected one call with value=Integer({expected}), got {calls:?}"
            )));
        }
        Ok(calls.iter().map(instruction_call_to_py).collect())
    }

    #[pyfunction]
    pub(crate) fn _test_register_native_empty(format: &str) -> PyResult<()> {
        qodec::register(|_, _| Ok(Vec::new()), format).map_err(PyValueError::new_err)
    }
}
