//! deq runtime (software)
//!
//! This module provides a distributed decoding system runtime.

#[cfg(feature = "cli")]
pub mod benchmark;
#[cfg(feature = "cli")]
pub mod cli;
pub mod controller;
pub mod coordinator;
pub mod decoder;
pub mod jit;
pub mod misc;
#[cfg(all(feature = "python_binding", feature = "cli"))]
pub mod python;
#[cfg(feature = "cli")]
pub mod server;
pub mod signal_checker;
pub mod simulator;
pub mod visualizer;

pub use crate::signal_checker::SIGNAL_CHECKER;

#[allow(clippy::large_enum_variant)]
pub mod bin {
    include!("proto/deq.bin.rs");
}
pub mod util {
    include!("proto/deq.util.rs");
}

#[cfg(feature = "python_binding")]
use pyo3::prelude::*;

#[cfg(feature = "python_binding")]
#[pymodule]
fn deq_runtime(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<decoder::python_decoder::PyDecodingHypergraph>()?;
    m.add_class::<decoder::python_decoder::PyHyperedge>()?;
    #[cfg(feature = "cli")]
    {
        m.add_function(wrap_pyfunction!(cli_run, m)?)?;
        python::register(m)?;
    }
    m.add_function(wrap_pyfunction!(jit::py_static_jit_compile, m)?)?;
    Ok(())
}

#[cfg(all(feature = "python_binding", feature = "cli"))]
#[pyfunction]
#[pyo3(signature = (*args, **kwargs))]
fn cli_run(
    py: Python<'_>,
    args: &Bound<'_, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<()> {
    let mut parameters: Vec<String> = vec!["deq".to_string()];
    for arg in args.iter() {
        let arg_str: String = arg.extract()?;
        parameters.push(arg_str);
    }
    if let Some(kwargs) = kwargs {
        for (key, value) in kwargs.iter() {
            let key_str: String = key.extract()?;
            // Convert Python-style underscores to CLI-style dashes
            let key_str = key_str.replace('_', "-");
            let value_str: String = value.extract()?;
            parameters.push(format!("--{key_str}"));
            parameters.push(value_str);
        }
    }
    use crate::cli::Cli;
    use clap::Parser;
    use tokio::runtime::Runtime;

    let cli = Cli::try_parse_from(parameters).map_err(|e| {
        // Print the error/help message but don't exit the process
        let _ = e.print();
        pyo3::exceptions::PyValueError::new_err(e.to_string())
    })?;
    // Release the GIL so the signal checker can acquire it for Ctrl+C handling
    py.detach(move || {
        let rt = Runtime::new().unwrap();
        rt.block_on(async {
            cli.run().await;
        });
    });
    Ok(())
}
