//! PyO3 bindings for an in-process [`crate::server::LocalServer`].
//!
//! The Python `Runtime` class wraps a `LocalServer`. It owns the configured
//! decoder, coordinator and (optional) controller services in process so that
//! Python callers can drive them via the `Local` client variants — no gRPC
//! serialization, no loopback TCP. An optional [`bind`](PyRuntime::bind)
//! method spins up the tonic router on a TCP socket so external clients can
//! connect concurrently.
//!
//! The runtime exposes its services through namespaced sub-objects:
//!
//! * [`PyRuntime::coordinator`] — the lower-level `deq.bin` coordinator
//!   interface (`load_library` / `execute` / `decode` / `reset`), always
//!   available.
//! * [`PyRuntime::jit_controller`] — the JIT controller interface for dynamic
//!   circuits (`load_library` / `execute` / `batch_execute` / `decode` /
//!   `batch_decode` / `reset`), available when the runtime was constructed
//!   with `controller="jit"`.
//!
//! All RPC-style methods are coroutines (via
//! `pyo3_async_runtimes::tokio::future_into_py`). Request/response payloads
//! cross the FFI boundary as raw protobuf bytes; the high-level Python
//! wrapper in `deq/runtime/__init__.py` converts to/from `*_pb2` objects on
//! the Python side.

use crate::bin;
use crate::controller::JitController;
use crate::coordinator;
use crate::jit;
use crate::server::{LocalServer, ServerConfigs};
use clap::Parser;
use prost::Message;
use pyo3::exceptions::{PyAttributeError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use std::sync::Arc;
use tonic::Status;

// ── Runtime ─────────────────────────────────────────────────────────────────

/// In-process deq runtime.
///
/// Constructing the class builds the configured decoder, coordinator and
/// (optional) controller services and drives their `start()` methods. The
/// runtime itself does not bind a network port unless [`Self::bind`] is
/// called.
///
/// Services are exposed through namespaced getters:
///
/// * [`Self::coordinator`] — always present.
/// * [`Self::jit_controller`] — present iff constructed with `controller="jit"`.
#[pyclass(name = "Runtime")]
pub struct PyRuntime {
    inner: Arc<LocalServer>,
}

#[pymethods]
impl PyRuntime {
    /// Construct a new runtime.
    ///
    /// Configuration mirrors the `deq server` CLI: decoder, coordinator and
    /// controller types are selected by name with per-component JSON config
    /// strings. Pass `None` (or omit) to accept the defaults.
    ///
    /// Construction is synchronous because the services' `start()` methods
    /// must complete before the runtime is usable. The work runs on the
    /// dedicated `pyo3_async_runtimes::tokio` runtime with the GIL released.
    #[new]
    #[pyo3(signature = (
        *,
        decoder = None,
        decoder_config = None,
        coordinator = None,
        coordinator_config = None,
        controller = None,
        controller_config = None,
    ))]
    fn new(
        py: Python<'_>,
        decoder: Option<&str>,
        decoder_config: Option<&str>,
        coordinator: Option<&str>,
        coordinator_config: Option<&str>,
        controller: Option<&str>,
        controller_config: Option<&str>,
    ) -> PyResult<Self> {
        let configs = parse_configs(
            decoder,
            decoder_config,
            coordinator,
            coordinator_config,
            controller,
            controller_config,
        )?;
        let inner = py.detach(|| {
            let rt = pyo3_async_runtimes::tokio::get_runtime();
            rt.block_on(async move { configs.build_local().await })
        });
        Ok(Self { inner })
    }

    /// Bind a gRPC server on `addr` so external clients can connect
    /// concurrently with in-process Python callers. Returns the URL clients
    /// should use to connect (with the unspecified-address rewrite applied).
    ///
    /// `addr` defaults to `"[::]:0"`, which lets the OS pick a free port; read
    /// it back from the returned URL or via [`Self::bound_port`].
    #[pyo3(signature = (addr = "[::]:0".to_string()))]
    fn bind<'py>(&self, py: Python<'py>, addr: String) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let parsed: core::net::SocketAddr = addr
                .parse()
                .map_err(|e: core::net::AddrParseError| PyValueError::new_err(format!("invalid addr {addr:?}: {e}")))?;
            inner
                .bind_grpc(parsed)
                .await
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Return the port the gRPC server is currently bound to, or `None` if no
    /// network endpoint has been bound.
    fn bound_port(&self) -> Option<u16> {
        self.inner.bound_port()
    }

    /// Shut the runtime down cleanly: propagate cancellation into the
    /// in-process services so any pending `decode`/`execute` awaits resolve
    /// with a cancellation error, then stop the gRPC serve loop (if bound)
    /// and wait for it. Safe to call multiple times.
    ///
    /// Without this propagation, a Python coroutine awaiting `decode` on a
    /// gadget whose decoding window touches the open frontier would block
    /// forever and trigger a fatal Python error at interpreter shutdown
    /// when the leaked task is torn down.
    fn shutdown<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(
            py,
            async move { inner.shutdown().await.map_err(PyRuntimeError::new_err) },
        )
    }

    /// Namespaced access to the coordinator service (the `deq.bin` interface).
    /// Always present.
    #[getter]
    fn coordinator(&self) -> PyCoordinator {
        PyCoordinator {
            inner: self.inner.clone(),
        }
    }

    /// Namespaced access to the JIT controller (the `deq.jit` interface).
    /// Raises `AttributeError` if the runtime was not constructed with
    /// `controller="jit"`.
    #[getter]
    fn jit_controller(&self) -> PyResult<PyJitController> {
        match self.inner.jit_controller() {
            Some(controller) => Ok(PyJitController { controller }),
            None => Err(PyAttributeError::new_err(
                "jit_controller is not configured; \
                 pass controller=\"jit\" to Runtime(...) to enable it",
            )),
        }
    }

    /// True iff the JIT controller is available (i.e. `jit_controller` will
    /// not raise).
    fn has_jit_controller(&self) -> bool {
        self.inner.jit_controller().is_some()
    }

    fn __repr__(&self) -> String {
        let bound = self.inner.bound_port();
        let jit = if self.inner.jit_controller().is_some() {
            ", controller=jit"
        } else {
            ""
        };
        match bound {
            Some(port) => format!("Runtime(bound=:{port}{jit})"),
            None => format!("Runtime(unbound{jit})"),
        }
    }
}

// ── Coordinator ─────────────────────────────────────────────────────────────

/// Coordinator (`deq.bin`) interface for the in-process runtime.
///
/// Obtain instances via [`PyRuntime::coordinator`]. Each method takes a raw
/// protobuf-serialized payload; typed wrappers live in
/// `deq/runtime/__init__.py`.
#[pyclass(name = "Coordinator")]
pub struct PyCoordinator {
    inner: Arc<LocalServer>,
}

#[pymethods]
impl PyCoordinator {
    /// Load a [`bin::Library`] (protobuf-serialized bytes) into the
    /// coordinator. Defines all gadget, port, check-model and error-model
    /// types the subsequent program will reference.
    fn load_library<'py>(&self, py: Python<'py>, library: Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.coordinator_client();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let library = bin::Library::decode(library.as_slice())
                .map_err(|e| PyValueError::new_err(format!("failed to decode Library: {e}")))?;
            client.load_library(library).await.map_err(status_to_pyerr)
        })
    }

    /// Submit an [`bin::Instruction`] (protobuf-serialized bytes). Returns
    /// the assigned id (gid / cid / eid depending on the instruction kind).
    fn execute<'py>(&self, py: Python<'py>, instruction: Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.coordinator_client();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let instruction = bin::Instruction::decode(instruction.as_slice())
                .map_err(|e| PyValueError::new_err(format!("failed to decode Instruction: {e}")))?;
            client.execute(instruction).await.map(|resp| resp.id).map_err(status_to_pyerr)
        })
    }

    /// Submit measurement [`coordinator::Outcomes`] for a gadget and await the
    /// corresponding [`coordinator::Readouts`]. Returns the readouts as raw
    /// protobuf bytes; parse with `coordinator_pb2.Readouts.FromString`.
    fn decode<'py>(&self, py: Python<'py>, outcomes: Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.coordinator_client();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let outcomes = coordinator::Outcomes::decode(outcomes.as_slice())
                .map_err(|e| PyValueError::new_err(format!("failed to decode Outcomes: {e}")))?;
            let readouts = client.decode(outcomes).await.map_err(status_to_pyerr)?;
            Ok(encode_message(&readouts))
        })
    }

    /// Reset the coordinator. Defaults keep the loaded library and decoder
    /// instances; flip the corresponding flag to wipe them.
    #[pyo3(signature = (reset_library = false, reset_decoder_service = false, custom = String::new()))]
    fn reset<'py>(
        &self,
        py: Python<'py>,
        reset_library: bool,
        reset_decoder_service: bool,
        custom: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let client = self.inner.coordinator_client();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let req = coordinator::ResetRequest {
                reset_library,
                reset_decoder_service,
                custom,
            };
            client.reset(req).await.map_err(status_to_pyerr)
        })
    }

    fn __repr__(&self) -> String {
        "Coordinator()".to_string()
    }
}

// ── JIT controller ──────────────────────────────────────────────────────────

/// JIT controller (`deq.jit`) interface for dynamic-circuit decoding.
///
/// Obtain instances via [`PyRuntime::jit_controller`] (which raises
/// `AttributeError` if the runtime was not constructed with
/// `controller="jit"`). The JIT controller compiles `JitInstruction`s into
/// the coordinator's `bin` types on the fly, which is the surface most
/// online-decoding callers want.
#[pyclass(name = "JitController")]
pub struct PyJitController {
    controller: Arc<JitController>,
}

#[pymethods]
impl PyJitController {
    /// Load a [`jit::JitLibrary`] (protobuf-serialized bytes). The JIT
    /// compiler accumulates types across multiple `load_library` calls;
    /// duplicate type ids panic — load each type at most once. The underlying
    /// `bin` port/gadget types are forwarded to the coordinator so subsequent
    /// `execute` calls can reference them.
    fn load_library<'py>(&self, py: Python<'py>, library: Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        let controller = self.controller.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let library = jit::JitLibrary::decode(library.as_slice())
                .map_err(|e| PyValueError::new_err(format!("failed to decode JitLibrary: {e}")))?;
            controller.load_library(library).await.map_err(status_to_pyerr)
        })
    }

    /// Compile and execute a [`jit::JitInstruction`] (protobuf-serialized
    /// bytes). Returns the assigned `gid`.
    fn execute<'py>(&self, py: Python<'py>, instruction: Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        let controller = self.controller.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let instruction = jit::JitInstruction::decode(instruction.as_slice())
                .map_err(|e| PyValueError::new_err(format!("failed to decode JitInstruction: {e}")))?;
            Ok(controller.execute(instruction).await)
        })
    }

    /// Compile and execute a batch of [`jit::JitInstruction`]s, respecting
    /// connector-based dependencies between them. All instructions must
    /// specify a non-zero `gid`. Returns the assigned `gid`s in input order.
    fn batch_execute<'py>(&self, py: Python<'py>, instructions: Vec<Vec<u8>>) -> PyResult<Bound<'py, PyAny>> {
        let controller = self.controller.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let decoded: Vec<jit::JitInstruction> = instructions
                .into_iter()
                .enumerate()
                .map(|(idx, bytes)| {
                    jit::JitInstruction::decode(bytes.as_slice())
                        .map_err(|e| PyValueError::new_err(format!("failed to decode JitInstruction[{idx}]: {e}")))
                })
                .collect::<PyResult<_>>()?;
            controller
                .batch_execute(decoded)
                .await
                .map_err(|e| PyValueError::new_err(e.to_string()))
        })
    }

    /// Submit measurement [`coordinator::Outcomes`] (protobuf-serialized) for
    /// a previously-executed gadget. Returns the corresponding
    /// [`coordinator::Readouts`] as raw protobuf bytes.
    ///
    /// JIT decode waits for the gadget's background error-model loading task
    /// to finish before forwarding to the coordinator, so it is safe to
    /// invoke immediately after [`Self::execute`].
    fn decode<'py>(&self, py: Python<'py>, outcomes: Vec<u8>) -> PyResult<Bound<'py, PyAny>> {
        let controller = self.controller.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let outcomes = coordinator::Outcomes::decode(outcomes.as_slice())
                .map_err(|e| PyValueError::new_err(format!("failed to decode Outcomes: {e}")))?;
            let readouts = controller.decode_single(outcomes).await.map_err(status_to_pyerr)?;
            Ok(encode_message(&readouts))
        })
    }

    /// Decode multiple gadgets concurrently. Returns a list of readout byte
    /// blobs in the same order as the input.
    fn batch_decode<'py>(&self, py: Python<'py>, outcomes_list: Vec<Vec<u8>>) -> PyResult<Bound<'py, PyAny>> {
        let controller = self.controller.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let decoded: Vec<coordinator::Outcomes> = outcomes_list
                .into_iter()
                .enumerate()
                .map(|(idx, bytes)| {
                    coordinator::Outcomes::decode(bytes.as_slice())
                        .map_err(|e| PyValueError::new_err(format!("failed to decode Outcomes[{idx}]: {e}")))
                })
                .collect::<PyResult<_>>()?;
            let readouts = controller.batch_decode(decoded).await.map_err(status_to_pyerr)?;
            Ok(readouts.iter().map(encode_message).collect::<Vec<_>>())
        })
    }

    /// Reset the JIT controller (cancels pending error-model tasks, clears
    /// caches). Defaults keep the loaded library and decoder instances; flip
    /// the corresponding flag to wipe them.
    #[pyo3(signature = (reset_library = false, reset_decoder_service = false, custom = String::new()))]
    fn reset<'py>(
        &self,
        py: Python<'py>,
        reset_library: bool,
        reset_decoder_service: bool,
        custom: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        let controller = self.controller.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let req = coordinator::ResetRequest {
                reset_library,
                reset_decoder_service,
                custom,
            };
            controller.reset(req).await.map_err(status_to_pyerr)
        })
    }

    fn __repr__(&self) -> String {
        "JitController()".to_string()
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

fn encode_message<M: Message>(msg: &M) -> Vec<u8> {
    let mut buf = Vec::with_capacity(msg.encoded_len());
    msg.encode(&mut buf).expect("encode protobuf message");
    buf
}

/// Convert a tonic [`Status`] into a Python exception. We use `RuntimeError`
/// because tonic statuses cover many failure modes (invalid input, internal
/// error, cancelled, ...) — callers can pattern-match on the message if
/// needed.
fn status_to_pyerr(status: Status) -> PyErr {
    PyRuntimeError::new_err(format!("{}: {}", status.code(), status.message()))
}

/// Convert the keyword arguments accepted by `Runtime.__new__` into a
/// [`ServerConfigs`] by delegating to clap's parser. This keeps the set of
/// recognized configuration knobs in sync with the CLI without having to
/// reimplement the JSON-schema and enum parsing in Python.
fn parse_configs(
    decoder: Option<&str>,
    decoder_config: Option<&str>,
    coordinator: Option<&str>,
    coordinator_config: Option<&str>,
    controller: Option<&str>,
    controller_config: Option<&str>,
) -> PyResult<ServerConfigs> {
    let mut argv: Vec<String> = vec!["deq-runtime-python".to_string()];
    if let Some(value) = decoder {
        argv.push("--decoder".into());
        argv.push(value.into());
    }
    if let Some(value) = decoder_config {
        argv.push("--decoder-config".into());
        argv.push(value.into());
    }
    if let Some(value) = coordinator {
        argv.push("--coordinator".into());
        argv.push(value.into());
    }
    if let Some(value) = coordinator_config {
        argv.push("--coordinator-config".into());
        argv.push(value.into());
    }
    if let Some(value) = controller {
        argv.push("--controller".into());
        argv.push(value.into());
    }
    if let Some(value) = controller_config {
        argv.push("--controller-config".into());
        argv.push(value.into());
    }
    ServerConfigs::try_parse_from(argv).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Register the Python types defined in this module into `m`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRuntime>()?;
    m.add_class::<PyCoordinator>()?;
    m.add_class::<PyJitController>()?;
    #[cfg(feature = "simulator")]
    m.add_class::<PySampler>()?;
    Ok(())
}

// ── Sampler ─────────────────────────────────────────────────────────────────

/// Standalone measurement sampler driven by a Stim circuit string.
///
/// Unlike [`PyRuntime`], this class is not attached to a runtime — it owns
/// its own sampling backend and is used when you want physical measurement
/// outcomes from a Stim-format circuit and plan to feed them to a decoder
/// yourself (e.g. via [`PyCoordinator`] or [`PyJitController`]).
///
/// Stim is the **input format**, not the backend identity: the inner field
/// holds any [`crate::simulator::common::Sampler`] implementation. The
/// concrete backend is chosen via the ``simulator`` argument and
/// configured via ``simulator_config``, mirroring the
/// decoder/coordinator/controller selection pattern on [`PyRuntime`].
///
/// The constructor takes the Stim circuit as a Python string; if you have
/// a file path, read it first (`Path(path).read_text()`).
///
/// Each shot is returned as a protobuf-encoded
/// :class:`deq.proto.simulator_pb2.ShotSample` whose ``outcomes``
/// field holds the flat physical-measurement record. Callers that want
/// per-gadget chunks can slice it themselves using the per-gadget
/// measurement counts from their ``JitLibrary`` — the Python
/// :class:`deq.runtime.Sampler` wrapper provides ``split_outcomes``
/// for that.
#[cfg(feature = "simulator")]
#[pyclass(name = "Sampler")]
pub struct PySampler {
    inner: Arc<dyn crate::simulator::common::Sampler>,
}

#[cfg(feature = "simulator")]
#[pymethods]
impl PySampler {
    /// Construct a new sampler from a Stim circuit source string.
    ///
    /// Args:
    ///   circuit: The Stim circuit source (the contents of a `.stim` file).
    ///   simulator: Backend name. ``"stim"`` (default) uses Stim's compiled
    ///     measurement sampler, auto-wrapping with resample-on-failure when
    ///     the circuit has ``PREPARE { ... REQUIRE ... }`` blocks (QDK v1.30+).
    ///     ``"preselect"`` uses a tableau-based sampler that natively runs
    ///     each PREPARE block with retry-from-checkpoint semantics.
    ///   simulator_config: Optional JSON-string config for the backend.
    ///     Currently supports ``preselect_max_attempts`` (int).
    ///   seed: Optional deterministic seed. When None, a random seed is
    ///     drawn from the OS.
    ///   skip_shots: Number of initial shots to consume and discard. Useful
    ///     for resuming or for splitting a deterministic run across processes.
    #[new]
    #[pyo3(signature = (circuit, *, simulator = None, simulator_config = None, seed = None, skip_shots = 0))]
    fn new(
        circuit: &str,
        simulator: Option<&str>,
        simulator_config: Option<&str>,
        seed: Option<u64>,
        skip_shots: usize,
    ) -> PyResult<Self> {
        use crate::simulator::common::SamplerType;
        use rand::Rng;
        let name = simulator.unwrap_or("stim");
        let sampler_type = SamplerType::from_name(name).ok_or_else(|| {
            PyValueError::new_err(format!(
                "unknown simulator {name:?}; valid: {:?}",
                SamplerType::variant_names()
            ))
        })?;
        let config: serde_json::Value = match simulator_config {
            Some(s) if !s.is_empty() => {
                serde_json::from_str(s).map_err(|e| PyValueError::new_err(format!("invalid simulator_config JSON: {e}")))?
            }
            _ => serde_json::json!({}),
        };
        let seed = seed.unwrap_or_else(|| rand::rng().next_u64());
        let inner = sampler_type
            .create(circuit, seed, skip_shots, config)
            .map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    /// Sample `num_shots` shots from the circuit. Returns a list of
    /// protobuf-encoded :class:`deq.proto.simulator_pb2.ShotSample` byte
    /// strings — one per shot. The Python wrapper layer parses these into
    /// typed proto objects.
    ///
    /// The call releases the GIL while shots are produced, so other Python
    /// threads can make progress in parallel.
    fn sample<'py>(&self, py: Python<'py>, num_shots: usize) -> PyResult<Vec<Bound<'py, PyAny>>> {
        use crate::simulator::DeterministicRng;
        use pyo3::types::PyBytes;
        use rand::SeedableRng;
        let inner = self.inner.clone();
        let shots: Vec<Vec<u8>> = py.detach(|| {
            // The StimSampler backend ignores the rng (samples come from
            // its background thread), but the trait wants one.
            let mut rng = DeterministicRng::seed_from_u64(0);
            (0..num_shots)
                .map(|_| {
                    let error_set = inner.sample(&mut rng);
                    let shot = crate::simulator::common::error_set_to_shot_sample(&error_set);
                    let mut buf = Vec::with_capacity(shot.encoded_len());
                    shot.encode(&mut buf).expect("encoding ShotSample cannot fail");
                    buf
                })
                .collect()
        });
        Ok(shots.into_iter().map(|bytes| PyBytes::new(py, &bytes).into_any()).collect())
    }

    fn __repr__(&self) -> String {
        "Sampler()".to_string()
    }
}
