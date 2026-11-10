//! Decoder with thread pooling
//!

use super::worker_trace::WorkerTrace;
use crate::decoder::blackbox_decoder::{self, ParityFactor, black_box_decoder_server};
pub use crate::decoder::decoder_features::DecoderFeatures;
use crate::misc::bit_vector;
#[cfg(feature = "cli")]
use crate::misc::fastrace::Span;
use crate::misc::validation;
use crate::util::BitVector;
use blackbox_decoder::DecodingHypergraph;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::LinkedList;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore, oneshot, watch};
#[cfg(feature = "cli")]
use tonic::transport::server::Router;
use tonic::{Request, Response, Status};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
pub struct ThreadPoolingConfig {
    /// number of parallel threads in the pool, default to number of logical cores
    #[serde(default)]
    pub parallel: usize,
}

pub struct ThreadPoolingDecoder<T: DecoderInstance> {
    pub config: ThreadPoolingConfig,
    pub original_config: Arc<serde_json::Value>,
    pub thread_pool: Arc<rayon::ThreadPool>,
    work_slots: Arc<Semaphore>,
    caller: Option<std::thread::Thread>,
    loaded: Arc<Mutex<HashMap<u64, Loaded<T>>>>,
    next_hid: AtomicU64,
    decoding: watch::Sender<usize>,
    features: DecoderFeatures,
}

/// Immutable graph bound to one decoder/configuration on first use; owns its lazy preparation.
/// On successful validation, stores `Arc<PreparedHypergraph<T>>` erased behind `Any` so the
/// coordinator need not know the decoder type. `prepared::<T>` recovers it with a checked
/// runtime downcast; requesting another decoder type returns an error.
pub(crate) struct SharedDecodingHypergraph {
    hypergraph: Arc<DecodingHypergraph>,
    preparation: OnceLock<Result<Arc<dyn Any + Send + Sync>, Status>>,
}

impl SharedDecodingHypergraph {
    pub(crate) fn new(hypergraph: DecodingHypergraph) -> Self {
        Self {
            hypergraph: Arc::new(hypergraph),
            preparation: OnceLock::new(),
        }
    }

    pub(crate) fn graph(&self) -> &DecodingHypergraph {
        &self.hypergraph
    }

    fn prepared<T: DecoderInstance + 'static>(
        &self,
        configuration: &Arc<serde_json::Value>,
    ) -> Result<Arc<PreparedHypergraph<T>>, Status> {
        let prepared = self
            .preparation
            .get_or_init(|| {
                validation::validate_hypergraph(&self.hypergraph).map_err(Status::invalid_argument)?;
                T::validate_hypergraph(&self.hypergraph, configuration).map_err(Status::invalid_argument)?;
                Ok(Arc::new(PreparedHypergraph::<T> {
                    hypergraph: Arc::downgrade(&self.hypergraph),
                    configuration: Arc::downgrade(configuration),
                    factory: OnceLock::new(),
                }))
            })
            .as_ref()
            .map_err(Clone::clone)?;
        let prepared = Arc::clone(prepared)
            .downcast::<PreparedHypergraph<T>>()
            .map_err(|_| Status::invalid_argument("shared graph is bound to another decoder type"))?;
        if !prepared.matches(&self.hypergraph, configuration) {
            return Err(Status::invalid_argument(
                "shared graph is bound to another decoder configuration",
            ));
        }
        Ok(prepared)
    }
}

impl std::ops::Deref for SharedDecodingHypergraph {
    type Target = DecodingHypergraph;

    fn deref(&self) -> &Self::Target {
        self.graph()
    }
}

/// Immutable backend setup that constructs a fresh, independent decoder on each call.
/// Factories receive the same graph and configuration that were prepared.
pub type DecoderFactory<T> = Arc<dyn Fn(&DecodingHypergraph, &serde_json::Value) -> T + Send + Sync>;

struct PreparedHypergraph<T> {
    hypergraph: std::sync::Weak<DecodingHypergraph>,
    configuration: std::sync::Weak<serde_json::Value>,
    factory: OnceLock<Result<DecoderFactory<T>, String>>,
}

impl<T: DecoderInstance + 'static> PreparedHypergraph<T> {
    fn matches(&self, hypergraph: &Arc<DecodingHypergraph>, configuration: &Arc<serde_json::Value>) -> bool {
        std::ptr::eq(self.hypergraph.as_ptr(), Arc::as_ptr(hypergraph))
            && std::ptr::eq(self.configuration.as_ptr(), Arc::as_ptr(configuration))
    }

    fn new_instance(
        &self,
        hypergraph: &Arc<DecodingHypergraph>,
        configuration: &Arc<serde_json::Value>,
    ) -> Result<T, DecodeError> {
        if !self.matches(hypergraph, configuration) {
            return Err(DecodeError::InvalidInput(
                "prepared decoder graph or configuration changed".into(),
            ));
        }
        self.factory
            .get_or_init(|| T::prepare(hypergraph, configuration))
            .as_ref()
            .map(|factory| factory(hypergraph, configuration))
            .map_err(|error| DecodeError::InvalidInput(error.clone()))
    }
}

pub struct Loaded<T: DecoderInstance> {
    hypergraph: Arc<SharedDecodingHypergraph>,
    prepared: Arc<PreparedHypergraph<T>>,
    instances: LinkedList<T>,
}

// Queued requests count toward reset; admitted work retains its slot through worker completion.
struct DecodingGuard {
    tx: watch::Sender<usize>,
    slot: Option<OwnedSemaphorePermit>,
}

impl DecodingGuard {
    fn new(tx: watch::Sender<usize>) -> Self {
        Self { tx, slot: None }
    }
}

impl Drop for DecodingGuard {
    fn drop(&mut self) {
        drop(self.slot.take());
        self.tx.send_modify(|v| *v -= 1);
    }
}

impl<T: DecoderInstance> std::fmt::Debug for Loaded<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Loaded")
            .field("hypergraph", self.hypergraph.graph())
            .finish_non_exhaustive()
    }
}

pub struct DecodeRequest<'a> {
    pub syndrome: &'a BitVector,
    /// Optional deterministic seed for decoder randomness. Zero is valid.
    pub decoder_seed: Option<u64>,
    /// Shot-scoped prior assignments. Implementations must not let these values
    /// affect a later request served by the same pooled instance.
    pub reweights: &'a [(u64, f64)],
    /// Structured loss observation for this shot. Independent of `reweights`;
    /// a decoder advertising both features must accept both together.
    pub loss: Option<&'a blackbox_decoder::LossInfo>,
}

impl DecodeRequest<'_> {
    fn required_features(&self) -> DecoderFeatures {
        DecoderFeatures::required(self.decoder_seed.is_some(), !self.reweights.is_empty(), self.loss.is_some())
    }

    fn require_supported(&self, supported: DecoderFeatures) -> Result<(), DecodeError> {
        self.required_features()
            .require_supported_by(supported)
            .map_err(DecodeError::Unsupported)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    Unsupported(DecoderFeatures),
    InvalidInput(String),
    Backend(String),
}

impl fmt::Display for DecodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported(features) => write!(formatter, "unsupported decoder features: {features}"),
            Self::InvalidInput(message) => write!(formatter, "invalid decode request: {message}"),
            Self::Backend(message) => write!(formatter, "decoder backend failed: {message}"),
        }
    }
}

impl std::error::Error for DecodeError {}

pub trait DecoderInstance {
    #[must_use]
    fn supported_features(_config: &serde_json::Value) -> DecoderFeatures {
        DecoderFeatures::empty()
    }

    /// Validates constraints imposed by a specific decoder backend.
    ///
    /// # Errors
    ///
    /// Returns an error when the hypergraph cannot be represented by the backend.
    fn validate_hypergraph(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Result<(), String> {
        Ok(())
    }

    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self;

    /// Prepare reusable immutable setup for a validated graph and configuration.
    /// The factory must create independent mutable state for every instance.
    /// Backends without separate setup reuse their ordinary constructor.
    ///
    /// # Errors
    /// Returns an error when backend configuration or initialization data is invalid.
    fn prepare(_hypergraph: &DecodingHypergraph, _config: &serde_json::Value) -> Result<DecoderFactory<Self>, String>
    where
        Self: Sized + 'static,
    {
        Ok(Arc::new(Self::new))
    }

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError>;

    fn reset(&mut self);
}

impl<T: DecoderInstance + Send + 'static> ThreadPoolingDecoder<T> {
    async fn acquire_work_slot(&self) -> Result<DecodingGuard, Status> {
        self.decoding.send_modify(|count| *count += 1);
        let mut guard = DecodingGuard::new(self.decoding.clone());
        guard.slot = Some(
            Arc::clone(&self.work_slots)
                .acquire_owned()
                .await
                .map_err(|_| Status::unavailable("decoder execution closed"))?,
        );
        Ok(guard)
    }

    pub(crate) async fn unload_hypergraphs(&self, hids: &[u64]) {
        let mut loaded = self.loaded.lock().await;
        for hid in hids {
            loaded.remove(hid);
        }
    }

    fn schedule_work(&self, work: impl FnOnce() + Send + 'static) {
        if self.thread_pool.current_thread_index().is_some() && crate::misc::pool_executor::inline_cpu_allowed() {
            crate::misc::pool_executor::flush_local(&self.thread_pool);
            work();
        } else {
            self.thread_pool.spawn(move || crate::misc::pool_executor::execute_cpu(work));
            if let Some(caller) = &self.caller {
                caller.unpark();
            }
        }
    }

    #[cfg(all(test, feature = "cli"))]
    pub(crate) fn pending_decodes(&self) -> watch::Receiver<usize> {
        self.decoding.subscribe()
    }

    #[must_use]
    pub fn features(&self) -> DecoderFeatures {
        self.features
    }

    /// # Panics
    ///
    /// Panics when `original_config` is invalid for thread pooling or the Rayon
    /// pool cannot be created.
    #[must_use]
    pub fn new(original_config: serde_json::Value) -> Self {
        Self::with_thread_pool(original_config, None)
    }

    /// Construct a decoder using an existing worker pool, or create its own.
    ///
    /// A caller-participating pool can eliminate ingress handoffs without
    /// increasing the configured number of computation threads.
    /// Construct on that caller thread so queued jobs can wake its parked driver.
    ///
    /// # Panics
    /// Panics on invalid configuration or failure to create a worker pool.
    #[must_use]
    pub fn with_thread_pool(original_config: serde_json::Value, thread_pool: Option<Arc<rayon::ThreadPool>>) -> Self {
        let mut config: ThreadPoolingConfig = serde_json::from_value(original_config.clone()).unwrap();
        let features = T::supported_features(&original_config);
        let thread_pool = if let Some(thread_pool) = thread_pool {
            config.parallel = thread_pool.current_num_threads();
            thread_pool
        } else {
            let mut thread_pool_builder = rayon::ThreadPoolBuilder::new();
            if config.parallel != 0 {
                thread_pool_builder = thread_pool_builder.num_threads(config.parallel);
            }
            Arc::new(
                thread_pool_builder
                    .panic_handler(|error| {
                        eprintln!("rayon pool thread panicked: {error:?}");
                    })
                    .build()
                    .expect("creating thread pool failed"),
            )
        };
        let caller = thread_pool.current_thread_index().map(|_| std::thread::current());
        let work_slots = Arc::new(Semaphore::new(thread_pool.current_num_threads()));
        Self {
            config,
            original_config: Arc::new(original_config),
            thread_pool,
            work_slots,
            caller,
            loaded: Default::default(),
            next_hid: AtomicU64::new(1),
            decoding: watch::channel(0).0,
            features,
        }
    }

    #[cfg(feature = "cli")]
    #[must_use]
    pub fn add_service(self: &Arc<Self>, router: Router) -> Router {
        let service =
            black_box_decoder_server::BlackBoxDecoderServer::from_arc(self.clone()).max_decoding_message_size(usize::MAX);
        router.add_service(service)
    }
}

impl<T: DecoderInstance + Send + 'static> ThreadPoolingDecoder<T> {
    pub(crate) async fn decode_shared(
        &self,
        hypergraph: Arc<SharedDecodingHypergraph>,
        syndrome: BitVector,
        loss: Option<blackbox_decoder::LossInfo>,
        decoder_seed: Option<u64>,
    ) -> Result<ParityFactor, Status> {
        let prepared = hypergraph.prepared::<T>(&self.original_config)?;
        validation::validate_syndrome(&syndrome, hypergraph.vertex_num).map_err(Status::invalid_argument)?;
        validation::validate_loss(loss.as_ref(), hypergraph.hyperedges.len()).map_err(Status::invalid_argument)?;
        let request = DecodeRequest {
            syndrome: &syndrome,
            decoder_seed,
            reweights: &[],
            loss: loss.as_ref(),
        };
        request.require_supported(self.features).map_err(decode_error_status)?;
        // Observable metadata and optional request fields can affect a
        // zero-syndrome correction.
        if bit_vector::is_zero(&syndrome)
            && request.required_features().is_empty()
            && !self.features.contains(DecoderFeatures::OBSERVABLES)
        {
            #[cfg(feature = "cli")]
            let _span = Span::enter_with_local_parent("decoder_zero_syndrome");
            return Ok(ParityFactor { subgraph: vec![] });
        }
        let (tx, rx) = oneshot::channel::<Result<ParityFactor, DecodeError>>();
        let original_config = self.original_config.clone();
        let decoding_guard = self.acquire_work_slot().await?;
        let trace = WorkerTrace::new("decoder_job");
        self.schedule_work(move || {
            let _decoding_guard = decoding_guard;
            trace.event("worker_started");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut instance = prepared.new_instance(&hypergraph.hypergraph, &original_config)?;
                trace.event("initialized");
                let decoded = instance.decode(DecodeRequest {
                    syndrome: &syndrome,
                    decoder_seed,
                    reweights: &[],
                    loss: loss.as_ref(),
                });
                trace.solved(decoded.is_ok());
                decoded.and_then(|parity_factor| {
                    validation::validate_parity_factor(parity_factor, hypergraph.hyperedges.len())
                        .map_err(DecodeError::Backend)
                })
            }));
            trace.event("worker_finished");
            trace.finish();
            drop(prepared);
            drop(hypergraph);
            match result {
                Ok(decode_result) => {
                    let _ = tx.send(decode_result);
                }
                Err(_) => {
                    eprintln!("decoder panicked during decode");
                }
            }
        });
        rx.await
            .map_err(|_| Status::internal("decode panicked or was cancelled"))?
            .map_err(decode_error_status)
    }
}

#[tonic::async_trait]
impl<T: DecoderInstance + Send + 'static> black_box_decoder_server::BlackBoxDecoder for ThreadPoolingDecoder<T> {
    async fn get_capabilities(
        &self,
        _request: Request<()>,
    ) -> Result<Response<blackbox_decoder::DecoderCapabilities>, Status> {
        Ok(Response::new(self.features.to_proto()))
    }

    async fn decode(&self, request: Request<blackbox_decoder::DecodingProblem>) -> Result<Response<ParityFactor>, Status> {
        let problem = request.into_inner();
        let syndrome = problem.syndrome.ok_or_else(|| Status::invalid_argument("missing syndrome"))?;
        let hypergraph = problem
            .hypergraph
            .ok_or_else(|| Status::invalid_argument("missing hypergraph"))?;
        self.decode_shared(
            Arc::new(SharedDecodingHypergraph::new(hypergraph)),
            syndrome,
            problem.loss,
            problem.decoder_seed,
        )
        .await
        .map(Response::new)
    }

    async fn load_hypergraph(
        &self,
        request: Request<blackbox_decoder::DecodingHypergraph>,
    ) -> Result<Response<blackbox_decoder::LoadHypergraphResponse>, Status> {
        let decoding_guard = self.acquire_work_slot().await?;
        let hypergraph = Arc::new(SharedDecodingHypergraph::new(request.into_inner()));
        let prepared = hypergraph.prepared::<T>(&self.original_config)?;
        let hid = self.next_hid.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel::<Result<(T, Arc<SharedDecodingHypergraph>, DecodingGuard), Status>>();
        let original_config = self.original_config.clone();
        let preparing = Arc::clone(&prepared);
        let trace = WorkerTrace::new("decoder_load_job");
        trace.property("hid", || hid.to_string());
        self.schedule_work(move || {
            trace.event("worker_started");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                preparing.new_instance(&hypergraph.hypergraph, &original_config)
            }));
            trace.event("initialized");
            trace.property("instance_created", || result.as_ref().is_ok_and(Result::is_ok).to_string());
            trace.finish();
            let result = match result {
                Ok(Ok(instance)) => Ok((instance, hypergraph, decoding_guard)),
                Ok(Err(error)) => Err(decode_error_status(error)),
                Err(_) => {
                    eprintln!("decoder panicked during load_hypergraph (hid={hid})");
                    Err(Status::internal(format!("hid={hid} load panicked")))
                }
            };
            let _ = tx.send(result);
        });
        let (instance, hypergraph, decoding_guard) = rx
            .await
            .map_err(|_| Status::internal(format!("hid={hid} load was cancelled")))??;
        let mut instances = LinkedList::new();
        instances.push_back(instance);
        self.loaded.lock().await.insert(
            hid,
            Loaded {
                hypergraph,
                prepared,
                instances,
            },
        );
        drop(decoding_guard);
        Ok(Response::new(blackbox_decoder::LoadHypergraphResponse { hid }))
    }

    #[allow(
        clippy::too_many_lines,
        reason = "Keep pooled ownership and worker tracing in one cancellation-safe sequence."
    )]
    async fn decode_loaded(
        &self,
        request: Request<blackbox_decoder::LoadedDecodingProblem>,
    ) -> Result<Response<blackbox_decoder::ParityFactor>, Status> {
        let problem = request.into_inner();
        let syndrome = problem
            .syndrome
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing syndrome"))?;
        let reweights: Vec<(u64, f64)> = problem
            .reweights
            .iter()
            .map(|value| (value.edge, value.probability))
            .collect();
        let request = DecodeRequest {
            syndrome,
            decoder_seed: problem.decoder_seed,
            reweights: &reweights,
            loss: problem.loss.as_ref(),
        };
        request.require_supported(self.features).map_err(decode_error_status)?;
        let decoding_guard = self.acquire_work_slot().await?;
        let (instance, hypergraph, prepared, edge_count) = {
            let mut guard = self.loaded.lock().await;
            let Some(loaded) = guard.get_mut(&problem.hid) else {
                return Err(Status::not_found(format!("hid={}", problem.hid)));
            };
            let edge_count = loaded.hypergraph.hyperedges.len();
            validation::validate_syndrome(syndrome, loaded.hypergraph.vertex_num).map_err(Status::invalid_argument)?;
            validation::validate_reweights(&reweights, edge_count).map_err(Status::invalid_argument)?;
            validation::validate_loss(problem.loss.as_ref(), edge_count).map_err(Status::invalid_argument)?;
            // Validate the handle and side information before applying the
            // plain-request zero-syndrome shortcut.
            if bit_vector::is_zero(syndrome)
                && request.required_features().is_empty()
                && !self.features.contains(DecoderFeatures::OBSERVABLES)
            {
                #[cfg(feature = "cli")]
                let _span = Span::enter_with_local_parent("decoder_zero_syndrome");
                return Ok(Response::new(ParityFactor { subgraph: vec![] }));
            }
            let instance = loaded.instances.pop_back();
            let hypergraph = instance.is_none().then(|| Arc::clone(&loaded.hypergraph));
            let prepared = instance.is_none().then(|| Arc::clone(&loaded.prepared));
            (instance, hypergraph, prepared, edge_count)
        };
        let (tx, rx) = oneshot::channel::<Result<(ParityFactor, Option<T>, DecodingGuard), DecodeError>>();
        let original_config = instance.is_none().then(|| Arc::clone(&self.original_config));
        let hid = problem.hid;
        let trace = WorkerTrace::new("decoder_job");
        trace.property("hid", || hid.to_string());
        trace.property("created_instance", || instance.is_none().to_string());
        trace.property("loaded", || "true".to_string());
        self.schedule_work(move || {
            trace.event("worker_started");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut instance = if let Some(instance) = instance {
                    instance
                } else {
                    prepared
                        .as_ref()
                        .unwrap()
                        .new_instance(&hypergraph.as_ref().unwrap().hypergraph, original_config.as_ref().unwrap())?
                };
                trace.event("initialized");
                let decode_result = instance
                    .decode(DecodeRequest {
                        syndrome: problem.syndrome.as_ref().unwrap(),
                        decoder_seed: problem.decoder_seed,
                        reweights: &reweights,
                        loss: problem.loss.as_ref(),
                    })
                    .and_then(|parity_factor| {
                        validation::validate_parity_factor(parity_factor, edge_count).map_err(DecodeError::Backend)
                    });
                trace.solved(decode_result.is_ok());
                Ok((instance, decode_result))
            }));
            let response = match result {
                Ok(Ok((mut instance, Ok(parity_factor)))) => {
                    let reset_succeeded =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| instance.reset())).is_ok();
                    if !reset_succeeded {
                        eprintln!("decoder panicked during reset after decode_loaded (hid={hid})");
                    }
                    let instance = reset_succeeded.then_some(instance);
                    Some(Ok((parity_factor, instance, decoding_guard)))
                }
                Ok(Ok((_, Err(error))) | Err(error)) => Some(Err(error)),
                Err(_) => {
                    eprintln!("decoder panicked during decode_loaded (hid={hid})");
                    None
                }
            };
            trace.event("worker_finished");
            trace.finish();
            if let Some(response) = response {
                let _ = tx.send(response);
            }
        });
        let (parity_factor, instance, decoding_guard) = rx
            .await
            .map_err(|_| Status::internal("decode panicked or was cancelled"))?
            .map_err(decode_error_status)?;
        if let Some(instance) = instance {
            let mut guard = self.loaded.lock().await;
            if let Some(loaded) = guard.get_mut(&hid) {
                loaded.instances.push_back(instance);
            }
        }
        drop(decoding_guard);
        Ok(parity_factor.into())
    }

    async fn reset(&self, request: Request<blackbox_decoder::ResetRequest>) -> Result<Response<()>, Status> {
        let flags = request.into_inner();
        if flags.reset_hypergraphs {
            // Acquire the loaded lock and check the counter atomically.
            // Since decode_loaded increments the counter BEFORE acquiring
            // this lock, seeing counter==0 while holding the lock guarantees
            // no decode_loaded is active or about to start.
            loop {
                let mut loaded = self.loaded.lock().await;
                if *self.decoding.borrow() == 0 {
                    loaded.clear();
                    break;
                }
                // In-flight decodes need the lock to return instances,
                // so drop it and wait for them to finish.
                drop(loaded);
                let mut rx = self.decoding.subscribe();
                rx.wait_for(|v| *v == 0).await.unwrap();
            }
        } else if *self.decoding.borrow() > 0 {
            let mut rx = self.decoding.subscribe();
            rx.wait_for(|v| *v == 0).await.unwrap();
        }
        Ok(().into())
    }
}

fn decode_error_status(error: DecodeError) -> Status {
    match error {
        DecodeError::Unsupported(_) => Status::failed_precondition(error.to_string()),
        DecodeError::InvalidInput(_) => Status::invalid_argument(error.to_string()),
        DecodeError::Backend(_) => Status::internal(error.to_string()),
    }
}

#[cfg(test)]
#[path = "../../tests/unit/thread_pooling_test.rs"]
mod tests;
