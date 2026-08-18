//! Decoder with thread pooling
//!

use crate::decoder::blackbox_decoder::{self, ParityFactor, black_box_decoder_server};
pub use crate::decoder::decoder_features::DecoderFeatures;
use crate::misc::bit_vector;
#[cfg(debug_assertions)]
use crate::misc::validation;
use crate::util::BitVector;
use blackbox_decoder::DecodingHypergraph;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::collections::LinkedList;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::{Mutex, oneshot, watch};
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
    loaded: Arc<Mutex<HashMap<u64, Loaded<T>>>>,
    next_hid: AtomicU64,
    decoding: watch::Sender<usize>,
    features: DecoderFeatures,
}

pub struct Loaded<T: DecoderInstance> {
    hypergraph: Arc<DecodingHypergraph>,
    instances: LinkedList<T>,
}

// Cancellation-safe guard for the `decoding` counter.
// Decrements on drop unless `defuse()` is called (happy path).
struct DecodingGuard {
    tx: Option<watch::Sender<usize>>,
}

impl DecodingGuard {
    fn new(tx: watch::Sender<usize>) -> Self {
        Self { tx: Some(tx) }
    }

    fn defuse(&mut self) {
        self.tx.take();
    }
}

impl Drop for DecodingGuard {
    fn drop(&mut self) {
        if let Some(tx) = self.tx.take() {
            tx.send_modify(|v| {
                *v -= 1;
            });
        }
    }
}

impl<T: DecoderInstance> std::fmt::Debug for Loaded<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Loaded")
            .field("hypergraph", &self.hypergraph)
            .finish_non_exhaustive()
    }
}

pub struct DecodeRequest<'a> {
    pub syndrome: &'a BitVector,
    /// Shot-scoped prior assignments. Implementations must not let these values
    /// affect a later request served by the same pooled instance.
    pub reweights: &'a [(u64, f64)],
    /// Structured loss observation for this shot. Independent of `reweights`;
    /// a decoder advertising both features must accept both together.
    pub loss: Option<&'a blackbox_decoder::LossInfo>,
}

impl DecodeRequest<'_> {
    fn required_features(&self) -> DecoderFeatures {
        DecoderFeatures::required(!self.reweights.is_empty(), self.loss.is_some())
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

    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self;

    fn decode(&mut self, request: DecodeRequest<'_>) -> Result<ParityFactor, DecodeError>;

    fn reset(&mut self);
}

impl<T: DecoderInstance + Send + 'static> ThreadPoolingDecoder<T> {
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
        let config: ThreadPoolingConfig = serde_json::from_value(original_config.clone()).unwrap();
        let features = T::supported_features(&original_config);
        let mut thread_pool_builder = rayon::ThreadPoolBuilder::new();
        if config.parallel != 0 {
            thread_pool_builder = thread_pool_builder.num_threads(config.parallel);
        }
        let thread_pool = Arc::new(
            thread_pool_builder
                .panic_handler(|e| {
                    eprintln!("rayon pool thread panicked: {:?}", e);
                })
                .build()
                .expect("creating thread pool failed"),
        );
        Self {
            config,
            original_config: Arc::new(original_config),
            thread_pool,
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

#[tonic::async_trait]
impl<T: DecoderInstance + Send + 'static> black_box_decoder_server::BlackBoxDecoder for ThreadPoolingDecoder<T> {
    async fn get_capabilities(
        &self,
        _request: Request<()>,
    ) -> Result<Response<blackbox_decoder::DecoderCapabilities>, Status> {
        Ok(Response::new(self.features.to_proto()))
    }

    async fn decode(
        &self,
        request: Request<blackbox_decoder::DecodingProblem>,
    ) -> Result<Response<blackbox_decoder::ParityFactor>, Status> {
        let problem = request.into_inner();
        let syndrome = problem
            .syndrome
            .as_ref()
            .ok_or_else(|| Status::invalid_argument("missing syndrome"))?;
        if problem.hypergraph.is_none() {
            return Err(Status::invalid_argument("missing hypergraph"));
        }
        #[cfg(debug_assertions)]
        {
            let hypergraph = problem.hypergraph.as_ref().unwrap();
            validation::validate_hypergraph(hypergraph).map_err(Status::invalid_argument)?;
            validation::validate_syndrome(syndrome, hypergraph.vertex_num).map_err(Status::invalid_argument)?;
            validation::validate_loss(problem.loss.as_ref(), hypergraph.hyperedges.len())
                .map_err(Status::invalid_argument)?;
        }
        let request = DecodeRequest {
            syndrome,
            reweights: &[],
            loss: problem.loss.as_ref(),
        };
        request.require_supported(self.features).map_err(decode_error_status)?;
        // A plain zero syndrome needs no correction. Side information can still
        // change a loss-aware decoder's logical choice, so those requests must
        // reach the backend.
        if bit_vector::is_zero(syndrome) && problem.loss.is_none() {
            return Ok(Response::new(ParityFactor { subgraph: vec![] }));
        }
        let (tx, rx) = oneshot::channel::<Result<ParityFactor, DecodeError>>();
        let original_config = self.original_config.clone();
        self.decoding.send_modify(|v| {
            *v += 1;
        });
        // Use a drop guard so the counter is decremented even if this future
        // is cancelled (e.g. by tokio::select! picking a cancellation branch).
        // Without this, a cancelled decode leaks a +1 in the counter, causing
        // black_box_decoder.reset() to wait forever.
        let mut decoding_guard = DecodingGuard::new(self.decoding.clone());
        let decoding_tx = self.decoding.clone();
        self.thread_pool.spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let hypergraph = problem.hypergraph.as_ref().unwrap();
                let mut instance = T::new(hypergraph, &original_config);
                instance
                    .decode(DecodeRequest {
                        syndrome: problem.syndrome.as_ref().unwrap(),
                        reweights: &[],
                        loss: problem.loss.as_ref(),
                    })
                    .and_then(|parity_factor| {
                        #[cfg(debug_assertions)]
                        {
                            return validation::validate_parity_factor(parity_factor, hypergraph.hyperedges.len())
                                .map_err(DecodeError::Backend);
                        }
                        #[cfg(not(debug_assertions))]
                        {
                            Ok(parity_factor)
                        }
                    })
            }));
            match result {
                Ok(decode_result) => {
                    let _ = tx.send(decode_result);
                }
                Err(_) => {
                    eprintln!("decoder panicked during decode");
                }
            }
            decoding_tx.send_modify(|v| {
                *v -= 1;
            });
        });
        // The worker owns the decrement from here. A cancelled RPC drops its
        // receiver, but reset must still wait for the backend work to finish.
        decoding_guard.defuse();
        let parity_factor = rx
            .await
            .map_err(|_| Status::internal("decode panicked or was cancelled"))?
            .map_err(decode_error_status)?;
        Ok(parity_factor.into())
    }

    async fn load_hypergraph(
        &self,
        request: Request<blackbox_decoder::DecodingHypergraph>,
    ) -> Result<Response<blackbox_decoder::LoadHypergraphResponse>, Status> {
        let hypergraph = Arc::new(request.into_inner());
        #[cfg(debug_assertions)]
        validation::validate_hypergraph(&hypergraph).map_err(Status::invalid_argument)?;
        let hid = self.next_hid.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = oneshot::channel::<Result<(T, Arc<DecodingHypergraph>, DecodingGuard), Status>>();
        let original_config = self.original_config.clone();
        self.decoding.send_modify(|count| *count += 1);
        let decoding_guard = DecodingGuard::new(self.decoding.clone());
        self.thread_pool.spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| T::new(&hypergraph, &original_config)));
            let result = match result {
                Ok(instance) => Ok((instance, hypergraph, decoding_guard)),
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
        self.loaded.lock().await.insert(hid, Loaded { hypergraph, instances });
        drop(decoding_guard);
        Ok(Response::new(blackbox_decoder::LoadHypergraphResponse { hid }))
    }

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
            reweights: &reweights,
            loss: problem.loss.as_ref(),
        };
        request.require_supported(self.features).map_err(decode_error_status)?;
        // Increment counter BEFORE accessing the loaded map, so that
        // reset() always sees counter > 0 while we're processing.
        self.decoding.send_modify(|v| {
            *v += 1;
        });
        let decoding_guard = DecodingGuard::new(self.decoding.clone());
        let (instance, hypergraph) = {
            let mut guard = self.loaded.lock().await;
            let Some(loaded) = guard.get_mut(&problem.hid) else {
                return Err(Status::not_found(format!("hid={}", problem.hid)));
            };
            #[cfg(debug_assertions)]
            {
                let edge_count = loaded.hypergraph.hyperedges.len();
                validation::validate_syndrome(syndrome, loaded.hypergraph.vertex_num).map_err(Status::invalid_argument)?;
                validation::validate_reweights(&reweights, edge_count).map_err(Status::invalid_argument)?;
                validation::validate_loss(problem.loss.as_ref(), edge_count).map_err(Status::invalid_argument)?;
            }
            // Preserve the plain zero-syndrome fast path without discarding
            // shot-scoped priors or structured loss. Validate the HID and any
            // assignments first so malformed requests cannot bypass the API.
            if bit_vector::is_zero(syndrome) && reweights.is_empty() && problem.loss.is_none() {
                return Ok(Response::new(ParityFactor { subgraph: vec![] }));
            }
            let instance = loaded.instances.pop_back();
            let hypergraph = (instance.is_none() || cfg!(debug_assertions)).then(|| Arc::clone(&loaded.hypergraph));
            (instance, hypergraph)
        };
        let (tx, rx) = oneshot::channel::<Result<(ParityFactor, Option<T>, DecodingGuard), DecodeError>>();
        let original_config = instance.is_none().then(|| Arc::clone(&self.original_config));
        let hid = problem.hid;
        self.thread_pool.spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut instance =
                    instance.unwrap_or_else(|| T::new(hypergraph.as_deref().unwrap(), original_config.as_deref().unwrap()));
                let decode_result = instance
                    .decode(DecodeRequest {
                        syndrome: problem.syndrome.as_ref().unwrap(),
                        reweights: &reweights,
                        loss: problem.loss.as_ref(),
                    })
                    .and_then(|parity_factor| {
                        #[cfg(debug_assertions)]
                        {
                            return validation::validate_parity_factor(
                                parity_factor,
                                hypergraph.as_ref().unwrap().hyperedges.len(),
                            )
                            .map_err(DecodeError::Backend);
                        }
                        #[cfg(not(debug_assertions))]
                        {
                            Ok(parity_factor)
                        }
                    });
                (instance, decode_result)
            }));
            match result {
                Ok((mut instance, Ok(parity_factor))) => {
                    let reset_succeeded =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| instance.reset())).is_ok();
                    if !reset_succeeded {
                        eprintln!("decoder panicked during reset after decode_loaded (hid={hid})");
                    }
                    let instance = reset_succeeded.then_some(instance);
                    let _ = tx.send(Ok((parity_factor, instance, decoding_guard)));
                }
                Ok((_instance, Err(error))) => {
                    let _ = tx.send(Err(error));
                }
                Err(_) => {
                    eprintln!("decoder panicked during decode_loaded (hid={hid})");
                }
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
