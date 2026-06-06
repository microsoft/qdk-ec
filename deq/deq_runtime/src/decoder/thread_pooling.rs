//! Decoder with thread pooling
//!

use crate::decoder::blackbox_decoder::{self, ParityFactor, black_box_decoder_server};
use crate::util::BitVector;
use blackbox_decoder::DecodingHypergraph;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};
use std::collections::LinkedList;
use std::sync::Arc;
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::runtime::Handle;
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
    decoding: watch::Sender<usize>,
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

pub trait DecoderInstance {
    fn new(hypergraph: &DecodingHypergraph, config: &serde_json::Value) -> Self;

    fn decode(&mut self, syndrome: &BitVector) -> ParityFactor;

    fn reset(&mut self);
}

impl<T: DecoderInstance + Send + 'static> ThreadPoolingDecoder<T> {
    pub fn new(original_config: serde_json::Value) -> Self {
        let config: ThreadPoolingConfig = serde_json::from_value(original_config.clone()).unwrap();
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
            decoding: watch::channel(0).0,
        }
    }

    #[cfg(feature = "cli")]
    pub fn add_service(self: &Arc<Self>, router: Router) -> Router {
        let service =
            black_box_decoder_server::BlackBoxDecoderServer::from_arc(self.clone()).max_decoding_message_size(usize::MAX);
        router.add_service(service)
    }
}

#[tonic::async_trait]
impl<T: DecoderInstance + Send + 'static> black_box_decoder_server::BlackBoxDecoder for ThreadPoolingDecoder<T> {
    async fn decode(
        &self,
        request: Request<blackbox_decoder::DecodingProblem>,
    ) -> Result<Response<blackbox_decoder::ParityFactor>, Status> {
        let problem = request.into_inner();
        // Skip decoding entirely when syndrome has no defects
        if problem.syndrome.as_ref().is_some_and(|s| s.data.iter().all(|&b| b == 0)) {
            return Ok(Response::new(ParityFactor { subgraph: vec![] }));
        }
        let (tx, rx) = oneshot::channel::<ParityFactor>();
        let original_config = self.original_config.clone();
        self.decoding.send_modify(|v| {
            *v += 1;
        });
        // Use a drop guard so the counter is decremented even if this future
        // is cancelled (e.g. by tokio::select! picking a cancellation branch).
        // Without this, a cancelled decode leaks a +1 in the counter, causing
        // black_box_decoder.reset() to wait forever.
        let mut decoding_guard = DecodingGuard::new(self.decoding.clone());
        self.thread_pool.spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let hypergraph = problem.hypergraph.as_ref().unwrap();
                let mut instance = T::new(hypergraph, &original_config);
                let syndrome = problem.syndrome.as_ref().unwrap();
                instance.decode(syndrome)
            }));
            match result {
                Ok(parity_factor) => {
                    let _ = tx.send(parity_factor);
                }
                Err(_) => {
                    eprintln!("decoder panicked during decode");
                }
            }
        });
        let parity_factor = rx.await;
        // Defuse the guard — we'll decrement manually on the happy path.
        // If rx.await was cancelled (future dropped), the guard's Drop fires instead.
        decoding_guard.defuse();
        self.decoding.send_modify(|v| {
            *v -= 1;
        });
        let parity_factor = parity_factor.map_err(|_| Status::internal("decode panicked or was cancelled".to_string()))?;
        Ok(parity_factor.into())
    }

    async fn load_hypergraph(
        &self,
        request: Request<blackbox_decoder::DecodingHypergraph>,
    ) -> Result<Response<blackbox_decoder::LoadHypergraphResponse>, Status> {
        let hypergraph = Arc::new(request.into_inner());
        let mut loaded = self.loaded.lock().await;
        let hid: u64 = (loaded.len() as u64) + 1;
        loaded.insert(
            hid,
            Loaded {
                hypergraph: hypergraph.clone(),
                instances: [].into(),
            },
        );
        drop(loaded);
        let (tx, rx) = oneshot::channel::<T>();
        let original_config = self.original_config.clone();
        self.thread_pool.spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| T::new(&hypergraph, &original_config)));
            match result {
                Ok(instance) => {
                    let _ = tx.send(instance);
                }
                Err(_) => {
                    eprintln!("decoder panicked during load_hypergraph (hid={})", hid);
                }
            }
        });
        let instance = rx
            .await
            .map_err(|_| Status::internal(format!("hid={hid} load panicked or was cancelled")))?;
        self.loaded.lock().await.get_mut(&hid).unwrap().instances.push_back(instance);
        Ok(Response::new(blackbox_decoder::LoadHypergraphResponse { hid }))
    }

    async fn decode_loaded(
        &self,
        request: Request<blackbox_decoder::LoadedDecodingProblem>,
    ) -> Result<Response<blackbox_decoder::ParityFactor>, Status> {
        let problem = request.into_inner();
        // Skip decoding entirely when syndrome has no defects
        if problem.syndrome.as_ref().is_some_and(|s| s.data.iter().all(|&b| b == 0)) {
            return Ok(Response::new(ParityFactor { subgraph: vec![] }));
        }
        let (tx, rx) = oneshot::channel::<ParityFactor>();
        // Increment counter BEFORE accessing the loaded map, so that
        // reset() always sees counter > 0 while we're processing.
        self.decoding.send_modify(|v| {
            *v += 1;
        });
        let decoding_tx = self.decoding.clone();
        let mut instance = {
            let mut guard = self.loaded.lock().await;
            match guard.get_mut(&problem.hid) {
                Some(loaded) => {
                    if let Some(instance) = loaded.instances.pop_back() {
                        instance
                    } else {
                        let hypergraph = loaded.hypergraph.clone();
                        drop(guard);
                        DecoderInstance::new(&hypergraph, &self.original_config)
                    }
                }
                None => {
                    self.decoding.send_modify(|v| {
                        *v -= 1;
                    });
                    return Err(Status::not_found(format!("hid={}", problem.hid)));
                }
            }
        };
        let loaded_arc = self.loaded.clone();
        let handle = Handle::current();
        self.thread_pool.spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let syndrome = problem.syndrome.as_ref().unwrap();
                instance.decode(syndrome)
            }));
            match result {
                Ok(parity_factor) => {
                    let _ = tx.send(parity_factor);
                    // reset and put the instance back
                    instance.reset();
                    let _guard = handle.enter();
                    tokio::spawn(async move {
                        let mut guard = loaded_arc.lock().await;
                        if let Some(loaded) = guard.get_mut(&problem.hid) {
                            loaded.instances.push_back(instance);
                        }
                        decoding_tx.send_modify(|v| {
                            *v -= 1;
                        });
                    });
                }
                Err(_) => {
                    // tx is dropped, rx.await will return RecvError
                    eprintln!("decoder panicked during decode_loaded (hid={})", problem.hid);
                    // Instance may be in a bad state, don't return to pool
                    let _guard = handle.enter();
                    tokio::spawn(async move {
                        decoding_tx.send_modify(|v| {
                            *v -= 1;
                        });
                    });
                }
            }
        });
        let parity_factor = rx
            .await
            .map_err(|_| Status::internal("decode panicked or was cancelled".to_string()))?;
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
