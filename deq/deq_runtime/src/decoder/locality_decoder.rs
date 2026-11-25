//! Decoder state owned by CPU-local workers, with transport-neutral logical handles.

use super::DecoderFeatures;
use super::blackbox_decoder::{self, DecodingHypergraph, ParityFactor, black_box_decoder_server::BlackBoxDecoder};
use super::execution_pool::{CacheKey, CachedValue, ExecutionPools, next_identity};
use super::thread_pooling::{
    DecodeError, DecodeRequest, DecoderFactory, DecoderInstance, SharedDecodingHypergraph, ThreadPoolingConfig,
};
use super::worker_trace::WorkerTrace;
use crate::misc::{bit_vector, validation};
use crate::util::BitVector;
use hashbrown::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::sync::{RwLock, oneshot, watch};
use tonic::{Request, Response, Status};

struct Pending(watch::Sender<usize>);

impl Drop for Pending {
    fn drop(&mut self) {
        self.0.send_modify(|count| *count -= 1);
    }
}

#[derive(Clone)]
struct Graph {
    source: Arc<SharedDecodingHypergraph>,
    retired: Arc<AtomicBool>,
}

struct Prepared<T> {
    graph: DecodingHypergraph,
    configuration: serde_json::Value,
    factory: DecoderFactory<T>,
}

struct Instance<T> {
    decoder: T,
    prepared: Arc<Prepared<T>>,
    id: u64,
}

struct LocalityDecoder<T> {
    namespace: u64,
    configuration: Arc<serde_json::Value>,
    execution: Arc<ExecutionPools>,
    features: DecoderFeatures,
    loaded: RwLock<HashMap<u64, Graph>>,
    pending: watch::Sender<usize>,
    lifecycle: RwLock<()>,
    instance: PhantomData<fn() -> T>,
}

impl<T> Drop for LocalityDecoder<T> {
    fn drop(&mut self) {
        for graph in self.loaded.get_mut().values() {
            graph.retired.store(true, Ordering::Release);
        }
        if let Err(error) = self.execution.enqueue_eviction(self.namespace, None) {
            eprintln!("decoder cache cleanup failed: {error}");
        }
    }
}

#[tonic::async_trait]
trait LocalBackend: BlackBoxDecoder {
    fn features(&self) -> DecoderFeatures;
    fn execution(&self) -> &Arc<ExecutionPools>;
    async fn decode_shared(
        &self,
        graph: Arc<SharedDecodingHypergraph>,
        syndrome: BitVector,
        loss: Option<blackbox_decoder::LossInfo>,
        seed: Option<u64>,
    ) -> Result<ParityFactor, Status>;
    async fn unload(&self, handles: &[u64]) -> Result<(), Status>;
}

pub struct DynLocalityDecoder {
    inner: Arc<dyn LocalBackend>,
}

impl DynLocalityDecoder {
    pub(super) fn new<T: DecoderInstance + Send + 'static>(
        configuration: serde_json::Value,
        shared: Option<Arc<ExecutionPools>>,
    ) -> Result<Arc<Self>, String> {
        let config: ThreadPoolingConfig =
            serde_json::from_value(configuration.clone()).map_err(|error| error.to_string())?;
        if config.parallel != 0 && !config.pools.is_empty() {
            return Err("configure either parallel or pools, not both".into());
        }
        let execution = if let Some(execution) = shared {
            if config.parallel != 0 || !config.pools.is_empty() {
                return Err("a shared decoder executor owns the worker configuration".into());
            }
            execution
        } else {
            ExecutionPools::new(&config.pools).map_err(|error| error.to_string())?
        };
        Ok(Arc::new(Self {
            inner: Arc::new(LocalityDecoder::<T> {
                namespace: next_identity(),
                features: T::supported_features(&configuration),
                configuration: Arc::new(configuration),
                execution,
                loaded: RwLock::new(HashMap::new()),
                pending: watch::channel(0).0,
                lifecycle: RwLock::new(()),
                instance: PhantomData,
            }),
        }))
    }

    pub(super) fn supported_features(&self) -> DecoderFeatures {
        self.inner.features()
    }

    pub(super) fn execution(&self) -> &Arc<ExecutionPools> {
        self.inner.execution()
    }

    pub(super) async fn decode_shared(
        &self,
        graph: Arc<SharedDecodingHypergraph>,
        syndrome: BitVector,
        loss: Option<blackbox_decoder::LossInfo>,
        seed: Option<u64>,
    ) -> Result<ParityFactor, Status> {
        self.inner.decode_shared(graph, syndrome, loss, seed).await
    }

    pub(super) async fn unload_hypergraphs(&self, handles: &[u64]) -> Result<(), Status> {
        self.inner.unload(handles).await
    }

    #[cfg(feature = "cli")]
    pub(super) fn add_service(
        self: &Arc<Self>,
        router: tonic::transport::server::Router,
    ) -> tonic::transport::server::Router {
        router.add_service(
            blackbox_decoder::black_box_decoder_server::BlackBoxDecoderServer::from_arc(Arc::clone(self))
                .max_decoding_message_size(usize::MAX),
        )
    }
}

impl<T: DecoderInstance + Send + 'static> LocalityDecoder<T> {
    fn pending(&self) -> Arc<Pending> {
        self.pending.send_modify(|count| *count += 1);
        Arc::new(Pending(self.pending.clone()))
    }

    async fn with_instance<R: Send + 'static>(
        &self,
        graph: Graph,
        pending: &Arc<Pending>,
        name: &'static str,
        reset: bool,
        operation: impl FnOnce(&mut T, &Prepared<T>, &mut WorkerTrace) -> Result<R, DecodeError> + Send + 'static,
    ) -> Result<R, Status> {
        let mut trace = WorkerTrace::new(name);
        let permit = Arc::clone(&self.execution.admission)
            .acquire_owned()
            .await
            .map_err(|_| Status::unavailable("decoder executor closed"))?;
        trace.event("admitted");
        let key = CacheKey {
            decoder: self.namespace,
            graph: graph.source.identity(),
        };
        let lease = self.execution.reserve(key).await;
        let (pool, worker, cpu) = lease.identity();
        trace.property("pool", || pool.to_string());
        trace.property("worker", || worker.to_string());
        trace.property("cpu", || cpu.to_string());
        trace.property("graph", || key.graph.to_string());
        let configuration = Arc::clone(&self.configuration);
        let tracked = Arc::clone(pending);
        let (response, received) = oneshot::channel();
        lease.submit(move |cache, mut lease| {
            let _permit = permit;
            let _pending = tracked;
            trace.event("worker_started");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let cached = cache.remove(&key);
                let created = cached.is_none();
                trace.property("created_instance", || created.to_string());
                let mut instance = if let Some(cached) = cached {
                    cached
                        .value
                        .downcast::<Instance<T>>()
                        .expect("decoder cache namespace has one instance type")
                } else {
                    trace.resources("construction_start");
                    let prepared = lease.prepared(key, || {
                        let local = graph.source.graph().clone();
                        let configuration = configuration.as_ref().clone();
                        let factory = T::prepare(&local, &configuration).map_err(DecodeError::InvalidInput)?;
                        Ok::<_, DecodeError>(Prepared {
                            graph: local,
                            configuration,
                            factory,
                        })
                    })?;
                    trace.event("graph_prepared");
                    trace.resources("graph_prepared");
                    let decoder = (prepared.factory)(&prepared.graph, &prepared.configuration);
                    trace.resources("construction_end");
                    Box::new(Instance {
                        decoder,
                        prepared,
                        id: next_identity(),
                    })
                };
                trace.property("instance", || instance.id.to_string());
                trace.event("initialized");
                lease.initialized();
                let mut result = operation(&mut instance.decoder, &instance.prepared, &mut trace);
                if result.is_ok()
                    && reset
                    && std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| instance.decoder.reset())).is_err()
                {
                    eprintln!("decoder reset panicked in pool {pool}, worker {worker}");
                    result = Err(DecodeError::Backend("decoder reset panicked".into()));
                }
                if result.is_ok() && !graph.retired.load(Ordering::Acquire) && Arc::strong_count(&graph.source) > 1 {
                    cache.insert(
                        key,
                        CachedValue {
                            owner: Arc::downgrade(&graph.source),
                            value: instance,
                        },
                    );
                    if created {
                        lease.cache_inserted(key);
                    }
                } else {
                    lease.cache_removed(key);
                }
                result
            }));
            let result = if let Ok(result) = result {
                result
            } else {
                eprintln!("decoder panicked in pool {pool}, worker {worker}");
                Err(DecodeError::Backend("decoder worker panicked".into()))
            };
            if result.is_err() {
                lease.cache_removed(key);
            }
            trace.event("worker_finished");
            trace.finish();
            let _ = response.send(result);
        })?;
        received
            .await
            .map_err(|_| Status::internal("decoder worker stopped before returning a result"))?
            .map_err(|error| super::thread_pooling::decode_error_status(&error))
    }

    fn validate_request(&self, graph: &Graph, request: &DecodeRequest<'_>) -> Result<bool, Status> {
        request
            .require_supported(self.features)
            .map_err(|error| super::thread_pooling::decode_error_status(&error))?;
        validation::validate_syndrome(request.syndrome, graph.source.vertex_num).map_err(Status::invalid_argument)?;
        validation::validate_reweights(request.reweights, graph.source.hyperedges.len())
            .map_err(Status::invalid_argument)?;
        validation::validate_loss(request.loss, graph.source.hyperedges.len()).map_err(Status::invalid_argument)?;
        Ok(bit_vector::is_zero(request.syndrome)
            && request.required_features().is_empty()
            && !self.features.contains(DecoderFeatures::OBSERVABLES))
    }
}

#[tonic::async_trait]
impl<T: DecoderInstance + Send + 'static> LocalBackend for LocalityDecoder<T> {
    fn features(&self) -> DecoderFeatures {
        self.features
    }

    fn execution(&self) -> &Arc<ExecutionPools> {
        &self.execution
    }

    async fn decode_shared(
        &self,
        source: Arc<SharedDecodingHypergraph>,
        syndrome: BitVector,
        loss: Option<blackbox_decoder::LossInfo>,
        seed: Option<u64>,
    ) -> Result<ParityFactor, Status> {
        let _lifecycle = self.lifecycle.read().await;
        source.bind::<T>(&self.configuration)?;
        let graph = Graph {
            source,
            retired: Arc::new(AtomicBool::new(false)),
        };
        if self.validate_request(
            &graph,
            &DecodeRequest {
                syndrome: &syndrome,
                decoder_seed: seed,
                reweights: &[],
                loss: loss.as_ref(),
            },
        )? {
            #[cfg(feature = "cli")]
            let _span = crate::misc::fastrace::Span::enter_with_local_parent("decoder_zero_syndrome");
            return Ok(ParityFactor { subgraph: vec![] });
        }
        let pending = self.pending();
        self.with_instance(graph, &pending, "decoder_job", true, move |instance, prepared, trace| {
            let result = trace.measure_solver(|| {
                instance.decode(DecodeRequest {
                    syndrome: &syndrome,
                    decoder_seed: seed,
                    reweights: &[],
                    loss: loss.as_ref(),
                })
            });
            let result = result.and_then(|result| {
                validation::validate_parity_factor(result, prepared.graph.hyperedges.len()).map_err(DecodeError::Backend)
            });
            trace.solved(result.is_ok());
            result
        })
        .await
    }

    async fn unload(&self, handles: &[u64]) -> Result<(), Status> {
        let mut retired = HashSet::new();
        {
            let mut loaded = self.loaded.write().await;
            for handle in handles {
                if let Some(graph) = loaded.remove(handle) {
                    graph.retired.store(true, Ordering::Release);
                    retired.insert(graph.source.identity());
                }
            }
        }
        if !retired.is_empty() {
            self.execution.evict(self.namespace, Some(Arc::new(retired))).await?;
        }
        Ok(())
    }
}

#[tonic::async_trait]
impl<T: DecoderInstance + Send + 'static> BlackBoxDecoder for LocalityDecoder<T> {
    async fn get_capabilities(
        &self,
        _request: Request<()>,
    ) -> Result<Response<blackbox_decoder::DecoderCapabilities>, Status> {
        Ok(Response::new(self.features.to_proto()))
    }

    async fn decode(&self, request: Request<blackbox_decoder::DecodingProblem>) -> Result<Response<ParityFactor>, Status> {
        let problem = request.into_inner();
        let graph = problem
            .hypergraph
            .ok_or_else(|| Status::invalid_argument("missing hypergraph"))?;
        let syndrome = problem.syndrome.ok_or_else(|| Status::invalid_argument("missing syndrome"))?;
        self.decode_shared(
            Arc::new(SharedDecodingHypergraph::new(graph)),
            syndrome,
            problem.loss,
            problem.decoder_seed,
        )
        .await
        .map(Response::new)
    }

    async fn load_hypergraph(
        &self,
        request: Request<DecodingHypergraph>,
    ) -> Result<Response<blackbox_decoder::LoadHypergraphResponse>, Status> {
        let _lifecycle = self.lifecycle.read().await;
        let pending = self.pending();
        let source = Arc::new(SharedDecodingHypergraph::new(request.into_inner()));
        source.bind::<T>(&self.configuration)?;
        let graph = Graph {
            source,
            retired: Arc::new(AtomicBool::new(false)),
        };
        self.with_instance(graph.clone(), &pending, "decoder_load_job", false, |_, _, _| Ok(()))
            .await?;
        let handle = next_identity();
        self.loaded.write().await.insert(handle, graph);
        Ok(Response::new(blackbox_decoder::LoadHypergraphResponse { hid: handle }))
    }

    async fn decode_loaded(
        &self,
        request: Request<blackbox_decoder::LoadedDecodingProblem>,
    ) -> Result<Response<ParityFactor>, Status> {
        let _lifecycle = self.lifecycle.read().await;
        let problem = request.into_inner();
        let graph = self
            .loaded
            .read()
            .await
            .get(&problem.hid)
            .cloned()
            .ok_or_else(|| Status::not_found(format!("hid={}", problem.hid)))?;
        let syndrome = problem.syndrome.ok_or_else(|| Status::invalid_argument("missing syndrome"))?;
        let reweights: Vec<_> = problem
            .reweights
            .iter()
            .map(|value| (value.edge, value.probability))
            .collect();
        if self.validate_request(
            &graph,
            &DecodeRequest {
                syndrome: &syndrome,
                decoder_seed: problem.decoder_seed,
                reweights: &reweights,
                loss: problem.loss.as_ref(),
            },
        )? {
            #[cfg(feature = "cli")]
            let _span = crate::misc::fastrace::Span::enter_with_local_parent("decoder_zero_syndrome");
            return Ok(Response::new(ParityFactor { subgraph: vec![] }));
        }
        let pending = self.pending();
        self.with_instance(graph, &pending, "decoder_job", true, move |instance, prepared, trace| {
            let result = trace.measure_solver(|| {
                instance.decode(DecodeRequest {
                    syndrome: &syndrome,
                    decoder_seed: problem.decoder_seed,
                    reweights: &reweights,
                    loss: problem.loss.as_ref(),
                })
            });
            let result = result.and_then(|result| {
                validation::validate_parity_factor(result, prepared.graph.hyperedges.len()).map_err(DecodeError::Backend)
            });
            trace.solved(result.is_ok());
            result
        })
        .await
        .map(Response::new)
    }

    async fn reset(&self, request: Request<blackbox_decoder::ResetRequest>) -> Result<Response<()>, Status> {
        let _lifecycle = self.lifecycle.write().await;
        let mut pending = self.pending.subscribe();
        pending
            .wait_for(|count| *count == 0)
            .await
            .map_err(|_| Status::internal("decoder lifecycle closed"))?;
        if request.into_inner().reset_hypergraphs {
            let mut loaded = self.loaded.write().await;
            for graph in loaded.values() {
                graph.retired.store(true, Ordering::Release);
            }
            loaded.clear();
            drop(loaded);
            self.execution.evict(self.namespace, None).await?;
        }
        Ok(Response::new(()))
    }
}

#[tonic::async_trait]
impl BlackBoxDecoder for DynLocalityDecoder {
    async fn get_capabilities(
        &self,
        request: Request<()>,
    ) -> Result<Response<blackbox_decoder::DecoderCapabilities>, Status> {
        self.inner.get_capabilities(request).await
    }

    async fn decode(&self, request: Request<blackbox_decoder::DecodingProblem>) -> Result<Response<ParityFactor>, Status> {
        self.inner.decode(request).await
    }

    async fn load_hypergraph(
        &self,
        request: Request<DecodingHypergraph>,
    ) -> Result<Response<blackbox_decoder::LoadHypergraphResponse>, Status> {
        self.inner.load_hypergraph(request).await
    }

    async fn decode_loaded(
        &self,
        request: Request<blackbox_decoder::LoadedDecodingProblem>,
    ) -> Result<Response<ParityFactor>, Status> {
        self.inner.decode_loaded(request).await
    }

    async fn reset(&self, request: Request<blackbox_decoder::ResetRequest>) -> Result<Response<()>, Status> {
        self.inner.reset(request).await
    }
}

#[cfg(test)]
#[path = "../../tests/unit/locality_decoder_test.rs"]
mod tests;
