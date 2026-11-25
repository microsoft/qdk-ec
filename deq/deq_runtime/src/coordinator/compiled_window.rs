//! Precompiled, bounded dataflow decoding for immutable finite programs.
//!
//! This coordinator owns each gadget's errors exactly once. Its decoder window
//! contains their complete check support; incoming future errors are projected
//! onto that window. Earlier touching corrections are immutable dependencies.
//! Frames are published only after every input frame and owned correction exists.

use crate::coordinator::{self, coordinator_server::Coordinator};
use crate::decoder::DynDecoder;
use crate::misc::validation::validate_outcomes;
use crate::{bin, util::BitVector};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};

mod frame;
mod plan;
mod runtime;
use plan::Plan;
use runtime::Session;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
pub struct CompiledWindowConfig {
    /// CPU workers for syndrome, window and frame computation.
    pub worker_threads: usize,
    /// Maximum dataflow tasks per CPU batch.
    pub batch_size: usize,
    /// Reject programs exceeding this number of distinct decoder templates.
    pub max_templates: usize,
}

impl Default for CompiledWindowConfig {
    fn default() -> Self {
        Self {
            worker_threads: 24,
            batch_size: 8,
            max_templates: 2048,
        }
    }
}

pub struct CompiledWindowCoordinator {
    config: CompiledWindowConfig,
    decoder: DynDecoder,
    pool: Arc<rayon::ThreadPool>,
    plan: RwLock<Option<Arc<Plan>>>,
    session: RwLock<Option<Arc<Session>>>,
    lifecycle: tokio::sync::Mutex<()>,
}

impl CompiledWindowCoordinator {
    /// Construct an immutable-program coordinator.
    ///
    /// # Panics
    /// Panics for unsupported configuration or if its CPU pool cannot start.
    #[must_use]
    pub fn new(value: serde_json::Value, decoder: DynDecoder) -> Self {
        let config: CompiledWindowConfig = serde_json::from_value(value).expect("invalid compiled-window configuration");
        assert!(
            config.worker_threads > 0 && config.worker_threads <= 1024,
            "invalid compiled worker count"
        );
        assert!(
            config.batch_size > 0 && config.batch_size <= 256,
            "invalid compiled batch size"
        );
        assert!(config.max_templates > 0, "compiled template limit must be positive");
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.worker_threads)
            .thread_name(|index| format!("deq-fast-{index}"))
            .build()
            .expect("cannot create compiled CPU pool");
        Self {
            config,
            decoder,
            pool: Arc::new(pool),
            plan: RwLock::new(None),
            session: RwLock::new(None),
            lifecycle: tokio::sync::Mutex::new(()),
        }
    }

    /// Publish a contiguous input batch without creating per-gadget futures.
    ///
    /// # Errors
    /// Rejects unsupported features, invalid input boundaries, or a failed epoch.
    pub async fn ingest(
        &self,
        request: coordinator::Outcomes,
        through: Option<usize>,
    ) -> Result<Vec<coordinator::Readouts>, Status> {
        if request.loss_mask.is_some() || !request.modifiers.is_empty() {
            return Err(Status::invalid_argument(
                "compiled-window does not support losses or probability updates",
            ));
        }
        let outcomes: BitVector = request.outcomes.ok_or_else(|| Status::invalid_argument("missing outcomes"))?;
        validate_outcomes(&outcomes, None, outcomes.size).map_err(Status::invalid_argument)?;
        let session = self
            .session
            .read()
            .await
            .clone()
            .ok_or_else(|| Status::failed_precondition("compiled program not loaded"))?;
        session.ingest(outcomes, through, request.decoder_seed).await
    }

    pub async fn cancel_pending(&self) {
        if let Some(session) = self.session.read().await.clone() {
            session.stop().await;
        }
    }

    /// Read an available logical result without waiting for the rest of the program.
    ///
    /// Returns `None` while this gadget's frame is pending in the current epoch.
    /// Callers must coordinate resets with their readers; no cross-epoch handle is returned.
    ///
    /// # Errors
    /// Rejects unloaded or cancelled epochs and gadget IDs outside the loaded program.
    pub async fn readout_bits(&self, gid: u64) -> Result<Option<BitVector>, Status> {
        let session = self.session.read().await;
        session
            .as_ref()
            .ok_or_else(|| Status::failed_precondition("compiled program not loaded"))?
            .epoch
            .readout_bits(gid)
    }
}

#[tonic::async_trait]
impl Coordinator for CompiledWindowCoordinator {
    async fn load_library(&self, request: Request<bin::Library>) -> Result<Response<()>, Status> {
        let _lifecycle = self.lifecycle.lock().await;
        if self.plan.read().await.is_some() {
            return Err(Status::already_exists(
                "compiled program is immutable; create a new coordinator to replace it",
            ));
        }

        let maximum = self.config.max_templates;
        let plan = tokio::task::spawn_blocking(move || Plan::compile(request.into_inner(), maximum))
            .await
            .map_err(|error| Status::internal(format!("compiled preparation failed: {error}")))?
            .map_err(Status::invalid_argument)?;
        for template in &plan.templates {
            let loaded = self.decoder.load_hypergraph(template.graph.graph().clone()).await?;
            template
                .hid
                .set(loaded.hid)
                .map_err(|_| Status::internal("template loaded twice"))?;
        }
        let plan = Arc::new(plan);
        let session = Session::new(Arc::clone(&plan), self.decoder.clone(), Arc::clone(&self.pool), &self.config)?;
        *self.plan.write().await = Some(plan);
        *self.session.write().await = Some(session);
        Ok(Response::new(()))
    }

    async fn reset(&self, request: Request<coordinator::ResetRequest>) -> Result<Response<()>, Status> {
        let flags = request.into_inner();
        if flags.reset_library || flags.reset_decoder_service {
            return Err(Status::invalid_argument(
                "compiled reset retains its immutable library and decoder templates",
            ));
        }
        let _lifecycle = self.lifecycle.lock().await;
        let old = self.session.write().await.take();
        if let Some(old) = old {
            old.stop().await;
        }
        if let Some(plan) = self.plan.read().await.clone() {
            let session = Session::new(plan, self.decoder.clone(), Arc::clone(&self.pool), &self.config)?;
            *self.session.write().await = Some(session);
        }
        Ok(Response::new(()))
    }

    async fn execute(&self, _: Request<bin::Instruction>) -> Result<Response<coordinator::ExecuteResponse>, Status> {
        Err(Status::unimplemented(
            "compiled-window requires a complete library and local static-controller ingestion",
        ))
    }

    async fn decode(&self, _: Request<coordinator::Outcomes>) -> Result<Response<coordinator::Readouts>, Status> {
        Err(Status::unimplemented(
            "compiled-window uses bulk local static-controller ingestion",
        ))
    }

    async fn unload(&self, _: Request<coordinator::UnloadLibrary>) -> Result<Response<()>, Status> {
        Err(Status::unimplemented(
            "compiled programs are immutable until coordinator destruction",
        ))
    }

    async fn get_memory_stats(&self, _: Request<()>) -> Result<Response<coordinator::MemoryStats>, Status> {
        let plan = self.plan.read().await;
        let count = plan.as_ref().map_or(0, |plan| u64::try_from(plan.gadgets.len()).unwrap());
        Ok(Response::new(coordinator::MemoryStats {
            automatic_collection: false,
            persistent_decoder: true,
            gadgets: count,
            check_models: count,
            error_models: count,
            frame_gadgets: count,
            ..Default::default()
        }))
    }
}

#[cfg(all(test, feature = "mwpm"))]
mod tests;
