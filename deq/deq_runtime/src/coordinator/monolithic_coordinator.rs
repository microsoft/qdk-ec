//! Monolithic coordinator
//!
//! This coordinator decodes the entire connected decoding hypergraph as a whole,
//! when all gadgets within this connected subgraph are loaded with measurements
//! and all output ports are connected. Note that once a connected subgraph is
//! decoded, the gadget instances and others will be deleted immediately to free
//! up memory. That means binding new check models to an already decoded region
//! is hazardous and not allowed. One should always make sure that all check models
//! and error models are loaded prior to loading all the measurement outcomes.
//!
//! Note that we do not aim to check all the possible errors or trying to return
//! them explicitly. Instead, we might return common errors while just panicking
//! for uncommon errors to reduce the code size; Also, the coordinator may simply
//! hang forever if an invalid program is provided, e.g., if an error model refers
//! to a remote check model that is never binding to the remote gadget, the coordinator
//! simply wait there forever. This behavior is by design because we don't require
//! user to provide a binding within any deadline. To make sure the coordinator
//! makes progress, users should always make sure the program is valid, e.g.,
//! by running the `deq.spec.program_validity.is_program_valid` check. Note that
//! although the spec-check tool only works for static program, it's always possible
//! to record the program sequence and run the validity check offline.
//!

use crate::bin;
use crate::coordinator;
use crate::coordinator::loss_handler::{RawLossSite, apply_loss_random_imputation, has_loss_model};
use crate::coordinator::reweight_handler::{
    ProjectedErrors, apply_reweights, decode_projected, deduplicate_decoder_input, load_projected_decoder,
    probability_reweights,
};
use crate::coordinator::{
    DecoderCacheKey, DecoderReweighting, FingerprintSource, LoadedDecoder, LossHandler, LossStrategy,
    build_modifier_fingerprints,
};
use crate::decoder::DynDecoder;
use crate::decoder::blackbox_decoder::{self, DecodingHypergraph, Hyperedge};
use crate::decoder::blackbox_util::assert_parity_factor;
use crate::misc::bit_vector::{self, get_bit, set_bit};
use crate::misc::index::{ErrorIndex, WILDCARD};
use crate::misc::pauli_frame_tracker::PauliFrameTracker;
use crate::misc::relative_program::{self, RelativeMapping, RelativeProgram};
use crate::misc::sync::{TaskCounter, check_or_receiver, get_or_receiver, get_value};
use crate::misc::union_find::{UnionFindGeneric, UnionNodeTrait};
use crate::misc::validation::{
    apply_check_model_reroutes, apply_error_model_reroutes, validate_outcomes, validate_probability_modifier,
};
use crate::util::BitVector;
use binar::{BitVec, BitwiseMut};
use hashbrown::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use std::sync::Arc;
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::{Mutex, RwLock, oneshot, watch};
use tokio_util::sync::CancellationToken;
use tonic::{Request, Response, Status};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct MonolithicCoordinatorConfig {
    /// if sanity check on the parity factor result from the decoder: every decoder
    /// should return a parity factor that exactly produces the observed syndrome
    #[serde(default)]
    pub assert_parity_factor: bool,
    /// Merge hyperedges with the same syndrome (enabled by default). Ignored by
    /// the ``handoff`` loss strategy because structured loss addresses distinct
    /// error mechanisms by edge index and deduplication would destroy that
    /// distinction.
    #[serde(default = "default_true")]
    pub merge_hyperedges: bool,
    /// by default, we expand the remote references prior to loading the outcomes,
    /// but one can disable this option which will reduce the amount of async tasks
    #[serde(default = "default_true")]
    pub async_expand: bool,
    /// by default, we load the hypergraph to the decoder service and use it
    /// thereafter; disabling this option will force the coordinator to build
    /// the decoding hypergraph every time and force the decoder service to
    /// build the decoder data structure every time, which could be time consuming
    #[serde(default = "default_true")]
    pub persistent_decoder: bool,
    /// Transport policy for shot-scoped probability updates: ``auto`` uses
    /// loaded reweights when supported and otherwise materializes a one-shot
    /// graph; ``enabled`` requires decoder support; ``disabled`` always
    /// materializes updates.
    #[serde(default)]
    pub decoder_reweighting: DecoderReweighting,
    /// when ``true`` (the default), each bit of ``Outcomes.outcomes`` whose
    /// position is set in the accompanying ``Outcomes.loss_mask`` is replaced
    /// with a uniformly random bit before the coordinator computes the syndrome.
    #[serde(default = "default_true")]
    pub loss_random_imputation: bool,
    /// optional seed for the loss-random-imputation RNG.  When ``None``, the
    /// RNG is seeded from the OS entropy pool at coordinator construction.
    #[serde(default)]
    pub loss_random_imputation_seed: Option<u64>,
    /// what to do with the atom losses heralded by ``Outcomes.loss_mask``:
    /// ``"reweight"`` (the default) raises the observed losses' Pauli-envelope
    /// generator edges to a comparable weight and hands the decoder an ordinary
    /// problem, ``"ignore"`` drops the information, and ``"handoff"`` passes the
    /// assembled loss sites to a loss-aware decoder as a structured
    /// [`LossInfo`](crate::decoder::blackbox_decoder::LossInfo). Orthogonal to
    /// ``loss_random_imputation``, which applies under every strategy. Under
    /// ``reweight`` and ``handoff`` the per-gadget-type loss models come from
    /// ``GadgetType.loss_model`` in the loaded library — a program whose gadget
    /// types declare none costs nothing.
    #[serde(default)]
    pub loss_strategy: LossStrategy,
    /// options for the chosen ``loss_strategy``, parsed and validated at
    /// construction. ``ignore`` and ``handoff`` take none; ``reweight`` takes
    /// ``{"weight_fraction": f}`` with ``f`` in ``(0, 1]`` (default ``0.5``),
    /// the fraction of its own scale *weight* each loss edge is lowered to. An
    /// option that belongs to another strategy is an error, not a silent no-op.
    #[serde(default)]
    #[cfg_attr(feature = "cli", structdoc(leaf = "JSON object"))]
    pub loss_config: serde_json::Value,
}

fn default_true() -> bool {
    true
}

/// to prevent deadlock, all of the following locks must be acquired in the order of
/// the fields defined below
pub struct MonolithicCoordinator {
    pub config: MonolithicCoordinatorConfig,
    /// library data
    pub port_types: RwLock<HashMap<u64, Arc<bin::PortType>>>,
    pub gadget_types: RwLock<HashMap<u64, Arc<bin::GadgetType>>>,
    pub check_model_types: RwLock<HashMap<u64, Arc<bin::CheckModelType>>>,
    pub error_model_types: RwLock<HashMap<u64, Arc<bin::ErrorModelType>>>,
    /// execution data
    pub gadgets: Arc<RwLock<HashMap<u64, Gadget>>>,
    pub check_models: Arc<RwLock<HashMap<u64, CheckModel>>>,
    pub error_models: Arc<RwLock<HashMap<u64, ErrorModel>>>,
    /// next id counters for auto-assignment
    pub next_gid: Mutex<u64>,
    pub next_cid: Mutex<u64>,
    pub next_eid: Mutex<u64>,
    /// the connected subgraph that is not decoded yet
    pub pending_subgraphs: Mutex<UnionFindGeneric<MonolithicUnionNode>>,
    /// mapping from gid to union find index (for efficient sparse gid handling)
    pub gid_to_union_index: Mutex<HashMap<u64, usize>>,
    /// the loaded decoders, keyed by `(RelativeProgram, mapping.global_eid_of)`.
    ///
    /// `RelativeProgram` alone is insufficient: two windows with the same
    /// relative-program structure can still produce different merged hypergraphs
    /// because each window reads per-`error_model` modifier state
    /// (`instance.modifier.probability_modifier`, `modified_remote_check_models[].check_bias`)
    /// at hypergraph-construction time. Those fields are set once per `eid` at
    /// error-model creation, so including the per-window `global_eid_of` vector
    /// in the cache key disambiguates windows that bind different `eid`s — and
    /// therefore different modifier state — into the same relative slot.
    pub loaded_decoders: RwLock<HashMap<DecoderCacheKey, LoadedDecoder>>,
    /// the decoder service
    pub decoder: DynDecoder,
    /// Pauli frame tracker
    pub pauli_frame_tracker: Mutex<PauliFrameTracker>,
    /// Cancelled on reset()/drop to abort all pending decode/expand tasks.
    pub cancellation: RwLock<CancellationToken>,
    /// Tracks active spawned tasks; reset() waits for all to finish before clearing state.
    pub task_counter: Arc<TaskCounter>,
    /// Deterministic RNG used by `apply_loss_random_imputation` when
    /// ``config.loss_random_imputation`` is enabled.  Seeded once at
    /// construction from ``config.loss_random_imputation_seed`` (or from OS
    /// entropy when no seed was supplied).  ``None`` when imputation is
    /// disabled, so the field doesn't even allocate.
    pub loss_imputation_rng: Option<Mutex<crate::simulator::DeterministicRng>>,
    /// Validated loss strategy, built from ``config.loss_strategy`` and
    /// ``config.loss_config`` at construction.
    pub loss_handler: LossHandler,
    /// Whether this decoder accepts shot-scoped loaded reweights.
    pub use_loaded_reweights: bool,
}

/// Per-coordinator [`FingerprintSource`] adapter for the monolithic
/// `ErrorModel` wrapper.  See [`crate::coordinator::build_modifier_fingerprints`].
impl FingerprintSource for ErrorModel {
    fn instance(&self) -> &bin::ErrorModel {
        &self.instance
    }
    fn modified_remote_check_models(&self) -> &Arc<Vec<Option<bin::error_model_type::RemoteCheckModel>>> {
        &self.modified_remote_check_models
    }
}

pub struct Gadget {
    pub instance: bin::Gadget,
    pub outcomes: Option<BitVector>,
    pub probability_modifiers: Vec<(u64, bin::ProbabilityModifier)>,
    pub loss_mask: Option<BitVector>,
    /// the check model's cid that is binding to this gadget
    pub binding_cid: watch::Sender<Option<u64>>,
    /// the peer gadgets' gid connected to each output port
    pub outputs: Vec<watch::Sender<Option<bin::gadget::Connector>>>,
    /// oneshot channel to send over the readout values; note that only the last
    /// loaded gadget is responsible for running the actual decoding, while the rest
    /// of them simply listen to the receiver channel,
    pub tx: oneshot::Sender<BitVector>,
    /// the receiver of the channel will be taken out by the async task
    pub rx: Option<oneshot::Receiver<BitVector>>,
}

pub struct CheckModel {
    pub instance: bin::CheckModel,
    /// the list of eid attaching to this check model
    pub attaching_eid_vec: Vec<u64>,
    /// the modified remote gadgets
    pub modified_remote_gadgets: Arc<Vec<Option<bin::check_model_type::RemoteGadget>>>,
    /// the expanded remote gadgets
    pub expanded_remote_gadgets: watch::Sender<Option<Vec<Option<u64>>>>,
}

pub struct ErrorModel {
    pub instance: bin::ErrorModel,
    /// the modified remote check models
    pub modified_remote_check_models: Arc<Vec<Option<bin::error_model_type::RemoteCheckModel>>>,
    /// the expanded remote check models
    pub expanded_remote_check_models: watch::Sender<Option<Vec<Option<u64>>>>,
}

impl MonolithicCoordinator {
    pub fn new(config: serde_json::Value, decoder: DynDecoder) -> Self {
        let config: MonolithicCoordinatorConfig = serde_json::from_value(config).unwrap();
        let use_loaded_reweights = config
            .decoder_reweighting
            .use_loaded(config.persistent_decoder, decoder.features())
            .unwrap_or_else(|error| panic!("invalid decoder reweighting configuration: {error}"));
        let loss_imputation_rng = if config.loss_random_imputation {
            use rand::{Rng, SeedableRng};
            let seed = config.loss_random_imputation_seed.unwrap_or_else(|| rand::rng().next_u64());
            Some(Mutex::new(crate::simulator::DeterministicRng::seed_from_u64(seed)))
        } else {
            None
        };
        let loss_handler = LossHandler::new(config.loss_strategy, config.loss_config.clone())
            .unwrap_or_else(|error| panic!("invalid loss configuration: {error}"));
        Self {
            config,
            port_types: Default::default(),
            gadget_types: Default::default(),
            check_model_types: Default::default(),
            error_model_types: Default::default(),
            gadgets: Default::default(),
            check_models: Default::default(),
            error_models: Default::default(),
            next_gid: Mutex::new(1),
            next_cid: Mutex::new(1),
            next_eid: Mutex::new(1),
            pending_subgraphs: Mutex::new(UnionFindGeneric::new(0)),
            gid_to_union_index: Mutex::new(HashMap::new()),
            loaded_decoders: Default::default(),
            decoder,
            pauli_frame_tracker: Default::default(),
            cancellation: RwLock::new(CancellationToken::new()),
            task_counter: TaskCounter::new(),
            loss_imputation_rng,
            loss_handler,
            use_loaded_reweights,
        }
    }

    /// Fire the cancellation token to abort pending decode tasks. Used by
    /// [`crate::server::LocalServer::shutdown`] to propagate runtime shutdown
    /// into the service layer. Unlike the `reset` RPC handler, this does not
    /// wait for tasks to finish, does not refresh the token, and does not
    /// clear coordinator state — it just signals every cancellable point to
    /// bail.
    pub async fn cancel_pending(&self) {
        let token = self.cancellation.read().await;
        token.cancel();
    }

    /// gather all the gadgets in the connected subgraph starting from the given gid;
    /// note that all the gadget must already be connected, otherwise this function panics
    async fn get_subgraph(&self, gid: u64) -> HashSet<u64> {
        let gadgets = self.gadgets.read().await;
        let mut subgraph: HashSet<u64> = HashSet::new();
        subgraph.insert(gid);
        let mut boundary_gadgets: Vec<u64> = vec![gid];
        while !boundary_gadgets.is_empty() {
            let mut new_boundary_gadgets = vec![];
            for boundary_gid in boundary_gadgets.into_iter() {
                let gadget = gadgets.get(&boundary_gid).unwrap();
                for next in gadget
                    .outputs
                    .iter()
                    .map(|x| x.borrow().unwrap())
                    .chain(gadget.instance.connectors.iter().copied())
                {
                    if !subgraph.contains(&next.gid) {
                        subgraph.insert(next.gid);
                        new_boundary_gadgets.push(next.gid);
                    }
                }
            }
            boundary_gadgets = new_boundary_gadgets;
        }
        subgraph
    }

    /// take the gadgets, check models, and error models out of the global data
    async fn take_subgraph(&self, gid: u64) -> (HashMap<u64, Gadget>, HashMap<u64, CheckModel>, HashMap<u64, ErrorModel>) {
        let subgraph = self.get_subgraph(gid).await;

        // wait for all the async jobs to finish before taking the objects out of the global dict
        if self.config.async_expand {
            let token = self.cancellation.read().await.clone();
            let mut handles = vec![];
            let gadgets = self.gadgets.read().await;
            let check_models = self.check_models.read().await;
            let error_models = self.error_models.read().await;
            for &gid in subgraph.iter() {
                let gadget = &gadgets[&gid];
                if let Some(&cid) = gadget.binding_cid.borrow().as_ref() {
                    let check_model = &check_models[&cid];
                    if let Err(receiver) = check_or_receiver(&check_model.expanded_remote_gadgets, token.clone()) {
                        handles.push(receiver);
                    }
                    for &eid in check_model.attaching_eid_vec.iter() {
                        let error_model = &error_models[&eid];
                        match check_or_receiver(&error_model.expanded_remote_check_models, token.clone()) {
                            Ok(..) => {}
                            Err(receiver) => handles.push(receiver),
                        }
                    }
                }
            }
            drop(gadgets);
            drop(check_models);
            drop(error_models);
            futures_util::future::join_all(handles).await;
        }

        let gadgets: HashMap<u64, Gadget> = {
            let mut gadgets = self.gadgets.write().await;
            subgraph.iter().map(|gid| (*gid, gadgets.remove(gid).unwrap())).collect()
        };

        let check_models: HashMap<u64, CheckModel> = {
            let mut check_models = self.check_models.write().await;
            subgraph
                .iter()
                .filter_map(|gid| {
                    let gadget = &gadgets[gid];
                    if let Some(&cid) = gadget.binding_cid.borrow().as_ref() {
                        Some((cid, check_models.remove(&cid).unwrap()))
                    } else {
                        None
                    }
                })
                .collect()
        };

        let error_models: HashMap<u64, ErrorModel> = {
            let mut error_models = self.error_models.write().await;
            check_models
                .iter()
                .flat_map(|(_, check_model)| {
                    check_model
                        .attaching_eid_vec
                        .iter()
                        .map(|eid| {
                            let error_model = error_models.remove(eid).unwrap();
                            (*eid, error_model)
                        })
                        .collect::<Vec<_>>()
                        .into_iter()
                })
                .collect()
        };

        (gadgets, check_models, error_models)
    }

    async fn batch_expand(
        &self,
        gadgets: HashMap<u64, Gadget>,
        mut check_models: HashMap<u64, CheckModel>,
        mut error_models: HashMap<u64, ErrorModel>,
    ) -> (HashMap<u64, Gadget>, HashMap<u64, CheckModel>, HashMap<u64, ErrorModel>) {
        let token = self.cancellation.read().await.clone();
        let gadgets_locked = RwLock::new(gadgets);
        for check_model in check_models.values_mut() {
            let expanded_remote_gadgets = Self::expand_remote_gadgets(
                &check_model.instance,
                &check_model.modified_remote_gadgets,
                &gadgets_locked,
                token.clone(),
            )
            .await;
            check_model
                .expanded_remote_gadgets
                .send_replace(Some(expanded_remote_gadgets));
        }

        let check_models_locked = RwLock::new(check_models);
        for error_model in error_models.values_mut() {
            let expanded_remote_check_models = Self::expand_remote_check_models(
                &error_model.instance,
                &error_model.modified_remote_check_models,
                &gadgets_locked,
                &check_models_locked,
                token.clone(),
            )
            .await;
            error_model
                .expanded_remote_check_models
                .send_replace(Some(expanded_remote_check_models));
        }

        (gadgets_locked.into_inner(), check_models_locked.into_inner(), error_models)
    }

    async fn decode_subgraph(&self, gid: u64) {
        // take the gadgets, check models, and error models out of the global data
        let (mut gadgets, mut check_models, mut error_models) = self.take_subgraph(gid).await;

        // expand the check models and error models when they are not expanded asynchronously
        if !self.config.async_expand {
            (gadgets, check_models, error_models) = self.batch_expand(gadgets, check_models, error_models).await;
        }

        let mut expanded_gadgets: Vec<relative_program::ExpandedGadget> = vec![];
        let mut gid_vec: Vec<_> = gadgets.keys().cloned().collect();
        gid_vec.sort();
        let token = self.cancellation.read().await.clone();
        for &gid in gid_vec.iter() {
            let gadget = gadgets.get(&gid).unwrap();
            let inputs: Vec<_> = gadget.instance.connectors.iter().cloned().map(Some).collect();
            let outputs: Vec<_> = gadget.outputs.iter().map(|v| v.borrow().unwrap()).map(Some).collect();
            let gtype = gadget.instance.gtype;
            let cid = gadget.binding_cid.borrow().as_ref().cloned();
            let (check_model, error_models) = if let Some(cid) = cid {
                let check_model = check_models.get(&cid).unwrap();
                let remote_gadgets = get_value(&check_model.expanded_remote_gadgets, token.clone()).await;
                let Some(remote_gadgets) = remote_gadgets else { return };
                let expanded_check_model = relative_program::ExpandedCheckModel {
                    cid,
                    ctype: check_model.instance.ctype,
                    remote_gadgets,
                    count_checks: self
                        .check_model_types
                        .read()
                        .await
                        .get(&check_model.instance.ctype)
                        .unwrap()
                        .checks
                        .len(),
                };
                let mut expanded_error_models = vec![];
                for &eid in check_model.attaching_eid_vec.iter() {
                    let error_model = error_models.get(&eid).unwrap();
                    let remote_check_models = get_value(&error_model.expanded_remote_check_models, token.clone()).await;
                    let Some(remote_check_models) = remote_check_models else {
                        return;
                    };
                    expanded_error_models.push(relative_program::ExpandedErrorModel {
                        eid,
                        etype: error_model.instance.etype,
                        remote_check_models,
                    });
                }
                (Some(expanded_check_model), expanded_error_models)
            } else {
                (None, vec![])
            };
            expanded_gadgets.push(relative_program::ExpandedGadget {
                gid,
                gtype,
                inputs,
                outputs,
                check_model,
                error_models,
            });
        }
        let (relative_program, mapping) = RelativeProgram::new(&expanded_gadgets);

        let (parity_factor, errors) = self
            .decode_parity_factor(&relative_program, &mapping, &gadgets, &check_models, &error_models)
            .await;

        let updates = self
            .update_pauli_frame(&parity_factor, &errors, &relative_program, &mapping, &error_models)
            .await;

        for (gid, readouts) in updates {
            let gadget = gadgets.remove(&gid).unwrap();
            let _ = gadget.tx.send(readouts);
        }
    }

    async fn update_pauli_frame(
        &self,
        parity_factor: &blackbox_decoder::ParityFactor,
        errors: &ProjectedErrors,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        error_models: &HashMap<u64, ErrorModel>,
    ) -> Vec<(u64, BitVector)> {
        let error_model_types = self.error_model_types.read().await;
        let mut tracker = self.pauli_frame_tracker.lock().await;

        // construct the residuals and readout flips
        let mut residual_vec: Vec<BitVec> = Vec::with_capacity(relative_program.local_gadgets.len());
        let mut readout_flips_vec: Vec<BitVec> = Vec::with_capacity(relative_program.local_gadgets.len());
        for &gid in mapping.global_gid_of.iter() {
            let Some(gadget) = tracker.gadgets.get(&gid) else {
                // Tracker was reset while decode was in flight — bail out
                return vec![];
            };
            residual_vec.push(BitVec::zeros(gadget.num_output_observables()));
            readout_flips_vec.push(BitVec::zeros(gadget.num_readouts()));
        }

        // for each error, apply the effect
        for &ei in parity_factor.subgraph.iter() {
            let local_error = &errors[ei as usize];
            let local_eid = local_error.eid;
            let eid = mapping.global_eid_of[local_eid];
            let error_index = local_error.error_index;
            let error_model = error_models.get(&eid).unwrap();
            let error_model_type = error_model_types.get(&error_model.instance.etype).unwrap();
            let error = &error_model_type.errors[error_index];
            // update the corresponding gadget's residual and readout flips
            let local_gid = mapping.local_gid_of_local_eid[local_eid];
            let residual = &mut residual_vec[local_gid];
            let readout_flips = &mut readout_flips_vec[local_gid];
            for &ri in error.residual.iter() {
                residual.negate_index(ri as usize);
            }
            for &ri in error.readout_flips.iter() {
                readout_flips.negate_index(ri as usize);
            }
        }

        // update the pauli frame tracker to get responses
        // we're expecting one return value per update because the gadgets are in order
        let mut updates = vec![];
        for ((&gid, residual), readout_flips) in mapping.global_gid_of.iter().zip(residual_vec).zip(readout_flips_vec) {
            let mut single_update = tracker.load_correction(gid, residual, readout_flips);
            debug_assert_eq!(single_update.keys().cloned().collect::<Vec<_>>(), vec![gid]);
            updates.push((gid, single_update.remove(&gid).unwrap()));
        }
        updates
    }

    async fn decode_parity_factor(
        &self,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        gadgets: &HashMap<u64, Gadget>,
        check_models: &HashMap<u64, CheckModel>,
        error_models: &HashMap<u64, ErrorModel>,
    ) -> (blackbox_decoder::ParityFactor, ProjectedErrors) {
        // calculate syndrome
        let syndrome = self.get_syndrome(relative_program, mapping, gadgets, check_models).await;

        // Assemble the observed atom losses into loss sites. These are
        // shot-dependent, but only in their *content*: the loaded graph stays
        // fixed and the shot's loss travels with the decode request, so a loss
        // shot is served from the cache like any other.
        let loss_sites = self.build_loss_sites(mapping, gadgets, check_models).await;

        // Deduplication collapses edges that share a syndrome, which renumbers
        // them and, worse for a loss-aware decoder, merges a loss generator with
        // any ordinary Pauli error sharing its syndrome. A coordinator that hands
        // losses to the decoder therefore never deduplicates -- not even on a shot
        // carrying no loss, since the graph it loads then serves later shots that
        // do.
        let deduplicate = self.config.merge_hyperedges && !self.loss_handler.hands_off_to_decoder();

        let cache_key = if self.config.persistent_decoder {
            let error_model_types = self.error_model_types.read().await;
            // Construction-time modifiers define the base graph and therefore
            // its cache identity. Outcomes.modifiers are shot-scoped assignments
            // projected below; including them here would defeat decoder reuse.
            Some(DecoderCacheKey {
                relative_program: relative_program.clone(),
                error_model_fingerprints: build_modifier_fingerprints(mapping, error_models, &error_model_types),
                committing_local_cids: Vec::new(),
            })
        } else {
            None
        };

        if let Some(ref cache_key) = cache_key {
            let loaded = self.loaded_decoders.read().await.get(cache_key).cloned();
            if let Some(loaded) = loaded {
                let probability_reweights = loaded
                    .projection
                    .probability_reweights(Self::shot_probability_modifiers(mapping, gadgets));
                let projected = self
                    .loss_handler
                    .project_shot(&loaded.projection, &probability_reweights, &loss_sites);
                let parity_factor = decode_projected(
                    &self.decoder,
                    &loaded,
                    syndrome.clone(),
                    projected.reweights,
                    projected.loss,
                    self.use_loaded_reweights,
                )
                .await
                .unwrap();
                if self.config.assert_parity_factor {
                    assert_parity_factor(loaded.decoding_hypergraph.as_ref().unwrap(), &parity_factor, &syndrome);
                }
                return (parity_factor, projected.errors);
            }
        }

        // when the decoder is not available, construct a monolithic decoding hypergraph
        // and instantiate such a decoder
        let (decoding_hypergraph, errors) = self
            .decoding_hypergraph(relative_program, mapping, check_models, error_models)
            .await;

        let Some(cache_key) = cache_key else {
            let probability_reweights = Self::shot_probability_reweights(mapping, gadgets, &errors);
            // Without a persistent decoder every shot ships its whole problem, so
            // probability updates are applied before loss derives its live priors.
            let mut decoding_hypergraph = decoding_hypergraph;
            apply_reweights(&mut decoding_hypergraph, &probability_reweights);
            let (mut decoding_hypergraph, loss) = self.loss_handler.apply_sites(decoding_hypergraph, &loss_sites, &errors);
            let mut errors = errors;
            if deduplicate {
                let prepared = deduplicate_decoder_input(&decoding_hypergraph, &errors);
                decoding_hypergraph = prepared.hypergraph;
                errors = prepared.representatives;
            }
            let parity_factor = self
                .decoder
                .decode(blackbox_decoder::DecodingProblem {
                    hypergraph: Some(decoding_hypergraph.clone()),
                    syndrome: Some(syndrome.clone()),
                    loss,
                })
                .await
                .unwrap();
            if self.config.assert_parity_factor {
                assert_parity_factor(&decoding_hypergraph, &parity_factor, &syndrome);
            }
            return (parity_factor, errors.into());
        };

        // Load the stable base graph before any shot's loss is applied, so the
        // cache entry can serve every later loss pattern.
        let retain_decoding_hypergraph = !self.use_loaded_reweights || self.config.assert_parity_factor;
        let loaded = load_projected_decoder(
            &self.decoder,
            decoding_hypergraph,
            errors,
            deduplicate,
            retain_decoding_hypergraph,
            false,
        )
        .await
        .unwrap();
        let probability_reweights = loaded
            .projection
            .probability_reweights(Self::shot_probability_modifiers(mapping, gadgets));
        let projected = self
            .loss_handler
            .project_shot(&loaded.projection, &probability_reweights, &loss_sites);
        let mut loaded_decoders = self.loaded_decoders.write().await;
        loaded_decoders.insert(cache_key, loaded.clone());
        drop(loaded_decoders);
        let parity_factor = decode_projected(
            &self.decoder,
            &loaded,
            syndrome.clone(),
            projected.reweights,
            projected.loss,
            self.use_loaded_reweights,
        )
        .await
        .unwrap();
        if self.config.assert_parity_factor {
            assert_parity_factor(loaded.decoding_hypergraph.as_ref().unwrap(), &parity_factor, &syndrome);
        }
        (parity_factor, projected.errors)
    }

    async fn bind_probability_modifiers(
        &self,
        gid: u64,
        modifiers: &[bin::ProbabilityModifier],
    ) -> Result<Vec<(u64, bin::ProbabilityModifier)>, Status> {
        if modifiers.is_empty() {
            return Ok(vec![]);
        }
        let cid = self
            .gadgets
            .read()
            .await
            .get(&gid)
            .and_then(|gadget| *gadget.binding_cid.borrow())
            .ok_or_else(|| Status::failed_precondition(format!("gid={gid} has no binding check model")))?;
        let eids = self
            .check_models
            .read()
            .await
            .get(&cid)
            .map(|check_model| check_model.attaching_eid_vec.clone())
            .ok_or_else(|| Status::failed_precondition(format!("cid={cid} is not loaded")))?;
        if modifiers.len() > eids.len() {
            return Err(Status::invalid_argument(format!(
                "gid={gid} supplied {} probability modifiers for {} attached error models",
                modifiers.len(),
                eids.len()
            )));
        }
        let error_model_types = self.error_model_types.read().await;
        let error_models = self.error_models.read().await;
        let mut bound = Vec::with_capacity(modifiers.len());
        for (&eid, modifier) in eids.iter().zip(modifiers) {
            let error_model = error_models
                .get(&eid)
                .ok_or_else(|| Status::failed_precondition(format!("eid={eid} is not loaded")))?;
            let error_model_type = error_model_types
                .get(&error_model.instance.etype)
                .ok_or_else(|| Status::failed_precondition(format!("etype={} is not loaded", error_model.instance.etype)))?;
            validate_probability_modifier(modifier, error_model_type.errors.len()).map_err(Status::invalid_argument)?;
            bound.push((eid, modifier.clone()));
        }
        Ok(bound)
    }

    fn shot_probability_reweights(
        mapping: &RelativeMapping,
        gadgets: &HashMap<u64, Gadget>,
        error_reference: &[ErrorIndex],
    ) -> Vec<(u64, f64)> {
        probability_reweights(error_reference, Self::shot_probability_modifiers(mapping, gadgets))
    }

    fn shot_probability_modifiers<'a>(
        mapping: &RelativeMapping,
        gadgets: &'a HashMap<u64, Gadget>,
    ) -> Vec<(usize, &'a bin::ProbabilityModifier)> {
        let mut modifiers: Vec<_> = mapping
            .global_gid_of
            .iter()
            .filter_map(|gid| gadgets.get(gid))
            .flat_map(|gadget| gadget.probability_modifiers.iter())
            .filter_map(|(eid, modifier)| mapping.local_eid_of.get(eid).map(|&local_eid| (local_eid, modifier)))
            .collect();
        modifiers.sort_unstable_by_key(|&(local_eid, _)| local_eid);
        modifiers
    }

    async fn get_syndrome(
        &self,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        gadgets: &HashMap<u64, Gadget>,
        check_models: &HashMap<u64, CheckModel>,
    ) -> BitVector {
        let mut syndrome: BitVector = bit_vector::from_sparse_indices(relative_program.count_checks as u64, &[]);
        let check_model_types = self.check_model_types.read().await;
        for (&cid, &start_index) in mapping.global_cid_of.iter().zip(mapping.start_indices.iter()) {
            let check_model = check_models.get(&cid).unwrap();
            let check_model_type = check_model_types.get(&check_model.instance.ctype).unwrap();
            let gid = check_model.instance.gid;
            let gadget = gadgets.get(&gid).unwrap();
            let expanded_remote_ref = check_model.expanded_remote_gadgets.borrow();
            let expanded_remotes = expanded_remote_ref.as_ref().unwrap();
            let local_outcomes = gadget.outcomes.as_ref().unwrap();
            // calculate the syndrome bits
            for (check_index, check) in check_model_type.checks.iter().enumerate() {
                let mut is_defect = check.naturally_flipped;
                for measurement in &check.measurements {
                    if let Some(ri) = measurement.remote_gadget {
                        let remote_gid = expanded_remotes[ri as usize].unwrap();
                        let remote_gadget = gadgets.get(&remote_gid).unwrap();
                        is_defect ^= get_bit(
                            remote_gadget.outcomes.as_ref().unwrap(),
                            measurement.measurement_index
                                + check_model.modified_remote_gadgets[ri as usize]
                                    .as_ref()
                                    .unwrap()
                                    .measurement_bias,
                        );
                    } else {
                        is_defect ^= get_bit(local_outcomes, measurement.measurement_index);
                    }
                }
                set_bit(&mut syndrome, (start_index + check_index) as u64, is_defect);
            }
        }
        syndrome
    }

    /// Build the possible loss sites for this decode window from the observed atom losses.
    ///
    /// Returns an empty list under [`LossStrategy::Ignore`], or when no gadget in
    /// this window recorded observed losses. Otherwise it collects each gadget's
    /// possible loss sites, kept ungrouped for either coordinator-side reweighting
    /// or handoff to a loss-aware decoder.
    async fn build_loss_sites(
        &self,
        mapping: &RelativeMapping,
        gadgets: &HashMap<u64, Gadget>,
        check_models: &HashMap<u64, CheckModel>,
    ) -> Vec<RawLossSite> {
        use crate::jit::loss_compiler::{GadgetLoss, build_cross_gadget_loss_sites, build_cross_gadget_output_links};
        let mut loss_sites = Vec::new();
        if !self.loss_handler.tracks_losses() || !gadgets.values().any(|g| g.loss_mask.is_some()) {
            return loss_sites;
        }

        // Gadgets in this window that carry a loss model, in program order. A
        // gadget with no *observed* loss is still included: a loss can pass
        // through it unheralded and only be resolved downstream, so it is needed
        // to assemble the cross-gadget chain.
        let gadget_types = self.gadget_types.read().await;
        let mut gid_of_index: Vec<u64> = Vec::new();
        let mut index_of_gid: HashMap<u64, usize> = HashMap::new();
        for &gid in &mapping.global_gid_of {
            let Some(gadget) = gadgets.get(&gid) else { continue };
            if !has_loss_model(&gadget_types, gadget.instance.gtype) {
                continue;
            }
            index_of_gid.insert(gid, gid_of_index.len());
            gid_of_index.push(gid);
        }
        if gid_of_index.is_empty() {
            return loss_sites;
        }

        // The error model whose error list a site's generators index: the first
        // one attached to the gadget, per the `GadgetType.loss_model` contract.
        let local_eid_of_index: Vec<Option<usize>> = gid_of_index
            .iter()
            .map(|&gid| {
                let cid = (*gadgets.get(&gid)?.binding_cid.borrow())?;
                let eid = *check_models.get(&cid)?.attaching_eid_vec.first()?;
                mapping.local_eid_of.get(&eid).copied()
            })
            .collect();

        let loss_masks: Vec<BitVector> = gid_of_index
            .iter()
            .map(|&gid| gadgets.get(&gid).and_then(|g| g.loss_mask.clone()).unwrap_or_default())
            .collect();

        // Cross-gadget links: each gadget's output flat slot that a loss can leave
        // on, mapped to the (downstream gadget index, input flat slot) it enters.
        let port_types = self.port_types.read().await;
        let gadget_instances: Vec<&bin::Gadget> =
            gid_of_index.iter().map(|&gid| &gadgets.get(&gid).unwrap().instance).collect();
        let output_links_vec = build_cross_gadget_output_links(&gadget_instances, &index_of_gid, &gadget_types, &port_types);

        let gadget_losses: Vec<GadgetLoss> = (0..gid_of_index.len())
            .map(|index| {
                let gtype = gadgets.get(&gid_of_index[index]).unwrap().instance.gtype;
                GadgetLoss {
                    loss_model: gadget_types.get(&gtype).unwrap().loss_model.as_ref().unwrap(),
                    observed: &loss_masks[index],
                    output_links: &output_links_vec[index],
                }
            })
            .collect();

        // Assemble the possible loss chains across gadgets. Each site maps 1:1
        // onto a returned `RawLossSite` so its `children` (positions in the site
        // list) stay valid for a loss-aware decoder; the envelope-matching
        // flatten simply unions the edges and ignores `children`.
        for site in build_cross_gadget_loss_sites(&gadget_losses) {
            let local_eid = local_eid_of_index[site.gadget_index];
            loss_sites.push(RawLossSite::from_compiled(site, local_eid));
        }
        loss_sites
    }

    async fn decoding_hypergraph(
        &self,
        relative_program: &RelativeProgram,
        mapping: &RelativeMapping,
        check_models: &HashMap<u64, CheckModel>,
        error_models: &HashMap<u64, ErrorModel>,
    ) -> (DecodingHypergraph, Arc<Vec<ErrorIndex>>) {
        // note that we will not compute the effect of an error (in terms of the readout flips)
        // because the parity factor is usually sparse and it's more efficient to just propagate
        // them once. Precomputing them takes O(N^2) time because an error must propagate along
        // all the gadgets. Besides, a dynamic decoding system should indeed propagate the
        // Pauli frame at runtime to minimize latency in the absence of a static program.
        let error_model_types = self.error_model_types.read().await;

        let mut hyperedges: Vec<Hyperedge> = vec![];
        let mut error_reference: Vec<ErrorIndex> = vec![];
        for (local_cid, &cid) in mapping.global_cid_of.iter().enumerate() {
            let check_model = check_models.get(&cid).unwrap();
            for &eid in &check_model.attaching_eid_vec {
                let local_eid = mapping.local_eid_of[&eid];
                let error_model = error_models.get(&eid).unwrap();
                let error_model_type = error_model_types.get(&error_model.instance.etype).unwrap();
                let expanded_remote_ref = error_model.expanded_remote_check_models.borrow();
                let expanded_remotes = expanded_remote_ref.as_ref().unwrap();
                let mut errors = &error_model_type.errors;
                // only when there is modifier to the errors, copy the list of errors and modify
                let modified_errors: Option<Vec<bin::error_model_type::Error>>;
                if let Some(modifier) = &error_model.instance.modifier
                    && let Some(probability_modifier) = &modifier.probability_modifier
                {
                    let mut new_errors = errors.clone();
                    for (error_index, &probability) in probability_modifier.probabilities.iter().enumerate() {
                        new_errors[error_index].probability = probability;
                    }
                    for (&error_index, &probability) in probability_modifier
                        .sparse_indices
                        .iter()
                        .zip(probability_modifier.sparse_probabilities.iter())
                    {
                        new_errors[error_index as usize].probability = probability;
                    }
                    modified_errors = Some(new_errors);
                    errors = modified_errors.as_ref().unwrap();
                }
                let local_start_index = mapping.start_indices[local_cid] as u64;
                for (error_index, error) in errors.iter().enumerate() {
                    // Zero-probability errors are kept, at their prior probability.
                    // They exist for a reason -- an atom loss activates its
                    // Pauli-envelope generators, and a caller may reweight any edge
                    // -- so the hyperedge set stays independent of what happened in
                    // a given shot. That keeps edge indices stable across shots, so
                    // `LossInfo` can reference them and a loaded decoder stays
                    // reusable; the reweighting raises them from infinite weight to
                    // a usable value.
                    let probability = error.probability;
                    let mut vertices: Vec<u64> = vec![];
                    for check in &error.checks {
                        if let Some(ri) = check.remote_check_model {
                            let remote_cid = expanded_remotes[ri as usize].unwrap();
                            let remote_local_cid = mapping.local_cid_of[&remote_cid];
                            let remote_start_index = mapping.start_indices[remote_local_cid] as u64;
                            vertices.push(
                                remote_start_index
                                    + check.check_index
                                    + error_model.modified_remote_check_models[ri as usize]
                                        .as_ref()
                                        .unwrap()
                                        .check_bias,
                            );
                        } else {
                            vertices.push(local_start_index + check.check_index);
                        }
                    }
                    if vertices.is_empty() {
                        continue; // skip the no-effect errors
                    }
                    error_reference.push(ErrorIndex {
                        eid: local_eid,
                        error_index,
                    });
                    hyperedges.push(Hyperedge { vertices, probability });
                }
            }
        }
        let hypergraph = DecodingHypergraph {
            vertex_num: relative_program.count_checks as u64,
            hyperedges,
        };
        (hypergraph, Arc::new(error_reference))
    }

    /// expand the remote gadgets referred by the check model; note that this function will
    /// be waiting for the gadget if it has not been connected yet, thus it should be called
    /// in a separate async task without blocking the gRPC request.
    async fn expand_remote_gadgets(
        check_model: &bin::CheckModel,
        modified_remote_gadgets: &Vec<Option<bin::check_model_type::RemoteGadget>>,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        token: CancellationToken,
    ) -> Vec<Option<u64>> {
        // expand the remote gadgets
        let mut expanded_remote_gid_vec: Vec<Option<u64>> = vec![None; modified_remote_gadgets.len()];
        for ri in 0..modified_remote_gadgets.len() {
            Self::expand_remote_gadget(
                &mut expanded_remote_gid_vec,
                ri,
                modified_remote_gadgets,
                check_model.gid,
                gadgets,
                token.clone(),
            )
            .await;
        }
        expanded_remote_gid_vec
    }

    async fn expand_remote_gadget(
        expanded_remote_gid_vec: &mut Vec<Option<u64>>,
        ri: usize,
        remote_gadgets: &Vec<Option<bin::check_model_type::RemoteGadget>>,
        gid: u64,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        token: CancellationToken,
    ) {
        if expanded_remote_gid_vec[ri].is_some() || remote_gadgets[ri].is_none() {
            return; // already expanded or nothing to expand
        }
        let remote_gadget = remote_gadgets[ri].as_ref().unwrap();
        // if absolute_gid is provided, use it directly
        if let Some(absolute_gid) = remote_gadget.absolute_gid {
            expanded_remote_gid_vec[ri] = Some(absolute_gid);
            return;
        }
        // expand the dependent remote gadget first
        // (we do not check circular dependency here for simplicity, see ProgSpec)
        let previous = if let Some(previous) = remote_gadget.previous_remote_gadget {
            Box::pin(Self::expand_remote_gadget(
                expanded_remote_gid_vec,
                previous as usize,
                remote_gadgets,
                gid,
                gadgets,
                token.clone(),
            ))
            .await;
            expanded_remote_gid_vec[previous as usize].unwrap()
        } else {
            gid
        };
        let gadgets = gadgets.read().await;
        let gadget = gadgets.get(&previous).unwrap();
        match remote_gadget.port.unwrap() {
            bin::check_model_type::remote_gadget::Port::Output(port) => {
                let next = get_or_receiver(&gadget.outputs[port as usize], token);
                drop(gadgets); // release the read lock
                let next = match next {
                    Ok(next) => Some(next),
                    Err(handle) => handle.await.unwrap_or(None),
                };
                if let Some(next) = next {
                    expanded_remote_gid_vec[ri] = Some(next.gid);
                }
            }
            bin::check_model_type::remote_gadget::Port::Input(port) => {
                let connector = &gadget.instance.connectors[port as usize];
                expanded_remote_gid_vec[ri] = Some(connector.gid);
            }
        }
    }

    /// expand the remote check models referred by the error model; note that this function will
    /// be waiting for the gadget if the remote has not been connected yet, thus it should be called
    /// in a separate async task without blocking the gRPC request.
    async fn expand_remote_check_models(
        error_model: &bin::ErrorModel,
        modified_remote_check_models: &Vec<Option<bin::error_model_type::RemoteCheckModel>>,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        check_models: &RwLock<HashMap<u64, CheckModel>>,
        token: CancellationToken,
    ) -> Vec<Option<u64>> {
        // expand the remote check models
        let gid = check_models.read().await.get(&error_model.cid).unwrap().instance.gid;
        let mut expanded_remote_gid_vec: Vec<Option<u64>> = vec![None; modified_remote_check_models.len()];
        for ri in 0..modified_remote_check_models.len() {
            Self::expand_remote_check_model(
                &mut expanded_remote_gid_vec,
                ri,
                modified_remote_check_models,
                gid,
                gadgets,
                token.clone(),
            )
            .await;
        }
        let mut expanded_remote_cid_vec = Vec::with_capacity(modified_remote_check_models.len());
        let mut gadgets_read = gadgets.read().await;
        for (ri, gid) in expanded_remote_gid_vec.into_iter().enumerate() {
            if let Some(gid) = gid {
                // Check if this is the sentinel for absolute_cid
                if gid == u64::MAX - 1 {
                    let absolute_cid = modified_remote_check_models[ri]
                        .as_ref()
                        .unwrap()
                        .absolute_cid
                        .expect("absolute_cid should be present when sentinel is used");
                    expanded_remote_cid_vec.push(Some(absolute_cid));
                    continue;
                }
                let gadget = gadgets_read.get(&gid).unwrap();
                let cid = if let Some(&cid) = gadget.binding_cid.borrow().as_ref() {
                    cid
                } else {
                    let mut rx = gadget.binding_cid.subscribe();
                    // release the read lock and wait for the gadget to bind to some check model
                    drop(gadgets_read);
                    let cid = tokio::select! {
                        result = rx.wait_for(|v| v.is_some()) => {
                            match result {
                                Ok(v) => v.unwrap(),
                                Err(_) => return expanded_remote_cid_vec,
                            }
                        }
                        _ = token.cancelled() => { return expanded_remote_cid_vec; }
                    };
                    gadgets_read = gadgets.read().await;
                    cid
                };
                expanded_remote_cid_vec.push(Some(cid));
            } else {
                expanded_remote_cid_vec.push(None);
            }
        }
        expanded_remote_cid_vec
    }

    async fn expand_remote_check_model(
        expanded_remotes: &mut Vec<Option<u64>>,
        ri: usize,
        remote_check_models: &Vec<Option<bin::error_model_type::RemoteCheckModel>>,
        gid: u64,
        gadgets: &RwLock<HashMap<u64, Gadget>>,
        token: CancellationToken,
    ) {
        if expanded_remotes[ri].is_some() || remote_check_models[ri].is_none() {
            return; // already expanded or nothing to expand
        }
        let remote_check_model = remote_check_models[ri].as_ref().unwrap();
        // if absolute_cid is provided, use it directly (but we need to find the gid first)
        // Note: absolute_cid refers to the check model, but we expand to gid here;
        // the conversion to cid happens in expand_remote_check_models after this function
        if remote_check_model.absolute_cid.is_some() {
            // For absolute_cid, we mark as expanded with a special sentinel;
            // the caller will handle the cid lookup directly
            expanded_remotes[ri] = Some(u64::MAX - 1); // sentinel for absolute_cid
            return;
        }
        // expand the dependent remote check model first
        // (we do not check circular dependency here for simplicity, see ProgSpec)
        let previous = if let Some(previous) = remote_check_model.previous_remote_check_model {
            Box::pin(Self::expand_remote_check_model(
                expanded_remotes,
                previous as usize,
                remote_check_models,
                gid,
                gadgets,
                token.clone(),
            ))
            .await;
            expanded_remotes[previous as usize].unwrap()
        } else {
            gid
        };
        let gadgets = gadgets.read().await;
        let gadget = gadgets.get(&previous).unwrap();
        match remote_check_model.port.unwrap() {
            bin::error_model_type::remote_check_model::Port::Output(port) => {
                let next = get_or_receiver(&gadget.outputs[port as usize], token);
                drop(gadgets); // release the read lock
                let next = match next {
                    Ok(gid) => Some(gid),
                    Err(handle) => handle.await.unwrap_or(None),
                };
                if let Some(next) = next {
                    expanded_remotes[ri] = Some(next.gid);
                }
            }
            bin::error_model_type::remote_check_model::Port::Input(port) => {
                let connector = &gadget.instance.connectors[port as usize];
                expanded_remotes[ri] = Some(connector.gid);
            }
        }
    }
}

#[tonic::async_trait]
impl coordinator::coordinator_server::Coordinator for MonolithicCoordinator {
    async fn load_library(&self, request: Request<bin::Library>) -> Result<Response<()>, Status> {
        let _task_guard = self
            .task_counter
            .try_guard()
            .ok_or_else(|| Status::unavailable("coordinator reset in progress"))?;
        let library = request.into_inner();
        self.loss_handler
            .validate_capability(&library.gadget_types, self.decoder.features())?;
        let mut port_types = self.port_types.write().await;
        for port_type in library.port_types.into_iter() {
            if port_types.contains_key(&port_type.ptype) {
                return Err(Status::already_exists(format!("ptype={}", port_type.ptype)));
            }
            port_types.insert(port_type.ptype, Arc::new(port_type));
        }
        drop(port_types);
        let mut gadget_types = self.gadget_types.write().await;
        for gadget_type in library.gadget_types.into_iter() {
            if gadget_types.contains_key(&gadget_type.gtype) {
                return Err(Status::already_exists(format!("gtype={}", gadget_type.gtype)));
            }
            gadget_types.insert(gadget_type.gtype, Arc::new(gadget_type));
        }
        drop(gadget_types);
        let mut check_model_types = self.check_model_types.write().await;
        for check_model_type in library.check_model_types.into_iter() {
            if check_model_types.contains_key(&check_model_type.ctype) {
                return Err(Status::already_exists(format!("ctype={}", check_model_type.ctype)));
            }
            check_model_types.insert(check_model_type.ctype, Arc::new(check_model_type));
        }
        drop(check_model_types);
        let mut error_model_types = self.error_model_types.write().await;
        for error_model_type in library.error_model_types.into_iter() {
            if error_model_types.contains_key(&error_model_type.etype) {
                return Err(Status::already_exists(format!("etype={}", error_model_type.etype)));
            }
            error_model_types.insert(error_model_type.etype, Arc::new(error_model_type));
        }
        drop(error_model_types);
        Ok(().into())
    }

    async fn unload(&self, _unload: Request<coordinator::UnloadLibrary>) -> Result<Response<()>, Status> {
        unimplemented!()
    }

    async fn execute(&self, request: Request<bin::Instruction>) -> Result<Response<coordinator::ExecuteResponse>, Status> {
        let _task_guard = self
            .task_counter
            .try_guard()
            .ok_or_else(|| Status::unavailable("coordinator reset in progress"))?;
        let instruction = request.into_inner();
        let create = instruction
            .create
            .ok_or_else(|| Status::invalid_argument("unknown instruction"))?;
        let id = match create {
            bin::instruction::Create::Gadget(gadget) => {
                let port_types = self.port_types.read().await;
                let gadget_types = self.gadget_types.read().await;
                let mut gadgets = self.gadgets.write().await;
                let gid = if gadget.gid == 0 {
                    // Auto-assign: find next unused gid
                    let mut next_gid = self.next_gid.lock().await;
                    while gadgets.contains_key(&*next_gid) {
                        *next_gid += 1;
                    }
                    let gid = *next_gid;
                    *next_gid += 1;
                    gid
                } else {
                    // User-provided gid
                    gadget.gid
                };
                let gadget_type = gadget_types
                    .get(&gadget.gtype)
                    .ok_or_else(|| Status::not_found(format!("gtype={}", gadget.gtype)))?;
                debug_assert!(gadget.connectors.len() == gadget_type.inputs.len());
                // add a union find node with indirect mapping
                let mut pending_subgraphs = self.pending_subgraphs.lock().await;
                let mut gid_to_union_index = self.gid_to_union_index.lock().await;
                let union_index = pending_subgraphs.payload.len();
                pending_subgraphs.insert(MonolithicUnionNode::default());
                gid_to_union_index.insert(gid, union_index);
                // update the clusters
                for (port, connector) in gadget.connectors.iter().enumerate() {
                    debug_assert!(gadgets.contains_key(&connector.gid));
                    debug_assert!({
                        let peer_outputs = &gadgets[&connector.gid].outputs;
                        (connector.port as usize) < peer_outputs.len()
                            && peer_outputs[connector.port as usize].borrow().is_none()
                    });
                    let peer_union_index = gid_to_union_index[&connector.gid];
                    pending_subgraphs.union(union_index, peer_union_index);
                    gadgets.get_mut(&connector.gid).unwrap().outputs[connector.port as usize]
                        .send_replace(Some(bin::gadget::Connector { gid, port: port as u64 }));
                }
                let node = pending_subgraphs.get_mut(union_index);
                node.num_unconnected_outputs += gadget_type.outputs.len();
                node.num_unconnected_outputs -= gadget.connectors.len();
                node.num_unloaded_gadgets += 1;
                let mut tracker = self.pauli_frame_tracker.lock().await;
                tracker.add_gadget(gid, gadget_type, gadget.modifier.as_ref(), &port_types, &gadget.connectors);
                let (tx, rx) = oneshot::channel();
                let mut gadget = gadget;
                gadget.gid = gid;
                gadgets.insert(
                    gid,
                    Gadget {
                        instance: gadget,
                        outcomes: None,
                        probability_modifiers: vec![],
                        loss_mask: None,
                        binding_cid: watch::channel(None).0,
                        // important: we should not use vec![;len] syntax because it will create clones
                        outputs: gadget_type.outputs.iter().map(|_| watch::channel(None).0).collect(),
                        tx,
                        rx: Some(rx),
                    },
                );
                gid
            }
            bin::instruction::Create::CheckModel(check_model) => {
                let check_model_types = self.check_model_types.read().await;
                let mut gadgets = self.gadgets.write().await;
                let mut check_models = self.check_models.write().await;
                let check_model_type = check_model_types
                    .get(&check_model.ctype)
                    .ok_or_else(|| Status::not_found(format!("ctype={}", check_model.ctype)))?;
                let modified_remote = Arc::new(
                    apply_check_model_reroutes(&check_model_type.remote_gadgets, check_model.modifier.as_ref())
                        .map_err(Status::invalid_argument)?,
                );
                let cid = if check_model.cid == 0 {
                    // Auto-assign: find next unused cid
                    let mut next_cid = self.next_cid.lock().await;
                    while check_models.contains_key(&*next_cid) {
                        *next_cid += 1;
                    }
                    let cid = *next_cid;
                    *next_cid += 1;
                    cid
                } else {
                    // User-provided cid
                    check_model.cid
                };
                let gadget = gadgets.get_mut(&check_model.gid).ok_or_else(|| {
                    Status::invalid_argument(format!("cid={cid} binding to unknown gid={}", check_model.gid))
                })?;
                debug_assert!(check_model_type.gtype == WILDCARD || check_model_type.gtype == gadget.instance.gtype);
                debug_assert!(gadget.binding_cid.borrow().is_none());
                gadget.binding_cid.send_replace(Some(cid));
                let mut check_model = check_model;
                check_model.cid = cid;
                check_models.insert(
                    cid,
                    CheckModel {
                        instance: check_model.clone(),
                        attaching_eid_vec: vec![],
                        modified_remote_gadgets: modified_remote.clone(),
                        expanded_remote_gadgets: watch::channel(None).0,
                    },
                );
                // expanding the remote gadgets may not be immediately possible if the gadgets
                // are not instantiated yet, so we spawn an async task to do it.
                let gadgets = self.gadgets.clone();
                let check_models = self.check_models.clone();
                if self.config.async_expand {
                    let token = self.cancellation.read().await.clone();
                    let _guard = self.task_counter.guard();
                    tokio::spawn(async move {
                        let _guard = _guard;
                        let expanded_remote_gadgets =
                            Self::expand_remote_gadgets(&check_model, &modified_remote, gadgets.as_ref(), token).await;
                        let mut check_models = check_models.write().await;
                        if let Some(cm) = check_models.get_mut(&cid) {
                            cm.expanded_remote_gadgets.send_replace(Some(expanded_remote_gadgets));
                        }
                    });
                }
                cid
            }
            bin::instruction::Create::ErrorModel(error_model) => {
                let error_model_types = self.error_model_types.read().await;
                let mut check_models = self.check_models.write().await;
                let mut error_models = self.error_models.write().await;
                let error_model_type = error_model_types
                    .get(&error_model.etype)
                    .ok_or_else(|| Status::not_found(format!("etype={}", error_model.etype)))?;
                if let Some(probability_modifier) = error_model
                    .modifier
                    .as_ref()
                    .and_then(|modifier| modifier.probability_modifier.as_ref())
                {
                    validate_probability_modifier(probability_modifier, error_model_type.errors.len())
                        .map_err(Status::invalid_argument)?;
                }
                let modified_remote = Arc::new(
                    apply_error_model_reroutes(&error_model_type.remote_check_models, error_model.modifier.as_ref())
                        .map_err(Status::invalid_argument)?,
                );
                let eid = if error_model.eid == 0 {
                    // Auto-assign: find next unused eid
                    let mut next_eid = self.next_eid.lock().await;
                    while error_models.contains_key(&*next_eid) {
                        *next_eid += 1;
                    }
                    let eid = *next_eid;
                    *next_eid += 1;
                    eid
                } else {
                    // User-provided eid
                    error_model.eid
                };
                let check_model = check_models.get_mut(&error_model.cid).ok_or_else(|| {
                    Status::invalid_argument(format!("eid={eid} attaching to unknown cid={}", error_model.cid))
                })?;
                debug_assert!(error_model_type.ctype == WILDCARD || error_model_type.ctype == check_model.instance.ctype);
                check_model.attaching_eid_vec.push(eid);
                let mut error_model = error_model;
                error_model.eid = eid;
                error_models.insert(
                    eid,
                    ErrorModel {
                        instance: error_model.clone(),
                        modified_remote_check_models: modified_remote.clone(),
                        expanded_remote_check_models: watch::channel(None).0,
                    },
                );
                // expanding the remote check models may not be immediately possible if the gadgets
                // are not instantiated yet, so we spawn an async task to do it.
                let gadgets = self.gadgets.clone();
                let check_models = self.check_models.clone();
                let error_models = self.error_models.clone();
                if self.config.async_expand {
                    let token = self.cancellation.read().await.clone();
                    let _guard = self.task_counter.guard();
                    tokio::spawn(async move {
                        let _guard = _guard;
                        let expanded_remote_check_models = Self::expand_remote_check_models(
                            &error_model,
                            &modified_remote,
                            gadgets.as_ref(),
                            check_models.as_ref(),
                            token,
                        )
                        .await;
                        let mut error_models = error_models.write().await;
                        if let Some(em) = error_models.get_mut(&eid) {
                            em.expanded_remote_check_models
                                .send_replace(Some(expanded_remote_check_models));
                        }
                    });
                }
                eid
            }
        };
        Ok((coordinator::ExecuteResponse { id }).into())
    }

    async fn decode(&self, request: Request<coordinator::Outcomes>) -> Result<Response<coordinator::Readouts>, Status> {
        let outcomes = request.into_inner();
        let _task_guard = self
            .task_counter
            .try_guard()
            .ok_or_else(|| Status::unavailable("coordinator reset in progress"))?;
        let gid = outcomes.gid;
        let probability_modifiers = self.bind_probability_modifiers(gid, &outcomes.modifiers).await?;
        let gadget_types = self.gadget_types.read().await;
        let mut gadgets = self.gadgets.write().await;
        let gadget = gadgets
            .get_mut(&gid)
            .ok_or_else(|| Status::not_found(format!("gid={}", gid)))?;
        if gadget.outcomes.is_some() {
            return Err(Status::already_exists(format!("gid={} outcomes loaded", gid)));
        }
        // load the outcome
        let mut outcome_data = outcomes
            .outcomes
            .ok_or_else(|| Status::invalid_argument("missing outcomes"))?;
        let gadget_type = gadget_types
            .get(&gadget.instance.gtype)
            .ok_or_else(|| Status::failed_precondition(format!("gtype={} is not loaded", gadget.instance.gtype)))?;
        validate_outcomes(
            &outcome_data,
            outcomes.loss_mask.as_ref(),
            u64::try_from(gadget_type.measurements.len()).unwrap(),
        )
        .map_err(Status::invalid_argument)?;
        // Apply loss-random-imputation before storing the outcomes: every
        // downstream consumer (syndrome calculation, pauli-frame tracker,
        // ...) reads `gadget.outcomes` and benefits from a single
        // consistent imputed value per measurement bit.
        if let (Some(rng_lock), Some(loss_mask)) = (self.loss_imputation_rng.as_ref(), outcomes.loss_mask.as_ref()) {
            let mut rng = rng_lock.lock().await;
            apply_loss_random_imputation(&mut outcome_data, loss_mask, &mut *rng);
        }
        // Record the observed atom losses for the loss pipeline: the set of local
        // measurement indices flagged as losses. Only kept when the strategy uses
        // them and this gadget type actually carries a loss model.
        if self.loss_handler.tracks_losses()
            && has_loss_model(&gadget_types, gadget.instance.gtype)
            && let Some(loss_mask) = outcomes.loss_mask.as_ref()
            && (0..loss_mask.size).any(|index| get_bit(loss_mask, index))
        {
            gadget.loss_mask = Some(loss_mask.clone());
        }
        gadget.outcomes.replace(outcome_data);
        gadget.probability_modifiers = probability_modifiers;
        let mut pending_subgraphs = self.pending_subgraphs.lock().await;
        let gid_to_union_index = self.gid_to_union_index.lock().await;
        let union_index = gid_to_union_index[&gid];
        let node = pending_subgraphs.get_mut(union_index);
        node.num_unloaded_gadgets -= 1;
        // release all the locks and get them in order later to prevent deadlocks
        let is_final_gadget = node.num_unloaded_gadgets == 0 && node.num_unconnected_outputs == 0;
        let rx = gadget.rx.take().unwrap();
        // calculate the raw readouts (before error correction);
        let mut readouts = Vec::with_capacity(gadget_type.readouts.len());
        let data: &BitVector = gadget.outcomes.as_ref().unwrap();
        for readout in gadget_type.readouts.iter() {
            let mut value = false;
            for &mi in readout.measurement_indices.iter() {
                value ^= get_bit(data, mi);
            }
            readouts.push(value);
        }
        self.pauli_frame_tracker.lock().await.load_raw(gid, &readouts, data);
        drop(gid_to_union_index);
        drop(pending_subgraphs);
        drop(gadgets);
        drop(gadget_types);
        if is_final_gadget {
            // this is the last gadget, it is responsible for doing the decoding work
            // and inform all other async tasks
            self.decode_subgraph(gid).await;
        }
        let readouts = rx.await.map_err(|_| Status::internal(format!("gid={} receive error", gid)))?;
        return Ok((coordinator::Readouts {
            gid,
            readouts: Some(readouts),
            ..Default::default()
        })
        .into());
    }

    async fn reset(&self, request: Request<coordinator::ResetRequest>) -> Result<Response<()>, Status> {
        let flags = request.into_inner();
        let _pause = self
            .task_counter
            .try_pause()
            .ok_or_else(|| Status::unavailable("coordinator reset already in progress"))?;
        // Cancel all pending async tasks, wait for them to finish, then
        // install a fresh token so post-reset operations proceed normally.
        {
            let token = self.cancellation.read().await;
            token.cancel();
        }
        self.task_counter.wait_for_zero().await;
        {
            let mut token = self.cancellation.write().await;
            *token = CancellationToken::new();
        }
        if flags.reset_library {
            self.port_types.write().await.clear();
            self.gadget_types.write().await.clear();
            self.check_model_types.write().await.clear();
            self.error_model_types.write().await.clear();
        }
        self.gadgets.write().await.clear();
        self.check_models.write().await.clear();
        self.error_models.write().await.clear();
        *self.next_gid.lock().await = 1;
        *self.next_cid.lock().await = 1;
        *self.next_eid.lock().await = 1;
        let mut pending_subgraphs = self.pending_subgraphs.lock().await;
        pending_subgraphs.remove_all();
        self.gid_to_union_index.lock().await.clear();
        self.pauli_frame_tracker.lock().await.reset();
        if flags.reset_library || flags.reset_decoder_service {
            self.loaded_decoders.write().await.clear();
        }
        // since decoders reset asynchronously, wait for all the decoders to finish
        self.decoder
            .reset(blackbox_decoder::ResetRequest {
                reset_hypergraphs: flags.reset_decoder_service,
                ..Default::default()
            })
            .await
            .map_err(|e| Status::internal(format!("reset decoder service error: {}", e)))?;
        Ok(().into())
    }
}

/// define your own union-find node data structure like this
#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct MonolithicUnionNode {
    pub set_size: usize,
    pub num_unloaded_gadgets: usize,
    pub num_unconnected_outputs: usize,
}

/// example trait implementation
impl UnionNodeTrait for MonolithicUnionNode {
    #[inline]
    fn union(left: &Self, right: &Self) -> (bool, Self) {
        let result = Self {
            set_size: left.set_size + right.set_size,
            num_unloaded_gadgets: left.num_unloaded_gadgets + right.num_unloaded_gadgets,
            num_unconnected_outputs: left.num_unconnected_outputs + right.num_unconnected_outputs,
        };
        // if left size is larger, choose left (weighted union)
        (left.set_size >= right.set_size, result)
    }
    #[inline]
    fn clear(&mut self) {
        self.set_size = 1;
    }
    #[inline]
    fn default() -> Self {
        Self {
            set_size: 1,
            num_unloaded_gadgets: 0,
            num_unconnected_outputs: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit tests for the `MonolithicCoordinator`'s cache-key helpers.
    //!
    //! Drives `build_modifier_fingerprints` directly with hand-built
    //! `RelativeMapping` / `ErrorModel` / `error_model_types` inputs so the
    //! invariant
    //!
    //!   different per-eid modifier or etype structure ⇒ different fingerprints
    //!
    //! can be verified without running the full async coordinator.
    use super::*;
    use crate::bin::error_model::ErrorModelModifier;
    use crate::bin::error_model_type::Error;

    fn mapping_with_eids(global_eid_of: Vec<u64>) -> RelativeMapping {
        RelativeMapping {
            global_eid_of,
            ..Default::default()
        }
    }

    fn pm_dense(probabilities: Vec<f64>) -> bin::ProbabilityModifier {
        bin::ProbabilityModifier {
            probabilities,
            sparse_indices: vec![],
            sparse_probabilities: vec![],
        }
    }

    fn make_error_model_instance(eid: u64, etype: u64, modifier: Option<bin::ProbabilityModifier>) -> bin::ErrorModel {
        bin::ErrorModel {
            eid,
            etype,
            cid: 1,
            modifier: modifier.map(|p| ErrorModelModifier {
                probability_modifier: Some(p),
                reroute_remote_check_models: vec![],
            }),
            ..Default::default()
        }
    }

    fn make_error_model(instance: bin::ErrorModel) -> ErrorModel {
        let (sender, _receiver) = watch::channel(None);
        ErrorModel {
            instance,
            modified_remote_check_models: Arc::new(vec![]),
            expanded_remote_check_models: sender,
        }
    }

    fn make_emt(etype: u64, errors: Vec<Error>) -> bin::ErrorModelType {
        bin::ErrorModelType {
            etype,
            ctype: 1,
            errors,
            remote_check_models: vec![],
            ..Default::default()
        }
    }

    fn make_error(probability: f64) -> Error {
        Error {
            checks: vec![bin::error_model_type::RemoteCheck {
                remote_check_model: None,
                check_index: 0,
            }],
            probability,
            ..Default::default()
        }
    }

    /// Build a fingerprint vector indexed by `local_eid` and verify it
    /// picks up the per-eid modifier state.  Replaces the old key, which
    /// only saw the `RelativeProgram` and would have produced the same
    /// fingerprint vector regardless of modifier.
    #[test]
    fn build_modifier_fingerprints_picks_up_probability_modifier() {
        let mapping = mapping_with_eids(vec![1]);
        let mut emts: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

        let mut models_a: HashMap<u64, ErrorModel> = HashMap::new();
        models_a.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.1])))),
        );

        let mut models_b: HashMap<u64, ErrorModel> = HashMap::new();
        models_b.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.2])))),
        );

        let fps_a = build_modifier_fingerprints(&mapping, &models_a, &emts);
        let fps_b = build_modifier_fingerprints(&mapping, &models_b, &emts);
        assert_ne!(fps_a, fps_b);
        assert_eq!(fps_a.len(), 1);
    }

    /// Two error-model types with the same `etype` id but different
    /// structural contents must produce different fingerprints.  Old key
    /// stored only the `etype` id and would have collided.
    #[test]
    fn build_modifier_fingerprints_picks_up_etype_structure() {
        let mapping = mapping_with_eids(vec![1]);
        let mut models: HashMap<u64, ErrorModel> = HashMap::new();
        models.insert(1, make_error_model(make_error_model_instance(1, 1, None)));

        let mut emts_v1: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts_v1.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

        let mut emts_v2: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts_v2.insert(1, Arc::new(make_emt(1, vec![make_error(0.1), make_error(0.2)])));

        let fps_v1 = build_modifier_fingerprints(&mapping, &models, &emts_v1);
        let fps_v2 = build_modifier_fingerprints(&mapping, &models, &emts_v2);
        assert_ne!(fps_v1, fps_v2);
    }

    /// Fingerprint vector is positional: swapping which `eid` lives at a
    /// given local-eid slot must change the fingerprints, otherwise two
    /// windows that bind the same set of error models in different orders
    /// would alias.
    #[test]
    fn build_modifier_fingerprints_is_positional() {
        let mut models: HashMap<u64, ErrorModel> = HashMap::new();
        models.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.1])))),
        );
        models.insert(
            2,
            make_error_model(make_error_model_instance(2, 1, Some(pm_dense(vec![0.9])))),
        );

        let mut emts: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

        let mapping_ab = mapping_with_eids(vec![1, 2]);
        let mapping_ba = mapping_with_eids(vec![2, 1]);
        let fps_ab = build_modifier_fingerprints(&mapping_ab, &models, &emts);
        let fps_ba = build_modifier_fingerprints(&mapping_ba, &models, &emts);
        assert_ne!(fps_ab, fps_ba);
    }

    #[test]
    fn build_modifier_fingerprints_equal_for_identical_state() {
        let mapping = mapping_with_eids(vec![1]);
        let mut emts: HashMap<u64, Arc<bin::ErrorModelType>> = HashMap::new();
        emts.insert(1, Arc::new(make_emt(1, vec![make_error(0.1)])));

        let mut models_a: HashMap<u64, ErrorModel> = HashMap::new();
        models_a.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.1])))),
        );
        let mut models_b: HashMap<u64, ErrorModel> = HashMap::new();
        models_b.insert(
            1,
            make_error_model(make_error_model_instance(1, 1, Some(pm_dense(vec![0.1])))),
        );

        let fps_a = build_modifier_fingerprints(&mapping, &models_a, &emts);
        let fps_b = build_modifier_fingerprints(&mapping, &models_b, &emts);
        assert_eq!(fps_a, fps_b);
    }
}
