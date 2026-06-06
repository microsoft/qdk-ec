use crate::util::BitVector;
use crate::{bin, coordinator};
use hashbrown::{HashMap, HashSet};
use rand::RngExt;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "cli")]
use structdoc::StructDoc;
use tokio::sync::Mutex;
use tonic::{Request, Response, Status};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "cli", derive(StructDoc))]
#[serde(deny_unknown_fields)]
pub struct NaiveCoordinatorConfig {
    /// by default the coordinator returns random readouts, disabling it will return all 0 readouts;
    /// we default to random so that people do not mistake all-0 readouts as correct decoding
    #[serde(default)]
    pub disable_random_readouts: bool,
    /// bias the gid, cid and eid, default to 1001 which is helpful to test the controller
    #[serde(default = "default_id_bias")]
    pub id_bias: u64,
}

fn default_id_bias() -> u64 {
    1001
}

pub struct NaiveCoordinator {
    pub config: NaiveCoordinatorConfig,
    /// gtype -> number of readouts
    pub gadget_types: Mutex<HashMap<u64, usize>>,
    /// gid -> number of readouts
    pub gadgets: Mutex<HashMap<u64, usize>>,
    /// used cids
    pub check_models: Mutex<HashSet<u64>>,
    /// used eids
    pub error_models: Mutex<HashSet<u64>>,
    pub next_gid: AtomicU64,
    pub next_cid: AtomicU64,
    pub next_eid: AtomicU64,
}

impl NaiveCoordinator {
    pub fn new(config: serde_json::Value) -> Self {
        let config: NaiveCoordinatorConfig = serde_json::from_value(config).unwrap();
        let id_bias = config.id_bias;
        Self {
            config,
            gadget_types: Default::default(),
            gadgets: Default::default(),
            check_models: Default::default(),
            error_models: Default::default(),
            next_gid: AtomicU64::new(id_bias),
            next_cid: AtomicU64::new(id_bias),
            next_eid: AtomicU64::new(id_bias),
        }
    }
}

#[tonic::async_trait]
impl coordinator::coordinator_server::Coordinator for NaiveCoordinator {
    async fn load_library(&self, request: Request<bin::Library>) -> Result<Response<()>, Status> {
        let library = request.into_inner();
        let mut gadget_types = self.gadget_types.lock().await;
        for gadget_type in library.gadget_types.iter() {
            if gadget_types.contains_key(&gadget_type.gtype) {
                return Err(Status::already_exists(format!("gtype={}", gadget_type.gtype)));
            }
            gadget_types.insert(gadget_type.gtype, gadget_type.readouts.len());
        }
        Ok(().into())
    }

    async fn unload(&self, _unload: Request<coordinator::UnloadLibrary>) -> Result<Response<()>, Status> {
        unimplemented!()
    }

    async fn execute(&self, request: Request<bin::Instruction>) -> Result<Response<coordinator::ExecuteResponse>, Status> {
        let instruction = request.into_inner();
        let create = instruction
            .create
            .ok_or_else(|| Status::invalid_argument("unknown instruction"))?;
        let id = match create {
            bin::instruction::Create::Gadget(gadget) => {
                let mut gadgets = self.gadgets.lock().await;
                let gid = if gadget.gid == 0 {
                    loop {
                        let gid = self.next_gid.fetch_add(1, Ordering::Relaxed);
                        if !gadgets.contains_key(&gid) {
                            break gid;
                        }
                    }
                } else {
                    gadget.gid
                };
                let gadget_types = self.gadget_types.lock().await;
                if let Some(&count_readouts) = gadget_types.get(&gadget.gtype) {
                    assert!(gadgets.insert(gid, count_readouts).is_none());
                } else {
                    return Err(Status::not_found(format!("gtype={}", gadget.gtype)));
                }
                gid
            }
            bin::instruction::Create::CheckModel(check_model) => {
                let mut check_models = self.check_models.lock().await;
                let cid = if check_model.cid == 0 {
                    loop {
                        let cid = self.next_cid.fetch_add(1, Ordering::Relaxed);
                        if !check_models.contains(&cid) {
                            break cid;
                        }
                    }
                } else {
                    check_model.cid
                };
                assert!(check_models.insert(cid));
                cid
            }
            bin::instruction::Create::ErrorModel(error_model) => {
                let mut error_models = self.error_models.lock().await;
                let eid = if error_model.eid == 0 {
                    loop {
                        let eid = self.next_eid.fetch_add(1, Ordering::Relaxed);
                        if !error_models.contains(&eid) {
                            break eid;
                        }
                    }
                } else {
                    error_model.eid
                };
                assert!(error_models.insert(eid));
                eid
            }
        };
        Ok((coordinator::ExecuteResponse { id }).into())
    }

    async fn decode(&self, request: Request<coordinator::Outcomes>) -> Result<Response<coordinator::Readouts>, Status> {
        let outcomes = request.into_inner();
        let mut gadgets = self.gadgets.lock().await;
        let count_readouts = gadgets
            .remove(&outcomes.gid)
            .ok_or_else(|| Status::not_found(format!("gid={}", outcomes.gid)))?;
        let mut bits = vec![false; count_readouts];
        if !self.config.disable_random_readouts {
            let mut rng = rand::rng();
            for v in bits.iter_mut() {
                *v = rng.random_range(0..2) == 1;
            }
        }
        let bit_vector = BitVector {
            size: count_readouts as u64,
            data: crate::misc::bit_vector::pack_bits(&bits),
        };
        Ok((coordinator::Readouts {
            gid: outcomes.gid,
            readouts: Some(bit_vector),
            ..Default::default()
        })
        .into())
    }

    async fn reset(&self, _request: Request<coordinator::ResetRequest>) -> Result<Response<()>, Status> {
        let flags = _request.into_inner();
        if flags.reset_library {
            self.gadget_types.lock().await.clear();
        }
        self.gadgets.lock().await.clear();
        self.check_models.lock().await.clear();
        self.error_models.lock().await.clear();
        self.next_gid.store(self.config.id_bias, Ordering::Relaxed);
        self.next_cid.store(self.config.id_bias, Ordering::Relaxed);
        self.next_eid.store(self.config.id_bias, Ordering::Relaxed);
        Ok(().into())
    }
}
