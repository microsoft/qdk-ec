use crate::bin::gadget::Connector;
use hashbrown::HashMap;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ExpandedGadget {
    pub gid: u64,
    pub gtype: u64,
    pub inputs: Vec<Option<Connector>>,
    pub outputs: Vec<Option<Connector>>,
    /// binding check model
    pub check_model: Option<ExpandedCheckModel>,
    /// attached error models
    pub error_models: Vec<ExpandedErrorModel>,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ExpandedCheckModel {
    pub cid: u64,
    pub ctype: u64,
    pub remote_gadgets: Vec<Option<u64>>,
    pub count_checks: usize,
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ExpandedErrorModel {
    pub eid: u64,
    pub etype: u64,
    pub remote_check_models: Vec<Option<u64>>,
}

/// a program that is agnostic to the global ids (gid, cid, eid): each gadget,
/// check model and error model is assigned by the order they are instantiated
/// and thus as long as the input program sequence is the same, the relative-id
/// program is the same; this is useful to get a cached or prepared decoder
/// instance
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RelativeProgram {
    pub local_gadgets: Vec<ExpandedGadget>,
    pub count_checks: usize,
}

#[derive(Clone, Debug, Default)]
pub struct RelativeMapping {
    // local ids starts from 0
    pub local_gid_of: HashMap<u64, usize>,
    pub global_gid_of: Vec<u64>,
    pub local_cid_of: HashMap<u64, usize>,
    pub global_cid_of: Vec<u64>,
    pub local_eid_of: HashMap<u64, usize>,
    pub global_eid_of: Vec<u64>,
    pub local_gid_of_local_eid: Vec<usize>,
    // the following fields are useful for calculating the syndrome
    pub start_indices: Vec<usize>,
    // from local_gid to the eid bias, useful to locate the error models attached to a gadget
    pub local_eid_bias: Vec<usize>,
}

impl RelativeProgram {
    pub fn new(expanded_gadgets: &[ExpandedGadget]) -> (RelativeProgram, RelativeMapping) {
        let mut mapping: RelativeMapping = Default::default();
        let mut local_gadgets = Vec::with_capacity(expanded_gadgets.len());
        let mut count_checks = 0;
        for gadget in expanded_gadgets.iter() {
            let local_gid = mapping.global_gid_of.len();
            let check_model = gadget.check_model.as_ref().map(|check_model| {
                let local_cid = mapping.global_cid_of.len();
                mapping.local_cid_of.insert(check_model.cid, local_cid);
                mapping.global_cid_of.push(check_model.cid);
                mapping.start_indices.push(count_checks);
                count_checks += check_model.count_checks;
                ExpandedCheckModel {
                    cid: local_cid as u64,
                    ctype: check_model.ctype,
                    // translate to local index later
                    remote_gadgets: check_model.remote_gadgets.clone(),
                    count_checks: check_model.count_checks,
                }
            });
            // Track eid bias per gadget (indexed by local_gid, not local_cid)
            // so that error-only gadgets (no check_model) are also covered.
            mapping.local_eid_bias.push(mapping.global_eid_of.len());
            let error_models = gadget
                .error_models
                .iter()
                .map(|error_model| {
                    let local_eid = mapping.global_eid_of.len();
                    mapping.local_eid_of.insert(error_model.eid, local_eid);
                    mapping.global_eid_of.push(error_model.eid);
                    mapping.local_gid_of_local_eid.push(local_gid);
                    ExpandedErrorModel {
                        eid: local_eid as u64,
                        etype: error_model.etype,
                        // translate to local index later
                        remote_check_models: error_model.remote_check_models.clone(),
                    }
                })
                .collect();
            mapping.local_gid_of.insert(gadget.gid, local_gid);
            mapping.global_gid_of.push(gadget.gid);
            local_gadgets.push(ExpandedGadget {
                gid: local_gid as u64,
                gtype: gadget.gtype,
                inputs: gadget.inputs.clone(),   // translate to local index later
                outputs: gadget.outputs.clone(), //  translate to local index later
                check_model,
                error_models,
            });
        }
        // now we have constructed all the local indices, we can update the indices
        for local_gadget in local_gadgets.iter_mut() {
            for connector in local_gadget.inputs.iter_mut().chain(local_gadget.outputs.iter_mut()) {
                if let Some(&mut Connector { gid: global_gid, port }) = connector.as_mut() {
                    if let Some(local_gid) = mapping.local_gid_of.get(&global_gid) {
                        connector.replace(Connector {
                            gid: *local_gid as u64,
                            port,
                        });
                    } else {
                        *connector = None; // drop unknown global index
                    }
                }
            }
            if let Some(check_model) = local_gadget.check_model.as_mut() {
                for gid in check_model.remote_gadgets.iter_mut() {
                    if let Some(global_gid) = gid.as_mut() {
                        if let Some(local_gid) = mapping.local_gid_of.get(global_gid) {
                            gid.replace(*local_gid as u64);
                        } else {
                            *gid = None; // drop unknown global index
                        }
                    }
                }
            }
            for error_model in local_gadget.error_models.iter_mut() {
                for cid in error_model.remote_check_models.iter_mut() {
                    if let Some(global_cid) = cid.as_mut() {
                        if let Some(local_cid) = mapping.local_cid_of.get(global_cid) {
                            cid.replace(*local_cid as u64);
                        } else {
                            *cid = None;
                        }
                    }
                }
            }
        }
        let program = RelativeProgram {
            local_gadgets,
            count_checks,
        };
        (program, mapping)
    }
}
