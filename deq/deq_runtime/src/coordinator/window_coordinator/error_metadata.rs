//! Facts derived once from an error-model type and reused by many decoding windows.
//!
//! An error model lists possible faults and the syndrome checks each fault can
//! flip. A check can belong to the same gadget or to another gadget. For the
//! latter, `remote_check_model` is a slot in the type's `remote_check_models`
//! list, not a global gadget or check ID. Each model instance binds those slots
//! to its own circuit locations.
//!
//! Window decoding handles only part of the circuit at a time. When preparing a
//! window, it needs to find the remote checks affected by its faults, validate a
//! possible terminal fallback, and identify the graph it can reuse. Neighboring
//! windows often use the same type, so rescanning every fault for these facts
//! would repeat work even though the type has not changed.

use crate::bin;
use crate::coordinator::decoder_cache_key::etype_digest;
use hashbrown::HashSet;
use std::sync::{Arc, Weak};

/// Reusable metadata for one immutable error-model type, not for one window.
///
/// For example, suppose a type declares four remote slots and its faults refer
/// directly to slots `[2, 0, 2]`. `indices` is `[0, 2]`. Window exploration can
/// start resolving those two slots without scanning the fault list again.
/// Resolution still follows any predecessor slots needed to reach their targets.
///
/// At an open circuit boundary, a terminal fallback can refer to local checks
/// and the reserved future-check placeholder. If its largest local check index
/// is 2, it fits an owner with three checks but not one with two. `terminal_shape`
/// stores that bound and whether the remote references have the required form.
///
/// `digest` summarizes the type's decoding-relevant contents for graph-cache
/// keys. Changing a fault probability from 0.01 to 0.02 changes graph weights;
/// reusing the numeric type ID alone would not detect that change.
///
/// These facts belong to the exact `Arc` allocation passed to `new`. A replacement
/// type, even with the same numeric ID, must be checked again. The `Weak` keeps
/// the allocation's identity from being reused while allowing the old type's
/// payload to be freed. It also makes `Arc::make_mut` detach before mutation.
/// Lookups return `None` for a different allocation so callers recompute the
/// metadata instead of using stale facts.
pub(super) struct CachedErrorRemotes {
    model_type: Weak<bin::ErrorModelType>,
    indices: Arc<[u64]>,
    terminal_shape: (Option<u64>, bool),
    digest: u64,
}

impl CachedErrorRemotes {
    pub(super) fn new(model_type: &Arc<bin::ErrorModelType>) -> Self {
        Self {
            model_type: Arc::downgrade(model_type),
            indices: referenced_error_remotes(model_type),
            terminal_shape: terminal_check_shape(model_type),
            digest: etype_digest(model_type),
        }
    }

    pub(super) fn indices_for_type(&self, model_type: &Arc<bin::ErrorModelType>) -> Option<Arc<[u64]>> {
        std::ptr::eq(self.model_type.as_ptr(), Arc::as_ptr(model_type)).then(|| Arc::clone(&self.indices))
    }

    pub(super) fn terminal_for_type(&self, model_type: &Arc<bin::ErrorModelType>, checks: usize) -> Option<bool> {
        std::ptr::eq(self.model_type.as_ptr(), Arc::as_ptr(model_type))
            .then(|| terminal_shape_fits(self.terminal_shape, checks))
    }

    pub(super) fn digest_for_type(&self, model_type: &Arc<bin::ErrorModelType>) -> Option<u64> {
        std::ptr::eq(self.model_type.as_ptr(), Arc::as_ptr(model_type)).then_some(self.digest)
    }
}

/// The largest local check index (if any), and validity of all future-check references.
///
/// The fallback uses check 0 of `FUTURE_CHECK_CID` as a boundary placeholder,
/// rather than following input/output ports to a check that may not exist yet.
pub(super) fn terminal_check_shape(model_type: &bin::ErrorModelType) -> (Option<u64>, bool) {
    let remote_count = u64::try_from(model_type.remote_check_models.len()).unwrap();
    let mut valid = model_type.remote_check_models.iter().all(|remote| {
        remote.absolute_cid == Some(crate::misc::index::FUTURE_CHECK_CID)
            && remote.previous_remote_check_model.is_none()
            && remote.port.is_none()
            && remote.check_bias == 0
    });
    let mut largest_local = None;
    for check in model_type.errors.iter().flat_map(|error| &error.checks) {
        if let Some(index) = check.remote_check_model {
            valid &= index < remote_count && check.check_index == 0;
        } else {
            largest_local = Some(largest_local.map_or(check.check_index, |index: u64| index.max(check.check_index)));
        }
    }
    (largest_local, valid)
}

pub(super) fn terminal_shape_fits((largest_local, valid): (Option<u64>, bool), checks: usize) -> bool {
    valid && largest_local.is_none_or(|index| index < u64::try_from(checks).unwrap())
}

/// Sorted, distinct remote slots mentioned directly by faults; local checks add no slot.
pub(super) fn referenced_error_remotes(model_type: &bin::ErrorModelType) -> Arc<[u64]> {
    let used: HashSet<_> = model_type
        .errors
        .iter()
        .flat_map(|error| &error.checks)
        .filter_map(|check| check.remote_check_model)
        .collect();
    let mut indices: Vec<_> = used.into_iter().collect();
    indices.sort_unstable();
    indices.into()
}
