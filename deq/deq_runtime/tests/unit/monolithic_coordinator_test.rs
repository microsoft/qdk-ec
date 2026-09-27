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
