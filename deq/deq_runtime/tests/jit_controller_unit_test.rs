//! Unit tests for JitController type keys and caching

use deq_runtime::bin::{self, check_model_type, error_model_type};
use deq_runtime::controller::jit_controller::{CheckModelTypeKey, ErrorModelTypeKey, TypeCache};
use deq_runtime::visualizer::Position;

#[test]
fn test_check_model_type_key_equality() {
    let cmt1 = bin::CheckModelType {
        ctype: 1,
        name: "check1".to_string(),
        gtype: 5,
        checks: vec![check_model_type::Check {
            tag: "tag1".to_string(),
            measurements: vec![check_model_type::RemoteMeasurement {
                remote_gadget: None,
                measurement_index: 0,
            }],
            naturally_flipped: false,
            relative: None,
            ..Default::default()
        }],
        ..Default::default()
    };

    let cmt2 = bin::CheckModelType {
        ctype: 2,
        name: "check2".to_string(),
        gtype: 5,
        checks: vec![check_model_type::Check {
            tag: "different_tag".to_string(),
            measurements: vec![check_model_type::RemoteMeasurement {
                remote_gadget: None,
                measurement_index: 0,
            }],
            naturally_flipped: false,
            relative: Some(Position::default()),
            ..Default::default()
        }],
        ..Default::default()
    };

    let key1 = CheckModelTypeKey(cmt1);
    let key2 = CheckModelTypeKey(cmt2);

    assert_eq!(
        key1, key2,
        "Keys should be equal despite different ctype, name, tag, and relative"
    );
}

#[test]
fn test_check_model_type_key_inequality() {
    let cmt1 = bin::CheckModelType {
        ctype: 1,
        gtype: 5,
        checks: vec![check_model_type::Check {
            measurements: vec![check_model_type::RemoteMeasurement {
                remote_gadget: None,
                measurement_index: 0,
            }],
            ..Default::default()
        }],
        ..Default::default()
    };

    let cmt2 = bin::CheckModelType {
        ctype: 1,
        gtype: 5,
        checks: vec![check_model_type::Check {
            measurements: vec![check_model_type::RemoteMeasurement {
                remote_gadget: None,
                measurement_index: 1,
            }],
            ..Default::default()
        }],
        ..Default::default()
    };

    let key1 = CheckModelTypeKey(cmt1);
    let key2 = CheckModelTypeKey(cmt2);

    assert_ne!(key1, key2, "Keys should differ when measurement_index differs");
}

#[test]
fn test_error_model_type_key_equality() {
    let emt1 = bin::ErrorModelType {
        etype: 1,
        name: "error1".to_string(),
        ctype: 5,
        errors: vec![error_model_type::Error {
            tag: "tag1".to_string(),
            checks: vec![error_model_type::RemoteCheck {
                remote_check_model: None,
                check_index: 0,
            }],
            residual: vec![1],
            readout_flips: vec![],
            probability: 0.01,
            relative: None,
            ..Default::default()
        }],
        ..Default::default()
    };

    let emt2 = bin::ErrorModelType {
        etype: 2,
        name: "error2".to_string(),
        ctype: 5,
        errors: vec![error_model_type::Error {
            tag: "different_tag".to_string(),
            checks: vec![error_model_type::RemoteCheck {
                remote_check_model: None,
                check_index: 0,
            }],
            residual: vec![1],
            readout_flips: vec![],
            probability: 0.01,
            relative: Some(Position::default()),
            ..Default::default()
        }],
        ..Default::default()
    };

    let key1 = ErrorModelTypeKey(emt1);
    let key2 = ErrorModelTypeKey(emt2);

    assert_eq!(
        key1, key2,
        "Keys should be equal despite different etype, name, tag, and relative"
    );
}

#[test]
fn test_error_model_type_key_inequality_probability() {
    let emt1 = bin::ErrorModelType {
        etype: 1,
        ctype: 5,
        errors: vec![error_model_type::Error {
            probability: 0.01,
            ..Default::default()
        }],
        ..Default::default()
    };

    let emt2 = bin::ErrorModelType {
        etype: 1,
        ctype: 5,
        errors: vec![error_model_type::Error {
            probability: 0.02,
            ..Default::default()
        }],
        ..Default::default()
    };

    let key1 = ErrorModelTypeKey(emt1);
    let key2 = ErrorModelTypeKey(emt2);

    assert_ne!(key1, key2, "Keys should differ when probability differs");
}

#[test]
fn test_hashmap_usage() {
    use std::collections::HashMap;

    let cmt1 = bin::CheckModelType {
        ctype: 1,
        gtype: 5,
        checks: vec![check_model_type::Check {
            tag: "tag1".to_string(),
            measurements: vec![check_model_type::RemoteMeasurement {
                remote_gadget: None,
                measurement_index: 0,
            }],
            ..Default::default()
        }],
        ..Default::default()
    };

    let cmt2 = bin::CheckModelType {
        ctype: 2,
        gtype: 5,
        checks: vec![check_model_type::Check {
            tag: "different_tag".to_string(),
            measurements: vec![check_model_type::RemoteMeasurement {
                remote_gadget: None,
                measurement_index: 0,
            }],
            ..Default::default()
        }],
        ..Default::default()
    };

    let mut cache: HashMap<CheckModelTypeKey, u64> = HashMap::new();
    cache.insert(CheckModelTypeKey(cmt1), 100);

    assert_eq!(cache.get(&CheckModelTypeKey(cmt2)), Some(&100));
}

#[test]
fn test_type_cache_check_model() {
    let mut cache = TypeCache::new();

    let cmt1 = bin::CheckModelType {
        ctype: 1,
        gtype: 5,
        checks: vec![check_model_type::Check {
            tag: "tag1".to_string(),
            measurements: vec![check_model_type::RemoteMeasurement {
                remote_gadget: None,
                measurement_index: 0,
            }],
            ..Default::default()
        }],
        ..Default::default()
    };

    assert_eq!(cache.get_check_model_type(&cmt1), None);

    cache.insert_check_model_type(&cmt1, 42);
    assert_eq!(cache.get_check_model_type(&cmt1), Some(42));

    let cmt2 = bin::CheckModelType {
        ctype: 99,
        gtype: 5,
        checks: vec![check_model_type::Check {
            tag: "different_tag".to_string(),
            measurements: vec![check_model_type::RemoteMeasurement {
                remote_gadget: None,
                measurement_index: 0,
            }],
            ..Default::default()
        }],
        ..Default::default()
    };
    assert_eq!(
        cache.get_check_model_type(&cmt2),
        Some(42),
        "Should find same ctype despite different metadata"
    );
}

#[test]
fn test_type_cache_error_model() {
    let mut cache = TypeCache::new();

    let emt1 = bin::ErrorModelType {
        etype: 1,
        ctype: 5,
        errors: vec![error_model_type::Error {
            probability: 0.01,
            residual: vec![1],
            ..Default::default()
        }],
        ..Default::default()
    };

    assert_eq!(cache.get_error_model_type(&emt1), None);

    cache.insert_error_model_type(&emt1, 99);
    assert_eq!(cache.get_error_model_type(&emt1), Some(99));

    let emt2 = bin::ErrorModelType {
        etype: 999,
        name: "different".to_string(),
        ctype: 5,
        errors: vec![error_model_type::Error {
            tag: "different_tag".to_string(),
            probability: 0.01,
            residual: vec![1],
            ..Default::default()
        }],
        ..Default::default()
    };
    assert_eq!(
        cache.get_error_model_type(&emt2),
        Some(99),
        "Should find same etype despite different metadata"
    );
}

#[test]
fn test_type_cache_clear() {
    let mut cache = TypeCache::new();

    let cmt = bin::CheckModelType {
        ctype: 1,
        gtype: 5,
        checks: vec![],
        ..Default::default()
    };

    let emt = bin::ErrorModelType {
        etype: 1,
        ctype: 5,
        errors: vec![],
        ..Default::default()
    };

    cache.insert_check_model_type(&cmt, 42);
    cache.insert_error_model_type(&emt, 99);

    assert!(cache.get_check_model_type(&cmt).is_some());
    assert!(cache.get_error_model_type(&emt).is_some());

    cache.clear();

    assert!(cache.get_check_model_type(&cmt).is_none());
    assert!(cache.get_error_model_type(&emt).is_none());
}
