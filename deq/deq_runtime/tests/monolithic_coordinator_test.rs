//! Tests for MonolithicCoordinator using MockDecoder

use deq_runtime::bin::{self, instruction};
use deq_runtime::coordinator::coordinator_server::Coordinator;
use deq_runtime::coordinator::monolithic_coordinator::MonolithicCoordinator;
use deq_runtime::decoder::{BlackBoxDecoderClient, MockDecoder};
use deq_runtime::util::{BitMatrix, BitVector};
use std::sync::Arc;
use tonic::Request;

fn make_mock_decoder() -> Arc<MockDecoder> {
    Arc::new(MockDecoder::new())
}

fn make_decoder_client(mock: Arc<MockDecoder>) -> BlackBoxDecoderClient {
    BlackBoxDecoderClient::from_mock(mock)
}

fn make_coordinator(mock: Arc<MockDecoder>) -> MonolithicCoordinator {
    let config = serde_json::json!({
        "persistent_decoder": false,
        "merge_hyperedges": false
    });
    MonolithicCoordinator::new(config, make_decoder_client(mock))
}

fn make_gadget(gid: u64, gtype: u64, connectors: Vec<(u64, u64)>) -> bin::Gadget {
    bin::Gadget {
        gid,
        gtype,
        connectors: connectors
            .into_iter()
            .map(|(gid, port)| bin::gadget::Connector { gid, port })
            .collect(),
        ..Default::default()
    }
}

fn make_check_model(cid: u64, ctype: u64, gid: u64) -> bin::CheckModel {
    bin::CheckModel {
        cid,
        ctype,
        gid,
        ..Default::default()
    }
}

fn make_error_model(eid: u64, etype: u64, cid: u64) -> bin::ErrorModel {
    bin::ErrorModel {
        eid,
        etype,
        cid,
        ..Default::default()
    }
}

/// Creates a canonical-style library similar to the Python test's default_library_canonical.
/// This uses a port type with NO observables, which simplifies the pauli frame tracker.
///
/// Matrix dimensions for gadget type with:
/// - 0 input observables (no input ports)
/// - 0 output observables (port type has no observables)
/// - 1 readout
///
/// correction_propagation: rows=0 (output obs), cols=1 (input obs + 1)
/// readout_propagation: rows=1 (readouts), cols=1 (input obs + 1)
/// logical_correction: rows=0 (output obs), cols=1 (readouts)
fn make_canonical_library() -> bin::Library {
    // Port type with NO observables - this is key for simplifying tests
    let port_type = bin::PortType {
        ptype: 1,
        observables: vec![], // No observables!
        ..Default::default()
    };

    // Gadget type similar to the canonical form
    let gadget_type = bin::GadgetType {
        gtype: 1,
        inputs: vec![],
        outputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        measurements: vec![bin::gadget_type::Measurement::default()],
        readouts: vec![bin::gadget_type::Readout {
            measurement_indices: vec![0],
            ..Default::default()
        }],
        // rows=0 (output obs), cols=1 (input obs + 1)
        correction_propagation: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        // rows=1 (readouts), cols=1 (input obs + 1)
        readout_propagation: Some(BitMatrix {
            rows: 1,
            cols: 1,
            ..Default::default()
        }),
        // rows=0 (output obs), cols=1 (readouts)
        logical_correction: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        physical_correction: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        ..Default::default()
    };

    // Check model type with one check
    let check_model_type = bin::CheckModelType {
        ctype: 1,
        gtype: 1,
        checks: vec![bin::check_model_type::Check {
            measurements: vec![bin::check_model_type::RemoteMeasurement {
                measurement_index: 0,
                remote_gadget: None,
            }],
            naturally_flipped: false,
            ..Default::default()
        }],
        remote_gadgets: vec![],
        ..Default::default()
    };

    // Error model type with one error
    let error_model_type = bin::ErrorModelType {
        etype: 1,
        ctype: 1,
        errors: vec![bin::error_model_type::Error {
            probability: 0.1,
            checks: vec![bin::error_model_type::RemoteCheck {
                check_index: 0,
                remote_check_model: None,
            }],
            residual: vec![],
            readout_flips: vec![],
            ..Default::default()
        }],
        remote_check_models: vec![],
        ..Default::default()
    };

    bin::Library {
        port_types: vec![port_type],
        gadget_types: vec![gadget_type],
        check_model_types: vec![check_model_type],
        error_model_types: vec![error_model_type],
        ..Default::default()
    }
}

#[tokio::test]
async fn test_monolithic_coordinator_load_library() {
    let mock = make_mock_decoder();
    let coordinator = make_coordinator(mock.clone());

    Coordinator::load_library(&coordinator, Request::new(make_canonical_library()))
        .await
        .unwrap();

    // Verify types were loaded
    let gadget_types = coordinator.gadget_types.read().await;
    assert!(gadget_types.contains_key(&1));

    let check_model_types = coordinator.check_model_types.read().await;
    assert!(check_model_types.contains_key(&1));

    let error_model_types = coordinator.error_model_types.read().await;
    assert!(error_model_types.contains_key(&1));
}

#[tokio::test]
async fn test_monolithic_coordinator_reset() {
    let mock = make_mock_decoder();
    let coordinator = make_coordinator(mock.clone());

    Coordinator::load_library(&coordinator, Request::new(make_canonical_library()))
        .await
        .unwrap();

    // Reset with library
    Coordinator::reset(
        &coordinator,
        Request::new(deq_runtime::coordinator::ResetRequest {
            reset_library: true,
            reset_decoder_service: true,
            ..Default::default()
        }),
    )
    .await
    .unwrap();

    // Verify library was cleared
    let gadget_types = coordinator.gadget_types.read().await;
    assert!(gadget_types.is_empty());

    // Verify ID counters were reset
    let next_gid = *coordinator.next_gid.lock().await;
    assert_eq!(next_gid, 1);

    // Verify decoder was reset
    let state = mock.state.read().await;
    assert_eq!(state.reset_count, 1);
}

#[tokio::test]
async fn test_monolithic_coordinator_auto_assigned_gid() {
    let mock = make_mock_decoder();
    let coordinator = make_coordinator(mock.clone());

    Coordinator::load_library(&coordinator, Request::new(make_canonical_library()))
        .await
        .unwrap();

    // Auto-assign (gid=0)
    let gadget = make_gadget(0, 1, vec![]);
    let response = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(gadget)),
        }),
    )
    .await
    .unwrap();

    let gid = response.into_inner().id;
    assert_eq!(gid, 1);

    // Verify gadget exists
    let gadgets = coordinator.gadgets.read().await;
    assert!(gadgets.contains_key(&1));
}

#[tokio::test]
async fn test_monolithic_coordinator_user_provided_gid() {
    let mock = make_mock_decoder();
    let coordinator = make_coordinator(mock.clone());

    Coordinator::load_library(&coordinator, Request::new(make_canonical_library()))
        .await
        .unwrap();

    // User provides gid=100
    let gadget = make_gadget(100, 1, vec![]);
    let response = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(gadget)),
        }),
    )
    .await
    .unwrap();

    assert_eq!(response.into_inner().id, 100);

    // Verify gadget exists with correct id
    let gadgets = coordinator.gadgets.read().await;
    assert!(gadgets.contains_key(&100));
}

#[tokio::test]
async fn test_monolithic_coordinator_mixed_gid_assignment() {
    let mock = make_mock_decoder();
    let coordinator = make_coordinator(mock.clone());

    Coordinator::load_library(&coordinator, Request::new(make_canonical_library()))
        .await
        .unwrap();

    // User provides gid=1 (would conflict with first auto-assignment)
    let gadget1 = make_gadget(1, 1, vec![]);
    let response = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(gadget1)),
        }),
    )
    .await
    .unwrap();
    assert_eq!(response.into_inner().id, 1);

    // Auto-assign should skip 1 and use 2
    let gadget2 = make_gadget(0, 1, vec![]);
    let response = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(gadget2)),
        }),
    )
    .await
    .unwrap();
    assert_eq!(response.into_inner().id, 2);

    // Verify both gadgets exist
    let gadgets = coordinator.gadgets.read().await;
    assert!(gadgets.contains_key(&1));
    assert!(gadgets.contains_key(&2));
}

#[tokio::test]
async fn test_monolithic_coordinator_cid_assignment() {
    let mock = make_mock_decoder();
    let coordinator = make_coordinator(mock.clone());

    Coordinator::load_library(&coordinator, Request::new(make_canonical_library()))
        .await
        .unwrap();

    // Create gadget first
    let gadget = make_gadget(0, 1, vec![]);
    let gid_response = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(gadget)),
        }),
    )
    .await
    .unwrap();
    let gid = gid_response.into_inner().id;

    // User provides cid=50
    let check_model = make_check_model(50, 1, gid);
    let response = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(check_model)),
        }),
    )
    .await
    .unwrap();

    assert_eq!(response.into_inner().id, 50);

    // Verify check model exists
    let check_models = coordinator.check_models.read().await;
    assert!(check_models.contains_key(&50));
}

#[tokio::test]
async fn test_monolithic_coordinator_eid_assignment() {
    let mock = make_mock_decoder();
    let coordinator = make_coordinator(mock.clone());

    Coordinator::load_library(&coordinator, Request::new(make_canonical_library()))
        .await
        .unwrap();

    // Create gadget
    let gadget = make_gadget(0, 1, vec![]);
    let gid = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(gadget)),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id;

    // Create check model
    let check_model = make_check_model(0, 1, gid);
    let cid = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(check_model)),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id;

    // User provides eid=99
    let error_model = make_error_model(99, 1, cid);
    let response = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::ErrorModel(error_model)),
        }),
    )
    .await
    .unwrap();

    assert_eq!(response.into_inner().id, 99);

    // Verify error model exists
    let error_models = coordinator.error_models.read().await;
    assert!(error_models.contains_key(&99));
}

/// Creates the default_library from the Python test suite (library_validator_test.py).
/// This is a complete library with:
/// - 2 port types (rep-code with 1 observable, surface-code with 2 observables)
/// - 3 gadget types (initialize, cnot, measure)
/// - 2 check model types
/// - 2 error model types
fn make_default_library() -> bin::Library {
    // Port types
    let port_type_rep = bin::PortType {
        ptype: 1,
        name: "rep-code".to_string(),
        observables: vec![bin::port_type::Observable {
            tag: "logical_z".to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };

    let port_type_surface = bin::PortType {
        ptype: 2,
        name: "surface-code".to_string(),
        observables: vec![
            bin::port_type::Observable {
                tag: "logical_x".to_string(),
                ..Default::default()
            },
            bin::port_type::Observable {
                tag: "logical_z".to_string(),
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    // Gadget type 1: initialize (no inputs, 1 output of ptype=1)
    // correction_propagation: rows=1 (output obs), cols=1 (0 input obs + 1)
    let gadget_type_initialize = bin::GadgetType {
        gtype: 1,
        name: "initialize".to_string(),
        inputs: vec![],
        outputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        measurements: vec![
            bin::gadget_type::Measurement {
                tag: "m1".to_string(),
                ..Default::default()
            },
            bin::gadget_type::Measurement {
                tag: "m2".to_string(),
                ..Default::default()
            },
        ],
        // correction_propagation: rows=1 (output obs), cols=1 (0 input obs + 1)
        correction_propagation: Some(BitMatrix {
            rows: 1,
            cols: 1,
            ..Default::default()
        }),
        // logical_correction: rows=1 (output obs), cols=0 (0 readouts)
        logical_correction: Some(BitMatrix {
            rows: 1,
            cols: 0,
            ..Default::default()
        }),
        // readout_propagation: rows=0 (0 readouts), cols=1 (0 input obs + 1)
        readout_propagation: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        physical_correction: Some(BitMatrix {
            rows: 1,
            cols: 2,
            ..Default::default()
        }),
        ..Default::default()
    };

    // Gadget type 2: cnot (1 input of ptype=1, 1 output of ptype=2)
    // input has 1 observable, output has 2 observables
    // 2 measurements, 0 readouts
    let gadget_type_cnot = bin::GadgetType {
        gtype: 2,
        name: "cnot".to_string(),
        inputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        outputs: vec![bin::gadget_type::Port {
            ptype: 2,
            ..Default::default()
        }],
        measurements: vec![
            bin::gadget_type::Measurement {
                tag: "m3".to_string(),
                ..Default::default()
            },
            bin::gadget_type::Measurement {
                tag: "m4".to_string(),
                ..Default::default()
            },
        ],
        // correction_propagation: rows=2 (output obs), cols=2 (1 input obs + 1)
        // bits set at (0,0) and (0,1)
        correction_propagation: Some(BitMatrix {
            rows: 2,
            cols: 2,
            i: vec![0, 0],
            j: vec![0, 1],
        }),
        // logical_correction: rows=2 (output obs), cols=0 (0 readouts)
        logical_correction: Some(BitMatrix {
            rows: 2,
            cols: 0,
            ..Default::default()
        }),
        // readout_propagation: rows=0 (0 readouts), cols=2 (1 input obs + 1)
        readout_propagation: Some(BitMatrix {
            rows: 0,
            cols: 2,
            ..Default::default()
        }),
        physical_correction: Some(BitMatrix {
            rows: 2,
            cols: 2,
            ..Default::default()
        }),
        ..Default::default()
    };

    // Gadget type 3: measure (1 input of ptype=2, no outputs)
    // input has 2 observables
    // readout_propagation: rows=1 (readouts), cols=3 (2 input obs + 1)
    let gadget_type_measure = bin::GadgetType {
        gtype: 3,
        name: "measure".to_string(),
        inputs: vec![bin::gadget_type::Port {
            ptype: 2,
            ..Default::default()
        }],
        outputs: vec![],
        measurements: vec![
            bin::gadget_type::Measurement {
                tag: "m5".to_string(),
                ..Default::default()
            },
            bin::gadget_type::Measurement {
                tag: "m6".to_string(),
                ..Default::default()
            },
            bin::gadget_type::Measurement {
                tag: "m7".to_string(),
                ..Default::default()
            },
        ],
        readouts: vec![bin::gadget_type::Readout {
            tag: "r1".to_string(),
            measurement_indices: vec![0, 2],
            ..Default::default()
        }],
        // rows=1, cols=3, bits set at (0,0) and (0,2)
        readout_propagation: Some(BitMatrix {
            rows: 1,
            cols: 3,
            i: vec![0, 0],
            j: vec![0, 2],
        }),
        // correction_propagation: rows=0, cols=3 (no outputs)
        correction_propagation: Some(BitMatrix {
            rows: 0,
            cols: 3,
            ..Default::default()
        }),
        // logical_correction: rows=0, cols=1
        logical_correction: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        physical_correction: Some(BitMatrix {
            rows: 0,
            cols: 3,
            ..Default::default()
        }),
        ..Default::default()
    };

    // Check model type 1: attached to gtype=2
    let check_model_type_1 = bin::CheckModelType {
        ctype: 1,
        gtype: 2,
        remote_gadgets: vec![
            bin::check_model_type::RemoteGadget {
                port: Some(bin::check_model_type::remote_gadget::Port::Input(0)),
                expecting_gtype: 1,
                measurement_bias: 1,
                ..Default::default()
            },
            bin::check_model_type::RemoteGadget {
                port: Some(bin::check_model_type::remote_gadget::Port::Output(0)),
                previous_remote_gadget: Some(0),
                expecting_gtype: 2,
                ..Default::default()
            },
            bin::check_model_type::RemoteGadget {
                port: Some(bin::check_model_type::remote_gadget::Port::Output(0)),
                previous_remote_gadget: None,
                expecting_gtype: 3,
                ..Default::default()
            },
        ],
        checks: vec![
            bin::check_model_type::Check {
                tag: "c1".to_string(),
                measurements: vec![
                    bin::check_model_type::RemoteMeasurement {
                        measurement_index: 1,
                        remote_gadget: None,
                    },
                    bin::check_model_type::RemoteMeasurement {
                        measurement_index: 0,
                        remote_gadget: Some(0),
                    },
                    bin::check_model_type::RemoteMeasurement {
                        measurement_index: 0,
                        remote_gadget: Some(1),
                    },
                    bin::check_model_type::RemoteMeasurement {
                        measurement_index: 1,
                        remote_gadget: Some(2),
                    },
                ],
                ..Default::default()
            },
            bin::check_model_type::Check {
                tag: "c2".to_string(),
                ..Default::default()
            },
            bin::check_model_type::Check {
                tag: "c3".to_string(),
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    // Check model type 2: attached to gtype=3
    let check_model_type_2 = bin::CheckModelType {
        ctype: 2,
        gtype: 3,
        remote_gadgets: vec![bin::check_model_type::RemoteGadget {
            port: Some(bin::check_model_type::remote_gadget::Port::Input(0)),
            expecting_gtype: 2,
            ..Default::default()
        }],
        checks: vec![
            bin::check_model_type::Check {
                tag: "c4".to_string(),
                measurements: vec![
                    bin::check_model_type::RemoteMeasurement {
                        measurement_index: 0,
                        remote_gadget: None,
                    },
                    bin::check_model_type::RemoteMeasurement {
                        measurement_index: 1,
                        remote_gadget: None,
                    },
                    bin::check_model_type::RemoteMeasurement {
                        measurement_index: 2,
                        remote_gadget: None,
                    },
                    bin::check_model_type::RemoteMeasurement {
                        measurement_index: 0,
                        remote_gadget: Some(0),
                    },
                ],
                ..Default::default()
            },
            bin::check_model_type::Check {
                tag: "c5".to_string(),
                ..Default::default()
            },
            bin::check_model_type::Check {
                tag: "c6".to_string(),
                ..Default::default()
            },
        ],
        ..Default::default()
    };

    // Error model type 1: attached to ctype=1
    let error_model_type_1 = bin::ErrorModelType {
        etype: 1,
        ctype: 1,
        remote_check_models: vec![
            bin::error_model_type::RemoteCheckModel {
                port: Some(bin::error_model_type::remote_check_model::Port::Output(0)),
                expecting_ctype: 2,
                ..Default::default()
            },
            bin::error_model_type::RemoteCheckModel {
                port: Some(bin::error_model_type::remote_check_model::Port::Input(0)),
                previous_remote_check_model: Some(0),
                expecting_ctype: 1,
                ..Default::default()
            },
            bin::error_model_type::RemoteCheckModel {
                port: Some(bin::error_model_type::remote_check_model::Port::Output(0)),
                expecting_ctype: 2,
                ..Default::default()
            },
        ],
        errors: vec![bin::error_model_type::Error {
            probability: 0.1,
            residual: vec![0],
            checks: vec![
                bin::error_model_type::RemoteCheck {
                    check_index: 0,
                    remote_check_model: None,
                },
                bin::error_model_type::RemoteCheck {
                    check_index: 1,
                    remote_check_model: Some(1),
                },
                bin::error_model_type::RemoteCheck {
                    check_index: 1,
                    remote_check_model: Some(2),
                },
            ],
            ..Default::default()
        }],
        ..Default::default()
    };

    // Error model type 2: attached to ctype=2
    let error_model_type_2 = bin::ErrorModelType {
        etype: 2,
        ctype: 2,
        remote_check_models: vec![bin::error_model_type::RemoteCheckModel {
            port: Some(bin::error_model_type::remote_check_model::Port::Input(0)),
            expecting_ctype: 1,
            ..Default::default()
        }],
        errors: vec![bin::error_model_type::Error {
            probability: 0.1,
            readout_flips: vec![0],
            ..Default::default()
        }],
        ..Default::default()
    };

    bin::Library {
        port_types: vec![port_type_rep, port_type_surface],
        gadget_types: vec![gadget_type_initialize, gadget_type_cnot, gadget_type_measure],
        check_model_types: vec![check_model_type_1, check_model_type_2],
        error_model_types: vec![error_model_type_1, error_model_type_2],
        ..Default::default()
    }
}

/// Test that executing the default_library program produces the correct internal state.
/// This mirrors the structure from Python test library_validator_test.py.
///
/// The program creates:
/// - 3 gadgets (initialize, cnot, measure)
/// - 2 check models (ctype=1 attached to gid=2, ctype=2 attached to gid=3)
/// - 2 error models (etype=1 attached to cid=1, etype=2 attached to cid=2)
///
/// The check model types have:
/// - ctype=1: 3 checks (c1, c2, c3)
/// - ctype=2: 3 checks (c4, c5, c6)
/// Total: 6 checks
///
/// The error model types have:
/// - etype=1: 1 error touching 3 checks
/// - etype=2: 1 error with readout flip
/// Total: 2 errors
#[tokio::test]
async fn test_hypergraph_construction_with_default_library() {
    let mock = make_mock_decoder();
    let coordinator = make_coordinator(mock.clone());

    // Load the library
    Coordinator::load_library(&coordinator, Request::new(make_default_library()))
        .await
        .unwrap();

    // Execute the program as in default_library:
    // 1. gadget(gtype=1, tag="initialize")
    let gid1 = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(bin::Gadget {
                gid: 0,
                gtype: 1,
                tag: "initialize".to_string(),
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id;
    assert_eq!(gid1, 1);

    // 2. gadget(gtype=2, tag="idle", connector to gid=1 port=0)
    let gid2 = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(bin::Gadget {
                gid: 0,
                gtype: 2,
                tag: "idle".to_string(),
                connectors: vec![bin::gadget::Connector { gid: 1, port: 0 }],
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id;
    assert_eq!(gid2, 2);

    // 3. check_model(ctype=1, gid=2)
    let cid1 = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(bin::CheckModel {
                cid: 0,
                ctype: 1,
                gid: 2,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id;
    assert_eq!(cid1, 1);

    // 4. error_model(etype=1, cid=1)
    let eid1 = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 0,
                etype: 1,
                cid: 1,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id;
    assert_eq!(eid1, 1);

    // 5. gadget(gtype=3, tag="measure", connector to gid=2 port=0)
    let gid3 = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(bin::Gadget {
                gid: 0,
                gtype: 3,
                tag: "measure".to_string(),
                connectors: vec![bin::gadget::Connector { gid: 2, port: 0 }],
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id;
    assert_eq!(gid3, 3);

    // 6. check_model(ctype=2, gid=3)
    let cid2 = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(bin::CheckModel {
                cid: 0,
                ctype: 2,
                gid: 3,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id;
    assert_eq!(cid2, 2);

    // 7. error_model(etype=2, cid=2)
    let eid2 = Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 0,
                etype: 2,
                cid: 2,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap()
    .into_inner()
    .id;
    assert_eq!(eid2, 2);

    // Verify the internal state was correctly constructed
    // 3 gadgets
    let gadgets = coordinator.gadgets.read().await;
    assert_eq!(gadgets.len(), 3, "Expected 3 gadgets");
    assert!(gadgets.contains_key(&1), "gid=1 should exist");
    assert!(gadgets.contains_key(&2), "gid=2 should exist");
    assert!(gadgets.contains_key(&3), "gid=3 should exist");

    // Verify gadget types
    assert_eq!(gadgets.get(&1).unwrap().instance.gtype, 1);
    assert_eq!(gadgets.get(&2).unwrap().instance.gtype, 2);
    assert_eq!(gadgets.get(&3).unwrap().instance.gtype, 3);

    // Verify connectors
    assert!(gadgets.get(&1).unwrap().instance.connectors.is_empty());
    assert_eq!(gadgets.get(&2).unwrap().instance.connectors.len(), 1);
    assert_eq!(gadgets.get(&2).unwrap().instance.connectors[0].gid, 1);
    assert_eq!(gadgets.get(&3).unwrap().instance.connectors.len(), 1);
    assert_eq!(gadgets.get(&3).unwrap().instance.connectors[0].gid, 2);
    drop(gadgets);

    // 2 check models
    let check_models = coordinator.check_models.read().await;
    assert_eq!(check_models.len(), 2, "Expected 2 check models");
    assert!(check_models.contains_key(&1), "cid=1 should exist");
    assert!(check_models.contains_key(&2), "cid=2 should exist");

    // Verify check model bindings
    assert_eq!(check_models.get(&1).unwrap().instance.gid, 2);
    assert_eq!(check_models.get(&1).unwrap().instance.ctype, 1);
    assert_eq!(check_models.get(&2).unwrap().instance.gid, 3);
    assert_eq!(check_models.get(&2).unwrap().instance.ctype, 2);
    drop(check_models);

    // 2 error models
    let error_models = coordinator.error_models.read().await;
    assert_eq!(error_models.len(), 2, "Expected 2 error models");
    assert!(error_models.contains_key(&1), "eid=1 should exist");
    assert!(error_models.contains_key(&2), "eid=2 should exist");

    // Verify error model bindings
    assert_eq!(error_models.get(&1).unwrap().instance.cid, 1);
    assert_eq!(error_models.get(&1).unwrap().instance.etype, 1);
    assert_eq!(error_models.get(&2).unwrap().instance.cid, 2);
    assert_eq!(error_models.get(&2).unwrap().instance.etype, 2);
    drop(error_models);

    // Verify the check model types have the expected number of checks
    let check_model_types = coordinator.check_model_types.read().await;
    let cmt1 = check_model_types.get(&1).unwrap();
    let cmt2 = check_model_types.get(&2).unwrap();
    assert_eq!(cmt1.checks.len(), 3, "ctype=1 should have 3 checks");
    assert_eq!(cmt2.checks.len(), 3, "ctype=2 should have 3 checks");
    drop(check_model_types);

    // Verify the error model types have the expected number of errors
    let error_model_types = coordinator.error_model_types.read().await;
    let emt1 = error_model_types.get(&1).unwrap();
    let emt2 = error_model_types.get(&2).unwrap();
    assert_eq!(emt1.errors.len(), 1, "etype=1 should have 1 error");
    assert_eq!(emt2.errors.len(), 1, "etype=2 should have 1 error");

    // Verify error details
    assert!(
        (emt1.errors[0].probability - 0.1).abs() < 1e-9,
        "etype=1 error probability should be 0.1"
    );
    assert!(
        (emt2.errors[0].probability - 0.1).abs() < 1e-9,
        "etype=2 error probability should be 0.1"
    );

    // etype=1 error should touch 3 checks
    assert_eq!(emt1.errors[0].checks.len(), 3, "etype=1 error should reference 3 checks");

    // etype=2 error should have 1 readout flip
    assert_eq!(
        emt2.errors[0].readout_flips.len(),
        1,
        "etype=2 error should have 1 readout flip"
    );
}

/// Test that invoking decode triggers hypergraph construction and sends it to the decoder.
/// Uses the default_library and provides measurement outcomes to trigger the decode flow.
#[tokio::test]
async fn test_decode_triggers_hypergraph_construction() {
    let mock = make_mock_decoder();
    let coordinator = make_coordinator(mock.clone());

    // Load the library
    Coordinator::load_library(&coordinator, Request::new(make_default_library()))
        .await
        .unwrap();

    // Execute the full program
    // 1. gadget(gtype=1)
    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(bin::Gadget {
                gid: 0,
                gtype: 1,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    // 2. gadget(gtype=2, connector to gid=1)
    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(bin::Gadget {
                gid: 0,
                gtype: 2,
                connectors: vec![bin::gadget::Connector { gid: 1, port: 0 }],
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    // 3. check_model(ctype=1, gid=2)
    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(bin::CheckModel {
                cid: 0,
                ctype: 1,
                gid: 2,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    // 4. error_model(etype=1, cid=1)
    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 0,
                etype: 1,
                cid: 1,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    // 5. gadget(gtype=3, connector to gid=2)
    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(bin::Gadget {
                gid: 0,
                gtype: 3,
                connectors: vec![bin::gadget::Connector { gid: 2, port: 0 }],
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    // 6. check_model(ctype=2, gid=3)
    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(bin::CheckModel {
                cid: 0,
                ctype: 2,
                gid: 3,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    // 7. error_model(etype=2, cid=2)
    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 0,
                etype: 2,
                cid: 2,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    // Now load outcomes for all gadgets to trigger decode
    // gid=1 has 2 measurements, gid=2 has 2 measurements, gid=3 has 3 measurements
    // Note: decode() awaits until the final gadget completes decoding.
    // We must call all decodes concurrently, otherwise non-final gadgets block forever.

    let coordinator_clone1 = &coordinator;
    let coordinator_clone2 = &coordinator;
    let coordinator_clone3 = &coordinator;

    // Spawn all decode calls concurrently using tokio::join!
    let (result1, result2, result3) = tokio::join!(
        async {
            Coordinator::decode(
                coordinator_clone1,
                Request::new(deq_runtime::coordinator::Outcomes {
                    gid: 1,
                    outcomes: Some(BitVector { data: vec![0], size: 2 }),
                    ..Default::default()
                }),
            )
            .await
        },
        async {
            Coordinator::decode(
                coordinator_clone2,
                Request::new(deq_runtime::coordinator::Outcomes {
                    gid: 2,
                    outcomes: Some(BitVector { data: vec![0], size: 2 }),
                    ..Default::default()
                }),
            )
            .await
        },
        async {
            Coordinator::decode(
                coordinator_clone3,
                Request::new(deq_runtime::coordinator::Outcomes {
                    gid: 3,
                    outcomes: Some(BitVector { data: vec![0], size: 3 }),
                    ..Default::default()
                }),
            )
            .await
        }
    );

    result1.unwrap();
    result2.unwrap();
    result3.unwrap();

    // Verify the hypergraph was sent to the decoder
    let state = mock.state.read().await;

    // With persistent_decoder=false, decode_calls should have been made
    // (not load_hypergraph + decode_loaded)
    assert!(
        !state.decode_calls.is_empty() || !state.loaded_hypergraphs.is_empty(),
        "Expected either decode_calls or loaded_hypergraphs"
    );

    // Check the hypergraph structure
    if !state.decode_calls.is_empty() {
        let hypergraph = &state.decode_calls[0].hypergraph;

        // The canonical form has 6 checks (vertices)
        assert_eq!(hypergraph.vertex_num, 6, "Expected 6 vertices (checks) in the hypergraph");

        // Only errors with non-trivial syndrome (checks) are included in hypergraph.
        // etype=2 only has readout_flips and no checks, so it's not included.
        // Thus we expect 1 hyperedge (from etype=1).
        assert_eq!(
            hypergraph.hyperedges.len(),
            1,
            "Expected 1 hyperedge (error with checks) in the hypergraph"
        );

        // Verify error probability
        let first_error = &hypergraph.hyperedges[0];
        assert!(
            (first_error.probability - 0.1).abs() < 1e-9,
            "Expected probability 0.1, got {}",
            first_error.probability
        );

        // First error should touch 3 checks (c1, c2, c5 = indices 0, 1, 4)
        assert_eq!(first_error.vertices.len(), 3, "First error should touch 3 checks");
        assert!(first_error.vertices.contains(&0), "Error should touch check 0");
        assert!(first_error.vertices.contains(&1), "Error should touch check 1");
        assert!(first_error.vertices.contains(&4), "Error should touch check 4");
    }
}

/// Test remote_conditional_correction with a chain of gadgets where the correction actually affects
/// the output observable.
///
/// Setup:
/// - Three gadgets: gid=1 -> gid=2 -> gid=3
/// - Gadget 2 has remote_conditional_correction referencing gadget 1's readout
/// - This tests that the remote correction is correctly applied across multiple gadgets
#[tokio::test]
async fn test_remote_conditional_correction_affects_residual() {
    let mock = make_mock_decoder();
    let coordinator = make_coordinator(mock.clone());

    // Port type with 1 observable
    let port_type = bin::PortType {
        ptype: 1,
        observables: vec![bin::port_type::Observable {
            tag: "Z".to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };

    // Gadget type 1: source gadget with 1 output, 1 readout
    let gadget_type_1 = bin::GadgetType {
        gtype: 1,
        inputs: vec![],
        outputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        measurements: vec![bin::gadget_type::Measurement::default()],
        readouts: vec![bin::gadget_type::Readout {
            measurement_indices: vec![0],
            ..Default::default()
        }],
        correction_propagation: Some(BitMatrix {
            rows: 1,
            cols: 1,
            ..Default::default()
        }),
        readout_propagation: Some(BitMatrix {
            rows: 1,
            cols: 1,
            ..Default::default()
        }),
        logical_correction: Some(BitMatrix {
            rows: 1,
            cols: 1,
            ..Default::default()
        }),
        physical_correction: Some(BitMatrix {
            rows: 1,
            cols: 1,
            ..Default::default()
        }),
        ..Default::default()
    };

    // Gadget type 2: middle gadget with 1 input, 1 output, 1 readout
    let gadget_type_2 = bin::GadgetType {
        gtype: 2,
        inputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        outputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        measurements: vec![bin::gadget_type::Measurement::default()],
        readouts: vec![bin::gadget_type::Readout {
            measurement_indices: vec![0],
            ..Default::default()
        }],
        // correction_propagation: pass through the input observable
        // rows=1 (output obs), cols=2 (1 input obs + 1)
        // Set bit at (0, 0) for output[0] = input[0]
        correction_propagation: Some(BitMatrix {
            rows: 1,
            cols: 2,
            i: vec![0],
            j: vec![0],
        }),
        readout_propagation: Some(BitMatrix {
            rows: 1,
            cols: 2,
            ..Default::default()
        }),
        logical_correction: Some(BitMatrix {
            rows: 1,
            cols: 1,
            ..Default::default()
        }),
        physical_correction: Some(BitMatrix {
            rows: 1,
            cols: 1,
            ..Default::default()
        }),
        ..Default::default()
    };

    // Gadget type 3: sink gadget with 1 input, no outputs, 1 readout
    let gadget_type_3 = bin::GadgetType {
        gtype: 3,
        inputs: vec![bin::gadget_type::Port {
            ptype: 1,
            ..Default::default()
        }],
        outputs: vec![],
        measurements: vec![bin::gadget_type::Measurement::default()],
        readouts: vec![bin::gadget_type::Readout {
            measurement_indices: vec![0],
            ..Default::default()
        }],
        correction_propagation: Some(BitMatrix {
            rows: 0,
            cols: 2,
            ..Default::default()
        }),
        // Readout XORs with input observable
        // Set bit at (0, 0) for readout[0] XOR= input_observable[0]
        readout_propagation: Some(BitMatrix {
            rows: 1,
            cols: 2,
            i: vec![0],
            j: vec![0],
        }),
        logical_correction: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        physical_correction: Some(BitMatrix {
            rows: 0,
            cols: 1,
            ..Default::default()
        }),
        ..Default::default()
    };

    let check_model_type = bin::CheckModelType {
        ctype: 1,
        gtype: 0, // WILDCARD
        checks: vec![],
        ..Default::default()
    };

    let error_model_type = bin::ErrorModelType {
        etype: 1,
        ctype: 0, // WILDCARD
        errors: vec![],
        ..Default::default()
    };

    let library = bin::Library {
        port_types: vec![port_type],
        gadget_types: vec![gadget_type_1, gadget_type_2, gadget_type_3],
        check_model_types: vec![check_model_type],
        error_model_types: vec![error_model_type],
        ..Default::default()
    };

    Coordinator::load_library(&coordinator, Request::new(library)).await.unwrap();

    // Create gadget 1
    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(bin::Gadget {
                gid: 1,
                gtype: 1,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    // Check model and error model for gadget 1
    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(bin::CheckModel {
                cid: 1,
                ctype: 1,
                gid: 1,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 1,
                etype: 1,
                cid: 1,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    // Create gadget 2 (middle, with remote_conditional_correction from gadget 1)
    // The remote_conditional_correction XORs gadget 1's readout[0] into gadget 2's output observable
    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(bin::Gadget {
                gid: 2,
                gtype: 2,
                connectors: vec![bin::gadget::Connector { gid: 1, port: 0 }],
                modifier: Some(bin::GadgetModifier {
                    remote_conditional_correction: Some(bin::RemoteConditionalCorrection {
                        remote_readouts: vec![bin::remote_conditional_correction::RemoteReadout {
                            gid: 1,
                            readout_index: 0,
                        }],
                        // Correction matrix: 1 row (1 output observable), 1 col (1 remote readout)
                        // output[0] XOR= remote_readout[0]
                        // Set bit at (0, 0)
                        correction: Some(BitMatrix {
                            rows: 1,
                            cols: 1,
                            i: vec![0],
                            j: vec![0],
                        }),
                    }),
                    ..Default::default()
                }),
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(bin::CheckModel {
                cid: 2,
                ctype: 1,
                gid: 2,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 2,
                etype: 1,
                cid: 2,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    // Create gadget 3 (sink)
    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(bin::Gadget {
                gid: 3,
                gtype: 3,
                connectors: vec![bin::gadget::Connector { gid: 2, port: 0 }],
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(bin::CheckModel {
                cid: 3,
                ctype: 1,
                gid: 3,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    Coordinator::execute(
        &coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 3,
                etype: 1,
                cid: 3,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    // Test case 1: gadget 1 readout = 1
    // Expected flow:
    // - Gadget 1: measurement=1, readout=1, residual=0 (no logical_correction effect)
    // - Gadget 2: has remote_conditional_correction that XORs gadget 1's readout (=1) into output observable
    //   So gadget 2's residual (output observable) = 1 XOR 0 = 1
    // - Gadget 3: readout XORs with input observable (from gadget 2 = 1)
    //   So gadget 3's readout = measurement[0] XOR input_observable[0] = 0 XOR 1 = 1
    //
    // Note: BitVector uses MSB-first encoding: bit 0 is at position 7 in the first byte.
    // So to set bit 0 = 1, use 0x80 (0b10000000), not 0x01.
    let (result1, result2, result3) = tokio::join!(
        async {
            Coordinator::decode(
                &coordinator,
                Request::new(deq_runtime::coordinator::Outcomes {
                    gid: 1,
                    outcomes: Some(BitVector {
                        data: vec![0x80], // measurement 0 = 1 (MSB-first: bit 0 at position 7)
                        size: 1,
                    }),
                    ..Default::default()
                }),
            )
            .await
        },
        async {
            Coordinator::decode(
                &coordinator,
                Request::new(deq_runtime::coordinator::Outcomes {
                    gid: 2,
                    outcomes: Some(BitVector {
                        data: vec![0x00], // measurement 0 = 0
                        size: 1,
                    }),
                    ..Default::default()
                }),
            )
            .await
        },
        async {
            Coordinator::decode(
                &coordinator,
                Request::new(deq_runtime::coordinator::Outcomes {
                    gid: 3,
                    outcomes: Some(BitVector {
                        data: vec![0x00], // measurement 0 = 0
                        size: 1,
                    }),
                    ..Default::default()
                }),
            )
            .await
        }
    );

    let readouts1 = result1.unwrap().into_inner().readouts.unwrap();
    let readouts2 = result2.unwrap().into_inner().readouts.unwrap();
    let readouts3 = result3.unwrap().into_inner().readouts.unwrap();

    // Verify decode completes successfully with expected sizes
    assert_eq!(readouts1.size, 1, "Gadget 1 should have 1 readout");
    assert_eq!(readouts2.size, 1, "Gadget 2 should have 1 readout");
    assert_eq!(readouts3.size, 1, "Gadget 3 should have 1 readout");

    // The remote_conditional_correction XORs gadget 1's readout into gadget 2's output observable.
    // This affects gadget 3's readout via readout_propagation (which XORs with input observable).
    //
    // Trace through the computation:
    // - Gadget 1: measurement=1, readout = raw XOR readout_propagation(input) = 1 XOR 0 = 1
    // - Gadget 2: residual = correction_propagation(input) XOR remote_cc(gadget1.readout)
    //             = 0 XOR 1 = 1 (the remote_conditional_correction applies gadget 1's readout)
    // - Gadget 3: readout = raw XOR readout_propagation(input) = 0 XOR gadget2.residual[0] = 0 XOR 1 = 1
    //
    // Note: Readout BitVector uses MSB-first: bit 0 is at position 7 in the first byte (0x80).
    // Verify actual readout values:
    assert_eq!(
        readouts1.data[0] & 0x80,
        0x80,
        "Gadget 1 readout should be 1 (from measurement)"
    );
    assert_eq!(
        readouts2.data[0] & 0x80,
        0,
        "Gadget 2 readout should be 0 (measurement=0, no readout_propagation)"
    );
    assert_eq!(
        readouts3.data[0] & 0x80,
        0x80,
        "Gadget 3 readout should be 1 (0 XOR input_observable from gadget 2's residual)"
    );
}

// ─── persistent_decoder cache-key regression tests ──────────────────────────
//
// These tests exercise the `MonolithicCoordinator::loaded_decoders` cache
// with `persistent_decoder: true`.  The cache is keyed by `DecoderCacheKey`,
// which (after the `fix/window-coordinator-cache-key` fix) must distinguish
// shots that share a `RelativeProgram` but differ in their resolved
// error-model state.  Under the previous key (which keyed only on
// `RelativeProgram`), the second shot below would incorrectly reuse the
// first shot's loaded hypergraph despite using a different probability
// modifier.

fn make_persistent_coordinator(mock: Arc<MockDecoder>) -> MonolithicCoordinator {
    let config = serde_json::json!({
        "persistent_decoder": true,
        "merge_hyperedges": false
    });
    MonolithicCoordinator::new(config, make_decoder_client(mock))
}

/// Build the canonical three-gadget program (initialize → cnot → measure) and
/// attach error models with the supplied probability modifiers, then trigger
/// the decode pipeline by submitting outcomes concurrently for all gadgets.
async fn run_canonical_shot(
    coordinator: &MonolithicCoordinator,
    modifier_for_etype_1: Option<bin::ProbabilityModifier>,
    modifier_for_etype_2: Option<bin::ProbabilityModifier>,
) {
    let wrap_modifier = |pm: Option<bin::ProbabilityModifier>| {
        pm.map(|p| bin::error_model::ErrorModelModifier {
            probability_modifier: Some(p),
            reroute_remote_check_models: vec![],
        })
    };

    Coordinator::execute(
        coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(bin::Gadget {
                gid: 0,
                gtype: 1,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    Coordinator::execute(
        coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(bin::Gadget {
                gid: 0,
                gtype: 2,
                connectors: vec![bin::gadget::Connector { gid: 1, port: 0 }],
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    Coordinator::execute(
        coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(bin::CheckModel {
                cid: 0,
                ctype: 1,
                gid: 2,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    Coordinator::execute(
        coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 0,
                etype: 1,
                cid: 1,
                modifier: wrap_modifier(modifier_for_etype_1),
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    Coordinator::execute(
        coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::Gadget(bin::Gadget {
                gid: 0,
                gtype: 3,
                connectors: vec![bin::gadget::Connector { gid: 2, port: 0 }],
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    Coordinator::execute(
        coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::CheckModel(bin::CheckModel {
                cid: 0,
                ctype: 2,
                gid: 3,
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    Coordinator::execute(
        coordinator,
        Request::new(bin::Instruction {
            create: Some(instruction::Create::ErrorModel(bin::ErrorModel {
                eid: 0,
                etype: 2,
                cid: 2,
                modifier: wrap_modifier(modifier_for_etype_2),
                ..Default::default()
            })),
        }),
    )
    .await
    .unwrap();

    // All three decode calls must be submitted concurrently — see the note
    // in `test_decode_triggers_hypergraph_construction`.
    let (r1, r2, r3) = tokio::join!(
        async {
            Coordinator::decode(
                coordinator,
                Request::new(deq_runtime::coordinator::Outcomes {
                    gid: 1,
                    outcomes: Some(BitVector { data: vec![0], size: 2 }),
                    ..Default::default()
                }),
            )
            .await
        },
        async {
            Coordinator::decode(
                coordinator,
                Request::new(deq_runtime::coordinator::Outcomes {
                    gid: 2,
                    outcomes: Some(BitVector { data: vec![0], size: 2 }),
                    ..Default::default()
                }),
            )
            .await
        },
        async {
            Coordinator::decode(
                coordinator,
                Request::new(deq_runtime::coordinator::Outcomes {
                    gid: 3,
                    outcomes: Some(BitVector { data: vec![0], size: 3 }),
                    ..Default::default()
                }),
            )
            .await
        }
    );
    r1.unwrap();
    r2.unwrap();
    r3.unwrap();
}

/// Reset between shots, keeping the library and the persisted decoder cache.
async fn reset_keeping_library_and_decoder(coordinator: &MonolithicCoordinator) {
    Coordinator::reset(
        coordinator,
        Request::new(deq_runtime::coordinator::ResetRequest {
            reset_library: false,
            reset_decoder_service: false,
            ..Default::default()
        }),
    )
    .await
    .unwrap();
}

/// Regression test for `loaded_decoders` cache-key correctness.
///
/// Drives two shots with identical `RelativeProgram` but with the
/// probability modifier on `etype=1` switched from `0.1` to `0.2` in the
/// second shot.  Under the new `DecoderCacheKey` the
/// `ErrorModelFingerprint` for that slot must differ, forcing a second
/// `load_hypergraph` call.  Under the old key (which keyed only on
/// `RelativeProgram`) the second shot would incorrectly reuse the cached
/// hypergraph built with the `0.1` probability.
#[tokio::test]
async fn test_persistent_decoder_distinguishes_probability_modifier_across_shots() {
    let mock = make_mock_decoder();
    let coordinator = make_persistent_coordinator(mock.clone());

    Coordinator::load_library(&coordinator, Request::new(make_default_library()))
        .await
        .unwrap();

    // Shot 1: probability modifier p = 0.1 on etype=1.
    run_canonical_shot(
        &coordinator,
        Some(bin::ProbabilityModifier {
            probabilities: vec![0.1],
            ..Default::default()
        }),
        None,
    )
    .await;

    reset_keeping_library_and_decoder(&coordinator).await;

    // Shot 2: same program shape, but probability modifier p = 0.2 on
    // etype=1.  Different `DecoderCacheKey` → new `load_hypergraph` call.
    run_canonical_shot(
        &coordinator,
        Some(bin::ProbabilityModifier {
            probabilities: vec![0.2],
            ..Default::default()
        }),
        None,
    )
    .await;

    let mock_state = mock.state.read().await;
    assert_eq!(
        mock_state.loaded_hypergraphs.len(),
        2,
        "Expected 2 distinct loaded hypergraphs (one per modifier), got {}; old cache key \
         would reuse the first one and report 1",
        mock_state.loaded_hypergraphs.len(),
    );
    let loaded_decoders = coordinator.loaded_decoders.read().await;
    assert_eq!(
        loaded_decoders.len(),
        2,
        "Expected 2 distinct DecoderCacheKey entries (one per modifier), got {}",
        loaded_decoders.len(),
    );
}

/// Positive control: two shots with *identical* error-model modifier must
/// reuse the cached hypergraph (cache hit on the second shot), so exactly
/// one `load_hypergraph` call is observed and a `decode_loaded` call is
/// served from the cache.  This guards the other direction — the new key
/// must not be over-strict.
#[tokio::test]
async fn test_persistent_decoder_reuses_cache_when_modifier_unchanged() {
    let mock = make_mock_decoder();
    let coordinator = make_persistent_coordinator(mock.clone());

    Coordinator::load_library(&coordinator, Request::new(make_default_library()))
        .await
        .unwrap();

    let modifier = bin::ProbabilityModifier {
        probabilities: vec![0.1],
        ..Default::default()
    };

    run_canonical_shot(&coordinator, Some(modifier.clone()), None).await;
    reset_keeping_library_and_decoder(&coordinator).await;
    run_canonical_shot(&coordinator, Some(modifier), None).await;

    let mock_state = mock.state.read().await;
    assert_eq!(
        mock_state.loaded_hypergraphs.len(),
        1,
        "Expected 1 cached hypergraph reused across both shots, got {}",
        mock_state.loaded_hypergraphs.len(),
    );
    assert!(
        !mock_state.decode_loaded_calls.is_empty(),
        "Expected the second shot to hit the cache and call decode_loaded",
    );
    let loaded_decoders = coordinator.loaded_decoders.read().await;
    assert_eq!(loaded_decoders.len(), 1, "Expected a single cache entry");
}
