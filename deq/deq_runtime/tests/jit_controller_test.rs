//! Integration tests for JitController with MockCoordinator
//!
//! Tests verify that JIT controller correctly compiles instructions and sends
//! appropriate types/instances to the coordinator, with and without caching.
//! Also tests timing (error models blocked until outputs connected) and
//! correctness (comparing against static_jit_compile output).

mod common;

use deq_runtime::bin::{self, check_model_type, error_model_type, gadget_type};
use deq_runtime::controller::jit_controller::JitController;
use deq_runtime::coordinator::{CoordinatorClient, MockCoordinator};
use deq_runtime::jit::{self, jit_gadget_type};
use std::sync::Arc;
use tokio::time::{Duration, timeout};

/// Create a basic JIT library for testing with prepare_z and measure_z gadgets.
fn basic_jit_library() -> jit::JitLibrary {
    jit::JitLibrary {
        description: String::new(),
        port_types: vec![jit::JitPortType {
            base: Some(bin::PortType {
                ptype: 1,
                observables: vec![bin::port_type::Observable::default(); 2],
                ..Default::default()
            }),
            stabilizers: vec![jit::jit_port_type::Stabilizer::default(); 2],
            k: 1,
        }],
        gadget_types: vec![
            // Gadget type 1: prepare_z (no inputs, one output)
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 1,
                    name: "prepare_z".to_string(),
                    measurements: vec![gadget_type::Measurement::default(); 2],
                    outputs: vec![gadget_type::Port {
                        ptype: 1,
                        ..Default::default()
                    }],
                    correction_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 2,
                        cols: 1,
                        i: vec![],
                        j: vec![],
                    }),
                    readout_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 1,
                        i: vec![],
                        j: vec![],
                    }),
                    physical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 2,
                        cols: 2,
                        i: vec![],
                        j: vec![],
                    }),
                    ..Default::default()
                }),
                finished_checks: vec![
                    jit_gadget_type::Check {
                        base: Some(check_model_type::Check {
                            tag: "check_a".to_string(),
                            ..Default::default()
                        }),
                        measurements: vec![jit_gadget_type::PresentMeasurement {
                            measurement_index: 0,
                            ..Default::default()
                        }],
                    },
                    jit_gadget_type::Check {
                        base: Some(check_model_type::Check {
                            tag: "check_b".to_string(),
                            ..Default::default()
                        }),
                        measurements: vec![jit_gadget_type::PresentMeasurement {
                            measurement_index: 1,
                            ..Default::default()
                        }],
                    },
                ],
                unfinished_checks: vec![
                    jit_gadget_type::Check {
                        base: Some(check_model_type::Check::default()),
                        measurements: vec![jit_gadget_type::PresentMeasurement {
                            measurement_index: 0,
                            ..Default::default()
                        }],
                    },
                    jit_gadget_type::Check {
                        base: Some(check_model_type::Check::default()),
                        measurements: vec![jit_gadget_type::PresentMeasurement {
                            measurement_index: 1,
                            ..Default::default()
                        }],
                    },
                ],
                errors: vec![jit_gadget_type::Error {
                    base: Some(error_model_type::Error {
                        residual: vec![1],
                        readout_flips: vec![],
                        probability: 0.01,
                        ..Default::default()
                    }),
                    finished_checks: vec![1],
                    unfinished_checks: vec![1],
                }],
            },
            // Gadget type 2: measure_z (one input, no outputs)
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 2,
                    name: "measure_z".to_string(),
                    measurements: vec![gadget_type::Measurement::default(); 3],
                    inputs: vec![gadget_type::Port {
                        ptype: 1,
                        ..Default::default()
                    }],
                    correction_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 3,
                        i: vec![],
                        j: vec![],
                    }),
                    readout_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
                        cols: 3,
                        i: vec![0],
                        j: vec![0],
                    }),
                    logical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 1,
                        i: vec![],
                        j: vec![],
                    }),
                    readouts: vec![gadget_type::Readout {
                        measurement_indices: vec![0, 1, 2],
                        ..Default::default()
                    }],
                    physical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 3,
                        i: vec![],
                        j: vec![],
                    }),
                    ..Default::default()
                }),
                finished_checks: vec![
                    jit_gadget_type::Check {
                        base: Some(check_model_type::Check::default()),
                        measurements: vec![
                            jit_gadget_type::PresentMeasurement {
                                input_port: Some(0),
                                measurement_index: 0,
                            },
                            jit_gadget_type::PresentMeasurement {
                                measurement_index: 0,
                                ..Default::default()
                            },
                            jit_gadget_type::PresentMeasurement {
                                measurement_index: 1,
                                ..Default::default()
                            },
                        ],
                    },
                    jit_gadget_type::Check {
                        base: Some(check_model_type::Check::default()),
                        measurements: vec![
                            jit_gadget_type::PresentMeasurement {
                                input_port: Some(0),
                                measurement_index: 1,
                            },
                            jit_gadget_type::PresentMeasurement {
                                measurement_index: 1,
                                ..Default::default()
                            },
                            jit_gadget_type::PresentMeasurement {
                                measurement_index: 2,
                                ..Default::default()
                            },
                        ],
                    },
                ],
                ..Default::default()
            },
        ],
        program: vec![],
    }
}

/// Create a JIT instruction for instantiating a gadget.
fn make_jit_instruction(gtype: u64, gid: u64, connectors: Vec<bin::gadget::Connector>) -> jit::JitInstruction {
    jit::JitInstruction {
        gadget: Some(bin::Gadget {
            gtype,
            gid,
            connectors,
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// Helper to set up JitController with MockCoordinator.
async fn setup_controller(library: jit::JitLibrary, cache_enabled: bool) -> (Arc<JitController>, Arc<MockCoordinator>) {
    let mock = MockCoordinator::new();
    let client = CoordinatorClient::from_mock(mock.clone());
    let controller = JitController::new_from_library(library, cache_enabled);
    controller.start(client).await;
    (controller, mock)
}

#[tokio::test]
async fn test_basic_compilation_cache_disabled() {
    let library = basic_jit_library();
    let (controller, mock) = setup_controller(library, false).await;

    let instruction = make_jit_instruction(1, 1, vec![]);
    controller.execute(instruction).await;

    let state = mock.state.read().await;

    // All gadget types are loaded upfront in start()
    assert_eq!(state.gadget_types.len(), 2, "all gadget types should be loaded upfront");
    assert!(state.gadget_types.contains_key(&1));
    assert!(state.gadget_types.contains_key(&2));

    assert_eq!(state.gadgets.len(), 1, "should have 1 gadget instance");
    assert!(state.gadgets.contains_key(&1));

    assert_eq!(state.check_model_types.len(), 1, "should have 1 check model type");

    assert_eq!(state.check_models.len(), 1, "should have 1 check model instance");
    assert!(state.check_models.contains_key(&1));

    assert!(
        state.error_model_types.is_empty(),
        "the error model should not be created until output is connected"
    );
}

#[tokio::test]
async fn test_cache_enabled_reuses_type_id() {
    let library = basic_jit_library();
    let (controller, mock) = setup_controller(library, true).await;

    let instruction1 = make_jit_instruction(1, 1, vec![]);
    controller.execute(instruction1).await;

    let instruction2 = make_jit_instruction(1, 2, vec![]);
    controller.execute(instruction2).await;

    let state = mock.state.read().await;

    assert_eq!(state.gadgets.len(), 2, "should have 2 gadget instances");

    assert_eq!(
        state.check_model_types.len(),
        1,
        "cache enabled: should reuse same check model type"
    );

    assert_eq!(state.check_models.len(), 2, "should have 2 check model instances");

    let ctype_1 = state.check_models.get(&1).unwrap().ctype;
    let ctype_2 = state.check_models.get(&2).unwrap().ctype;
    assert_eq!(ctype_1, ctype_2, "both check models should use the same ctype");
}

#[tokio::test]
async fn test_cache_disabled_creates_new_types() {
    let library = basic_jit_library();
    let (controller, mock) = setup_controller(library, false).await;

    let instruction1 = make_jit_instruction(1, 1, vec![]);
    controller.execute(instruction1).await;

    let instruction2 = make_jit_instruction(1, 2, vec![]);
    controller.execute(instruction2).await;

    let state = mock.state.read().await;

    assert_eq!(state.gadgets.len(), 2, "should have 2 gadget instances");

    assert_eq!(
        state.check_model_types.len(),
        2,
        "cache disabled: should create new check model type for each"
    );

    let ctype_1 = state.check_models.get(&1).unwrap().ctype;
    let ctype_2 = state.check_models.get(&2).unwrap().ctype;
    assert_ne!(ctype_1, ctype_2, "each check model should use a different ctype");
}

#[tokio::test]
async fn test_modifier_patches_remote_gadgets() {
    let library = basic_jit_library();
    let (controller, mock) = setup_controller(library, false).await;

    let prepare_instruction = make_jit_instruction(1, 1, vec![]);
    controller.execute(prepare_instruction).await;

    let measure_instruction = make_jit_instruction(2, 2, vec![bin::gadget::Connector { gid: 1, port: 0 }]);
    controller.execute(measure_instruction).await;

    let state = mock.state.read().await;

    assert_eq!(state.gadgets.len(), 2);
    assert_eq!(state.check_models.len(), 2);

    let measure_check_model = state.check_models.get(&2).expect("measure_z check model should exist");

    let modifier = measure_check_model.modifier.as_ref();
    assert!(
        modifier.is_some(),
        "measure_z check model should have a modifier for remote gadget rerouting"
    );

    let reroutes = &modifier.unwrap().reroute_remote_gadgets;
    assert!(!reroutes.is_empty(), "modifier should contain remote gadget reroutes");

    let first_reroute = &reroutes[0];
    assert!(first_reroute.value.is_some());
    let remote_gadget = first_reroute.value.as_ref().unwrap();
    assert_eq!(
        remote_gadget.absolute_gid,
        Some(1),
        "remote gadget should be rerouted to prepare_z (gid=1)"
    );
}

#[tokio::test]
async fn test_error_model_timing_after_output_connection() {
    let library = basic_jit_library();
    let (controller, mock) = setup_controller(library, false).await;

    let prepare_instruction = make_jit_instruction(1, 1, vec![]);
    controller.execute(prepare_instruction).await;

    {
        let state = mock.state.read().await;
        let error_model_count_before = state.error_models.len();
        assert_eq!(
            error_model_count_before, 0,
            "error model should not be created before output is connected"
        );
    }

    let measure_instruction = make_jit_instruction(2, 2, vec![bin::gadget::Connector { gid: 1, port: 0 }]);
    controller.execute(measure_instruction).await;

    let error_model_created = timeout(Duration::from_millis(200), async {
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let state = mock.state.read().await;
            if state.error_models.len() > 1 {
                return true;
            }
        }
    })
    .await;

    assert!(
        error_model_created.is_ok(),
        "error model should be created after output is connected"
    );

    let state = mock.state.read().await;
    assert!(state.error_models.len() == 2, "both error models should be created");
}

#[tokio::test]
async fn test_effective_types_expand_modifiers() {
    let library = basic_jit_library();
    let (controller, mock) = setup_controller(library, false).await;

    let prepare_instruction = make_jit_instruction(1, 1, vec![]);
    controller.execute(prepare_instruction).await;

    let measure_instruction = make_jit_instruction(2, 2, vec![bin::gadget::Connector { gid: 1, port: 0 }]);
    controller.execute(measure_instruction).await;

    // Wait for error models to be created (they're spawned asynchronously)
    timeout(Duration::from_millis(100), async {
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let state = mock.state.read().await;
            if state.error_models.len() >= 2 {
                break;
            }
        }
    })
    .await
    .expect("error models should be created");

    let effective = mock.get_effective_types().await;

    assert_eq!(
        effective.check_model_types.len(),
        2,
        "should have effective types for both check models"
    );

    let measure_effective_cmt = effective.check_model_types.get(&2).expect("measure_z effective type");

    assert!(
        !measure_effective_cmt.remote_gadgets.is_empty(),
        "effective type should have resolved remote gadgets"
    );
    assert_eq!(
        measure_effective_cmt.remote_gadgets[0], 1,
        "remote gadget should resolve to gid=1 (prepare_z)"
    );
}

#[tokio::test]
async fn test_multiple_gadget_types_loaded() {
    let library = basic_jit_library();
    let (controller, mock) = setup_controller(library, false).await;

    let prepare_instruction = make_jit_instruction(1, 1, vec![]);
    controller.execute(prepare_instruction).await;

    let measure_instruction = make_jit_instruction(2, 2, vec![bin::gadget::Connector { gid: 1, port: 0 }]);
    controller.execute(measure_instruction).await;

    let state = mock.state.read().await;

    assert_eq!(state.gadget_types.len(), 2, "both gadget types should be loaded");
    assert!(state.gadget_types.contains_key(&1));
    assert!(state.gadget_types.contains_key(&2));
}

#[tokio::test]
async fn test_gadget_type_loaded_only_once() {
    let library = basic_jit_library();
    let (controller, mock) = setup_controller(library, false).await;

    let instr1 = make_jit_instruction(1, 1, vec![]);
    controller.execute(instr1).await;

    let instr2 = make_jit_instruction(1, 2, vec![]);
    controller.execute(instr2).await;

    let instr3 = make_jit_instruction(1, 3, vec![]);
    controller.execute(instr3).await;

    let state = mock.state.read().await;

    // All gadget types are loaded upfront in start(), so we expect 2 types
    // (all types from the library) regardless of how many instances are created
    assert_eq!(
        state.gadget_types.len(),
        2,
        "all gadget types should be loaded upfront from library"
    );
    assert_eq!(state.gadgets.len(), 3, "all 3 gadget instances should exist");
}

// ============================================================================
// Timing tests: verify error models are sent at correct timing
// ============================================================================

/// Test that error models are blocked until all output ports are connected.
/// This mirrors the timing test in jit_compiler_tests.rs but tests the full
/// JIT controller → MockCoordinator pipeline.
#[tokio::test]
async fn test_error_model_timing_blocked_until_output_connected() {
    let library = basic_jit_library();
    let (controller, mock) = setup_controller(library, false).await;

    // Execute prepare_z (gid 1) - has output port that needs connection
    controller.execute(make_jit_instruction(1, 1, vec![])).await;

    {
        let state = mock.state.read().await;
        assert_eq!(state.gadgets.len(), 1, "gadget should be created");
        assert_eq!(state.check_models.len(), 1, "check model should be created");
        assert_eq!(
            state.error_models.len(),
            0,
            "error model should NOT be created while output is unconnected"
        );
    }

    // Execute measure_z (gid 2) connected to prepare_z output
    controller
        .execute(make_jit_instruction(2, 2, vec![bin::gadget::Connector { gid: 1, port: 0 }]))
        .await;

    // Wait for error models to be created
    let error_models_created = timeout(Duration::from_millis(200), async {
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let state = mock.state.read().await;
            if state.error_models.len() >= 2 {
                return true;
            }
        }
    })
    .await;

    assert!(
        error_models_created.is_ok(),
        "both error models should be created after output is connected"
    );

    let state = mock.state.read().await;
    assert_eq!(state.error_models.len(), 2, "prepare and measure error models should exist");
}

/// Test timing with multiple gadgets in a chain: prepare → idle → measure.
/// When idle (which has syndrome extraction) is connected to prepare_z, the
/// prepare_z error model can resolve. The idle error model stays blocked until
/// its output is connected to measure.
#[tokio::test]
async fn test_error_model_timing_chain() {
    let library = extended_jit_library_with_idle();
    let (controller, mock) = setup_controller(library, false).await;

    // Execute prepare_z (gid 1)
    controller.execute(make_jit_instruction(1, 1, vec![])).await;

    {
        let state = mock.state.read().await;
        assert_eq!(state.gadgets.len(), 1);
        assert_eq!(
            state.error_models.len(),
            0,
            "prepare error model should be blocked (no output connection)"
        );
    }

    // Execute idle (gid 2) connected to prepare_z
    // Since idle has syndrome extraction (finished_checks), prepare_z error model can resolve
    controller
        .execute(make_jit_instruction(3, 2, vec![bin::gadget::Connector { gid: 1, port: 0 }]))
        .await;

    // Wait for prepare_z error model to be created
    let prepare_resolved = timeout(Duration::from_millis(100), async {
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let state = mock.state.read().await;
            if !state.error_models.is_empty() {
                return true;
            }
        }
    })
    .await;

    assert!(
        prepare_resolved.is_ok(),
        "prepare error model should resolve when idle (with syndrome extraction) is connected"
    );

    {
        let state = mock.state.read().await;
        assert_eq!(state.gadgets.len(), 2);
        // Only prepare_z error model should be created; idle is still blocked
        assert_eq!(
            state.error_models.len(),
            1,
            "only prepare error model should be created; idle is blocked"
        );
    }

    // Execute measure_z (gid 3) connected to idle output
    controller
        .execute(make_jit_instruction(2, 3, vec![bin::gadget::Connector { gid: 2, port: 0 }]))
        .await;

    let error_models_created = timeout(Duration::from_millis(200), async {
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let state = mock.state.read().await;
            if state.error_models.len() >= 3 {
                return true;
            }
        }
    })
    .await;

    assert!(
        error_models_created.is_ok(),
        "all error models should be created after full chain is connected"
    );
}

/// Test that measurement gadgets (no output ports) have error models created immediately.
#[tokio::test]
async fn test_error_model_immediate_for_no_output_gadget() {
    let library = basic_jit_library();
    let (controller, mock) = setup_controller(library, false).await;

    // Execute prepare_z first
    controller.execute(make_jit_instruction(1, 1, vec![])).await;

    // Execute measure_z connected to prepare_z
    controller
        .execute(make_jit_instruction(2, 2, vec![bin::gadget::Connector { gid: 1, port: 0 }]))
        .await;

    // The measure_z error model should be created very quickly since it has no outputs
    let measure_error_created = timeout(Duration::from_millis(100), async {
        loop {
            tokio::time::sleep(Duration::from_millis(5)).await;
            let state = mock.state.read().await;
            // Check if measure's error model (eid 2) exists
            if state.error_models.len() >= 2 {
                return true;
            }
        }
    })
    .await;

    assert!(
        measure_error_created.is_ok(),
        "measurement error model should be created quickly (no output blocking)"
    );
}

// ============================================================================
// Correctness tests: verify JIT controller output matches static_jit_compile
// ============================================================================

use common::effective_type_comparison::assert_effective_types_equivalent;
use deq_runtime::jit::static_jit_compile;

/// Test that JIT controller produces equivalent output to static_jit_compile
/// for a simple prepare → measure circuit.
#[tokio::test]
async fn test_correctness_simple_prepare_measure() {
    let mut library = basic_jit_library();

    // Set up program for static compilation
    library.program = vec![
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 1,
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 2,
                connectors: vec![bin::gadget::Connector { gid: 1, port: 0 }],
                ..Default::default()
            }),
            ..Default::default()
        },
    ];

    // Get expected output from static compilation
    let expected = static_jit_compile(library.clone()).await;

    // Clear program and run through JIT controller
    library.program.clear();
    let (controller, mock) = setup_controller(library, false).await;

    controller.execute(make_jit_instruction(1, 1, vec![])).await;
    controller
        .execute(make_jit_instruction(2, 2, vec![bin::gadget::Connector { gid: 1, port: 0 }]))
        .await;

    // Wait for all error models to be created
    timeout(Duration::from_millis(200), async {
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let state = mock.state.read().await;
            if state.error_models.len() >= 2 {
                break;
            }
        }
    })
    .await
    .expect("error models should be created");

    // Compare effective types against expected
    assert_effective_types_equivalent(&mock, &expected).await;
}

/// Test correctness with cache enabled - should still produce equivalent results.
#[tokio::test]
async fn test_correctness_with_cache_enabled() {
    let mut library = basic_jit_library();

    // Two prepare → measure pairs
    library.program = vec![
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 1,
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 2,
                connectors: vec![bin::gadget::Connector { gid: 1, port: 0 }],
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 1,
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 2,
                connectors: vec![bin::gadget::Connector { gid: 3, port: 0 }],
                ..Default::default()
            }),
            ..Default::default()
        },
    ];

    let expected = static_jit_compile(library.clone()).await;

    library.program.clear();
    let (controller, mock) = setup_controller(library, true).await;

    controller.execute(make_jit_instruction(1, 1, vec![])).await;
    controller
        .execute(make_jit_instruction(2, 2, vec![bin::gadget::Connector { gid: 1, port: 0 }]))
        .await;
    controller.execute(make_jit_instruction(1, 3, vec![])).await;
    controller
        .execute(make_jit_instruction(2, 4, vec![bin::gadget::Connector { gid: 3, port: 0 }]))
        .await;

    timeout(Duration::from_millis(200), async {
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let state = mock.state.read().await;
            if state.error_models.len() >= 4 {
                break;
            }
        }
    })
    .await
    .expect("all error models should be created");

    assert_effective_types_equivalent(&mock, &expected).await;
}

/// Test correctness for prepare → idle → measure chain.
#[tokio::test]
async fn test_correctness_with_idle_chain() {
    let mut library = extended_jit_library_with_idle();

    library.program = vec![
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 1,
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 3,
                connectors: vec![bin::gadget::Connector { gid: 1, port: 0 }],
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 2,
                connectors: vec![bin::gadget::Connector { gid: 2, port: 0 }],
                ..Default::default()
            }),
            ..Default::default()
        },
    ];

    let expected = static_jit_compile(library.clone()).await;

    library.program.clear();
    let (controller, mock) = setup_controller(library, false).await;

    controller.execute(make_jit_instruction(1, 1, vec![])).await;
    controller
        .execute(make_jit_instruction(3, 2, vec![bin::gadget::Connector { gid: 1, port: 0 }]))
        .await;
    controller
        .execute(make_jit_instruction(2, 3, vec![bin::gadget::Connector { gid: 2, port: 0 }]))
        .await;

    timeout(Duration::from_millis(200), async {
        loop {
            tokio::time::sleep(Duration::from_millis(10)).await;
            let state = mock.state.read().await;
            if state.error_models.len() >= 3 {
                break;
            }
        }
    })
    .await
    .expect("all error models should be created");

    assert_effective_types_equivalent(&mock, &expected).await;
}

// ============================================================================
// Helper: Extended JIT library with idle gadget
// ============================================================================

/// Create a JIT library with prepare_z, measure_z, and idle gadgets.
fn extended_jit_library_with_idle() -> jit::JitLibrary {
    let mut library = basic_jit_library();

    // Add idle gadget type (gtype 3): one input, one output, has syndrome extraction
    library.gadget_types.push(jit::JitGadgetType {
        base: Some(bin::GadgetType {
            gtype: 3,
            name: "idle".to_string(),
            measurements: vec![gadget_type::Measurement::default(); 2],
            inputs: vec![gadget_type::Port {
                ptype: 1,
                ..Default::default()
            }],
            outputs: vec![gadget_type::Port {
                ptype: 1,
                ..Default::default()
            }],
            correction_propagation: Some(deq_runtime::util::BitMatrix {
                rows: 2,
                cols: 2,
                i: vec![0, 1],
                j: vec![0, 1],
            }),
            readout_propagation: Some(deq_runtime::util::BitMatrix {
                rows: 0,
                cols: 2,
                i: vec![],
                j: vec![],
            }),
            logical_correction: Some(deq_runtime::util::BitMatrix {
                rows: 2,
                cols: 0,
                i: vec![],
                j: vec![],
            }),
            physical_correction: Some(deq_runtime::util::BitMatrix {
                rows: 2,
                cols: 2,
                i: vec![],
                j: vec![],
            }),
            ..Default::default()
        }),
        finished_checks: vec![
            jit_gadget_type::Check {
                base: Some(check_model_type::Check::default()),
                measurements: vec![
                    jit_gadget_type::PresentMeasurement {
                        input_port: Some(0),
                        measurement_index: 0,
                    },
                    jit_gadget_type::PresentMeasurement {
                        measurement_index: 0,
                        ..Default::default()
                    },
                ],
            },
            jit_gadget_type::Check {
                base: Some(check_model_type::Check::default()),
                measurements: vec![
                    jit_gadget_type::PresentMeasurement {
                        input_port: Some(0),
                        measurement_index: 1,
                    },
                    jit_gadget_type::PresentMeasurement {
                        measurement_index: 1,
                        ..Default::default()
                    },
                ],
            },
        ],
        unfinished_checks: vec![
            jit_gadget_type::Check {
                base: Some(check_model_type::Check::default()),
                measurements: vec![jit_gadget_type::PresentMeasurement {
                    measurement_index: 0,
                    ..Default::default()
                }],
            },
            jit_gadget_type::Check {
                base: Some(check_model_type::Check::default()),
                measurements: vec![jit_gadget_type::PresentMeasurement {
                    measurement_index: 1,
                    ..Default::default()
                }],
            },
        ],
        errors: vec![jit_gadget_type::Error {
            base: Some(error_model_type::Error {
                tag: "idle error".to_string(),
                residual: vec![0],
                probability: 0.001,
                ..Default::default()
            }),
            finished_checks: vec![0],
            unfinished_checks: vec![0],
        }],
    });

    library
}
