use deq_runtime::bin::{self, instruction::Create};
use deq_runtime::jit::{self, static_jit_compile};
use std::collections::HashSet;
use tokio_util::sync::CancellationToken;

#[tokio::test]
async fn test_empty_jit_compile() {
    let jit_library = jit::JitLibrary::default();
    let library = static_jit_compile(jit_library).await;
    assert_eq!(library, bin::Library::default());
}

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
            // Gadget type 1: prepare_z
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 1,
                    name: "prepare_z".to_string(),
                    measurements: vec![bin::gadget_type::Measurement::default(); 2],
                    outputs: vec![bin::gadget_type::Port {
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
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 0,
                            ..Default::default()
                        }],
                    },
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 1,
                            ..Default::default()
                        }],
                    },
                ],
                unfinished_checks: vec![
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 0,
                            ..Default::default()
                        }],
                    },
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 1,
                            ..Default::default()
                        }],
                    },
                ],
                errors: vec![jit::jit_gadget_type::Error {
                    base: Some(bin::error_model_type::Error {
                        residual: vec![1],
                        readout_flips: vec![],
                        probability: 0.01,
                        ..Default::default()
                    }),
                    finished_checks: vec![1],
                    unfinished_checks: vec![1],
                }],
            },
            // Gadget type 2: measure_z
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 2,
                    name: "measure_z".to_string(),
                    measurements: vec![bin::gadget_type::Measurement::default(); 3],
                    inputs: vec![bin::gadget_type::Port {
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
                    readouts: vec![bin::gadget_type::Readout {
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
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![
                            jit::jit_gadget_type::PresentMeasurement {
                                input_port: Some(0),
                                measurement_index: 0,
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                measurement_index: 0,
                                ..Default::default()
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                measurement_index: 1,
                                ..Default::default()
                            },
                        ],
                    },
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![
                            jit::jit_gadget_type::PresentMeasurement {
                                input_port: Some(0),
                                measurement_index: 1,
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                measurement_index: 1,
                                ..Default::default()
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                measurement_index: 2,
                                ..Default::default()
                            },
                        ],
                    },
                ],
                ..Default::default()
            },
            // Gadget type 3: idle
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 3,
                    name: "idle".to_string(),
                    measurements: vec![bin::gadget_type::Measurement::default(); 2],
                    inputs: vec![bin::gadget_type::Port {
                        ptype: 1,
                        ..Default::default()
                    }],
                    outputs: vec![bin::gadget_type::Port {
                        ptype: 1,
                        ..Default::default()
                    }],
                    correction_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 2,
                        cols: 3,
                        i: vec![0, 1],
                        j: vec![0, 1],
                    }),
                    readout_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 3,
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
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![
                            jit::jit_gadget_type::PresentMeasurement {
                                input_port: Some(0),
                                measurement_index: 0,
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                measurement_index: 0,
                                ..Default::default()
                            },
                        ],
                    },
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![
                            jit::jit_gadget_type::PresentMeasurement {
                                input_port: Some(0),
                                measurement_index: 1,
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                measurement_index: 1,
                                ..Default::default()
                            },
                        ],
                    },
                ],
                unfinished_checks: vec![
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 0,
                            ..Default::default()
                        }],
                    },
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 1,
                            ..Default::default()
                        }],
                    },
                ],
                ..Default::default()
            },
        ],
        program: vec![],
    }
}

#[tokio::test]
async fn test_basic_jit_compile() {
    // cargo test test_basic_jit_compile -- --nocapture
    let mut jit_library = basic_jit_library();
    jit_library.program = vec![
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 1,
                gid: 1,
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 2,
                gid: 2,
                connectors: vec![bin::gadget::Connector { gid: 1, port: 0 }],
                ..Default::default()
            }),
            ..Default::default()
        },
    ];

    let library = static_jit_compile(jit_library.clone()).await;

    assert_eq!(
        library.gadget_types[0],
        jit_library.gadget_types[0].base.as_ref().unwrap().clone()
    );
    assert_eq!(
        library.gadget_types[1],
        jit_library.gadget_types[1].base.as_ref().unwrap().clone()
    );

    assert_eq!(
        library.check_model_types[0],
        bin::CheckModelType {
            ctype: 1,
            gtype: 1,
            checks: vec![
                bin::check_model_type::Check {
                    measurements: vec![bin::check_model_type::RemoteMeasurement {
                        measurement_index: 0,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
                bin::check_model_type::Check {
                    measurements: vec![bin::check_model_type::RemoteMeasurement {
                        measurement_index: 1,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
    );

    // Check model type 1 - measurements may be in different order due to HashSet iteration
    let check_model_type_1 = &library.check_model_types[1];
    assert_eq!(check_model_type_1.ctype, 2);
    assert_eq!(check_model_type_1.gtype, 2);
    assert_eq!(check_model_type_1.remote_gadgets.len(), 1);
    assert_eq!(check_model_type_1.remote_gadgets[0].absolute_gid, Some(1));
    assert_eq!(check_model_type_1.remote_gadgets[0].expecting_gtype, 1);
    assert_eq!(check_model_type_1.checks.len(), 2);
    // Check 0: should have 3 measurements - one from remote gadget (index 0), two local (indices 0, 1)
    let check0_measurements: HashSet<_> = check_model_type_1.checks[0]
        .measurements
        .iter()
        .map(|m| (m.remote_gadget, m.measurement_index))
        .collect();
    assert!(check0_measurements.contains(&(Some(0), 0)));
    assert!(check0_measurements.contains(&(None, 0)));
    assert!(check0_measurements.contains(&(None, 1)));
    // Check 1: should have 3 measurements - one from remote gadget (index 1), two local (indices 1, 2)
    let check1_measurements: HashSet<_> = check_model_type_1.checks[1]
        .measurements
        .iter()
        .map(|m| (m.remote_gadget, m.measurement_index))
        .collect();
    assert!(check1_measurements.contains(&(Some(0), 1)));
    assert!(check1_measurements.contains(&(None, 1)));
    assert!(check1_measurements.contains(&(None, 2)));

    // Check error model type 0 - the order of checks may vary due to HashSet iteration
    let error_model_type_0 = &library.error_model_types[0];
    assert_eq!(error_model_type_0.etype, 1);
    assert_eq!(error_model_type_0.ctype, 0); // WILDCARD: error model type uses wildcard ctype
    assert_eq!(error_model_type_0.remote_check_models.len(), 1);
    assert_eq!(error_model_type_0.remote_check_models[0].expecting_ctype, 2);
    assert_eq!(error_model_type_0.remote_check_models[0].absolute_cid, Some(2));
    assert_eq!(error_model_type_0.remote_check_models[0].check_bias, 1);
    assert_eq!(error_model_type_0.errors.len(), 1);
    assert_eq!(error_model_type_0.errors[0].residual, vec![1]);
    assert!((error_model_type_0.errors[0].probability - 0.01).abs() < 1e-9);
    assert_eq!(error_model_type_0.errors[0].checks.len(), 2);
    // The order of checks is non-deterministic, so check both are present
    let checks = &error_model_type_0.errors[0].checks;
    let has_local = checks.iter().any(|c| c.remote_check_model.is_none() && c.check_index == 1);
    let has_remote = checks.iter().any(|c| c.remote_check_model == Some(0) && c.check_index == 0);
    assert!(has_local, "should have local check with index 1");
    assert!(has_remote, "should have remote check with index 0");

    assert_eq!(
        library.error_model_types[1],
        bin::ErrorModelType {
            etype: 2,
            ctype: 0, // WILDCARD: error model type uses wildcard ctype
            ..Default::default()
        }
    );

    assert_eq!(
        library.program[0],
        bin::Instruction {
            create: Some(Create::Gadget(bin::Gadget {
                gtype: 1,
                gid: 1,
                ..Default::default()
            }))
        }
    );
    assert_eq!(
        library.program[1],
        bin::Instruction {
            create: Some(Create::CheckModel(bin::CheckModel {
                gid: 1,
                ctype: 1,
                cid: 1,
                ..Default::default()
            }))
        }
    );
    assert_eq!(
        library.program[2],
        bin::Instruction {
            create: Some(Create::Gadget(bin::Gadget {
                gtype: 2,
                gid: 2,
                connectors: vec![bin::gadget::Connector { gid: 1, port: 0 }],
                ..Default::default()
            }))
        }
    );
    assert_eq!(
        library.program[3],
        bin::Instruction {
            create: Some(Create::CheckModel(bin::CheckModel {
                gid: 2,
                ctype: 2,
                cid: 2,
                ..Default::default()
            }))
        }
    );
    assert_eq!(
        library.program[4],
        bin::Instruction {
            create: Some(Create::ErrorModel(bin::ErrorModel {
                cid: 1,
                etype: 1,
                eid: 1,
                ..Default::default()
            }))
        }
    );
    assert_eq!(
        library.program[5],
        bin::Instruction {
            create: Some(Create::ErrorModel(bin::ErrorModel {
                cid: 2,
                etype: 2,
                eid: 2,
                ..Default::default()
            }))
        }
    );
}

fn check_propagation_jit_library() -> jit::JitLibrary {
    jit::JitLibrary {
        description: String::new(),
        port_types: vec![jit::JitPortType {
            base: Some(bin::PortType {
                ptype: 1,
                observables: vec![bin::port_type::Observable::default(); 1],
                ..Default::default()
            }),
            stabilizers: vec![jit::jit_port_type::Stabilizer::default(); 1],
            k: 1,
        }],
        gadget_types: vec![
            // Gadget type 1: prepare
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 1,
                    name: "prepare".to_string(),
                    measurements: vec![bin::gadget_type::Measurement::default(); 1],
                    outputs: vec![bin::gadget_type::Port {
                        ptype: 1,
                        ..Default::default()
                    }],
                    correction_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
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
                    logical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 0,
                        i: vec![],
                        j: vec![],
                    }),
                    physical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
                        cols: 1,
                        i: vec![],
                        j: vec![],
                    }),
                    ..Default::default()
                }),
                finished_checks: vec![jit::jit_gadget_type::Check {
                    base: Some(bin::check_model_type::Check::default()),
                    measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                        measurement_index: 0,
                        ..Default::default()
                    }],
                }],
                unfinished_checks: vec![jit::jit_gadget_type::Check {
                    base: Some(bin::check_model_type::Check::default()),
                    measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                        measurement_index: 0,
                        ..Default::default()
                    }],
                }],
                errors: vec![jit::jit_gadget_type::Error {
                    base: Some(bin::error_model_type::Error {
                        residual: vec![0],
                        readout_flips: vec![],
                        probability: 0.01,
                        ..Default::default()
                    }),
                    finished_checks: vec![0],
                    unfinished_checks: vec![0],
                }],
            },
            // Gadget type 2: merger (no measurement)
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 2,
                    name: "merger".to_string(),
                    inputs: vec![
                        bin::gadget_type::Port {
                            ptype: 1,
                            ..Default::default()
                        },
                        bin::gadget_type::Port {
                            ptype: 1,
                            ..Default::default()
                        },
                    ],
                    outputs: vec![bin::gadget_type::Port {
                        ptype: 1,
                        ..Default::default()
                    }],
                    correction_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
                        cols: 3,
                        i: vec![],
                        j: vec![],
                    }),
                    readout_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 3,
                        i: vec![],
                        j: vec![],
                    }),
                    logical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
                        cols: 0,
                        i: vec![],
                        j: vec![],
                    }),
                    physical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
                        cols: 0,
                        i: vec![],
                        j: vec![],
                    }),
                    ..Default::default()
                }),
                unfinished_checks: vec![jit::jit_gadget_type::Check {
                    base: Some(bin::check_model_type::Check::default()),
                    measurements: vec![
                        jit::jit_gadget_type::PresentMeasurement {
                            input_port: Some(0),
                            measurement_index: 0,
                        },
                        jit::jit_gadget_type::PresentMeasurement {
                            input_port: Some(1),
                            measurement_index: 0,
                        },
                    ],
                }],
                ..Default::default()
            },
            // Gadget type 3: measure
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 3,
                    name: "measure".to_string(),
                    measurements: vec![bin::gadget_type::Measurement::default(); 1],
                    inputs: vec![bin::gadget_type::Port {
                        ptype: 1,
                        ..Default::default()
                    }],
                    correction_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 2,
                        i: vec![],
                        j: vec![],
                    }),
                    readout_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
                        cols: 2,
                        i: vec![],
                        j: vec![],
                    }),
                    logical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 1,
                        i: vec![],
                        j: vec![],
                    }),
                    readouts: vec![bin::gadget_type::Readout {
                        measurement_indices: vec![0],
                        ..Default::default()
                    }],
                    physical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 1,
                        i: vec![],
                        j: vec![],
                    }),
                    ..Default::default()
                }),
                finished_checks: vec![jit::jit_gadget_type::Check {
                    base: Some(bin::check_model_type::Check::default()),
                    measurements: vec![
                        jit::jit_gadget_type::PresentMeasurement {
                            input_port: Some(0),
                            measurement_index: 0,
                        },
                        jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 0,
                            ..Default::default()
                        },
                    ],
                }],
                ..Default::default()
            },
        ],
        program: vec![],
    }
}

#[tokio::test]
async fn test_check_propagation_compile_two() {
    let mut jit_library = check_propagation_jit_library();
    jit_library.program = vec![
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 1,
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
                connectors: vec![
                    bin::gadget::Connector { gid: 1, port: 0 },
                    bin::gadget::Connector { gid: 2, port: 0 },
                ],
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 3,
                connectors: vec![bin::gadget::Connector { gid: 3, port: 0 }],
                ..Default::default()
            }),
            ..Default::default()
        },
    ];

    let library = static_jit_compile(jit_library).await;

    assert_eq!(
        library.check_model_types[0],
        bin::CheckModelType {
            ctype: 1,
            gtype: 1,
            checks: vec![bin::check_model_type::Check {
                measurements: vec![bin::check_model_type::RemoteMeasurement {
                    measurement_index: 0,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }
    );

    assert_eq!(
        library.check_model_types[1],
        bin::CheckModelType {
            ctype: 2,
            gtype: 1,
            checks: vec![bin::check_model_type::Check {
                measurements: vec![bin::check_model_type::RemoteMeasurement {
                    measurement_index: 0,
                    ..Default::default()
                }],
                ..Default::default()
            }],
            ..Default::default()
        }
    );

    assert_eq!(library.check_model_types[2].checks.len(), 0);

    let check_model_type_3 = &library.check_model_types[3];
    assert_eq!(check_model_type_3.ctype, 4);
    assert_eq!(check_model_type_3.gtype, 3);
    assert_eq!(check_model_type_3.checks.len(), 1);
    assert_eq!(check_model_type_3.checks[0].measurements.len(), 3);
    let expected_remote_gadgets: HashSet<_> = [
        bin::check_model_type::RemoteGadget {
            absolute_gid: Some(1),
            expecting_gtype: 1,
            ..Default::default()
        },
        bin::check_model_type::RemoteGadget {
            absolute_gid: Some(2),
            expecting_gtype: 1,
            ..Default::default()
        },
    ]
    .into_iter()
    .collect();
    let actual_remote_gadgets: HashSet<_> = check_model_type_3.remote_gadgets.iter().cloned().collect();
    assert_eq!(actual_remote_gadgets, expected_remote_gadgets);
    let expected_measurements: HashSet<_> = check_model_type_3.checks[0]
        .measurements
        .iter()
        .map(|measurement| {
            let source_gid = measurement
                .remote_gadget
                .map(|index| check_model_type_3.remote_gadgets[index as usize].absolute_gid.unwrap());
            (source_gid, measurement.measurement_index)
        })
        .collect();
    let expected_measurement_set: HashSet<_> = [(Some(1), 0), (Some(2), 0), (None, 0)].into_iter().collect();
    assert_eq!(expected_measurements, expected_measurement_set);
}

#[tokio::test]
async fn test_check_propagation_compile_three() {
    let mut jit_library = check_propagation_jit_library();
    jit_library.program = vec![
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 1,
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
                gtype: 1,
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 2,
                connectors: vec![
                    bin::gadget::Connector { gid: 2, port: 0 },
                    bin::gadget::Connector { gid: 3, port: 0 },
                ],
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 2,
                connectors: vec![
                    bin::gadget::Connector { gid: 4, port: 0 },
                    bin::gadget::Connector { gid: 1, port: 0 },
                ],
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 3,
                connectors: vec![bin::gadget::Connector { gid: 5, port: 0 }],
                ..Default::default()
            }),
            ..Default::default()
        },
    ];

    let library = static_jit_compile(jit_library).await;

    assert_eq!(library.check_model_types[0].checks.len(), 1);
    assert_eq!(library.check_model_types[1].checks.len(), 1);
    assert_eq!(library.check_model_types[2].checks.len(), 1);
    assert_eq!(library.check_model_types[3].checks.len(), 0);
    assert_eq!(library.check_model_types[4].checks.len(), 0);
    assert_eq!(library.check_model_types[5].checks.len(), 1);

    let check_model_type_5 = &library.check_model_types[5];
    assert_eq!(check_model_type_5.ctype, 6);
    assert_eq!(check_model_type_5.gtype, 3);
    assert_eq!(check_model_type_5.checks.len(), 1);
    assert_eq!(check_model_type_5.checks[0].measurements.len(), 4);
    let expected_remote_gadgets: HashSet<_> = [
        bin::check_model_type::RemoteGadget {
            absolute_gid: Some(1),
            expecting_gtype: 1,
            ..Default::default()
        },
        bin::check_model_type::RemoteGadget {
            absolute_gid: Some(2),
            expecting_gtype: 1,
            ..Default::default()
        },
        bin::check_model_type::RemoteGadget {
            absolute_gid: Some(3),
            expecting_gtype: 1,
            ..Default::default()
        },
    ]
    .into_iter()
    .collect();
    let actual_remote_gadgets: HashSet<_> = check_model_type_5.remote_gadgets.iter().cloned().collect();
    assert_eq!(actual_remote_gadgets, expected_remote_gadgets);
    let expected_measurements: HashSet<_> = check_model_type_5.checks[0]
        .measurements
        .iter()
        .map(|measurement| {
            let source_gid = measurement
                .remote_gadget
                .map(|index| check_model_type_5.remote_gadgets[index as usize].absolute_gid.unwrap());
            (source_gid, measurement.measurement_index)
        })
        .collect();
    let expected_measurement_set: HashSet<_> = [(Some(1), 0), (Some(2), 0), (Some(3), 0), (None, 0)].into_iter().collect();
    assert_eq!(expected_measurements, expected_measurement_set);
}

#[tokio::test]
async fn test_check_propagation_compile_four() {
    let mut jit_library = check_propagation_jit_library();
    jit_library.program = vec![
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 1,
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
                gtype: 1,
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
                connectors: vec![
                    bin::gadget::Connector { gid: 1, port: 0 },
                    bin::gadget::Connector { gid: 2, port: 0 },
                ],
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 2,
                connectors: vec![
                    bin::gadget::Connector { gid: 3, port: 0 },
                    bin::gadget::Connector { gid: 4, port: 0 },
                ],
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 2,
                connectors: vec![
                    bin::gadget::Connector { gid: 5, port: 0 },
                    bin::gadget::Connector { gid: 6, port: 0 },
                ],
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 3,
                connectors: vec![bin::gadget::Connector { gid: 7, port: 0 }],
                ..Default::default()
            }),
            ..Default::default()
        },
    ];

    let library = static_jit_compile(jit_library).await;

    assert_eq!(library.check_model_types[0].checks.len(), 1);
    assert_eq!(library.check_model_types[1].checks.len(), 1);
    assert_eq!(library.check_model_types[2].checks.len(), 1);
    assert_eq!(library.check_model_types[3].checks.len(), 1);
    assert_eq!(library.check_model_types[4].checks.len(), 0);
    assert_eq!(library.check_model_types[5].checks.len(), 0);
    assert_eq!(library.check_model_types[6].checks.len(), 0);
    assert_eq!(library.check_model_types[7].checks.len(), 1);

    let check_model_type_7 = &library.check_model_types[7];
    assert_eq!(check_model_type_7.ctype, 8);
    assert_eq!(check_model_type_7.gtype, 3);
    assert_eq!(check_model_type_7.checks.len(), 1);
    assert_eq!(check_model_type_7.checks[0].measurements.len(), 5);
    let expected_remote_gadgets: HashSet<_> = [
        bin::check_model_type::RemoteGadget {
            absolute_gid: Some(1),
            expecting_gtype: 1,
            ..Default::default()
        },
        bin::check_model_type::RemoteGadget {
            absolute_gid: Some(2),
            expecting_gtype: 1,
            ..Default::default()
        },
        bin::check_model_type::RemoteGadget {
            absolute_gid: Some(3),
            expecting_gtype: 1,
            ..Default::default()
        },
        bin::check_model_type::RemoteGadget {
            absolute_gid: Some(4),
            expecting_gtype: 1,
            ..Default::default()
        },
    ]
    .into_iter()
    .collect();
    let actual_remote_gadgets: HashSet<_> = check_model_type_7.remote_gadgets.iter().cloned().collect();
    assert_eq!(actual_remote_gadgets, expected_remote_gadgets);
    let expected_measurements: HashSet<_> = check_model_type_7.checks[0]
        .measurements
        .iter()
        .map(|measurement| {
            let source_gid = measurement
                .remote_gadget
                .map(|index| check_model_type_7.remote_gadgets[index as usize].absolute_gid.unwrap());
            (source_gid, measurement.measurement_index)
        })
        .collect();
    let expected_measurement_set: HashSet<_> = [(Some(1), 0), (Some(2), 0), (Some(3), 0), (Some(4), 0), (None, 0)]
        .into_iter()
        .collect();
    assert_eq!(expected_measurements, expected_measurement_set);
}

fn rep_code_jit_library() -> jit::JitLibrary {
    jit::JitLibrary {
        port_types: vec![jit::JitPortType {
            base: Some(bin::PortType {
                ptype: 1,
                // let's just focus on the Z observable, where every X error will flip
                // for debugging simplicity, we also consider just one data qubit error
                // on the left most and then a measurement flip error on the second ancilla
                observables: vec![bin::port_type::Observable::default(); 1],
                ..Default::default()
            }),
            stabilizers: vec![jit::jit_port_type::Stabilizer::default(); 2],
            k: 0,
        }],
        gadget_types: vec![
            // Gadget type 1: prepare_z
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 1,
                    name: "prepare_z".to_string(),
                    measurements: vec![bin::gadget_type::Measurement::default(); 2],
                    outputs: vec![bin::gadget_type::Port {
                        ptype: 1,
                        ..Default::default()
                    }],
                    correction_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
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
                    logical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 0,
                        i: vec![],
                        j: vec![],
                    }),
                    physical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
                        cols: 2,
                        i: vec![],
                        j: vec![],
                    }),
                    ..Default::default()
                }),
                finished_checks: vec![
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 0,
                            ..Default::default()
                        }],
                    },
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 1,
                            ..Default::default()
                        }],
                    },
                ],
                unfinished_checks: vec![
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 0,
                            ..Default::default()
                        }],
                    },
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 1,
                            ..Default::default()
                        }],
                    },
                ],
                errors: vec![
                    jit::jit_gadget_type::Error {
                        base: Some(bin::error_model_type::Error {
                            tag: "data qubit flip".to_string(),
                            residual: vec![0],
                            readout_flips: vec![],
                            probability: 0.01,
                            ..Default::default()
                        }),
                        finished_checks: vec![0],
                        unfinished_checks: vec![],
                    },
                    jit::jit_gadget_type::Error {
                        base: Some(bin::error_model_type::Error {
                            tag: "measurement flip".to_string(),
                            residual: vec![],
                            readout_flips: vec![],
                            probability: 0.01,
                            ..Default::default()
                        }),
                        finished_checks: vec![1],
                        unfinished_checks: vec![1],
                    },
                ],
            },
            // Gadget type 2: measure_z
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 2,
                    name: "measure_z".to_string(),
                    measurements: vec![bin::gadget_type::Measurement::default(); 3],
                    inputs: vec![bin::gadget_type::Port {
                        ptype: 1,
                        ..Default::default()
                    }],
                    correction_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 2,
                        i: vec![],
                        j: vec![],
                    }),
                    readout_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
                        cols: 2,
                        i: vec![0],
                        j: vec![0],
                    }),
                    logical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 1,
                        i: vec![],
                        j: vec![],
                    }),
                    readouts: vec![bin::gadget_type::Readout {
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
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![
                            jit::jit_gadget_type::PresentMeasurement {
                                input_port: Some(0),
                                measurement_index: 0,
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                measurement_index: 0,
                                ..Default::default()
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                measurement_index: 1,
                                ..Default::default()
                            },
                        ],
                    },
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![
                            jit::jit_gadget_type::PresentMeasurement {
                                input_port: Some(0),
                                measurement_index: 1,
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                measurement_index: 1,
                                ..Default::default()
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                measurement_index: 2,
                                ..Default::default()
                            },
                        ],
                    },
                ],
                errors: vec![jit::jit_gadget_type::Error {
                    base: Some(bin::error_model_type::Error {
                        tag: "data measurement flip".to_string(),
                        residual: vec![],
                        readout_flips: vec![0],
                        probability: 0.01,
                        ..Default::default()
                    }),
                    finished_checks: vec![0, 1],
                    unfinished_checks: vec![],
                }],
                ..Default::default()
            },
            // Gadget type 3: idle
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 3,
                    name: "idle".to_string(),
                    measurements: vec![bin::gadget_type::Measurement::default(); 2],
                    inputs: vec![bin::gadget_type::Port {
                        ptype: 1,
                        ..Default::default()
                    }],
                    outputs: vec![bin::gadget_type::Port {
                        ptype: 1,
                        ..Default::default()
                    }],
                    correction_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
                        cols: 2,
                        i: vec![0],
                        j: vec![0],
                    }),
                    readout_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 2,
                        i: vec![],
                        j: vec![],
                    }),
                    logical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
                        cols: 0,
                        i: vec![],
                        j: vec![],
                    }),
                    physical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 1,
                        cols: 2,
                        i: vec![],
                        j: vec![],
                    }),
                    ..Default::default()
                }),
                finished_checks: vec![
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![
                            jit::jit_gadget_type::PresentMeasurement {
                                input_port: Some(0),
                                measurement_index: 0,
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                measurement_index: 0,
                                ..Default::default()
                            },
                        ],
                    },
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![
                            jit::jit_gadget_type::PresentMeasurement {
                                input_port: Some(0),
                                measurement_index: 1,
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                measurement_index: 1,
                                ..Default::default()
                            },
                        ],
                    },
                ],
                unfinished_checks: vec![
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 0,
                            ..Default::default()
                        }],
                    },
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            measurement_index: 1,
                            ..Default::default()
                        }],
                    },
                ],
                ..Default::default()
            },
            // Gadget type 4: transversal CNOT without any SE
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 4,
                    name: "CNOT".to_string(),
                    measurements: vec![],
                    inputs: vec![
                        bin::gadget_type::Port {
                            ptype: 1,
                            ..Default::default()
                        },
                        bin::gadget_type::Port {
                            ptype: 1,
                            ..Default::default()
                        },
                    ],
                    outputs: vec![
                        bin::gadget_type::Port {
                            ptype: 1,
                            ..Default::default()
                        },
                        bin::gadget_type::Port {
                            ptype: 1,
                            ..Default::default()
                        },
                    ],
                    correction_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 2,
                        cols: 3,
                        i: vec![0, 1, 1],
                        j: vec![0, 0, 1],
                    }),
                    readout_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 3,
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
                        cols: 0,
                        i: vec![],
                        j: vec![],
                    }),
                    ..Default::default()
                }),
                // there is no finished checks since there is no SE
                finished_checks: vec![],
                unfinished_checks: vec![
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            input_port: Some(0),
                            measurement_index: 0,
                        }],
                    },
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![jit::jit_gadget_type::PresentMeasurement {
                            input_port: Some(0),
                            measurement_index: 1,
                        }],
                    },
                    // the checks on the target qubit must check 3 measurements
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![
                            jit::jit_gadget_type::PresentMeasurement {
                                input_port: Some(0),
                                measurement_index: 0,
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                input_port: Some(1),
                                measurement_index: 0,
                            },
                        ],
                    },
                    jit::jit_gadget_type::Check {
                        base: Some(bin::check_model_type::Check::default()),
                        measurements: vec![
                            jit::jit_gadget_type::PresentMeasurement {
                                input_port: Some(0),
                                measurement_index: 1,
                            },
                            jit::jit_gadget_type::PresentMeasurement {
                                input_port: Some(1),
                                measurement_index: 1,
                            },
                        ],
                    },
                ],
                errors: vec![
                    jit::jit_gadget_type::Error {
                        base: Some(bin::error_model_type::Error {
                            tag: "control data qubit flip".to_string(),
                            residual: vec![0, 1], // flip both control and target logical observables
                            readout_flips: vec![],
                            probability: 0.01,
                            ..Default::default()
                        }),
                        finished_checks: vec![],
                        unfinished_checks: vec![0, 2], // trigger both checks
                    },
                    jit::jit_gadget_type::Error {
                        base: Some(bin::error_model_type::Error {
                            tag: "target data qubit flip".to_string(),
                            residual: vec![1],
                            readout_flips: vec![],
                            probability: 0.01,
                            ..Default::default()
                        }),
                        finished_checks: vec![],
                        unfinished_checks: vec![2],
                    },
                ],
            },
        ],
        ..Default::default()
    }
}

#[tokio::test]
async fn test_repetition_code_jit_single_cnot() {
    // cargo test test_repetition_code_jit_single_cnot -- --nocapture
    let mut jit_library = rep_code_jit_library();
    jit_library.program = vec![
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 1,
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
                gtype: 4,
                connectors: vec![
                    bin::gadget::Connector { gid: 1, port: 0 },
                    bin::gadget::Connector { gid: 2, port: 0 },
                ],
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
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 2,
                connectors: vec![bin::gadget::Connector { gid: 3, port: 1 }],
                ..Default::default()
            }),
            ..Default::default()
        },
    ];

    let library = static_jit_compile(jit_library).await;

    assert!(
        library.check_model_types[2].checks.is_empty(),
        "transversalt CNOT doesn't have internel checks"
    );
    assert!(
        library.check_model_types[3].checks.len() == 2,
        "measure_z on control qubit should have 2 checks"
    );
    assert!(
        library.check_model_types[3].checks[0].measurements.len() == 3,
        "measure_z on control qubit check should have checks on 3 measurements"
    );
    assert!(
        library.check_model_types[4].checks.len() == 2,
        "measure_z on target qubit should have 2 checks"
    );
    assert!(
        library.check_model_types[4].checks[0].measurements.len() == 4,
        "measure_z on target qubit check should have checks on 4 measurements"
    );

    let error = &library.error_model_types[0].errors[0];
    assert!(error.tag == "data qubit flip");
    assert!(error.checks.len() == 1);
    assert!(error.checks[0].remote_check_model.is_none());
    let error = &library.error_model_types[0].errors[1];
    assert!(error.tag == "measurement flip");
    assert!(error.checks.len() == 3, "measurement on control flips 3 checks");

    let error = &library.error_model_types[1].errors[0];
    assert!(error.tag == "data qubit flip");
    assert!(error.checks.len() == 1);
    let error = &library.error_model_types[1].errors[1];
    assert!(error.tag == "measurement flip");
    assert!(error.checks.len() == 2, "measurement on target only flips 2 checks");

    let error = &library.error_model_types[2].errors[0];
    assert!(error.tag == "control data qubit flip");
    assert!(error.checks.len() == 2);
    assert!(error.checks[0].remote_check_model.is_some());
    let error = &library.error_model_types[2].errors[1];
    assert!(error.tag == "target data qubit flip");
    assert!(error.checks.len() == 1);
    assert!(error.checks[0].remote_check_model.is_some());

    let error = &library.error_model_types[3].errors[0];
    assert!(error.tag == "data measurement flip");
    assert!(error.checks.len() == 2, "the measurement is used in both checks");
    assert!(error.checks[0].remote_check_model.is_none());
    assert!(error.checks[1].remote_check_model.is_none());

    let error = &library.error_model_types[4].errors[0];
    assert!(error.tag == "data measurement flip");
    assert!(error.checks.len() == 2, "the measurement is used in both checks");
    assert!(error.checks[0].remote_check_model.is_none());
    assert!(error.checks[1].remote_check_model.is_none());
}

#[tokio::test]
async fn test_repetition_code_jit_self_two_cnot() {
    // cargo test test_repetition_code_jit_self_two_cnot -- --nocapture
    let mut jit_library = rep_code_jit_library();
    jit_library.program = vec![
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 1,
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
                gtype: 4,
                connectors: vec![
                    bin::gadget::Connector { gid: 1, port: 0 },
                    bin::gadget::Connector { gid: 2, port: 0 },
                ],
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 4,
                connectors: vec![
                    bin::gadget::Connector { gid: 3, port: 0 },
                    bin::gadget::Connector { gid: 3, port: 1 },
                ],
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 2,
                connectors: vec![bin::gadget::Connector { gid: 4, port: 0 }],
                ..Default::default()
            }),
            ..Default::default()
        },
        jit::JitInstruction {
            gadget: Some(bin::Gadget {
                gtype: 2,
                connectors: vec![bin::gadget::Connector { gid: 4, port: 1 }],
                ..Default::default()
            }),
            ..Default::default()
        },
    ];

    let library = static_jit_compile(jit_library).await;

    assert!(
        library.check_model_types[2].checks.is_empty(),
        "transversalt CNOT doesn't have internel checks"
    );
    assert!(
        library.check_model_types[3].checks.is_empty(),
        "transversalt CNOT doesn't have internel checks"
    );
    assert!(
        library.check_model_types[4].checks.len() == 2,
        "measure_z on control qubit should have 2 checks"
    );
    assert!(
        library.check_model_types[4].checks[0].measurements.len() == 3,
        "two CNOTs will cancel out each other and thus the check is on 3 measurements only"
    );
    assert!(
        library.check_model_types[5].checks.len() == 2,
        "measure_z on target qubit should have 2 checks"
    );
    assert!(
        library.check_model_types[5].checks[0].measurements.len() == 3,
        "two CNOTs will cancel out each other and thus the check is on 3 measurements only"
    );

    let error = &library.error_model_types[0].errors[0];
    assert!(error.tag == "data qubit flip");
    assert!(error.checks.len() == 1);
    assert!(error.checks[0].remote_check_model.is_none());
    let error = &library.error_model_types[0].errors[1];
    assert!(error.tag == "measurement flip");
    assert!(error.checks.len() == 2, "CNOTs cancel out, measurement only flips 2 checks");

    let error = &library.error_model_types[1].errors[0];
    assert!(error.tag == "data qubit flip");
    assert!(error.checks.len() == 1);
    let error = &library.error_model_types[1].errors[1];
    assert!(error.tag == "measurement flip");
    assert!(error.checks.len() == 2, "measurement on target only flips 2 checks");

    let error = &library.error_model_types[2].errors[0];
    assert!(error.tag == "control data qubit flip");
    assert!(error.checks.len() == 1);
    assert!(error.checks[0].remote_check_model.is_some());
    assert!(
        library.error_model_types[2].remote_check_models[error.checks[0].remote_check_model.unwrap() as usize].absolute_cid
            == Some(5)
    );
    let error = &library.error_model_types[2].errors[1];
    assert!(error.tag == "target data qubit flip");
    assert!(error.checks.len() == 1);
    assert!(error.checks[0].remote_check_model.is_some());
    assert!(
        library.error_model_types[2].remote_check_models[error.checks[0].remote_check_model.unwrap() as usize].absolute_cid
            == Some(6)
    );

    let error = &library.error_model_types[3].errors[0];
    assert!(error.tag == "control data qubit flip");
    assert!(error.checks.len() == 2, "error in between two CNOT will trigger both checks");
    assert!(error.checks[0].remote_check_model.is_some());
    assert!(error.checks[1].remote_check_model.is_some());
    let error = &library.error_model_types[3].errors[1];
    assert!(error.tag == "target data qubit flip");
    assert!(error.checks.len() == 1);
    assert!(error.checks[0].remote_check_model.is_some());

    let error = &library.error_model_types[4].errors[0];
    assert!(error.tag == "data measurement flip");
    assert!(error.checks.len() == 2, "the measurement is used in both checks");
    assert!(error.checks[0].remote_check_model.is_none());
    assert!(error.checks[1].remote_check_model.is_none());

    let error = &library.error_model_types[5].errors[0];
    assert!(error.tag == "data measurement flip");
    assert!(error.checks.len() == 2, "the measurement is used in both checks");
    assert!(error.checks[0].remote_check_model.is_none());
    assert!(error.checks[1].remote_check_model.is_none());
}

/// Test that error model futures are blocked until all output ports are connected
/// to gadgets with syndrome extraction (idle or measurement gadgets).
#[tokio::test]
async fn test_error_model_blocking_until_syndrome_extraction() {
    // cargo test test_error_model_blocking_until_syndrome_extraction -- --nocapture
    use deq_runtime::jit::jit_compiler::JitCompiler;
    use std::pin::pin;
    use std::time::Duration;

    let jit_library = rep_code_jit_library();
    let compiler = JitCompiler::new();
    compiler.load_library(jit_library).await;

    // Prepare two logical qubits (gid 1 and 2)
    let (_, _, _, error_model_future_1) = compiler
        .compile(
            jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gtype: 1,
                    ..Default::default()
                }),
                ..Default::default()
            },
            CancellationToken::new(),
        )
        .await;

    let (_, _, _, error_model_future_2) = compiler
        .compile(
            jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gtype: 1,
                    ..Default::default()
                }),
                ..Default::default()
            },
            CancellationToken::new(),
        )
        .await;

    // Apply CNOT between them (gid 3)
    let (cnot_gadget, _, _, error_model_future_cnot) = compiler
        .compile(
            jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gtype: 4,
                    connectors: vec![
                        bin::gadget::Connector { gid: 1, port: 0 },
                        bin::gadget::Connector { gid: 2, port: 0 },
                    ],
                    ..Default::default()
                }),
                ..Default::default()
            },
            CancellationToken::new(),
        )
        .await;

    let cnot_gid = cnot_gadget.gid;

    // Pin the futures so we can poll them multiple times with select!
    let mut future_1 = pin!(error_model_future_1);
    let mut future_2 = pin!(error_model_future_2);
    let mut future_cnot = pin!(error_model_future_cnot);

    // CHECK 1: All three futures should be blocked because CNOT's outputs are not connected
    tokio::select! {
        biased;
        _ = &mut future_1 => panic!("prepare_1 error model should be blocked before CNOT outputs are connected"),
        _ = &mut future_2 => panic!("prepare_2 error model should be blocked before CNOT outputs are connected"),
        _ = &mut future_cnot => panic!("CNOT error model should be blocked before its outputs are connected"),
        _ = tokio::time::sleep(Duration::from_millis(50)) => {
            // Good - all futures are blocked as expected
        }
    }

    // Add idle gates on both logical qubits (gid 4 and 5)
    let (idle_gadget_1, _, _, error_model_future_idle_1) = compiler
        .compile(
            jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gtype: 3,
                    connectors: vec![bin::gadget::Connector { gid: cnot_gid, port: 0 }],
                    ..Default::default()
                }),
                ..Default::default()
            },
            CancellationToken::new(),
        )
        .await;

    let (idle_gadget_2, _, _, error_model_future_idle_2) = compiler
        .compile(
            jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gtype: 3,
                    connectors: vec![bin::gadget::Connector { gid: cnot_gid, port: 1 }],
                    ..Default::default()
                }),
                ..Default::default()
            },
            CancellationToken::new(),
        )
        .await;

    let idle_gid_1 = idle_gadget_1.gid;
    let idle_gid_2 = idle_gadget_2.gid;

    // Pin idle futures
    let mut future_idle_1 = pin!(error_model_future_idle_1);
    let mut future_idle_2 = pin!(error_model_future_idle_2);

    // Give the runtime a chance to wake up the blocked futures now that connectors are set
    tokio::task::yield_now().await;

    // CHECK 2: Now CNOT and prepare futures should be ready (idle gates have syndrome extraction)
    // But idle futures should still be blocked (their outputs are not connected)
    // We use join! to run CNOT and prepare futures concurrently (they depend on each other)
    let ready_futures = futures_util::future::join3(&mut future_1, &mut future_2, &mut future_cnot);
    let ready_result = tokio::time::timeout(Duration::from_millis(100), ready_futures).await;
    assert!(
        ready_result.is_ok(),
        "prepare and CNOT error models should be ready after idle gates are added"
    );
    let (result_1, result_2, result_cnot) = ready_result.unwrap();

    // Verify the CNOT error model
    let (error_model_type_cnot, _) = result_cnot;
    assert_eq!(error_model_type_cnot.errors.len(), 2);
    assert_eq!(error_model_type_cnot.errors[0].tag, "control data qubit flip");
    assert_eq!(error_model_type_cnot.errors[1].tag, "target data qubit flip");

    // Verify prepare_1 error model
    let (error_model_type_1, _) = result_1;
    assert_eq!(error_model_type_1.errors.len(), 2);

    // Verify prepare_2 error model
    let (error_model_type_2, _) = result_2;
    assert_eq!(error_model_type_2.errors.len(), 2);

    // CHECK 3: Idle futures should still be blocked (their outputs are not connected to measurement)
    tokio::select! {
        biased;
        _ = &mut future_idle_1 => panic!("idle_1 error model should be blocked before measurement is connected"),
        _ = &mut future_idle_2 => panic!("idle_2 error model should be blocked before measurement is connected"),
        _ = tokio::time::sleep(Duration::from_millis(10)) => {
            // Good - idle futures are blocked as expected
        }
    }

    // Add measurement gates to complete the circuit
    let (_, _, _, error_model_future_measure_1) = compiler
        .compile(
            jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gtype: 2,
                    connectors: vec![bin::gadget::Connector {
                        gid: idle_gid_1,
                        port: 0,
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            },
            CancellationToken::new(),
        )
        .await;

    let (_, _, _, error_model_future_measure_2) = compiler
        .compile(
            jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gtype: 2,
                    connectors: vec![bin::gadget::Connector {
                        gid: idle_gid_2,
                        port: 0,
                    }],
                    ..Default::default()
                }),
                ..Default::default()
            },
            CancellationToken::new(),
        )
        .await;

    // CHECK 3: Now idle futures should be ready
    let result_idle_1 = tokio::time::timeout(Duration::from_millis(50), &mut future_idle_1).await;
    assert!(
        result_idle_1.is_ok(),
        "idle_1 error model should be ready after measurement is added"
    );

    let result_idle_2 = tokio::time::timeout(Duration::from_millis(50), &mut future_idle_2).await;
    assert!(
        result_idle_2.is_ok(),
        "idle_2 error model should be ready after measurement is added"
    );

    // CHECK 4: Measurement futures should complete immediately (no output ports)
    let all_measure_futures = vec![error_model_future_measure_1, error_model_future_measure_2];
    let measure_results =
        tokio::time::timeout(Duration::from_millis(50), futures_util::future::join_all(all_measure_futures))
            .await
            .expect("Measurement error models should be immediately ready");

    // Verify measurement error models
    let (error_model_type_measure_1, _) = &measure_results[0];
    assert_eq!(error_model_type_measure_1.errors.len(), 1);
    assert_eq!(error_model_type_measure_1.errors[0].tag, "data measurement flip");

    let (error_model_type_measure_2, _) = &measure_results[1];
    assert_eq!(error_model_type_measure_2.errors.len(), 1);
    assert_eq!(error_model_type_measure_2.errors[0].tag, "data measurement flip");
}

/// Test that error model future is blocked when outputs are not connected.
/// This test verifies blocking by showing that join_all times out when
/// output ports are not connected.
#[tokio::test]
async fn test_error_model_blocked_without_output_connection() {
    // cargo test test_error_model_blocked_without_output_connection -- --nocapture
    use deq_runtime::jit::jit_compiler::JitCompiler;
    use std::time::Duration;

    let jit_library = rep_code_jit_library();
    let compiler = JitCompiler::new();
    compiler.load_library(jit_library).await;

    // Prepare a logical qubit (gid 1)
    let (_, _, _, error_model_future_1) = compiler
        .compile(
            jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gtype: 1,
                    ..Default::default()
                }),
                ..Default::default()
            },
            CancellationToken::new(),
        )
        .await;

    // The prepare gadget's output port is not connected to anything.
    // Trying to await its error model should block indefinitely.
    let result = tokio::time::timeout(Duration::from_millis(50), error_model_future_1).await;

    assert!(
        result.is_err(),
        "Error model future should be blocked when output port is not connected"
    );
}

/// Test that connecting output to measurement gadget unblocks the error model.
#[tokio::test]
async fn test_error_model_unblocked_with_measurement() {
    // cargo test test_error_model_unblocked_with_measurement -- --nocapture
    use deq_runtime::jit::jit_compiler::JitCompiler;
    use std::time::Duration;

    let jit_library = rep_code_jit_library();
    let compiler = JitCompiler::new();
    compiler.load_library(jit_library).await;

    // Prepare a logical qubit (gid 1)
    let (_, _, _, error_model_future_1) = compiler
        .compile(
            jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gtype: 1,
                    ..Default::default()
                }),
                ..Default::default()
            },
            CancellationToken::new(),
        )
        .await;

    // Connect output to measurement gadget (gid 2)
    let (_, _, _, error_model_future_measure) = compiler
        .compile(
            jit::JitInstruction {
                gadget: Some(bin::Gadget {
                    gtype: 2,
                    connectors: vec![bin::gadget::Connector { gid: 1, port: 0 }],
                    ..Default::default()
                }),
                ..Default::default()
            },
            CancellationToken::new(),
        )
        .await;

    // Now both error model futures should complete
    let results = tokio::time::timeout(
        Duration::from_millis(100),
        futures_util::future::join_all(vec![error_model_future_1, error_model_future_measure]),
    )
    .await
    .expect("Error model futures should complete when connected to measurement");

    // Verify the prepare error model
    let (error_model_type_1, _) = &results[0];
    assert_eq!(error_model_type_1.errors.len(), 2);

    // Verify the measurement error model
    let (error_model_type_measure, _) = &results[1];
    assert_eq!(error_model_type_measure.errors.len(), 1);
    assert_eq!(error_model_type_measure.errors[0].tag, "data measurement flip");
}
