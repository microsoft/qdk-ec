//! Shared JIT library builders for integration tests.

use deq_runtime::bin::{self, check_model_type, error_model_type, gadget_type};
use deq_runtime::jit::{self, jit_gadget_type};

/// Build a JIT library with five gadget types:
/// - gtype 1 (opcode 0): prepare_z — no inputs, 1 output, 2 measurements
/// - gtype 2 (opcode 1): measure_z — 1 input, no outputs, 3 measurements, 1 readout
/// - gtype 3 (opcode 2): cnot — 2 inputs, 2 outputs, 0 measurements
/// - gtype 4 (opcode 3): identity — 1 input, 1 output, 0 measurements
/// - gtype 5 (opcode 4): idle — 1 input, 1 output, 2 measurements
pub fn test_jit_library() -> jit::JitLibrary {
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
            // gtype 1 (opcode 0): prepare_z — no inputs, 1 output
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
            // gtype 2 (opcode 1): measure_z — 1 input, no outputs
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
            // gtype 3 (opcode 2): cnot — 2 inputs, 2 outputs, 0 measurements
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 3,
                    name: "cnot".to_string(),
                    measurements: vec![],
                    inputs: vec![
                        gadget_type::Port {
                            ptype: 1,
                            ..Default::default()
                        },
                        gadget_type::Port {
                            ptype: 1,
                            ..Default::default()
                        },
                    ],
                    outputs: vec![
                        gadget_type::Port {
                            ptype: 1,
                            ..Default::default()
                        },
                        gadget_type::Port {
                            ptype: 1,
                            ..Default::default()
                        },
                    ],
                    correction_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 4,
                        cols: 5,
                        i: vec![0, 1, 2, 3],
                        j: vec![0, 1, 2, 3],
                    }),
                    readout_propagation: Some(deq_runtime::util::BitMatrix {
                        rows: 0,
                        cols: 5,
                        i: vec![],
                        j: vec![],
                    }),
                    logical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 4,
                        cols: 0,
                        i: vec![],
                        j: vec![],
                    }),
                    physical_correction: Some(deq_runtime::util::BitMatrix {
                        rows: 4,
                        cols: 0,
                        i: vec![],
                        j: vec![],
                    }),
                    ..Default::default()
                }),
                unfinished_checks: vec![
                    jit_gadget_type::Check {
                        base: Some(check_model_type::Check::default()),
                        measurements: vec![jit_gadget_type::PresentMeasurement {
                            input_port: Some(0),
                            measurement_index: 0,
                        }],
                    },
                    jit_gadget_type::Check {
                        base: Some(check_model_type::Check::default()),
                        measurements: vec![jit_gadget_type::PresentMeasurement {
                            input_port: Some(0),
                            measurement_index: 1,
                        }],
                    },
                    jit_gadget_type::Check {
                        base: Some(check_model_type::Check::default()),
                        measurements: vec![jit_gadget_type::PresentMeasurement {
                            input_port: Some(1),
                            measurement_index: 0,
                        }],
                    },
                    jit_gadget_type::Check {
                        base: Some(check_model_type::Check::default()),
                        measurements: vec![jit_gadget_type::PresentMeasurement {
                            input_port: Some(1),
                            measurement_index: 1,
                        }],
                    },
                ],
                ..Default::default()
            },
            // gtype 4 (opcode 3): identity — 1 input, 1 output, 0 measurements
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 4,
                    name: "identity".to_string(),
                    measurements: vec![],
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
                        cols: 0,
                        i: vec![],
                        j: vec![],
                    }),
                    ..Default::default()
                }),
                // Pass-through: propagate input stabilizers to output unchanged.
                unfinished_checks: vec![
                    jit_gadget_type::Check {
                        base: Some(check_model_type::Check::default()),
                        measurements: vec![jit_gadget_type::PresentMeasurement {
                            input_port: Some(0),
                            measurement_index: 0,
                        }],
                    },
                    jit_gadget_type::Check {
                        base: Some(check_model_type::Check::default()),
                        measurements: vec![jit_gadget_type::PresentMeasurement {
                            input_port: Some(0),
                            measurement_index: 1,
                        }],
                    },
                ],
                ..Default::default()
            },
            // gtype 5 (opcode 4): idle — 1 input, 1 output, 2 measurements (hop-counted)
            jit::JitGadgetType {
                base: Some(bin::GadgetType {
                    gtype: 5,
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
                        residual: vec![1],
                        readout_flips: vec![],
                        probability: 0.01,
                        ..Default::default()
                    }),
                    finished_checks: vec![1],
                    unfinished_checks: vec![1],
                }],
            },
        ],
        program: vec![],
    }
}
