use super::*;
use crate::{Span, Spanned};

/// Wraps a node in a `Spanned<T>` with a dummy span. Equality ignores spans, so
/// this compares equal to any parsed `Spanned<T>` carrying the same node.
fn sp<T>(node: T) -> Spanned<T> {
    Spanned::new(node, Span { start: 0, end: 0 })
}

/// Parsing a source, displaying it, and re-parsing must yield the same AST.
fn assert_roundtrip(input: &str) {
    let ast1: DeqFile = input.parse().unwrap_or_else(|e| panic!("parse failed:\n{input}\n{e}"));
    let serialized = ast1.to_string();
    let ast2: DeqFile = serialized
        .parse()
        .unwrap_or_else(|e| panic!("re-parse failed:\n{serialized}\n{e}"));
    assert_eq!(
        ast1, ast2,
        "roundtrip mismatch.\ninput:\n{input}\nserialized:\n{serialized}"
    );
}

#[test]
fn parse_small_example_ast() {
    let ast: DeqFile = "\
CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}

GADGET PrepareZ {
    R 0 1 2
    X_ERROR(0.03) 0 1 2
    OUTPUT RepetitionCode 0 1 2
}
"
    .parse()
    .unwrap();

    assert_eq!(ast.definitions.len(), 2);
    let Definition::Code(code) = &ast.definitions[0].node else {
        panic!("expected CODE");
    };
    assert_eq!(code.name, "RepetitionCode");
    assert_eq!((code.n, code.k, code.d), (3, 1, Some(1)));
    assert_eq!(code.logicals.len(), 1);
    assert_eq!(code.stabilizers.len(), 2);

    let Definition::Gadget(gadget) = &ast.definitions[1].node else {
        panic!("expected GADGET");
    };
    assert_eq!(gadget.name, "PrepareZ");
    assert_eq!(gadget.body.len(), 3);
}

#[test]
fn code_without_distance() {
    let ast: DeqFile = "CODE C [[96,6]] {\n    STABILIZER X0*X1\n}\n".parse().unwrap();
    let Definition::Code(code) = &ast.definitions[0].node else {
        panic!("expected CODE");
    };
    assert_eq!((code.n, code.k, code.d), (96, 6, None));
}

#[test]
fn imports_are_collected() {
    let ast: DeqFile = "IMPORT \"code.deq\"\nGADGET G {\n    R 0\n}\n".parse().unwrap();
    assert_eq!(ast.imports, vec!["code.deq".to_string()]);
    assert_eq!(ast.definitions.len(), 1);
}

#[test]
fn decorators_attach_to_definition() {
    let ast: DeqFile = "@GTYPE(1)\n@CHECKS(\"syndrome\")\nGADGET G {\n    R 0\n}\n"
        .parse()
        .unwrap();
    let Definition::Gadget(gadget) = &ast.definitions[0].node else {
        panic!("expected GADGET");
    };
    assert_eq!(gadget.decorators.len(), 2);
    assert_eq!(gadget.decorators[0].name, "GTYPE");
    assert_eq!(
        gadget.decorators[0].arguments,
        vec![DecoratorArg::Value(DecoratorValue::Int(1))]
    );
    assert_eq!(
        gadget.decorators[1].arguments,
        vec![DecoratorArg::Value(DecoratorValue::String("syndrome".into()))]
    );
}

#[test]
fn check_and_detector_conflate() {
    let a: DeqFile = "GADGET G {\n    CHECK rec[-1]\n}\n".parse().unwrap();
    let b: DeqFile = "GADGET G {\n    DETECTOR rec[-1]\n}\n".parse().unwrap();
    assert_eq!(a, b);
    let Definition::Gadget(gadget) = &a.definitions[0].node else {
        panic!("expected GADGET");
    };
    assert_eq!(
        gadget.body[0],
        GadgetStatement::Check(CheckStatement {
            targets: vec![sp(Target::MeasurementRecord { offset: 1 })],
            flip: false,
        })
    );
}

#[test]
fn readout_and_observable_include_conflate() {
    let a: DeqFile = "GADGET G {\n    READOUT LX0 FLIP\n}\n".parse().unwrap();
    let b: DeqFile = "GADGET G {\n    OBSERVABLE_INCLUDE LX0 FLIP\n}\n".parse().unwrap();
    assert_eq!(a, b);
    let Definition::Gadget(gadget) = &a.definitions[0].node else {
        panic!("expected GADGET");
    };
    assert_eq!(
        gadget.body[0],
        GadgetStatement::Readout(ReadoutStatement {
            targets: vec![sp(ReadoutTargetItem::Logical(LogicalPauliTarget {
                pauli: Pauli::X,
                index: 0,
                port: None,
            }))],
            flip: true,
        })
    );
}

#[test]
fn propagate_targets() {
    let ast: DeqFile = "GADGET G {\n    PROPAGATE LX0 FROM LZ0 IN0.DS3 rec[-2] M4 R1 FLIP\n}\n"
        .parse()
        .unwrap();
    let Definition::Gadget(gadget) = &ast.definitions[0].node else {
        panic!("expected GADGET");
    };
    assert_eq!(
        gadget.body[0],
        GadgetStatement::Propagate(PropagateStatement {
            target: sp(LogicalPauliTarget {
                pauli: Pauli::X,
                index: 0,
                port: None
            }),
            terms: vec![
                sp(PropagateTerm::Logical(LogicalPauliTarget {
                    pauli: Pauli::Z,
                    index: 0,
                    port: None
                })),
                sp(PropagateTerm::Destabilizer { port: 0, stabilizer: 3 }),
                sp(PropagateTerm::MeasurementRecord { offset: 2 }),
                sp(PropagateTerm::PhysicalMeasurement { index: 4 }),
                sp(PropagateTerm::Readout { index: 1 }),
            ],
            flip: true,
        })
    );
}

#[test]
fn readout_with_destabilizer_target() {
    let ast: DeqFile = "GADGET G {\n    READOUT LX0 IN1.DS2\n}\n".parse().unwrap();
    let Definition::Gadget(gadget) = &ast.definitions[0].node else {
        panic!("expected GADGET");
    };
    assert_eq!(
        gadget.body[0],
        GadgetStatement::Readout(ReadoutStatement {
            targets: vec![
                sp(ReadoutTargetItem::Logical(LogicalPauliTarget {
                    pauli: Pauli::X,
                    index: 0,
                    port: None
                })),
                sp(ReadoutTargetItem::Destabilizer { port: 1, stabilizer: 2 }),
            ],
            flip: false,
        })
    );
}

#[test]
fn conditional_correction_in_program_and_compose() {
    let ast: DeqFile = "\
COMPOSE C {
    CONDITIONAL rec[-1] X0*Y1 2
}

PROGRAM P {
    CONDITIONAL rec[-3] Z0 4
}
"
    .parse()
    .unwrap();

    let Definition::Compose(compose) = &ast.definitions[0].node else {
        panic!("expected COMPOSE");
    };
    assert_eq!(
        compose.body[0],
        ComposeStatement::ConditionalCorrection(ConditionalCorrection {
            readout_offset: 1,
            paulis: vec![(Pauli::X, 0), (Pauli::Y, 1)],
            wire: 2,
        })
    );

    let Definition::Program(program) = &ast.definitions[1].node else {
        panic!("expected PROGRAM");
    };
    assert_eq!(
        program.body[0],
        ProgramStatement::ConditionalCorrection(ConditionalCorrection {
            readout_offset: 3,
            paulis: vec![(Pauli::Z, 0)],
            wire: 4,
        })
    );
}

#[test]
fn port_scoped_logical() {
    let ast: DeqFile = "GADGET G {\n    VIRTUAL IN0.LX1 OUT2.LZ3\n}\n".parse().unwrap();
    let Definition::Gadget(gadget) = &ast.definitions[0].node else {
        panic!("expected GADGET");
    };
    assert_eq!(
        gadget.body[0],
        GadgetStatement::VirtualLogical(VirtualLogicalStatement {
            targets: vec![
                sp(LogicalPauliTarget {
                    pauli: Pauli::X,
                    index: 1,
                    port: Some(Port {
                        kind: PortKind::In,
                        index: 0
                    }),
                }),
                sp(LogicalPauliTarget {
                    pauli: Pauli::Z,
                    index: 3,
                    port: Some(Port {
                        kind: PortKind::Out,
                        index: 2
                    }),
                }),
            ],
        })
    );
}

#[test]
fn gadget_application_forms() {
    let ast: DeqFile = "\
COMPOSE C {
    Idle IN(0 1) OUT(2 3)
    Setup ()
    Shortcut 0 1
}
"
    .parse()
    .unwrap();
    let Definition::Compose(compose) = &ast.definitions[0].node else {
        panic!("expected COMPOSE");
    };
    assert_eq!(
        compose.body[0],
        ComposeStatement::GadgetApplication(GadgetApplication {
            gadget_name: sp("Idle".to_string()),
            in_indices: Some(vec![0, 1]),
            out_indices: Some(vec![2, 3]),
        })
    );
    assert_eq!(
        compose.body[1],
        ComposeStatement::GadgetApplication(GadgetApplication {
            gadget_name: sp("Setup".to_string()),
            in_indices: None,
            out_indices: None,
        })
    );
    // Shortcut form parses as a plain instruction.
    assert!(matches!(compose.body[2].node, ComposeStatement::Instruction(_)));
}

#[test]
fn pauli_products_with_combiners_and_identity() {
    let ast: DeqFile = "GADGET G {\n    MPP X1*Z2 !Y3\n}\n".parse().unwrap();
    let Definition::Gadget(gadget) = &ast.definitions[0].node else {
        panic!("expected GADGET");
    };
    let GadgetStatement::Instruction(instr) = &gadget.body[0].node else {
        panic!("expected instruction");
    };
    assert_eq!(instr.name, "MPP");
    assert_eq!(
        instr.targets,
        vec![
            sp(Target::Pauli {
                inverted: false,
                pauli: Pauli::X,
                index: 1
            }),
            sp(Target::Combiner),
            sp(Target::Pauli {
                inverted: false,
                pauli: Pauli::Z,
                index: 2
            }),
            sp(Target::Pauli {
                inverted: true,
                pauli: Pauli::Y,
                index: 3
            }),
        ]
    );
}

#[test]
fn identity_pauli_product() {
    let ast: DeqFile = "CODE C [[1,1]] {\n    LOGICAL _ Z0\n}\n".parse().unwrap();
    let Definition::Code(code) = &ast.definitions[0].node else {
        panic!("expected CODE");
    };
    assert_eq!(code.logicals[0].x_operator, PauliProduct::Identity);
}

// ── Roundtrip tests ──────────────────────────────────────────────────

#[test]
fn roundtrip_full_program() {
    assert_roundtrip(
        "\
IMPORT \"code.deq\"

CODE RepetitionCode [[3,1,1]] {
    LOGICAL X0*X1*X2 Z0*Z1*Z2
    STABILIZER Z0*Z1 Z1*Z2
}

@GTYPE(1)
@CHECKS(\"syndrome\")
GADGET Idle {
    INPUT RepetitionCode 0 2 4
    X_ERROR(0.03) 0 2 4
    R 1 3
    CX 0 1 2 3
    M(0.03) 1 3
    CHECK rec[-1] rec[-2] FLIP
    READOUT LX0
    ERROR(0.001) C0 R1 LX0
    CONDITIONAL R0 LZ0 LX1
    CONDITIONAL rec[-1] LZ0
    PRESELECT rec[-1] 0
    PROPAGATE LX0 FROM LZ0 IN0.DS3 rec[-2] FLIP
    VIRTUAL LZ0 OUT1.LX2
    OUTPUT RepetitionCode 0 2 4
}

COMPOSE Round {
    INPUT RepetitionCode 0 1 2
    Idle IN(0 1 2) OUT(0 1 2)
    Idle 0
    REPEAT 5 {
        Idle IN(0 1 2) OUT(0 1 2)
    }
    OUTPUT RepetitionCode 0 1 2
}

PROGRAM Simulation {
    PrepareZ 0
    Round IN(0 1 2) OUT(0 1 2)
    VIRTUAL X0*Y1 3
    ASSERT_EQ rec[-1] 0
}
",
    );
}

#[test]
fn roundtrip_tags_and_sweep() {
    assert_roundtrip("GADGET G {\n    TICK[hello\\Cworld\\n]\n    CX sweep[5] 1\n}\n");
}

#[test]
fn roundtrip_preselect_forms() {
    // Default parity (no trailing bit): Display emits an explicit ` 0`, which
    // must re-parse to the same AST (expected_value == 0).
    assert_roundtrip("GADGET G {\n    M 0\n    PRESELECT rec[-1]\n}\n");
    // Multiple physical targets, default and explicit parity.
    assert_roundtrip("GADGET G {\n    M 0 1\n    PRESELECT rec[-1] rec[-2]\n}\n");
    assert_roundtrip("GADGET G {\n    M 0 1\n    PRESELECT rec[-1] rec[-2] 1\n}\n");
    // Absolute physical measurement form.
    assert_roundtrip("GADGET G {\n    M 0\n    PRESELECT M0 1\n}\n");
}

#[test]
fn roundtrip_correlated_error_chain() {
    assert_roundtrip("GADGET G {\n    CORRELATED_ERROR(0.2) Z17\n    ELSE_CORRELATED_ERROR(0.25) Z158\n}\n");
}

#[test]
fn roundtrip_decorator_keyword_args() {
    assert_roundtrip("@TWEAK(level=\"high\", scale=2.5, count=3)\nGADGET G {\n    R 0\n}\n");
}

#[test]
fn decorator_string_escapes_are_decoded() {
    let ast: DeqFile = "@label(\"hello\\nworld\")\nGADGET G {\n    R 0\n}\n".parse().unwrap();
    let Definition::Gadget(gadget) = &ast.definitions[0].node else {
        panic!("expected GADGET");
    };
    assert_eq!(
        gadget.decorators[0].arguments,
        vec![DecoratorArg::Value(DecoratorValue::String("hello\nworld".to_string()))]
    );
    // The decoded newline must re-encode so the value round-trips.
    assert_roundtrip("@label(\"hello\\nworld\")\nGADGET G {\n    R 0\n}\n");
}

#[test]
fn import_string_escapes_are_decoded() {
    let ast: DeqFile = "IMPORT \"a\\tb.deq\"\n".parse().unwrap();
    assert_eq!(ast.imports, vec!["a\tb.deq".to_string()]);
    assert_roundtrip("IMPORT \"a\\tb.deq\"\n");
}

#[test]
fn roundtrip_integral_decorator_float() {
    // `2.0` must not collapse to `2` (which would re-parse as an Int).
    let ast: DeqFile = "@A(2.0)\nGADGET G {\n    R 0\n}\n".parse().unwrap();
    let Definition::Gadget(gadget) = &ast.definitions[0].node else {
        panic!("expected GADGET");
    };
    assert_eq!(
        gadget.decorators[0].arguments,
        vec![DecoratorArg::Value(DecoratorValue::Float(2.0))]
    );
    assert_roundtrip("@A(2.0)\nGADGET G {\n    R 0\n}\n");
}

#[test]
fn roundtrip_code_surgery_constructs() {
    assert_roundtrip(
        "\
GADGET G {
    READOUT LX0 IN0.DS1
    PROPAGATE LZ0 FROM IN0.DS0 R2 rec[-1] FLIP
}

COMPOSE C {
    CONDITIONAL rec[-2] X0*Z1 3
}

PROGRAM P {
    CONDITIONAL rec[-1] Y0 0
}
",
    );
}

#[test]
fn multiline_string_is_rejected() {
    assert!("IMPORT \"a\nb\"\n".parse::<DeqFile>().is_err());
}

// ── Error reporting (numeric out-of-range / non-finite) ──────────────

/// Parses `input`, expecting an `Err` whose rendered message contains `needle`.
/// Crucially, this must be a returned error — never a panic.
fn assert_error_contains(input: &str, needle: &str) {
    let err = input.parse::<DeqFile>().expect_err("expected a parse error");
    let rendered = err.to_string();
    assert!(
        rendered.contains(needle),
        "error did not mention {needle:?}:\n{rendered}"
    );
}

#[test]
fn oversized_qubit_index_errors_not_panics() {
    assert_error_contains("GADGET G {\n    R 99999999999999999999\n}\n", "integer out of range");
}

#[test]
fn oversized_decorator_int_errors() {
    assert_error_contains(
        "@A(99999999999999999999999)\nGADGET G {\n    R 0\n}\n",
        "integer out of range",
    );
}

#[test]
fn non_finite_instruction_argument_errors() {
    assert_error_contains("GADGET G {\n    X_ERROR(1e999) 0\n}\n", "finite");
}

#[test]
fn error_message_carries_line_and_column() {
    // The span-anchored pest formatting includes a `line:col` locator.
    assert_error_contains("GADGET G {\n    R 99999999999999999999\n}\n", "2:7");
}

// ── Source spans (G2) ────────────────────────────────────────────────

#[test]
fn spans_point_at_the_right_bytes() {
    let src = "GADGET G {\n    R 3 5\n}\n";
    let ast: DeqFile = src.parse().unwrap();

    // The gadget definition spans from `GADGET` to its closing brace.
    let def = &ast.definitions[0];
    assert_eq!(&src[def.span.start..def.span.end], src.trim_end());

    let Definition::Gadget(gadget) = &def.node else {
        panic!("expected GADGET");
    };
    // The `R 3 5` instruction statement.
    let stmt = &gadget.body[0];
    assert_eq!(&src[stmt.span.start..stmt.span.end], "R 3 5");

    let GadgetStatement::Instruction(instr) = &stmt.node else {
        panic!("expected instruction");
    };
    // Each target carries its own span.
    assert_eq!(&src[instr.targets[0].span.start..instr.targets[0].span.end], "3");
    assert_eq!(&src[instr.targets[1].span.start..instr.targets[1].span.end], "5");
    // line/col resolves against the source.
    assert_eq!(instr.targets[1].span.line_col(src), Some((2, 9)));
}

#[test]
fn spans_ignored_by_equality() {
    // Two sources that differ only in whitespace produce equal ASTs despite
    // carrying different byte offsets.
    let a: DeqFile = "GADGET G {\n    R 0\n}\n".parse().unwrap();
    let b: DeqFile = "GADGET G {  R 0  }\n".parse().unwrap();
    assert_eq!(a, b);
}
