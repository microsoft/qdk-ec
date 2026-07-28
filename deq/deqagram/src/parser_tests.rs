use pest::Parser;

use super::*;

fn parse_ok(input: &str) {
    DeqParser::parse(Rule::deq_file, input).unwrap_or_else(|e| panic!("parse failed:\n{input}\n{e}"));
}

#[test]
fn parse_small_example() {
    parse_ok(
        "\
CODE RepetitionCode [[3,1,1]] {
LOGICAL X0*X1*X2 Z0*Z1*Z2
STABILIZER Z0*Z1 Z1*Z2
}

GADGET PrepareZ {
R 0 1 2
X_ERROR(0.03) 0 1 2
OUTPUT RepetitionCode 0 1 2
}

GADGET MeasureZ {
INPUT RepetitionCode 0 1 2
M(0.03) 0 1 2
READOUT rec[-1] rec[-2] rec[-3]
}

PROGRAM Simulation {
PrepareZ 0
MeasureZ 0
ASSERT_EQ rec[-1] 0
}
",
    );
}

#[test]
fn parse_import() {
    parse_ok("IMPORT \"code.deq\"\n");
}

#[test]
fn parse_code_without_distance() {
    parse_ok("CODE C [[96,6]] {\n    STABILIZER X0*X1\n}\n");
}

#[test]
fn parse_decorated_gadget() {
    parse_ok("@GTYPE(1)\n@CHECKS(\"syndrome\")\nGADGET G {\n    R 0\n}\n");
}

#[test]
fn parse_check_and_readout_flip() {
    parse_ok("GADGET G {\n    CHECK rec[-1] rec[-2] FLIP\n    READOUT LX0 FLIP\n}\n");
}

#[test]
fn parse_error_and_conditional() {
    parse_ok("GADGET G {\n    ERROR(0.001) C0 R1 LX0\n    CONDITIONAL R0 LZ0 LX1\n    CONDITIONAL rec[-1] LZ0\n}\n");
}

#[test]
fn parse_propagate_and_preselect() {
    parse_ok("GADGET G {\n    PROPAGATE LX0 FROM LZ0 IN0.DS3 rec[-2] FLIP\n    PRESELECT rec[-1] 0\n}\n");
}

#[test]
fn parse_compose_and_gadget_application() {
    parse_ok(
        "COMPOSE C {\n    INPUT Code 0 1 2\n    Idle IN(0 1 2) OUT(0 1 2)\n    Idle 0\n    OUTPUT Code 0 1 2\n}\n",
    );
}

#[test]
fn parse_program_virtual_correction() {
    parse_ok("PROGRAM P {\n    VIRTUAL X0*Y1 3\n}\n");
}

#[test]
fn parse_instruction_tag() {
    parse_ok("GADGET G {\n    TICK[hello\\Cworld]\n    X_ERROR[t](0.1) 5 6\n}\n");
}

fn parse_err(input: &str) {
    assert!(
        DeqParser::parse(Rule::deq_file, input).is_err(),
        "expected a parse error, but this parsed:\n{input}"
    );
}

/// A keyword-led statement that fails to parse must raise, not fall back to the
/// generic `instruction` rule: PEG backtracking would otherwise re-read the
/// keyword as a gate name and resume mid-statement, turning a typo into two
/// statements instead of an error.
#[test]
fn keyword_led_statement_does_not_fall_back_to_instruction() {
    parse_err("COMPOSE C {\n    INPUT INPUT SurfaceCode 0\n}\n");
    parse_err("PROGRAM P {\n    INPUT INPUT SurfaceCode 0\n}\n");
    parse_err("COMPOSE C {\n    REPEAT\n}\n");
    parse_err("PROGRAM P {\n    VIRTUAL\n}\n");
}

/// A keyword must not match the prefix of a longer identifier: `CHECKM0` is one
/// instruction name, not `CHECK` followed by the target `M0`.
#[test]
fn keyword_does_not_match_identifier_prefix() {
    let input = "GADGET G {\n    CHECKM0 1\n}\n";
    let parsed = DeqParser::parse(Rule::deq_file, input).unwrap_or_else(|e| panic!("parse failed:\n{input}\n{e}"));
    assert!(
        parsed.as_str().contains("CHECKM0"),
        "CHECKM0 must stay a single instruction name"
    );
    assert!(
        !format!("{parsed:?}").contains("check_statement"),
        "CHECKM0 must not parse as a check_statement"
    );
}

/// Keywords are reserved per body context: `CHECK` and `PRESELECT` lead no
/// COMPOSE statement, so they stay usable as instruction names there.
#[test]
fn keywords_are_reserved_per_context() {
    parse_ok("COMPOSE C {\n    CHECK\n    PRESELECT\n}\n");
    parse_ok("GADGET G {\n    INPUT SurfaceCode 0\n}\n");
    parse_ok("GADGET Checker {\n}\nCOMPOSE C {\n    Checker()\n}\n");
}

/// An empty gadget-application argument list is the two-character token `()`,
/// not a parenthesis pair with implicit whitespace between: `Foo ()` is fine,
/// `Foo( )` is not.
#[test]
fn empty_gadget_application_arguments_reject_inner_whitespace() {
    parse_ok("GADGET Foo {\n}\nCOMPOSE C {\n    Foo()\n}\n");
    parse_ok("GADGET Foo {\n}\nCOMPOSE C {\n    Foo ()\n}\n");
    parse_err("GADGET Foo {\n}\nCOMPOSE C {\n    Foo( )\n}\n");
}
