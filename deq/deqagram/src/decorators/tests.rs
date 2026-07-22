use super::*;
use crate::ast::{Definition, DeqFile};

/// Parses `src` and returns the body of its single GADGET definition.
fn gadget_body(src: &str) -> Vec<Spanned<GadgetStatement>> {
    let ast: DeqFile = src.parse().unwrap();
    let Definition::Gadget(g) = ast.definitions.into_iter().next().unwrap().node else {
        panic!("expected a GADGET");
    };
    g.body
}

#[test]
fn attaches_to_following_statement() {
    let body = gadget_body("GADGET G {\n    @SIMULATE_ONLY\n    M 0 1\n}\n");
    let (attached, dangling) = attach_gadget_body(body);

    assert!(dangling.is_empty());
    assert_eq!(attached.len(), 1);
    assert_eq!(attached[0].decorators.len(), 1);
    assert_eq!(attached[0].decorators[0].name, "SIMULATE_ONLY");
    assert!(matches!(
        attached[0].statement.node,
        Attached::Statement(GadgetStatement::Instruction(_))
    ));
}

#[test]
fn multiple_decorators_accumulate_in_order() {
    let body = gadget_body("GADGET G {\n    @A\n    @B(1)\n    R 0\n}\n");
    let (attached, dangling) = attach_gadget_body(body);

    assert!(dangling.is_empty());
    assert_eq!(attached.len(), 1);
    let names: Vec<_> = attached[0].decorators.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, ["A", "B"]);
}

#[test]
fn undecorated_statements_get_empty_lists() {
    let body = gadget_body("GADGET G {\n    R 0\n    @X\n    M 1\n}\n");
    let (attached, _) = attach_gadget_body(body);

    assert_eq!(attached.len(), 2);
    assert!(attached[0].decorators.is_empty()); // R 0
    assert_eq!(attached[1].decorators.len(), 1); // @X M 1
    assert_eq!(attached[1].decorators[0].name, "X");
}

#[test]
fn decorators_removed_from_the_statement_stream() {
    // Parsed body has 4 items (2 decorators + 2 statements); attached has 2.
    let body = gadget_body("GADGET G {\n    @A\n    R 0\n    @B\n    M 1\n}\n");
    assert_eq!(body.len(), 4);
    let (attached, _) = attach_gadget_body(body);
    assert_eq!(attached.len(), 2);
    // No attached statement is a bare decorator.
    assert!(
        attached
            .iter()
            .all(|d| matches!(d.statement.node, Attached::Statement(_) | Attached::Repeat { .. }))
    );
}

#[test]
fn dangling_decorator_before_close_brace_is_returned() {
    let body = gadget_body("GADGET G {\n    R 0\n    @LEFTOVER\n}\n");
    let (attached, dangling) = attach_gadget_body(body);

    assert_eq!(attached.len(), 1); // R 0, no decorators
    assert!(attached[0].decorators.is_empty());
    assert_eq!(dangling.len(), 1);
    assert_eq!(dangling[0].node.name, "LEFTOVER");
}

#[test]
fn all_decorators_dangling_when_body_has_no_statement() {
    let body = gadget_body("GADGET G {\n    @A\n    @B\n}\n");
    let (attached, dangling) = attach_gadget_body(body);

    assert!(attached.is_empty());
    let names: Vec<_> = dangling.iter().map(|d| d.node.name.as_str()).collect();
    assert_eq!(names, ["A", "B"]);
}

#[test]
fn repeat_body_is_attached_recursively() {
    let body = gadget_body("GADGET G {\n    @OUTER\n    REPEAT 2 {\n        @INNER\n        R 0\n    }\n}\n");
    let (attached, dangling) = attach_gadget_body(body);

    assert!(dangling.is_empty());
    assert_eq!(attached.len(), 1);
    assert_eq!(attached[0].decorators[0].name, "OUTER");

    // The REPEAT's inner body is attached in place — @INNER folds onto R 0.
    let Attached::Repeat { count, body: inner } = &attached[0].statement.node else {
        panic!("expected REPEAT");
    };
    assert_eq!(*count, 2);
    assert_eq!(inner.len(), 1);
    assert_eq!(inner[0].decorators[0].name, "INNER");
    assert!(matches!(
        inner[0].statement.node,
        Attached::Statement(GadgetStatement::Instruction(_))
    ));
}

#[test]
fn dangling_inside_nested_repeat_bubbles_up_with_its_span() {
    let src = "GADGET G {\n    REPEAT 2 {\n        M 0\n        @DEEP\n    }\n}\n";
    let body = gadget_body(src);
    let (attached, dangling) = attach_gadget_body(body);

    // The REPEAT attached cleanly (M 0, no decorators); the dangling @DEEP
    // surfaced at the top level.
    assert_eq!(attached.len(), 1);
    assert_eq!(dangling.len(), 1);
    assert_eq!(dangling[0].node.name, "DEEP");
    // Its span still points at the original `@DEEP` text. (The `decorator` rule
    // ends in an optional argument list, so pest's span over-covers the trailing
    // whitespace — the start is what matters for diagnostics.)
    assert!(src[dangling[0].span.start..dangling[0].span.end].starts_with("@DEEP"));
}

#[test]
fn compose_and_program_bodies_attach_too() {
    let ast: DeqFile = "\
COMPOSE C {
    @CDECO
    Idle IN(0) OUT(0)
}

PROGRAM P {
    @PDECO
    ASSERT_EQ rec[-1] 0
}
"
    .parse()
    .unwrap();
    let mut defs = ast.definitions.into_iter();

    let Definition::Compose(c) = defs.next().unwrap().node else {
        panic!("expected COMPOSE");
    };
    let (c_attached, c_dangling) = attach_compose_body(c.body);
    assert!(c_dangling.is_empty());
    assert_eq!(c_attached[0].decorators[0].name, "CDECO");

    let Definition::Program(p) = defs.next().unwrap().node else {
        panic!("expected PROGRAM");
    };
    let (p_attached, p_dangling) = attach_program_body(p.body);
    assert!(p_dangling.is_empty());
    assert_eq!(p_attached[0].decorators[0].name, "PDECO");
}
