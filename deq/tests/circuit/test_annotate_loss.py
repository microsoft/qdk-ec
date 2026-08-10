"""Annotator tests for the ``LOSS`` block derived from the binary loss model."""

from deq.circuit.parser import parse
from deq.cli.strip_tags import strip_jit_library
from deq.transpiler.jit_annotate import annotate as render_annotated
from deq.transpiler.jit_library_builder import build_jit_library

_FAITHFUL_LOSS_SRC = """
CODE C[[3,1,1]] {
    LOGICAL X0 Z0
    STABILIZER
}

GADGET G {
    INPUT C 0 1 2
    M 0
    M 1
    OUTPUT C 0 1 2
    ERROR(0) LX0
    ERROR(0) LZ0
    LOSS(0.1) SE0 CE1 L1 OUT0.L2 M1
    LOSS(0.2) SE1 M0
    LOSS(IN0.L1) CE0 L0 OUT0.L0 M0
}
"""


def test_loss_block_is_emitted_with_labels() -> None:
    rendered = render_annotated(parse(_FAITHFUL_LOSS_SRC))
    # Error index labels for SE/CE references.
    assert "# E0" in rendered
    assert "# E1" in rendered
    # Source-loss labels for L<i> references.
    assert "# L0" in rendered
    assert "# L1" in rendered
    # The LOSS statements themselves.
    assert "LOSS(0.1) SE0 CE1 L1 OUT0.L2 M1" in rendered
    assert "LOSS(0.2) SE1 M0" in rendered
    assert "LOSS(IN0.L1) CE0 L0 OUT0.L0 M0" in rendered


def test_faithful_loss_round_trips_byte_equivalent() -> None:
    qfile = parse(_FAITHFUL_LOSS_SRC)
    rendered = render_annotated(qfile, keep_noise=False)
    orig, _ = strip_jit_library(build_jit_library(qfile))
    anno, _ = strip_jit_library(build_jit_library(parse(rendered)))
    assert orig.SerializeToString() == anno.SerializeToString()


def test_keep_noise_omits_expanded_loss_block() -> None:
    # Preserved noise reconstructs the loss model on re-transpilation, so the
    # expanded LOSS block is not emitted.
    rendered = render_annotated(parse(_FAITHFUL_LOSS_SRC), keep_noise=True)
    assert "LOSS(0.1)" not in rendered
    assert "# L0" not in rendered


def test_original_loss_statements_are_not_duplicated() -> None:
    rendered = render_annotated(parse(_FAITHFUL_LOSS_SRC))
    # Exactly the two source losses and one input loss appear once each.
    assert rendered.count("LOSS(0.1)") == 1
    assert rendered.count("LOSS(0.2)") == 1
    assert rendered.count("LOSS(IN0.L1)") == 1


def test_loss_free_gadget_emits_no_loss_block() -> None:
    rendered = render_annotated(
        parse(
            """
            CODE C[[1,1,1]] { LOGICAL X0 Z0 STABILIZER }
            GADGET G {
                INPUT C 0
                M 0
                OUTPUT C 0
            }
            """
        )
    )
    assert "LOSS(" not in rendered
    assert "# L0" not in rendered
