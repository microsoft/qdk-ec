"""Inferred loss compiler tests, driven by the annotated file text.

These assert on the ``annotate`` output (and its byte-equal round-trip) rather
than on loss-model proto internals, which keeps them compact and close to what
a user inspects.
"""

from google.protobuf.json_format import MessageToDict

from deq.circuit.parser import parse
from deq.cli.strip_tags import strip_jit_library
from deq.transpiler.jit_annotate import annotate as render_annotated
from deq.transpiler.jit_library_builder import build_jit_library
from deq.transpiler.loss import (
    GateLossPolicy,
    NeutralAtomLossModel,
    QdkLossConfig,
    TrappedIonLossModel,
)


class _PropagatingNeutralAtomLossModel(NeutralAtomLossModel):
    config = QdkLossConfig(gate_policies=(("cz", GateLossPolicy.PROPAGATE),))

# The loss-decoding paper's data-loss CNOT chain: qubit 0 is lost at five points
# along its lifetime, pushed through CNOTs, then measured.
_PAPER = """
CODE Reg [[5,5,1]] {
    LOGICAL X0 Z0
    LOGICAL X1 Z1
    LOGICAL X2 Z2
    LOGICAL X3 Z3
    LOGICAL X4 Z4
}
GADGET PaperDataLoss {
    INPUT Reg 0 1 2 3 4
    LOSS_ERROR(0.1) 0
    CX 1 0
    LOSS_ERROR(0.1) 0
    CX 0 2
    LOSS_ERROR(0.1) 0
    CX 0 3
    LOSS_ERROR(0.1) 0
    CX 4 0
    LOSS_ERROR(0.1) 0
    M 0
    OUTPUT Reg 0 1 2 3 4
}
"""

_MINIMAL = """
CODE Q [[1,1,1]] { LOGICAL X0 Z0 }
GADGET Loss1 {
    INPUT Q 0
    LOSS_ERROR(0.2) 0
    M 0
    OUTPUT Q 0
}
"""


def _loss_lines(rendered: str) -> list[str]:
    return [
        line.strip()
        for line in rendered.splitlines()
        if line.lstrip().startswith("LOSS(")
    ]


def test_forward_loss_annotation_round_trips() -> None:
    for source in (_PAPER, _MINIMAL):
        qfile = parse(source)
        rendered = render_annotated(qfile)
        original, _ = strip_jit_library(build_jit_library(qfile))
        annotated, _ = strip_jit_library(build_jit_library(parse(rendered)))
        assert original.SerializeToString() == annotated.SerializeToString()


def test_jit_library_records_loss_strategy_metadata() -> None:
    library = build_jit_library(parse(_MINIMAL))

    assert (
        MessageToDict(library.metadata)["loss_strategy"]
        == NeutralAtomLossModel.config.to_json_object()
    )


def test_neutral_atom_model_emits_source_envelope_and_herald_metadata() -> None:
    gadget = build_jit_library(
        parse(_MINIMAL), loss_model=NeutralAtomLossModel()
    ).gadget_types[0]
    (loss,) = gadget.base.loss_model.losses

    assert list(loss.loss_measurements) == [0]
    assert list(loss.source_errors)
    assert list(gadget.errors)


def test_neutral_atom_model_relocates_loss_through_swap() -> None:
    source = """
    CODE Pair [[2,2,1]] {
        LOGICAL X0 Z0
        LOGICAL X1 Z1
    }
    GADGET G {
        INPUT Pair 0 1
        LOSS_ERROR(0.1) 0
        SWAP 0 1
        M 0 1
        OUTPUT Pair 0 1
    }
    """
    gadget = build_jit_library(
        parse(source), loss_model=NeutralAtomLossModel()
    ).gadget_types[0]

    assert list(gadget.base.loss_model.losses[0].loss_measurements) == [1]


def test_propagated_branches_keep_causal_links_to_later_sources() -> None:
    source = """
    GADGET G {
        LOSS_ERROR(0.1) 0
        CZ 0 1
        LOSS_ERROR(0.2) 0
        LOSS_ERROR(0.3) 1
        M 0 1
    }
    """
    gadget = build_jit_library(
        parse(source), loss_model=_PropagatingNeutralAtomLossModel()
    ).gadget_types[0]
    parent, first_branch, second_branch = gadget.base.loss_model.losses

    assert list(parent.child_losses) == [1, 2]
    assert list(parent.loss_measurements) == []
    assert list(first_branch.loss_measurements) == [0]
    assert list(second_branch.loss_measurements) == [1]


def test_compose_preserves_propagated_branch_successor() -> None:
    source = """
    CODE C [[1,1,1]] { LOGICAL X0 Z0 }
    GADGET A {
        LOSS_ERROR(0.1) 0
        CZ 0 1
        LOSS_ERROR(0.2) 0
        R 1
        OUTPUT C 0
    }
    GADGET B {
        INPUT C 0
        M 0
    }
    COMPOSE Chain {
        A 0
        B 0
    }
    """
    library = build_jit_library(
        parse(source), loss_model=_PropagatingNeutralAtomLossModel()
    )
    chain = next(gadget for gadget in library.gadget_types if gadget.base.name == "Chain")
    parent, child = chain.base.loss_model.losses

    assert list(parent.child_losses) == [1]
    assert list(parent.loss_measurements) == []
    assert list(child.loss_measurements) == [0]


def test_trapped_ion_cz_residual_becomes_continuation_error() -> None:
    source = """
    CODE Pair [[2,2,1]] {
        LOGICAL X0 Z0
        LOGICAL X1 Z1
    }
    GADGET G {
        INPUT Pair 0 1
        LOSS_ERROR(0.1) 0
        CZ 0 1
        M 0
        OUTPUT Pair 0 1
    }
    """
    gadget = build_jit_library(
        parse(source), loss_model=TrappedIonLossModel()
    ).gadget_types[0]
    (loss,) = gadget.base.loss_model.losses
    continuation_errors = [gadget.errors[index] for index in loss.continuation_errors]

    assert any(list(error.base.residual) == [2] for error in continuation_errors)


def test_paper_loss_chain_has_one_herald_at_the_end() -> None:
    rendered = render_annotated(parse(_PAPER))
    # Source losses (one per LOSS_ERROR); the entering-loss ``LOSS(IN...)`` lines
    # for the input port are counted separately below.
    source_lines = [
        line for line in _loss_lines(rendered) if not line.startswith("LOSS(IN")
    ]
    # One declared source loss per LOSS_ERROR.
    assert len(source_lines) == 5
    # The chain is heralded by the single terminal measurement M0: only the
    # last loss detects it directly; the rest inherit it through child links.
    assert sum("M0" in line for line in source_lines) == 1
    assert "M0" in source_lines[-1]


def test_lost_controlled_pauli_target_dephases_control_in_z() -> None:
    for gate in ("CX", "CY"):
        source = f"""
        CODE Pair [[2,2,1]] {{
            LOGICAL X0 Z0
            LOGICAL X1 Z1
        }}
        GADGET G {{
            INPUT Pair 0 1
            LOSS_ERROR(0.1) 1
            {gate} 0 1
            M 1
            OUTPUT Pair 0 1
        }}
        """
        gadget = build_jit_library(parse(source)).gadget_types[0]
        (loss,) = gadget.base.loss_model.losses
        generator_errors = [
            gadget.errors[index]
            for index in (*loss.source_errors, *loss.continuation_errors)
        ]

        assert all(error.base.tag.endswith("1") for error in generator_errors)
        residual_span = {frozenset()}
        for error in generator_errors:
            residual = frozenset(error.base.residual)
            residual_span.update(
                candidate ^ residual for candidate in tuple(residual_span)
            )
        assert frozenset({0}) in residual_span
        assert frozenset({1}) not in residual_span


def test_lost_record_control_adds_target_continuation_error() -> None:
    source = """
    CODE Pair [[2,2,1]] {
        LOGICAL X0 Z0
        LOGICAL X1 Z1
    }
    GADGET G {
        INPUT Pair 0 1
        LOSS_ERROR(0.1) 1
        M 1
        CX rec[-1] 0
        OUTPUT Pair 0 1
    }
    """
    gadget = build_jit_library(parse(source)).gadget_types[0]
    (loss,) = gadget.base.loss_model.losses
    continuation_errors = [gadget.errors[index] for index in loss.continuation_errors]

    assert list(loss.loss_measurements) == [0]
    assert any(error.base.tag.endswith("X0") for error in continuation_errors)


def test_loss_error_is_followed_by_authoritative_loss_metadata() -> None:
    rendered = render_annotated(parse(_PAPER))
    assert (
        "@SIMULATE_ONLY\n"
        "    LOSS_ERROR(0.1) 0\n"
        "    LOSS(0.1) SE0 SE1 CE0 CE2 L1  # L0"
    ) in rendered
    assert "ERROR(0.0) OUT0.LX2 OUT0.LX3 OUT0.LX4  # E0" in rendered


def test_continuation_generators_reference_canonical_error_rows() -> None:
    lines = render_annotated(parse(_PAPER)).splitlines()
    loss_line = next(
        line for line in lines if line.strip().startswith("LOSS(0.1) SE0 SE1")
    )
    assert "CE2" in loss_line
    assert any(line.strip().endswith("# E2") for line in lines)


def test_degenerate_generators_share_canonical_error_indices() -> None:
    rendered = render_annotated(parse(_PAPER))
    source_losses = [
        line.strip()
        for line in rendered.splitlines()
        if line.lstrip().startswith("LOSS(0.1)")
    ]
    assert "CE4 CE5" in source_losses[3]
    assert "CE4 CE5" in source_losses[4]


def test_multi_target_gates_and_swap_round_trip() -> None:
    # Multi-pair CX (one merged decomposed layer) and a native SWAP (loss
    # relabelling) both used to overflow the loss-injection boundary and trip
    # the "each mechanism must be injected exactly once" assertion. Verify the
    # forward transpiler now annotates and round-trips them byte-for-byte.
    qfile = parse(
        """
        CODE C [[6,1,1]] { LOGICAL X0*X1*X2*X3*X4*X5 Z0 }
        GADGET G {
            INPUT C 0 1 2 3 4 5
            RZ 3 4
            CX 0 3 1 4
            SWAP 3 0 4 1
            LOSS_ERROR(0.1) 0 1
            CX 2 0 2 1
            MZ 3 4
            OUTPUT C 0 1 2 3 4 5
        }
        """
    )
    rendered = render_annotated(qfile)
    orig, _ = strip_jit_library(build_jit_library(qfile))
    anno, _ = strip_jit_library(build_jit_library(parse(rendered)))
    assert orig.SerializeToString() == anno.SerializeToString()


_PAPER_WITH_NOISE = """
CODE Reg [[5,5,1]] {
    LOGICAL X0 Z0
    LOGICAL X1 Z1
    LOGICAL X2 Z2
    LOGICAL X3 Z3
    LOGICAL X4 Z4
}
GADGET PaperDataLossNoise {
    INPUT Reg 0 1 2 3 4
    LOSS_ERROR(0.1) 0
    Z_ERROR(0.001) 0
    CX 1 0
    LOSS_ERROR(0.1) 0
    CX 0 2
    LOSS_ERROR(0.1) 0
    CX 0 3
    LOSS_ERROR(0.1) 0
    CX 4 0
    LOSS_ERROR(0.1) 0
    M 0
    OUTPUT Reg 0 1 2 3 4
}
"""


def test_regular_noise_errors_are_emitted_beside_source() -> None:
    lines = render_annotated(parse(_PAPER_WITH_NOISE)).splitlines()
    error_line = next(
        i for i, line in enumerate(lines) if "ERROR(0.001)" in line and "# E0" in line
    )
    output_line = next(
        i for i, line in enumerate(lines) if line.strip().startswith("OUTPUT ")
    )
    assert error_line < output_line


def test_loss_generators_dedup_against_regular_errors() -> None:
    rendered = render_annotated(parse(_PAPER_WITH_NOISE))
    # The Z source generator of the first loss has the same footprint as the
    # Z_ERROR(0.001) noise (error E0), so the LOSS line references E0 instead
    # of appending a duplicate generator row.
    first_loss = next(
        line.strip()
        for line in rendered.splitlines()
        if line.lstrip().startswith("LOSS(0.1)")
    )
    assert "SE0" in first_loss


def test_paper_with_noise_round_trips_byte_equivalent() -> None:
    qfile = parse(_PAPER_WITH_NOISE)
    rendered = render_annotated(qfile)
    orig, _ = strip_jit_library(build_jit_library(qfile))
    anno, _ = strip_jit_library(build_jit_library(parse(rendered)))
    assert orig.SerializeToString() == anno.SerializeToString()


def test_user_noise_and_loss_errors_round_trip_in_source_order() -> None:
    qfile = parse(
        """
        CODE C [[2,2,1]] {
            LOGICAL X0 Z0
            LOGICAL X1 Z1
        }
        GADGET G {
            INPUT C 0 1
            LOSS_ERROR(0.1) 0
            Z_ERROR(0.001) 0
            ERROR(0.002) LX1
            M 0
            OUTPUT C 0 1
        }
        """
    )

    rendered = render_annotated(qfile)
    original, _ = strip_jit_library(build_jit_library(qfile))
    annotated, _ = strip_jit_library(build_jit_library(parse(rendered)))

    lines = rendered.splitlines()
    declared_error_line = next(
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("ERROR(0.002)")
    )
    measurement_line = next(
        index for index, line in enumerate(lines) if line.strip() == "M 0"
    )
    noise_error_line = next(
        index
        for index, line in enumerate(lines)
        if line.lstrip().startswith("ERROR(0.001)")
    )
    assert noise_error_line < declared_error_line
    assert declared_error_line < measurement_line
    assert lines[noise_error_line].endswith("# E0")
    assert lines[declared_error_line].endswith("# E1")
    assert original.SerializeToString() == annotated.SerializeToString()


def test_loss_generators_are_probability_zero() -> None:
    rendered = render_annotated(parse(_PAPER))
    # Loss-induced generators are footprint-only: activated by the herald at
    # runtime, so their static ERROR rows carry probability 0.
    error_lines = [
        line.strip()
        for line in rendered.splitlines()
        if line.lstrip().startswith("ERROR(")
    ]
    assert error_lines
    assert all(line.startswith("ERROR(0.0)") for line in error_lines)


def test_minimal_forward_loss_emits_single_loss() -> None:
    rendered = render_annotated(parse(_MINIMAL))
    # One source loss from the single LOSS_ERROR; the input port additionally
    # yields one ``LOSS(IN...)`` entering-loss line.
    source_lines = [
        line for line in _loss_lines(rendered) if not line.startswith("LOSS(IN")
    ]
    assert len(source_lines) == 1
    assert source_lines[0].startswith("LOSS(0.2)")
    assert "M0" in source_lines[0]


def test_entering_loss_generator_uses_input_loss_origin() -> None:
    rendered = render_annotated(parse(_MINIMAL))
    assert "ERROR(0.0) OUT0.LZ0  # E0" in rendered
    assert "LOSS(IN0.L0) CE0 L0" in rendered
