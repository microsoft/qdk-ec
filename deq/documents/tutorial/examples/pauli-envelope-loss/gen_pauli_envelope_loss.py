"""Generate and verify annotations for the Pauli-envelope loss tutorial.

Run from the repository's ``deq/`` directory with::

    python documents/tutorial/examples/pauli-envelope-loss/gen_pauli_envelope_loss.py

In addition to ``deq annotate``'s byte-equivalent round-trip check, this script
asserts the metadata fragments explained by the chapter. Do not edit generated
``*.annotated.deq`` files by hand.
"""

from pathlib import Path

from deq.cli.annotate import annotate


HERE = Path(__file__).resolve().parent

EXAMPLES: tuple[tuple[str, str, str, tuple[str, ...], int, int], ...] = (
    (
        "01_single_qubit.deq",
        "01_single_qubit.annotated.deq",
        "neutral-atom",
        ("LOSS(0.1) SE0 CE0 M0", "ERROR(0.0) R0"),
        1,
        1,
    ),
    (
        "02_platform_cz.deq",
        "02_platform_cz.neutral_atom.annotated.deq",
        "neutral-atom",
        ("LOSS(0.1) SE0 CE1 M0",),
        1,
        2,
    ),
    (
        "02_platform_cz.deq",
        "02_platform_cz.trapped_ion.annotated.deq",
        "trapped-ion",
        ("LOSS(0.1) SE0 CE1 CE2 M0", "ERROR(0.0) R1"),
        1,
        3,
    ),
    (
        "02_platform_cz.deq",
        "02_platform_cz.custom.annotated.deq",
        str(HERE / "custom_loss_model.py"),
        ("LOSS(0.1) SE0 CE1 CE2 M0 M1",),
        1,
        3,
    ),
    (
        "03_cross_gadget.deq",
        "03_cross_gadget.annotated.deq",
        "neutral-atom",
        ("LOSS(0.1) SE0 SE1 CE0 CE1 OUT0.L0", "LOSS(IN0.L0) CE0 M0"),
        1,
        3,
    ),
    (
        "04_four_cx.deq",
        "04_four_cx.annotated.deq",
        "neutral-atom",
        ("L1  # L0", "L2  # L1", "L3  # L2", "L4  # L3", "M0  # L4"),
        5,
        15,
    ),
    (
        "05_backend_contract.deq",
        "05_backend_contract.annotated.deq",
        "neutral-atom",
        ("LOSS(1.0)", "OUT0.L0", "LOSS(IN0.L0)", "M0"),
        1,
        2,
    ),
)


def main() -> None:
    for (
        source_name,
        output_name,
        loss_model,
        expected_fragments,
        expected_source_losses,
        expected_errors,
    ) in EXAMPLES:
        print(f"Annotating {source_name} with {loss_model}...")
        output = HERE / output_name
        annotate(
            str(HERE / source_name),
            out=str(output),
            loss_model=loss_model,
        )
        rendered = output.read_text(encoding="utf-8")
        missing = [
            fragment for fragment in expected_fragments if fragment not in rendered
        ]
        if missing:
            raise AssertionError(
                f"{output_name} no longer contains documented metadata: {missing}"
            )
        source_losses = [
            line
            for line in rendered.splitlines()
            if line.lstrip().startswith("LOSS(")
            and not line.lstrip().startswith("LOSS(IN")
        ]
        envelope_errors = [
            line
            for line in rendered.splitlines()
            if line.lstrip().startswith("ERROR(0.0)")
        ]
        if (
            len(source_losses) != expected_source_losses
            or len(envelope_errors) != expected_errors
        ):
            raise AssertionError(
                f"{output_name} must contain {expected_source_losses} source "
                f"losses and {expected_errors} generator footprints, got "
                f"{len(source_losses)} and {len(envelope_errors)}"
            )


if __name__ == "__main__":
    main()
