"""The iceberg code [[k+2, k, 2]], constructed in Python and parameterized by k.

The iceberg family is the [[n, n-2, 2]] error-*detection* code (n even): the whole
codespace is fixed by just two stabilizers — the global X parity and the global Z
parity — so a block of n = k + 2 physical qubits carries k logical qubits. Being
distance 2 it detects any single fault but does not correct it, so it runs with
post-selection via per-parity detection flags (one per global parity).

This module builds the qodec for an arbitrary even ``k`` with
:func:`build_iceberg`. The committed ``iceberg.qodec.yaml`` in this directory is
the concrete ``k = 4`` (the [[6,4,2]] code) instance. It carries the same code,
logical instruction set, and gadgets that this builder produces for ``k = 4``,
but names a different physical layer: the committed bundle references the shared
``../stim.isa.yaml``, while the builder emits a compact physical instruction set
of its own so that it stays self-contained for any ``k``.

Run directly to build a few sizes and check each round-trips through disk:

    python iceberg.py

Reference: the [[n, n-2, 2]] "iceberg" code (M. Self et al. / Quantinuum);
see e.g. arXiv:2211.06703.
"""

from __future__ import annotations

from pathlib import Path

import qodec
from qodec.actions import Clifford, Observe, Stabilize
from qodec.codes import Code
from qodec.gadgets import Circuit, Encoding
from qodec.instructions import Block, BlockOperand, Instruction, InstructionSet


def _physical_isa() -> InstructionSet:
    """A compact Stim-compatible physical ISA (the bottom layer)."""
    qubit = [BlockOperand("qubit")]
    pair = [BlockOperand("qubit"), BlockOperand("qubit")]
    return InstructionSet(
        name="stim",
        description="Compact Stim-compatible physical ISA.",
        blocks=[Block("qubit", encodes=1)],
        instructions=[
            Instruction("R", description="Reset to |0>.", outputs=qubit, action=[Stabilize(["Z_0"])]),
            Instruction(
                "H",
                description="Hadamard.",
                inputs=qubit,
                outputs=qubit,
                action=[Clifford({"X_0": "Z_0", "Z_0": "X_0"})],
            ),
            Instruction(
                "CX",
                description="Controlled-X (CNOT).",
                inputs=pair,
                outputs=pair,
                action=[Clifford({"X_0": "X_0 X_1", "Z_1": "Z_0 Z_1"})],
            ),
            Instruction("M", description="Destructive Z-basis measurement.", inputs=qubit, action=[Observe(["Z_0"])]),
            Instruction("MX", description="Destructive X-basis measurement.", inputs=qubit, action=[Observe(["X_0"])]),
        ],
    )


def build_iceberg(k: int) -> qodec.Qodec:
    """Build the [[k+2, k, 2]] iceberg code as a two-layer qodec.

    ``k`` is the number of logical qubits and must be an even integer >= 2 (the
    iceberg family is defined for an even number of physical qubits n = k + 2, so
    that the global X and Z parities commute).
    """
    if k < 2 or k % 2 != 0:
        raise ValueError(f"the iceberg [[k+2, k, 2]] code requires an even k >= 2; got {k}")

    n = k + 2
    data = range(n)

    # ── Code: two global-parity stabilizers; k logical qubits paired to the two
    #    reference qubits 0 and n-1. ──────────────────────────────────────────
    code = Code(
        name="iceberg",
        description=(
            f"The [[{n},{k},2]] iceberg code: the global X parity and the global Z "
            f"parity, with {k} logical qubits each paired to reference qubits 0 and {n - 1}."
        ),
        stabilizers=[
            " ".join(f"X_{q}" for q in data),
            " ".join(f"Z_{q}" for q in data),
        ],
        x=[f"X_0 X_{i + 1}" for i in range(k)],
        z=[f"Z_{i + 1} Z_{n - 1}" for i in range(k)],
    )
    encoding = Encoding(code=code, support=[str(q) for q in data])

    # ── Logical ISA: prepare / detect / measure on the whole block. ──────────
    block = [BlockOperand("iceberg")]
    logical_isa = InstructionSet(
        name="iceberg",
        description=f"Logical instruction set for the [[{n},{k},2]] iceberg code ({k} logical qubits).",
        blocks=[Block("iceberg", encodes=k)],
        instructions=[
            Instruction(
                "prepare_z_all",
                description=f"Prepare the logical |{'0' * k}> state.",
                outputs=block,
                action=[Stabilize([f"Z_{i}" for i in range(k)])],
            ),
            Instruction(
                "idle",
                description="One detection round; two flags report whether each global parity changed.",
                inputs=block,
                outputs=block,
                flags=["detected_x", "detected_z"],
            ),
            Instruction(
                "measure_z_all",
                description="Destructive measurement of all logical Z observables.",
                inputs=block,
                action=[Observe([f"Z_{i}" for i in range(k)])],
            ),
        ],
    )

    physical_isa = _physical_isa()
    ancilla_x = n  # ancilla for the global X parity
    ancilla_z = n + 1  # ancilla for the global Z parity

    # ── prepare_z: reset all data (fixing the Z parity and every logical Z),
    #    then project by measuring the global X parity. ────────────────────────
    prepare_source = "\n".join(
        [
            "R " + " ".join(str(q) for q in range(n + 1)),
            f"H {ancilla_x}",
            "CX " + " ".join(f"{ancilla_x} {q}" for q in data),
            f"H {ancilla_x}",
            f"M {ancilla_x}",
        ]
    )
    prepare_z = qodec.Gadget(
        implements=logical_isa.instructions["prepare_z_all"],
        circuit=Circuit(physical_isa, prepare_source, format="stim"),
        outputs=[encoding],
        checks=[
            ["circuit.readouts[0]", "out[0].stabilizers[0]"],
            ["out[0].stabilizers[1]"],
        ],
    )

    # ── idle: measure both global parities. Two detection flags report whether each
    #    parity changed since the previous round; the carry-forward to the next
    #    round stays in `checks`. A consumer post-selects on either flag firing. ──
    idle_source = "\n".join(
        [
            f"R {ancilla_x} {ancilla_z}",
            f"H {ancilla_x}",
            "CX " + " ".join(f"{ancilla_x} {q}" for q in data),
            f"H {ancilla_x}",
            "CX " + " ".join(f"{q} {ancilla_z}" for q in data),
            f"M {ancilla_x} {ancilla_z}",
        ]
    )
    idle = qodec.Gadget(
        implements=logical_isa.instructions["idle"],
        circuit=Circuit(physical_isa, idle_source, format="stim"),
        inputs=[encoding],
        outputs=[encoding],
        checks=[
            ["circuit.readouts[0]", "out[0].stabilizers[0]"],
            ["circuit.readouts[1]", "out[0].stabilizers[1]"],
        ],
        readouts=[
            {"detected_x": ["circuit.readouts[0]", "in[0].stabilizers[0]"]},
            {"detected_z": ["circuit.readouts[1]", "in[0].stabilizers[1]"]},
        ],
    )

    # ── measure_z: read every data qubit in Z. The global Z parity becomes a
    #    deterministic check; each logical Z_i = Z_{i+1} Z_{n-1}. ──────────────
    measure_source = "M " + " ".join(str(q) for q in data)
    measure_z = qodec.Gadget(
        implements=logical_isa.instructions["measure_z_all"],
        circuit=Circuit(physical_isa, measure_source, format="stim"),
        inputs=[encoding],
        checks=[[f"circuit.readouts[0:{n}]", "in[0].stabilizers[1]"]],
        readouts=[
            [f"circuit.readouts[{i + 1}]", f"circuit.readouts[{n - 1}]", f"in[0].z[{i}]"] for i in range(k)
        ],
    )

    return qodec.Qodec(
        layers=[
            qodec.Layer(logical_isa, gadgets=[prepare_z, idle, measure_z]),
            qodec.Layer(physical_isa),
        ],
        name="iceberg",
        description=(
            f"The [[{n},{k},2]] iceberg detection code (built in Python): {k} logical "
            "qubits in one block, two global-parity stabilizers, post-selected via detection flags."
        ),
    )


def main() -> None:
    """Build several sizes and confirm each is a valid, reloadable qodec."""
    import tempfile

    print(build_iceberg(4))
    print()

    for k in (2, 4, 6):
        codec = build_iceberg(k)
        n = k + 2
        code = codec.codes["iceberg"]
        assert len(code.stabilizers) == 2, "iceberg has exactly two stabilizers"
        assert len(code.x) == len(code.z) == k, "k logical qubits"

        # Saving then reloading re-runs qodec's full validation — including the
        # stabilizer-commutation algebra — on the generated artifacts. (On save,
        # inline stim bodies are externalized to .stim sidecars, so the reloaded
        # circuit source is a path; the code and decoding surface round-trip
        # exactly, which is what we assert here.)
        with tempfile.TemporaryDirectory() as directory:
            codec.save(directory, single_file=True)
            reloaded = qodec.Qodec.load(Path(directory) / codec.manifest_filename)
        assert reloaded.codes["iceberg"].stabilizers == code.stabilizers
        assert reloaded.codes["iceberg"].x == code.x
        assert reloaded.codes["iceberg"].z == code.z
        assert reloaded.layers[0].gadgets.keys() == codec.layers[0].gadgets.keys()

        print(f"[[{n},{k},2]] iceberg: built, saved, reloaded, validated " f"({n} physical, {k} logical qubits)")


if __name__ == "__main__":
    main()
