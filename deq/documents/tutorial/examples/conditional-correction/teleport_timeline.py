"""Render the Bell-pair teleportation circuit figure used in the tutorial.

This script intentionally lives outside the ``gen_*.py`` family so that
``run_generators.py`` (which discovers ``gen_*.py`` recursively) does not
re-render the figure on every ``make tutorial`` invocation.  The output
PNG is committed to the repository; regenerate it manually with::

    python teleport_timeline.py

The circuit is drawn at the LOGICAL level: each ``q0`` / ``q1`` / ``q2``
line stands in for a 9-qubit distance-3 surface-code patch, and the
syndrome-extraction noise inside each patch is omitted so the timeline
emphasises the Bell-pair preparation, Bell-basis measurement, and
classically-conditioned Pauli corrections.

The circuit body matches the gadget body used by every variant in the
chapter (``TeleportRepropagate`` / ``TeleportConditional`` /
``TeleportProgramConditionalMemoryZ``):

* ``H 1`` + ``CX 1 2``         — prepare Bell pair on q1, q2
* ``CX 0 1`` + ``H 0``         — rotate q0/q1 into the Bell basis
* ``M 0`` (m_XX), ``M 1`` (m_ZZ) — Bell-basis measurement
* ``X 2`` if ``m_ZZ = 1``,
  ``Z 2`` if ``m_XX = 1``       — feedforward Pauli corrections

Qiskit's matplotlib drawer is used for a cleaner rendering than the
stim ``timeline-svg`` output.
"""

import os

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(THIS_DIR, "teleport_timeline.png")


def build_teleportation_circuit() -> QuantumCircuit:
    """Return the 3-qubit Bell-pair teleportation circuit."""
    q = QuantumRegister(3, "q")
    m_xx = ClassicalRegister(1, "m_XX")
    m_zz = ClassicalRegister(1, "m_ZZ")
    qc = QuantumCircuit(q, m_xx, m_zz)

    qc.h(1)
    qc.cx(1, 2)
    qc.barrier(label="Bell prep")

    qc.cx(0, 1)
    qc.h(0)
    qc.barrier(label="Bell meas")
    qc.measure(0, m_xx[0])
    qc.measure(1, m_zz[0])
    qc.barrier(label="feedforward")

    with qc.if_test((m_zz, 1)):
        qc.x(2)
    with qc.if_test((m_xx, 1)):
        qc.z(2)

    return qc


def generate_teleport_timeline_png(out_path: str = DEFAULT_OUTPUT) -> None:
    """Render the teleportation circuit as a PNG using qiskit's mpl drawer."""
    qc = build_teleportation_circuit()
    fig = qc.draw("mpl", style="iqp", fold=-1)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"    -> {os.path.basename(out_path)}")


if __name__ == "__main__":
    generate_teleport_timeline_png()
