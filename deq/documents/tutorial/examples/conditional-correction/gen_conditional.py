"""Generate outputs for the conditional-correction tutorial chapter.

Runs the CLI commands referenced in ``conditional-correction.md`` so
that breaking changes are caught by ``make tutorial``:

* transpile + annotate every variant ``.deq`` file
  (``01_teleport_repropagate.deq``,
  ``02_teleport_compose_conditional.deq``,
  ``03_teleport_program_conditional.deq``);
* extract per-block snippets (``TeleportRepropagate``,
  ``TeleportConditional``, ``MeasureBell``, the PROGRAM bodies) so
  the chapter can highlight one block at a time;
* run ``deq sample`` on the COMPOSE-level CONDITIONAL memory program
  (``TeleportConditionalMemoryZ``), capturing 20 noiseless shots
  so the chapter can quote the sample output verbatim;
* run ``deq simulate ler`` on the same program for 20 shots and
  capture the ``Logical errors: 0`` line.

The Bell-pair teleportation timeline figure (``teleport_timeline.png``)
is rendered by the standalone ``teleport_timeline.py`` script in this
directory and committed to the repository, so it is NOT regenerated on
every ``make tutorial`` invocation.
"""

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snippet_utils import extract_block, run_cli, write_snippet  # noqa: E402

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# 1. Transpile + annotate every variant
# ---------------------------------------------------------------------------

print("Transpiling + annotating variant .deq files...")

VARIANTS: list[tuple[str, str]] = [
    # (filename, program-or-None)
    ("01_teleport_repropagate.deq", "TeleportRepropagateMemoryZ"),
    ("02_teleport_compose_conditional.deq", "TeleportConditionalMemoryZ"),
    ("03_teleport_program_conditional.deq", "TeleportProgramConditionalMemoryZ"),
]

for deq_name, program in VARIANTS:
    deq_path = os.path.join(THIS_DIR, deq_name)
    jit_out = deq_path + ".jit"
    transpile_args = ["transpile", deq_name, "--out", jit_out]
    if program is not None:
        transpile_args += ["--program", program]
    run_cli(f"transpile {deq_name}", transpile_args, cwd=THIS_DIR)

    annotate_out = deq_path.replace(".deq", ".annotated.deq")
    run_cli(
        f"annotate {deq_name}",
        ["annotate", deq_name, "--out", annotate_out],
        cwd=THIS_DIR,
    )


# ---------------------------------------------------------------------------
# 2. Extract per-block snippets for inline display in the chapter
# ---------------------------------------------------------------------------

print("Extracting snippets...")

with open(
    os.path.join(THIS_DIR, "00_teleportation_library.deq"), encoding="utf-8"
) as f:
    library_text = f.read()

write_snippet(
    os.path.join(THIS_DIR, "snippet_prepare_bell.deq"),
    extract_block(library_text, "COMPOSE", "PrepareBell"),
)
write_snippet(
    os.path.join(THIS_DIR, "snippet_measure_bell.deq"),
    extract_block(library_text, "COMPOSE", "MeasureBell"),
)

with open(
    os.path.join(THIS_DIR, "01_teleport_repropagate.deq"), encoding="utf-8"
) as f:
    repropagate_text = f.read()
with open(
    os.path.join(THIS_DIR, "02_teleport_compose_conditional.deq"), encoding="utf-8"
) as f:
    compose_cond_text = f.read()
with open(
    os.path.join(THIS_DIR, "03_teleport_program_conditional.deq"), encoding="utf-8"
) as f:
    program_cond_text = f.read()

write_snippet(
    os.path.join(THIS_DIR, "snippet_teleport_repropagate.deq"),
    extract_block(repropagate_text, "COMPOSE", "TeleportRepropagate"),
)
write_snippet(
    os.path.join(THIS_DIR, "snippet_teleport_conditional.deq"),
    extract_block(compose_cond_text, "COMPOSE", "TeleportConditional"),
)
write_snippet(
    os.path.join(THIS_DIR, "snippet_teleport_program_conditional.deq"),
    extract_block(
        program_cond_text, "PROGRAM", "TeleportProgramConditionalMemoryZ"
    ),
)


# ---------------------------------------------------------------------------
# 3. Run ``deq sample`` on the COMPOSE-level CONDITIONAL memory program
# ---------------------------------------------------------------------------
#
# 20 noiseless shots with a fixed seed produces a deterministic
# transcript that the chapter can quote verbatim, so the reader sees
# both the random ``MeasureBell`` outcomes and the always-zero final
# ``MeasureZ`` that proves the CONDITIONAL absorbed the frame
# correction.

print("Sampling 20 noiseless shots of TeleportConditionalMemoryZ...")

_, sample_stdout, _ = run_cli(
    "deq sample (compose CONDITIONAL)",
    [
        "sample",
        "02_teleport_compose_conditional.deq",
        "--program",
        "TeleportConditionalMemoryZ",
        "--shots",
        "20",
        "--noiseless",
        "--interpret",
        "--seed",
        "42",
    ],
    cwd=THIS_DIR,
)
write_snippet(
    os.path.join(THIS_DIR, "teleport_conditional_sample.txt"),
    sample_stdout,
)


# ---------------------------------------------------------------------------
# 4. Run ``deq simulate ler`` on the same program (20 shots noiseless)
# ---------------------------------------------------------------------------

print("Running 20-shot noiseless LER simulation...")

_, simulate_stdout, _ = run_cli(
    "deq simulate ler (compose CONDITIONAL)",
    [
        "simulate",
        "ler",
        "02_teleport_compose_conditional.deq",
        "--program",
        "TeleportConditionalMemoryZ",
        "--shots",
        "20",
        "--errors",
        "100",
        "--batch-size",
        "20",
        "--seed",
        "42",
        "--jobs",
        "1",
    ],
    cwd=THIS_DIR,
)
m_shots = re.search(r"Shots:\s+(\d+)", simulate_stdout)
m_errs = re.search(r"Logical errors:\s+(\d+)", simulate_stdout)
if m_shots is None or m_errs is None:
    raise RuntimeError(
        f"could not parse simulator output:\n{simulate_stdout}"
    )
simulate_summary = (
    "=== Simulation Results ===\n"
    f"  Shots:          {m_shots.group(1)}\n"
    f"  Logical errors: {m_errs.group(1)}\n"
)
write_snippet(
    os.path.join(THIS_DIR, "teleport_conditional_simulate.txt"),
    simulate_summary,
)

print("done.")
