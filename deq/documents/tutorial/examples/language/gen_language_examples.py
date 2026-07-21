import json
import os
import sys
import deq_runtime
from deq.cli.jit import transpile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from snippet_utils import extract_block, write_snippet

this_dir = os.path.dirname(__file__)

examples = [
    "01_prepare_measure.deq",
    "02_noisy.deq",
    "03_with_idle.deq",
    "04_manual_checks.deq",
    "05_import.deq",
]

for example in examples:
    path = os.path.join(this_dir, example)
    out = os.path.join(this_dir, f"{example}.jit")
    print(f"Transpiling {example}...")
    transpile(path, out=out, program="Simulation", jobs=1)
    print(f"  -> {out}")
    print(f"  -> {out}.txt")


# ── Extract snippets from source files ─────────────────────────────────

with open(os.path.join(this_dir, "01_prepare_measure.deq"), encoding="utf-8") as f:
    src_01 = f.read()
write_snippet(
    os.path.join(this_dir, "snippet_prepare.deq"),
    extract_block(src_01, "GADGET", "PrepareZ"),
)
write_snippet(
    os.path.join(this_dir, "snippet_measure.deq"),
    extract_block(src_01, "GADGET", "MeasureZ"),
)
write_snippet(
    os.path.join(this_dir, "snippet_program.deq"),
    extract_block(src_01, "PROGRAM", "Simulation"),
)

with open(os.path.join(this_dir, "03_with_idle.deq"), encoding="utf-8") as f:
    src_03 = f.read()
write_snippet(
    os.path.join(this_dir, "snippet_idle.deq"),
    extract_block(src_03, "GADGET", "Idle"),
)


# also check that the commands in the tutorial actually runs
deq_runtime.cli_run(
    "server",
    "--addr",
    "[::]:0",
    "--decoder",
    "black-box-relay-bp",
    "--coordinator",
    "window",
    "--coordinator-config",
    json.dumps({"buffer_radius": 3}),
    "--controller",
    "jit",
    "--controller-config",
    json.dumps({"filepath": os.path.join(this_dir, "03_with_idle.deq.jit")}),
    "--simulator",
    "jit-static",
    "--simulator-config",
    json.dumps(
        {
            "filepath": os.path.join(this_dir, "03_with_idle.stim"),
            "jit_library_filepath": os.path.join(this_dir, "03_with_idle.deq.jit"),
            "shots": 100,  # a small number of shots just for testing
            "seed": 123,
        }
    ),
)
