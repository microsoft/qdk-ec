"""LER sweep for the Steane-style EC tutorial chapter.

Produces sweep_ler.txt with results at two physical error rates,
comparing monolithic / buffer=0 (no phantom) / buffer=0 (with phantom).
"""

import os
import subprocess
import sys

this_dir = os.path.dirname(os.path.abspath(__file__))
source = os.path.join(this_dir, "steane_code.deq")

CONFIGS = [
    # (label, p, coordinator, coordinator_config, mako_extras)
    ("monolithic  p=1e-3", "0.001", "monolithic", "{}", []),
    ("buffer=0    p=1e-3 (no phantom)", "0.001", "window", '{"buffer_radius":0}', []),
    (
        "buffer=0    p=1e-3 (phantom)",
        "0.001",
        "window",
        '{"buffer_radius":0}',
        ["has_phantom=1"],
    ),
]

results = []
for label, p, coordinator, coord_config, mako_extras in CONFIGS:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    mako_args = ["--mako", f"p={p}"]
    for m in mako_extras:
        mako_args += ["--mako", m]

    cmd = [
        sys.executable,
        "-m",
        "deq",
        "simulate",
        "ler",
        source,
        "--program",
        "Minimal",
        "--skip-mako-warning",
        "--coordinator",
        coordinator,
        "--coordinator-config",
        coord_config,
        "--decoder",
        "black-box-relay-bp",
        "--batch-size",
        "100",
        "--shots",
        "100000000000",
        "--errors",
        "100",
    ] + mako_args

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=this_dir)
    output = result.stdout + result.stderr
    print(output)

    # Extract error rate line
    for line in output.splitlines():
        if "Error rate:" in line or "error rate:" in line:
            results.append(f"{label:40s}  {line.strip()}")
            break
    else:
        results.append(f"{label:40s}  (no result)")

# Write summary
out_path = os.path.join(this_dir, "sweep_ler.txt")
with open(out_path, "w") as f:
    f.write("Steane [[7,1,3]] code — LER sweep (100 target errors per data point)\n")
    f.write("=" * 80 + "\n\n")
    for r in results:
        f.write(r + "\n")
    f.write("\n")

print(f"\nSummary written to {out_path}")
print("\n".join(results))
