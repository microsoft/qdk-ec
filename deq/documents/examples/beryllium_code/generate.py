import os
import time
from glob import glob
from concurrent.futures import ProcessPoolExecutor
from deq.cli.annotate import annotate
from deq.cli.jit import transpile

this_dir = os.path.dirname(__file__)

tasks = []
for deq_file in sorted(glob(os.path.join(this_dir, "*.deq"))):
    basename = os.path.basename(deq_file)
    if basename == "code.deq":
        continue
    if basename.endswith(".annotated.deq"):
        continue
    name = basename.removesuffix(".deq")
    out_annotated = os.path.join(this_dir, f"{name}.annotated.deq")
    out_jit = os.path.join(this_dir, f"{name}.deq.jit")
    tasks.append((deq_file, out_annotated, out_jit, basename, name))

jobs = max((os.cpu_count() or 1) - 2, 1)
print(f"Processing {len(tasks)} files with {jobs} workers...")


def _process(args: tuple[str, str, str, str, str]) -> str:
    deq_file, out_annotated, out_jit, basename, name = args
    t0 = time.perf_counter()
    annotate(deq_file, out=out_annotated)
    transpile(deq_file, out=out_jit)
    elapsed = time.perf_counter() - t0
    return f"{basename}  ({elapsed:.2f}s)"


with ProcessPoolExecutor(max_workers=jobs) as pool:
    for msg in pool.map(_process, tasks):
        print(f"  Done: {msg}")
