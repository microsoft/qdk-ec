# JIT compiler performance regression benchmark

This is an **occasional, opt-in scalability check**, not a timing gate for every
edit. The [JIT tutorial](../../chapters/jit-basics.md#measuring-jit-scalability)
introduces the workload and how to interpret the measurements.

This benchmark measures the production shared JIT compilation path only:
instruction/check compilation, deferred error-model resolution, and incremental
compiler garbage collection. It constructs **no controller, coordinator,
decoder, or simulator** and supplies no measurement outcomes.

The workload is the existing noisy distance-7 surface-code lattice-surgery
example at p=0.003: each reusable CNOT subprogram performs ZZ and XX surgery,
ancilla readout, and remote conditional Paulis. The large fixture is the actual
connected W=200, 100-layer brickwall, not disconnected copies of a small circuit.
A W=4, two-layer case provides a small control and exercises the same type library.

## Run and check

From the repository root, with no competing benchmark or decoder process:

```bash
python deq/documents/tutorial/examples/jit-basics/generate_jit_benchmark_fixture.py
RUSTFLAGS="-C target-cpu=native" cargo bench -p deq-runtime \
    --no-default-features --features bench-internals --bench jit_compiler
python deq/documents/tutorial/examples/jit-basics/report_jit_benchmark.py
```

Generate the fixtures once before running the benchmark, and regenerate after
changes to the workload generator or compiler that produces the JIT library.
The report command writes a concise Markdown summary under `target/jit-benchmark-report/` and exits
nonzero if the performance budget fails. It rejects stale measurements, changed
source/fixtures, incomplete case selections, fewer than ten timing samples, and
non-finite or low-confidence estimates. `--criterion` selects an alternate
Criterion output directory. Do not compare native-tuned results across different
CPU models without recording that difference.

Each compilation has a 360-second safety limit. If the final cache-disabled
control reaches it, the benchmark exits nonzero and records timeout provenance.
Run the report command separately in that case: it accepts the explicit timeout
only when every other case has complete results. The timed-out row is a lower
bound, not an extrapolated mean or a successful sample. A new invocation clears
the previous run's completion record before measuring, preventing failed runs
from silently reusing an old baseline.

The requirement is that the **upper confidence bound (at least 95%) of mean
full-program compilation time / epoch count stays at or below 390 ms** with
error-template caching enabled. Disabling that cache is an experimental control,
not the default configuration or a separate performance gate. The
W=200 workload has 302 epochs, giving a 117.78-second compiler-only budget.
Ancilla preparation uses one SE round overlapped with the preceding block;
initial data preparation and idle blocks still use d rounds.
This checks aggregate JIT throughput, not worst-case epoch latency or the
end-to-end 390 ms hardware deadline. Controller, window-graph, and backend
regressions still require the separate timed-emulation study.

Criterion also retains statistical comparisons against previous local runs.
The absolute budget is deliberately tied to the workload requirement, rather
than claiming that measurements on different machines should match.

## What is timed

Each iteration spawns dependency-ordered compilation tasks on 24 Tokio workers,
resolves **every** error-model future, and retires completed JIT state. There is
no pending-gadget capacity that could prevent future context from arriving.
The harness checks gadget, check, and physical-error-hypothesis counts and
requires the final live-gadget count to be zero.

Fixture decompression, library loading, instruction cloning, dependency-channel
allocation, and compiler creation are outside the measured interval. Both modes
use a fresh compiler for every iteration:

- **Cache enabled:** ratio 10 gives the error-template LRU 240 entries for this
  library's 24 registered gadget types. It starts empty and reuses templates
  within the program.
- **Cache disabled:** ratio 0 retains no templates, so every dynamic error
  template is rebuilt.

Both use the compiler's ordinary ratio setting, not a benchmark-specific
switch. Production controllers expose the same `cache_limit_ratio`, default 10,
for their generated-type/identity caches and compiler templates. Each cache
has its own budget of ratio times registered gadget types; zero disables both
components' lookup caches. Loading types grows the compiler budget; ordinary
shot resets preserve it, and library resets clear it.

Input-check-map and terminal-error-model caches remain enabled in both modes.
This harness has no controller or decoder, so it measures only the compiler's
response to the shared ratio. In production, ratio zero also disables controller
type caching. Neither mode retains templates from a preceding full pass.
The benchmark includes the asynchronous scheduling needed to drive the compiler,
but no intentional input delays.

The `bench-internals` feature exposes only a hidden, unstable harness. It is not
enabled by normal runtime builds and does not change the public compiler API.
The harness calls the same crate-private shared path used by the controller;
benchmarking only the owned-output compatibility wrapper would miss that path.

## Fixtures and correctness

Compressed protobuf fixtures, their manifest, and inspectable `.deq` source text
are generated locally under this example's `fixtures/` directory, which is git ignored.
These generated inputs are not required to compile the Rust benchmark; executing
it without fixtures fails with the preparation command, rather than silently skipping.
Generate them using your source-installed DEQ Python environment:

```bash
python deq/documents/tutorial/examples/jit-basics/generate_jit_benchmark_fixture.py
python deq/documents/tutorial/examples/jit-basics/generate_jit_benchmark_fixture.py --check
RUSTFLAGS="-C target-cpu=native" cargo test -p deq-runtime \
    --no-default-features --features bench-internals --test jit_benchmark_test
```

Generation uses the local [lattice-surgery helper](lattice_surgery.py) with
physical export disabled; it verifies the expanded quantum wiring and remote
conditional-Pauli references. The JIT tools need no decoder-performance files.
The decoder-performance tools reuse this helper for the same workload, keeping
decoder execution, timing, and resource monitoring outside the JIT example.

### Measurement-based syndrome extraction

The helper also supports `use_3aux=True` on `surgery_library`, `circuit_source`,
and `compile_case` (and on `extraction_round` for an individual round):

```python
from lattice_surgery import surgery_library

source = surgery_library(distance=3, probability=0.001, use_3aux=True)
```

This selects the pairwise-measurement 3aux circuits from
[arXiv:2310.12981v1, Figures 2-4 and 8](https://arxiv.org/abs/2310.12981v1)
for every preparation, idle round, and merged-patch round. Bulk checks use three
auxiliaries; weight-two boundary checks use one. The layout is transposed to keep
hook errors perpendicular to the existing logical strings. Extraction uses
no quantum CNOTs: its physical measurements are single-qubit X/Z and two-qubit
XX/ZZ. DEQ tracks the measurement-induced Pauli frame and combines the appropriate
measurement parities for surgery readouts.

Each standalone round prepares its scratch auxiliaries and includes the full
eight-step measurement ramp-up/down. This does **not** implement the paper's
four-step steady-state pipelining across gadget boundaries. `block_seconds`
retains its existing user-specified meaning; it is not rescaled for the new
physical schedule. Noise follows this example's convention: readout flips at
probability `p`, independent post-measurement depolarization at `p`, idle
depolarization at `p` per step, and reset errors at `p`. This is not the paper's
joint measurement-fault distribution.

`compile_case` records `use_3aux` in its configuration, and noiseless latency
reruns retain that selection. The default remains `False`; the fixed JIT
benchmark fixtures continue to use the original CNOT-based extraction.

### Compilation and timing

The generator creates `.deq` text and compiles it using the ordinary Python JIT
library/program builders. The existing hand-written lattice-surgery `.deq`
examples are distance 3, not this distance-7 workload; no alternate binary
source of truth or duplicate hand-written circuit is maintained here.
Each call carries native `@START` timing in integer-nanosecond-derived decimal
seconds. The compiler owns completion ordering and reference remapping; the
generator does not reorder or renumber compiled instructions. Epoch counts
are diagnostic block indices derived from native gadget completion times.
The compiler-only benchmark still performs no timed delivery or sleeping.
Compression has a fixed timestamp and no embedded filename. Rust verifies the
decoded fixture checksums before timing and records the manifest checksum for
the report to verify alongside SHA-256 fixture provenance.

## Reports

No timing baseline is published. Criterion maintains raw samples, estimates,
and statistical comparisons under `target/criterion/`, alongside run provenance.
The report command writes Markdown to `target/jit-benchmark-report/`.
The summary records
the base revision, dirty-worktree status, compiler, CPU, build flags, aggregate
timings, confidence bounds, and pass/fail decisions. `results/*.json` is ignored
to prevent accidental check-in of older generated reports.

Only the Rust harness and support module live in
[`deq_runtime/benches`](../../../../deq_runtime/benches).
The Python generation/report tools, their tests, this guide, and `.gitignore`
live here with the tutorial example. Generated fixtures,
Python bytecode, and raw measurement JSON are local artifacts only.
