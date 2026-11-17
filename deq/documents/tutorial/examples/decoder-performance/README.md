# Decoder performance evaluation

The main evaluation measures **readout latency**. It compares
JIT MWPF, JIT MWPM, static/persistent MWPM, and static/persistent MWPF c=10 on the same lattice-surgery
brickwall workload. See the [chapter](../../chapters/decoder-performance.md)
for the circuit and interpretation of the results.

## Logical error rates

The separate `evaluate_ler.py` sweep invokes the existing **`deq simulate ler`**
CLI with `--errors 20`, fresh physical sampling, `p=0.001`, distance 7, and
50 layers. MWPM and MWPF c=10 run at successive powers-of-two widths until
each whole-program LER first exceeds 80%. It rejects failed decoder shots
and points that did not reach 20 errors.

```bash
python deq/documents/tutorial/examples/decoder-performance/evaluate_ler.py \
    --jobs 1 --threads 24 --memory-gib 48 --results target/deq-ler/ler-results.json
```

The published [LER data](ler-results.json) contain 17 completed points,
9,141 shots, and 340 logical errors: MWPM crossed 80% at W=128 and MWPF at W=256.
Every point reached 20 errors with no failed decoder shots. The run used one
native worker with 24 decoder threads and peaked at 10.79 GiB. Delivery delays
are disabled for LER; source-defined execution order and physical noise are
unchanged. To redraw the measured points without running simulations:

```bash
python deq/documents/tutorial/examples/decoder-performance/evaluate_ler.py \
    --plot-only
```

The default plot-only publication path requires a complete evaluation.
`--allow-incomplete` is an explicit opt-in that labels unfinished curves; it
never relaxes shot-count or error-count validation. Error bars are approximate
95% Wilson intervals, not a guarantee under adaptive stopping. Raw CLI logs,
generated circuits, and intermediate files stay in the results directory's
ignored `ler-work/`; the published JSON retains counts and build/source hashes.

## Build and run

Use one [source-installed Python environment](../../../../README.md#install-from-source).
From the repository root:

```bash
export RUSTFLAGS="-C target-cpu=native"
(cd deq/deqagram/bindings/python && python -m maturin develop --release)
(cd deq/deq_runtime && python -m maturin develop --release)
python deq/deq/proto/compile.py
python -m pip install -e './deq[benchmark]'
python deq/documents/tutorial/examples/decoder-performance/run.py \
    --threads 24 --memory-gib 48
python deq/documents/tutorial/examples/decoder-performance/plot.py \
    --data deq/documents/tutorial/examples/decoder-performance/performance-data.json \
    --output deq/documents/tutorial/examples/decoder-performance/performance.png
```

Allow several hours; do not run competing CPU-heavy work. Tasks execute
sequentially in isolated processes, with process-tree RSS and time limits.
The default decoder worker count is 24. Choose a memory guard
that leaves room for the OS and other work. Each task receives at least one
hour by default; `--task-timeout-seconds` increases this minimum for expensive
large-circuit preparation. The timeout also grows with the requested shots and
scheduled delivery time. Memory and time guards remain active without a width cap.

### Larger machines

Rebuild the parser and runtime on the target machine using the source-install
commands above; do not copy binaries built with another machine's
`target-cpu=native`. Start a fresh results directory so hardware/build identities
are not mixed with an earlier run.

For a machine with 120 available CPU cores and 400 GB of RAM:

```bash
python deq/documents/tutorial/examples/decoder-performance/run.py \
    --threads 120 --memory-gib 320 --task-timeout-seconds 14400 \
    --results target/decoder-performance-120core/results.json
python deq/documents/tutorial/examples/decoder-performance/plot.py \
    --data target/decoder-performance-120core/performance-data.json \
    --output target/decoder-performance-120core/performance.png
```

There is no default width limit: eligible curves continue through W=2048,
4096, and higher until their mean latency exceeds 1,000 ms. The 320 GiB guard
is approximately 344 GB, leaving headroom on a 400 GB host. Allow ample disk
space for source exports and traces; raw output size grows with width.
Use only cores and memory allocated to this run, and avoid competing work.
Larger hardware does not guarantee a proportional increase in sustainable width.

After timing completes, run LER separately to avoid contention:

```bash
python deq/documents/tutorial/examples/decoder-performance/evaluate_ler.py \
    --jobs 1 --threads 120 --memory-gib 320 \
    --results target/decoder-performance-120core/ler/ler-results.json
```

All configurations invoke `deq simulate ler` with one `.deq` source, using
`--simulator jit-static` or `--simulator static`. The native compiler produces
the compact static library.
Preparation records source and JIT metadata only; the CLI exports the physical
circuit for each run, avoiding an unused additional large Stim export.
Each point uses `--jobs 1 --shots QUOTA --batch-size QUOTA --errors QUOTA+1`
with `--strict-timing`, `--trace timing.pb`, and
`--simulator-trace-output shots.pb`. The unreachable error target prevents
early stopping on logical errors. A per-batch CLI timeout and an independent
whole-process-tree time/RSS guard bound every task.

To inspect individual solver calls from an existing native trace:

```bash
python deq/documents/tutorial/examples/decoder-performance/readout_latency.py \
    path/to/timing.pb --output target/solver-distribution.json
```

The report prints call count, mean, median, p90, p99, and maximum in microseconds,
and saves distribution quantiles in milliseconds. It excludes dispatch,
instance construction, and reset; loaded-decoder traces include result validation
in the solver stage. Zero-syndrome shortcuts are counted separately, not inserted
as zero-duration solver calls. These timings do not include scheduled-input
lateness or coordinator overhead.

## Measurement contract

- Distance 7, noise parameter 0.001, seed 42, 50 CNOT layers.
- Window buffer radius 1, lookahead 0; persistence only in the static series.
- Width doubles from 1. Each series stops independently after its first mean
  intermediate or final latency above 1 second (1,000 ms), including that point.
  There is no default width cap. `--max-width` can impose an optional
  power-of-two safety cap; reaching that cap does not establish overload.
- At least 100 final and 1,001 intermediate readouts per point, except W=1,
  which has no intermediate readouts. Logical-error shots are retained.
- Latency is window commitment minus its last required **scheduled** input.
  Missing, duplicate, failed, or incomplete readouts are errors, never samples.
- JIT metadata is registered up to two blocks early, clipped at shot start.
  There is no extra startup grace or external adjustment to the circuit clock.
- Physical sampling finishes before each shot's native `start_decoding` event.
  Static program setup precedes that event and is **excluded** from latency.
  Subsequent graph preparation and decoder execution remain included.
- Every configuration at **every width** uses fresh physical samples.

Runs use three d-round blocks per CNOT layer, with one-round ancilla
preparation overlapping preceding work. Initial data preparation remains d
rounds. This gives 152 timed blocks for 50 layers. The commands above regenerate the
published data and figure.
`@START(at=seconds,duration=seconds)` annotations in the generated source define
the schedule. CNOT's origin precedes the data boundary by one ancilla round;
its local preparation starts at zero, followed by aligned ZZ, XX and readout
blocks. Times are computed in integer nanoseconds and printed as decimal seconds.
An optional generator `block_seconds` sets test cadence; fractional-nanosecond
round divisions use the integer floor, while the full block boundary stays exact.
`@DELAY(seconds)` advances the local source cursor; omitted durations are zero,
not inherited. Corrections need no annotations: the compiler places them at
their dependency completion. Native compilation orders completed gadgets and
remaps references; neither the generator nor runner sorts/re-IDs the program or
inserts a separate timed-Stim pass.

## Outputs and recovery

The published timing data are unchanged and used the previous 10-second
threshold, not the current 1-second stopping rule. MWPF crossed that threshold at
W=64, MWPM at W=512, and static/persistent MWPF at W=256. Static/persistent
MWPM reached the W=1024 limit below the threshold. Peak process-tree memory
was 28.07 GiB. No failed task contributes a latency point.
`summarize.py --allow-incomplete` permits publishing a raw evaluation marked
`stopped`, retaining sample validation and labeling every unfinished curve.

`results.json` checkpoints completed points and failure details.
`performance-data.json` is written only after all series finish and their sample
counts, means, and stopping points validate. It holds the figure's distributions
and build/source hashes. `work/` retains generated circuits, commands, inputs,
native traces, and task logs; large raw artifacts are ignored by Git.

Use `--resume` with the same results path to continue. Environment, source hashes,
settings, and native binaries must match exactly. An incomplete task requires
`--resume --retry-failed`: its partial outputs are archived before retrying,
while completed tasks are retained. For changed code or settings, start in a new
directory.

The plotter reads only the compact data and writes PNG. To recreate compact data
from a completed raw evaluation, use `summarize.py --results <results.json>`.

## Code map

| File | Responsibility |
| --- | --- |
| `run.py` | Four-series sweep, isolated tasks, provenance and checkpoints |
| `study_utils.py` | Sample quotas, atomic JSON writes and process guards |
| `lattice_surgery_latency.py` | Unified native source-CLI invocation and completed-run validation |
| `readout_latency.py` | Commit-time extraction and readout coverage |
| `summarize.py`, `plot.py` | Validated compact data and PNG plotting |

Circuit generation and wiring validation live in the shared
[JIT workload helper](../jit-basics/lattice_surgery.py). Worker-phase breakdowns,
preload probes, and CPU utilization studies are deliberately outside this
evaluation; raw native traces remain available for separate investigations.

## Circuits, diagrams, and tests

```bash
python deq/documents/tutorial/examples/decoder-performance/generate_circuits.py --width 128
python deq/documents/tutorial/examples/decoder-performance/draw_circuits.py
(cd deq && python -m pytest documents/tutorial/examples/decoder-performance -q)
```

The workload generator and circuit exporter emit positional calls directly
(`MZZ 0 2`, `PrepareXNonFT 2`), without rewriting generated text,
with explicit PROGRAM boundary declarations and verifies that the
brickwall and expanded surgery compile identically. Data wire `i` is placed at
`2i`, with surgery ancillas between adjacent data patches. Matplotlib draws the
small explanatory diagrams; no large `.deq` exports need to be committed.
`PrepareXNonFT` is a single gadget, not a COMPOSE wrapper, and has no data inputs.
Its one SE round is not independently fault-tolerant and finishes at the start
of the first ZZ merge, with both dependencies and timing expressed in DEQ.

Tests cover circuit distance and noiseless behavior, source/reference alignment, timing
coverage, native execution, resource guards, resume, and PNG-only plotting.
The independent [JIT benchmark](../jit-basics/README.md) measures compiler-only
throughput and is not part of this decoder sweep.
