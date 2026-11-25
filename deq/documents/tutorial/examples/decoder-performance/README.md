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

### Precomputed warm-only replay

`precompute_replay.py` prepares ten physical shots at each power-of-two width
through W=8192 before decoding starts. It submits larger widths first to a
spawned process pool and batches shots within each width for Stim SIMD
efficiency. Physical gadgets stream through small in-memory chunks; no expanded
Stim file or global tableau is created. The CSS reference transformation follows
the native runtime's `stim_reference.rs`; unsupported physical gates fail rather
than falling back to a tableau.

Use the same rebuilt Python environment for preparation and replay:

```bash
python deq/documents/tutorial/examples/decoder-performance/precompute_replay.py \
    --output target/warm-replay/samples --workers 14 --memory-gib 320
python deq/documents/tutorial/examples/decoder-performance/warm_replay.py \
    --samples target/warm-replay/samples --output target/warm-replay/evaluation
```

The replay runner targets the qualified 120-CPU topology: frontend CPUs 0-29,
24 Tokio threads, and three 24-worker decoder pools on CPUs 30-53, 60-83, and
90-113. Each pool permits two concurrent initializations. History remains until
shot reset (`automatic_collection=false`). JIT decoders remain nonpersistent;
excluding warm-up shots does not change that policy.
Static libraries use `static_jit_compiler(..., compact=True)` in a separate
process, matching the native execution path rather than the debugging expander.

Each shot is a headerless, LSB-first packed `.b8` file. Manifests record dimensions,
hashes, source identities, a deterministic batch/reference seed pair, and each
shot's SIMD column. Reproduction requires the same Stim build and execution
shape. W=8192 produces 82,579,856 bytes per shot; all fourteen widths and ten
shots total 1,651,052,790 bytes. Every generated file is limited to 10 GB.
The pool has an aggregate memory guard, a time guard, atomic checkpoints,
and per-width progress logs.

Replay cycles the ten files: files 0 and 1 warm up the process, followed by
files 2-9, 0, and 1 for ten measured shots. All twelve shots must complete with
full readout coverage; only the declared warm-ups are excluded. Logical-error
shots are retained. The Python/native sampler boundary accepts packed bytes
directly, and native PROGRAM metadata supplies measurement counts and absolute
delivery deadlines without a Stim input. Readout-only Rhai assertions avoid
expanding every physical measurement into script values.

`DEQ_TRACE_MODE=latency` retains readout-window timing and shot anchors but
omits solver diagnostics and redundant payloads. Required records bypass the
best-effort tracing collector through a bounded, lossless writer channel and
use a shared monotonic clock. `DEQ_TRACE_MAX_BYTES` limits
both native trace files. Trace analysis streams records and validates warm-ups
before excluding them. Solver-call distributions require a full trace instead.
Each curve still stops independently at its first warm mean above one second,
or at W=8192, which is labeled as a safety cap rather than overload.

Preparation supports `--resume` with input/hash verification. Replay supports
`--resume`; add `--retry-failed` to archive an incomplete decoding attempt before
retrying it. `--wait-for-samples` can wait for an active preparation campaign.
These warm-only results must not be merged with the original cold-inclusive
evaluation.

### Deep single-width replay

For a depth study, both preparer and evaluator accept `--layers` and
`--min-width`. Both also accept `--probability` (default `0.001`); use the
same value for preparation and evaluation. Changing it requires fresh physical
samples and a static library with matching error-model probabilities, not
merely a relabeled result or a resampled shot with the old decoder weights.
The evaluator also accepts `--series`. For example:

```bash
python deq/documents/tutorial/examples/decoder-performance/precompute_replay.py \
  --output target/deep-replay/samples --min-width 1024 --max-width 1024 \
  --layers 1000 --shots 10 --workers 1 --memory-gib 320
python deq/documents/tutorial/examples/decoder-performance/warm_replay.py \
  --samples target/deep-replay/samples --output target/deep-replay/evaluation \
  --min-width 1024 --max-width 1024 --layers 1000 --series persistent \
  --memory-gib 320 --task-timeout-seconds 21600
```

Profile shallower depths first: compact files do not guarantee low peak
compilation or runtime memory. Depth counts brickwall layers, each containing
three timed blocks. A depth-1000 shot spans 210.14 scheduled seconds at the
default 70 ms/block cadence. Samples remain headerless packed measurement bits;
the per-file 10 GB limit and full warm-up/readout validation still apply.

For an error-rate comparison, pass (for example) `--probability 1e-5` to
both commands and use fresh output directories. A lower error rate does not
reduce the number of measurements, gadgets, or scheduled input batches.
Report backend-call timing separately from full readout latency: input
processing, syndrome computation, frame propagation, and coordination still
run even when most syndromes require no matching work.

On Linux/glibc, `warm_replay.py --trim-heap` optionally calls `malloc_trim(0)`
before reading each physical shot. This releases unused allocator pages, not
live decoder instances or circuit state. It runs before the native latency
clock and writes RSS-before/after and trimming time to `allocator.jsonl`.
Total process wall time still includes it. The default is disabled; record
the policy when comparing runs, and do not assume it fixes live-memory growth.

A single static/persistent MWPM point additionally produces
`intermediate-latency.png` and `intermediate-latency.json`: pooled and
readout-type CCDFs, plus layer-wise p50/p95/p99/p99.9. All observations contribute
to statistics; only plotting knots are reduced. Readouts sharing a shot or
window are correlated, not independent trials. Final readouts remain validated.

CPU nice levels are recorded in execution settings. If permitted, raise the
supervisor's normal-scheduler priority before launch so children inherit it.
Do not use real-time scheduling or change unrelated processes. Follow the
[NUMA guide](../../chapters/numa-performance.md) for affinity and memory policy.

### System-tail diagnostics

For a short diagnostic native-server run, `DEQ_TRACE_MODE=latency` together
with `DEQ_TRACE_DIAGNOSTICS=1` adds lossless records without enabling large
solver payload dumps. The ordinary replay driver explicitly disables these
extra records; its `run_point` helper accepts `system_diagnostics=True` in
the recorded settings for focused experiments.

`input_delivery` and `deadline_wakeup` distinguish late submission from timer
wakeup delay. `static_batch` separates controller locking, buffering,
dispatch, and completion draining. Setting `trace_pipeline=true` in the
coordinator configuration also records its existing 1-in-64 gadget phases.
`decoder_job` records backend-call boundaries as `solver_started_offset_ns`
and `solver_returned_offset_ns`, excluding construction, queueing, result
validation, and reset. Linux records thread CPU time, faults, and context
switches around that call. Zero-syndrome bypasses are not fake zero-duration
solver calls.

`frame_operation` samples every 64th frame-owner request. Its
`submission_started`, `worker_started`, `worker_finished`, and `caller_resumed`
offsets distinguish queue/lock wait,
frame computation, and completion-to-caller delay. The owner captures only
timestamps; diagnostic string allocation is deferred until after caller
resumption. These are diagnostic observations, not replacements for
scheduled-input-to-readout latency.

`DEQ_TRACE_WORKER_SAMPLE_RATE=N` retains every Nth worker record (default 1);
the rate is written into each retained record. System/root phase records are
not reduced by that setting. Use a recorded `worker_sample_rate` in `run_point`
settings when diagnosing a large case, and label sampled tails accordingly.

Keep diagnostic runs short and retain the 10 GB/file guard. Additional
recording has overhead, so compare against an unchanged, non-diagnostic
baseline. A window's `decoding` to `decoded` duration is a decoder-request
measurement, not a pure solver measurement. Delay before window creation
can include input delivery, readiness, history, or reservation waits.
Inherited `GLIBC_TUNABLES` are recorded in each native command receipt.
See the [allocator-tail guidance](../../chapters/numa-performance.md#diagnose-allocator-tails-separately-from-numa-placement)
before treating long allocator pauses as decoder overload.

For controlled runtime comparisons, `run_point` can use a fixed `static_input`
containing its `path` and original `compiler_sha256`. The original receipt,
JIT-input hash, compact mode, and library hash must all match; receipts are
never rewritten. The consumer runtime is recorded separately. Ordinary cache
reuse still requires the current compiler; this is an explicit immutable
input, not an automatic stale-cache fallback.

### Original fresh-sampling protocol

- Distance 7, noise parameter 0.001, seed 42, 50 CNOT layers.
- Window buffer radius 1, lookahead 0; persistence only in the static series.
- Width doubles from 1. Each series stops independently after its first mean
  intermediate or final latency above 1 second (1,000 ms), including that point.
  There is no default width cap. `--max-width` can impose an optional
  power-of-two safety cap; reaching that cap does not establish overload.
- At least 100 final and 1,001 intermediate readouts per point, except W=1,
  which has no intermediate readouts. Logical-error shots are retained.
- Latency is window commitment minus its last required **scheduled** input.
  Windows that commit no readouts do not contribute latency observations.
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
