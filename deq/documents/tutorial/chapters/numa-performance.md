# Tuning deq on NUMA CPUs

The best configuration measured on the example host separates the frontend from
three decoder pools, keeps mutable solvers on their owning workers, and limits
cold initialization. It uses **72 decoder workers, not 120**. Giving every
logical CPU to the decoder leaves insufficient room for compilation, scheduling,
frame propagation, and I/O.

This is a measured starting point, not a universal optimum. The low-latency
static results below use a **finite program with history retained until reset**.
That configuration must not be copied unchanged into an unbounded service.

## Discover the allocation, not just the processor model

Run these commands in the same container or job allocation as the service:

```bash
lscpu
lscpu -e=CPU,NODE,SOCKET,CORE
grep Cpus_allowed_list /proc/self/status
```

Also inspect the job/container CPU and memory limits. On cgroup v2, the current
cgroup's `cpuset.cpus.effective` and `memory.max` are relevant. A thread affinity
mask is not the same as the allocation's cpuset.

The example VM reports an **AMD EPYC 7V73X 64-Core Processor**, with 120 online
logical CPUs, one thread per exposed core, two exposed sockets, and four NUMA
nodes. These are guest-visible properties; they do not establish the physical
host's cache layout or exclusive ownership.

| NUMA node | Visible CPUs | Measured assignment |
| --- | --- | --- |
| 0 | 0-29 | Frontend; 24 Tokio workers, 2 global Rayon workers, and other service threads |
| 1 | 30-59 | Decoder pool on 30-53 |
| 2 | 60-89 | Decoder pool on 60-83 |
| 3 | 90-119 | Decoder pool on 90-113 |

The remaining six CPUs on each decoder node are outside these pools; they are
not automatically reserved for this process. Do not copy these numbers onto
another host without inspecting its topology. SMT siblings are not independent
physical cores, so start with one worker per physical core when SMT is exposed.

## Build and prepare locally

Follow [installation from source](../../../README.md#install-from-source) in the
selected Python environment. Keep the checkout, Cargo target directory, temporary
compilation output, and active traces on local storage. Installing a finished
wheel into a shared environment is different from building on NFS.

From a checkout verified to be on local storage, the example build flags are:

```bash
findmnt -T "$PWD"
export CARGO_TARGET_DIR="$PWD/target/numa-release"
export CARGO_BUILD_JOBS=24
export RUSTFLAGS="-C target-cpu=znver3"
export CFLAGS="-march=znver3"
export CXXFLAGS="-march=znver3"
(cd deq/deq_runtime && maturin develop --release)
```

`znver3` is appropriate for this host. Use a target supported by **every**
destination machine when sharing binaries. A wheel's platform tag does not
guarantee compatibility with instructions enabled by `target-cpu=native`.
Pin the package/compiler versions and record the actual loaded extension path
and hash; an inherited installation can otherwise shadow a new build.

For static execution, use compact libraries:

Run this preparer in a fresh process with
`TOKIO_WORKER_THREADS=4 RAYON_NUM_THREADS=2`. These are preparation settings,
not the 24-thread frontend configuration used for decoding.

```python
from pathlib import Path
from deq.compiler.jit_compiler import static_jit_compiler
from deq.proto.deq_jit_pb2 import JitLibrary

library = JitLibrary.FromString(Path("program.deq.jit").read_bytes())
compiled = static_jit_compiler(library, compact=True)
Path("program.deq.bin").write_bytes(compiled.SerializeToString())
```

The default `compact=False` is a debugging expansion, not the latency benchmark
configuration. Distinct per-instance check/error type IDs defeat much of the
persistent cache reuse. In the example workload at W=8, the debugging library
was about 936 MB with 2,091 check types and 2,091 error types; the compact library
was 4.72 MB with 11 and 8 types.

For large libraries, compile in a separate process and let it exit before
measurement, releasing compiler allocator arenas. The replay driver does this
with four Tokio threads. Do not run builds or physical sampling concurrently
with a decoder latency benchmark.

## Configure execution domains explicitly

This Linux command starts a static server for a **bounded, already-compiled**
program. A client must supply the outcomes and reset between finite executions;
no simulator is started by this command.

```bash
TOKIO_WORKER_THREADS=24 \
RAYON_NUM_THREADS=2 \
OMP_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 \
taskset --cpu-list 0-29 \
python -m deq.runtime server \
  --addr 127.0.0.1:50051 \
  --decoder mwpm \
  --decoder-config '{
    "pools": [
      {"name":"numa-1","cpus":"30-53","max_initializations":2},
      {"name":"numa-2","cpus":"60-83","max_initializations":2},
      {"name":"numa-3","cpus":"90-113","max_initializations":2}
    ]
  }' \
  --coordinator window \
  --coordinator-config '{
    "buffer_radius":1,
    "lookahead_radius":0,
    "persistent_decoder":true,
    "automatic_collection":false,
    "assert_parity_factor":true
  }' \
  --controller static \
  --controller-config '{"filepath":"program.deq.bin"}'
```

Start a fresh process with these environment variables and affinity settings,
before runtimes and worker threads are created. `RAYON_NUM_THREADS` controls the
global Rayon pool, not the number of explicitly pinned decoder workers.

The decoder workers override the inherited `taskset` mask, but cannot escape
the kernel's cpuset restrictions. Allocate the union of frontend and decoder
CPUs to the job/container. A container restricted to CPUs 0-29 cannot run
decoder workers on 30-113, even though its main thread has the intended mask.

Do not combine `pools` with `parallel`; omit `pools` entirely to select the
legacy Rayon configuration. Pool lists must be nonempty, disjoint, and within
one NUMA node. CPU IDs above 1023 are unsupported. Invalid configuration or
failed affinity binding is an error, not an unbound fallback.

Each worker owns its mutable instances through construction, decoding, reset,
and eviction. Factories are shared only within a pool. Scheduling transfers
requests and results, not mutable solver objects. A Rust ownership move alone
does not copy a heap graph; remote memory access, first-touch placement,
allocation bursts, and synchronization are separate costs.

CPU binding encourages local first-touch allocation; it is **not a strict
memory-binding guarantee**. In particular, do not add `numactl --membind=0` to
this command: it would force the remote decoder pools to allocate on node 0.
Blanket memory interleaving is not the default recommendation either. Inspect
actual placement before changing memory policy.

## Compiled dataflow for finite static programs

For high offered load, the opt-in `compiled-window` coordinator removes
per-gadget async task creation, repeated model binding, dynamic window discovery,
and the single frame-owner queue. Preparation compiles dependency adjacency,
shared hypergraph templates, and inline frame transforms once. Bounded CPU
batches compute syndromes, corrections, and frames; the existing NUMA-local
decoder pools still own the mutable solvers.

In the server command above, set `TOKIO_WORKER_THREADS=2`, replace
`--coordinator window` with `--coordinator compiled-window`, and replace
the `--coordinator-config` value with:

```json
{"worker_threads":24,"batch_size":8,"max_templates":2048}
```

Set the environment variable on the actual server invocation, before its runtime
starts. The 24 compiled CPU workers inherit the frontend affinity, independently
of the two Tokio workers. `worker_threads` accepts 1-1024 and `batch_size` accepts
1-256; `max_templates` is a positive preparation-time resource guard, not an
evicting cache. Start with the example settings rather than assuming that more
workers or larger batches always reduce latency.

This is a **different decoding-window policy**, not merely a faster implementation
of the dynamic `window` coordinator. Each window commits only its centre gadget's
errors. It contains their complete detector support and projects touching future
errors onto those detectors as noncommitting explanations. Earlier touching
corrections are immutable dependencies. Both the detector dependencies and
transitive frame dependencies must be satisfied before a logical result is ready.
The existing backend handles hyperedges; graphlike arity is not assumed.

Eligibility is checked before execution: a complete, nonempty static program;
ascending dense gadget IDs; exactly one check and error model per gadget with
`gid=cid=eid`; fixed finite error probabilities in `[0, 0.5)`; no loss models,
loss masks, probability updates, or forced-gap decoder. Frame inputs have at most
127 bits plus the bias bit; outputs, readouts, and remote-readout operands each
have at most 128 bits. Static reroutes are supported, but a remote path entry may
only refer to an earlier path entry. Unsupported inputs fail explicitly.

Only local static-controller ingestion is supported. Incremental/JIT execution,
remote-controller ingestion, library replacement, and unloading are not supported.
Reset drains accepted CPU work, clears the epoch's outcomes and frames, and
retains the immutable plan and decoder caches. Resetting the library or decoder
service is rejected. State is proportional to the finite program; this is not
an unbounded streaming service or a networked decoder pool.

Rust integrations can call `CompiledWindowCoordinator::readout_bits(gid)` to
obtain a ready logical result without waiting for the entire program. `None`
means pending; invalid IDs, cancellation, and an unloaded program are errors.
Readers must coordinate resets because this query addresses the current epoch.
The ordinary static-controller bulk response still returns at program completion;
there is no new streaming gRPC or Python readout-query API.

Latency traces record actual frame-slot availability, not just correction
commitment. Report both latency from the latest required scheduled input and
latency from the readout gadget's own scheduled end: the latter includes
algorithmic lookahead. Neither metric measures external client/network delivery.
Validate every warm-up and measured result, and qualify logical accuracy for the
chosen window policy rather than inferring it from low latency.

## Verify placement and tune one variable at a time

For the service's known PID, inspect thread placement:

```bash
ps -L -p <PID> -o tid,comm,psr
taskset -pc <TID>
numastat -p <PID>
```

`numastat` is supplied by the optional system `numactl` tools. `PSR` is the last
CPU a thread ran on, not proof of its allowed affinity mask. Decoder thread
names start with `deq-p`; each should have exactly its configured single CPU.
Verify frontend threads separately. Do not overwrite all thread masks after
the decoder pools have started, and do not change cpusets or hotplug CPUs
during a measured run.

Take memory-placement snapshots outside timed execution. On upstream Linux
5.14, reading `/proc/<PID>/numa_maps` walks page tables while holding the
process's memory-map read lock; this can block concurrent mapping changes.
`numastat -p` uses that interface. Repeated scans of a large process are not a
free observer, even when the monitoring process runs on a spare CPU. Compare
latency with the observer disabled before attributing new tails to the runtime.

A useful tuning sequence is:

1. Establish a 24-worker single-node baseline with fixed inputs and workload.
2. Compare the three 24-worker pools while holding the frontend allocation,
   decoder, timing schedule, and measurement protocol constant.
3. Start with one cold initialization per pool; compare two on this host.
   Higher limits can amplify allocation/page-fault bursts even if workers
   are otherwise available.
4. Inspect preparation, queueing, solve, commitment, and reset separately.
   More decoder workers cannot fix a saturated compiler or frame owner.
5. Change admission only after identifying the bottleneck. `max_ready_windows`
   defaults to twice the execution capacity, or 144 here. It bounds ready
   preparation/decode/commit work, **not all registered gadgets or input waits**.

Keep upstream queues and the number of outstanding requests bounded. A slow or
overloaded endpoint needs backpressure, not unlimited queued futures. Hard and
forced-gap decoders share the execution capacity but have separate caches;
configure pools only on the hard decoder.

## Distinguish async scheduling from NUMA migration

An async task can wait milliseconds even when every poll runs on one NUMA
node. In a W=1024, depth-50, p=1e-5 control on this host, inherited frontend
affinity 0-29 kept all observed Tokio and frame execution on node 0.
Individually pinning the 24 Tokio workers to 0-23 and the frame owner to 24
removed observed frontend CPU changes, but changed the mean only from
8.56 to 8.43 ms. Both arms used two warm-ups and ten measured shots.
Remote decoder-completion notifications are not evidence of remote execution
of the receiving async task. Aggregate process NUMA pages also include decoder
caches and file mappings; they do not isolate frontend memory traffic.

The static controller's `decode_task_batch_size` controls how many pending
gadget futures share a Tokio task. The default is **16**; set a positive
integer in `--controller-config`, for example:

```json
{"filepath":"program.deq.bin","decode_task_batch_size":16}
```

One creates an independent task per pending gadget. Larger groups reduce task
creation but can increase delay within a group. In the tested futures-util
0.3.34, `try_join_all` switches from direct polling to `FuturesOrdered` above
30 children, adding another internal scheduling layer. That threshold is an
implementation detail, not a stable API. On matched two-measured-shot pilots,
groups of 32 and 16 averaged 8.81 and 6.42 ms respectively; sizes 4, 8,
and 30 did not beat 16 in the trials. More frequent Tokio global-queue polling
did not materially help and is not exposed as a tuning option.

Tune the grouping against the actual workload rather than the CPU count alone.
Grouping preserves input availability, ordered results, cancellation, and reset;
it does not delay an input until an entire group has arrived. It does not imply
that end-to-end latency is below one millisecond.

The full-depth follow-up used the same binary and input shots at depth 1000,
with two warm-ups and two measured shots per configuration. Each arm retained
all 3,069,000 measured intermediate readouts. Continuous page-map scans were
disabled in these final controls:

| Task group size | Mean | p99 | p99.9 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| 32 | 9.32 ms | 15.34 ms | 152.88 ms | 1170.25 ms |
| 16 | 6.66 ms | 12.26 ms | 14.01 ms | 513.60 ms |

The mean improved by 28.6%, but **the 500 us target was not reached**. Early-layer
outliers remain even without page-map scans; their complete cause is not
isolated. This is a two-measured-shot comparison, not a production SLA.
Mean admission and frame-commit intervals fell from 0.542/1.085 ms to
0.018/0.514 ms, while 5.973 ms of the new mean still occurs before window
creation. That interval includes input publication, dependency readiness, and
scheduling; it is not a measurement of XOR arithmetic alone.

The grouping also does not improve every scheduler metric. Separate short,
CPU-only diagnostic controls had mean first-poll delays of 0.769/2.053 ms for
groups of 32/16. Choose using complete readout latency and resource costs,
not one isolated queue metric. Data and the matched full-depth figure are in
`target/evaluation-120core/async-locality-20261124/final-investigation.json`
and `full-cpu-only-comparison.png` in that directory.

With lossless latency tracing, `DEQ_TRACE_DIAGNOSTICS=1` adds systematically
sampled `gadget_polls` records: first-poll delay, wake-to-poll delay, poll wall
time, and CPUs at polls and the first wake before each poll. Later coalesced
wake CPUs are not recorded. These traces change the observer cost; keep
their results separate from uninstrumented qualification. Wake-to-poll includes
possible overlap with a preceding poll and is not pure OS runqueue time; poll
wall time is not CPU time. Do not sum waits across unrelated gadgets and call
that one readout's latency.

## Choose the memory and lifecycle policy deliberately

| Workload | Policy |
| --- | --- |
| Finite static experiment with explicit resets | `persistent_decoder=true`; `automatic_collection=false` can avoid repeated whole-program collection scans, with a measured memory budget |
| Long-running or unbounded input stream | Keep `automatic_collection=true`; establish a bounded working set and validate sustained collection performance |
| JIT comparison in this study | `persistent_decoder=false`; compilation remains in the timed path, and warm-ups do not turn it into the static configuration |

Disabling collection retains execution history until reset. It is not a
general production streaming optimization. Worker-local graph replicas and
long-lived graph identities can also increase cache memory. Monitor RSS,
faults, allocator pressure, and time spent resetting/evicting, not just solve
time. A resource limit is a safety boundary, not evidence that a workload meets
its latency target.

Cancellation does not make an already-running native solver stop immediately.
Capacity remains occupied until it actually finishes. Abandoning a decode after
its input has been accepted cancels the whole shot, including while publication
or readouts are pending; reset is required. Reset drains accepted work before
clearing state, and a failed reset must not admit a fresh uncancelled epoch.
Backend reset/worker failures
must be handled as errors, not successful corrections. Use process-level
supervision for a backend that can hang indefinitely.

## Diagnose allocator tails separately from NUMA placement

Fast decoding does not guarantee fast readouts. A pause in input delivery,
coordination, or the frame owner can stall dependent work while decoder workers
remain idle. Separate scheduled-input delay, queueing, backend execution, and
frame publication before adding workers.

On this host's **glibc 2.34**, focused CPU stacks during long frame-owner
operations showed `realloc`, `_int_free`, and `malloc_consolidate`: deferred
heap coalescing, not decoder computation or copying a solver between NUMA
nodes. Diagnostic formatting also allocates, so confirm proposed remedies
without additional diagnostic instrumentation.

For this allocator, test disabling fastbins at process startup:

```bash
export GLIBC_TUNABLES=glibc.malloc.mxfast=0
```

Then launch the same server or replay command. If `GLIBC_TUNABLES` already
contains other settings, preserve them and replace/add only the `mxfast`
entry in its colon-separated list. This changes only new processes launched
with that environment; it does not change the kernel or other users' jobs.
The thread-local allocation cache remains enabled.

A matched W=128, depth-1000 comparison used the same native binary, input
shots, 70 ms/block schedule, and 72-worker placement, with two warm-ups and
two measured shots per configuration. Each distribution contains 381,000
intermediate readouts:

| Latency | Default allocator | Fastbins disabled |
| --- | ---: | ---: |
| Median | 6.07 ms | 6.28 ms |
| p99 | 10.30 ms | 8.33 ms |
| p99.9 | 94.64 ms | 10.09 ms |
| p99.99 | 107.16 ms | 13.00 ms |
| Maximum | 110.15 ms | 18.69 ms |

This trades a small median increase for much smaller observed tails; it is
not a worst-case guarantee or an independent-trial statistical bound.
All measured observations were retained. Recheck throughput, RSS, and tails
on the deployment workload. The setting is **not a universal allocator
recommendation**: newer glibc releases have removed fastbins, and other
allocators have different controls. See the
[glibc 2.34 tunable documentation](https://sourceware.org/glibc/manual/2.34/html_node/Memory-Allocation-Tunables.html).

At W=1024, disabling fastbins alone still left long startup transients. On
this Linux 5.14 host, adding **process-local `numactl --localalloc`** removed
the remaining multi-second tails in the matched trial:

| W=1024 setting | Mean | p99 | p99.99 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Default allocator and memory policy | 244.26 ms | 2654.31 ms | 4525.61 ms | 4582.97 ms |
| Fastbins disabled | 70.71 ms | 762.75 ms | 1808.99 ms | 1896.63 ms |
| Fastbins disabled, explicit local policy | 47.56 ms | 65.21 ms | 115.39 ms | 135.29 ms |

Each arm contains 3,069,000 intermediate readouts from the same two measured
shots, with the same two warm-ups. These are not the earlier ten-shot
summary's cohort. No initial layers or slow observations were removed.
The combined setting's p99.9 was 66.94 ms; its system p99.99 was **not**
below 100 ms. Peak process-tree RSS was 236.32 GiB, with the same logged,
untimed heap trimming used by the controls.

Place `numactl --localalloc` before the Python server command, or select
`--numa-policy local` in `warm_replay.py`. This is local-first allocation
with fallback, not `--membind=0`. On the tested kernel it also suppresses
automatic NUMA hinting for these inherited policies; thread hint/migration
counters were zero. It does not change global NUMA-balancing settings.
Compare on the deployment kernel rather than assuming identical behavior.
The local policy alone did not remove W=128's allocator tails.

The measured binary was
`9e838f14639c4ea4e1d48f20aa811629eb0d81d6b6fa34db13bd67a899a1aef8`.
Matched data and the figure are retained under
`target/evaluation-120core/latency-recovery-20261122/allocator-w128-comparison.json`
and the adjacent PNG. `allocator-tail-comparison.png` and
`final-tail-report.json` in that directory include both widths and all three
W=1024 arms. Replay command receipts record inherited
`GLIBC_TUNABLES` for subsequent experiments.

## Measure the quantity you intend to deploy

Use the [warm-only replay workflow](../examples/decoder-performance/README.md#precomputed-warm-only-replay)
to separate physical sampling from decoding. It verifies all twelve shots,
excludes exactly two warm-ups, and reports ten measured shots. Logical-error
shots and slow measured shots are retained. The latency origin is the native
scheduled-input clock, not the time the sampler returned or the solver started.

For the same d=7, p=0.001, 50-layer, 70 ms/block workload, the completed
pre-hardening study on this host observed:

| Configuration | Largest observed width with sub-second means | First observed mean crossing |
| --- | ---: | ---: |
| JIT MWPM | 128 | 256 |
| JIT MWPF c=10 | 128 | 256 |
| Static/persistent MWPF c=10 | 512 | 1024 |
| Static/persistent MWPM | 1024 | 2048 |

At W=1024, static/persistent MWPM averaged **42.50 ms intermediate** and
**33.90 ms final** latency. These are warmed means, not cold-start, p99, maximum,
or end-to-end shot-time guarantees. Static setup and physical sampling are
excluded. The doubling sweep does not locate the exact saturation width.
The sharp jumps at overload are a reason to operate with headroom.

The study contains 41 points and 5,413,890 measured readout observations. Its
native build was
`1f0a2c921a184602265846c87b298a6d0c2bd6721dcbfc8f0d052640a8e5eba3`;
these numbers are not a new benchmark of subsequent review fixes. Local
artifacts are retained under
`target/evaluation-120core/warm-replay-20261118/evaluation-lossless/`.
They are not shipped with the source checkout. Record fresh source, binary,
CPU allocation, input, and configuration identities for a deployment.

For timing qualification, use `DEQ_TRACE_MODE=latency` with `--trace` and an
appropriate `DEQ_TRACE_MAX_BYTES`. Required shot/window/frame records use a lossless,
bounded writer and a common monotonic clock, not the best-effort diagnostic
collector. Slow storage can apply backpressure: use local storage and check
the observer's overhead. Missing clocks, incomplete coverage, and trace I/O
failures invalidate a measured run.

The latency-only writer has 16,384 queued-record slots so a burst of thousands
of readouts need not wait for the writer to be scheduled. The queue remains
bounded and flush waits for accepted records; a persistently slow sink still
applies backpressure. Buffering does not change captured timestamps or permit
dropped records. Full diagnostic tracing retains its separate, smaller queue.

Use a full trace for sampled construction/CPU/fault diagnostics or solver-call
distributions. `trace_pipeline=true` is a diagnostic option, not a free
performance setting. Validate cold starts and tails separately, and do not
merge cold-inclusive results with warm-only results.

## Deployment acceptance

### Check offered load, not just solver timing

Fast matching does not establish system throughput. A controlled W=1024,
depth-1000, d=7, p=1e-5 experiment compressed every PROGRAM start and duration
by exactly ten: **70 to 7 ms per block**. A CNOT layer contains three blocks.
The same physical samples were reused; undoing the scaling restored every
field of the original compiled library, including model weights and IDs.

With the same 24-worker frontend, 72 decoder workers, and two measured shots
after two warm-ups, mean window-commit latency rose from **6.66 ms to
24.91 seconds**, and p99 from **12.26 ms to 38.02 seconds**. The input schedule
ended at 21.014 seconds, but about 1.29-1.34 million logical readouts remained
uncommitted; their last commit was at 57.75-59.22 seconds. All early layers
and slow observations were retained.

The overloaded diagnostic run still had a **97 us backend-call p99**.
It nevertheless processes 5,631,548 gadget instances and about 3.21 billion
parity terms per shot. Cached model types do not remove per-shot publication,
readiness/history tracking, allocation, or frame propagation. During active
processing the frontend used about 20-21 cores, while all decoder workers
together used about 0.24-0.32 cores. Giving matching more workers does not
address that imbalance.

Also distinguish commitment from completion. An unnecessary global
readout-completion mutex was still used with collection disabled. The runtime
now skips that GC bookkeeping in retained-history mode; active collection
keeps its existing tracking. Matched diagnostic readouts' mean delay from
window marker to coordinator return fell from **30.49 seconds to 0.219 ms**.
Whole-program decode results returned in about 65 seconds instead of 86.
This is a partial improvement: window-commit mean was 27.08 seconds in the
candidate run, not better than 24.91 seconds. Neither configuration meets
the requested sub-millisecond system-latency target at 7 ms/block.

The existing static interface gathers results after the whole program; an
internal commit marker is not a network/client-delivery measurement.
The exact data, separate diagnostic populations, and reproduction tools are
under `target/evaluation-120core/seven-ms-20261124/`, including
`cadence-comparison.png`, `readout-return-comparison.png`, and
`final-investigation.json`.

### Compiled-coordinator qualification at 7 ms/block

The opt-in compiled mode above sustains this same W=1024, depth-1000, d=7,
p=1e-5 workload without accumulating backlog. Ten measured raw-sample replays after
two warm-ups produced **15,345,000 intermediate observations**, with all twelve
shots and every readout validated and no observed logical errors.

| Frame-availability clock | Mean | p99 | Maximum |
| --- | ---: | ---: | ---: |
| Latest required scheduled input | 1.756 ms | 3.131 ms | 15.429 ms |
| Own operation's scheduled end, including lookahead | 6.423 ms | 9.927 ms | 22.429 ms |

The first clock's p99.99 is 3.630 ms. No initial layers or slow values were
removed. The 21.014-second input schedule drains its final frames in another
3.32-4.83 ms. At most 1,030 scheduled-input-ready logical readouts are pending
at once, approximately one wave rather than the general coordinator's
million-readout backlog. The bulk response still arrives roughly
0.2 seconds later; use the local early-readout API for feedback rather
than treating the bulk response as a low-latency stream.

Coordinator preparation and replay peak at 51.43 GiB process-tree RSS. During measured
execution, frontend work averages about 7.08 CPU-core equivalents, the scheduler
0.33, decoder workers collectively 0.71, and the trace writer 0.15. These are
measured averages, not a guarantee of headroom under different burst patterns.

The initial compiled pilot still suffered a 188 ms spike with the old
64-entry lossless trace queue. In a matched two-shot control, increasing its
bounded burst capacity to 16,384 reduced mean/p99/max from 3.928/7.693/188.008 ms
to 1.705/3.051/6.009 ms. The subsequent ten-shot cohort above retains its larger
15.429 ms maximum. Tracing is part of the critical path when its producer queue
blocks; neither this buffer nor CPU affinity provides a hard real-time guarantee.

This qualification uses a different, explicitly restricted window policy; it
does not establish logical-error-rate equivalence to the general coordinator.
Readouts within a shot are correlated, and this finite shared-VM experiment is
not a production SLA. The retained native hash is
`78750ba771557b4f796b13af4b2b1456ace346a6cd496b50a311ffbde829b971`.
Figures, exact per-shot drain/coverage, inputs, source identities, and reproduction
tools are under `target/evaluation-120core/compiled-coordinator-20261124/`:
`compiled-performance.png`, `compiled-comparison.png`, and
`final-investigation.json`.

### Narrow workloads need fewer frontend workers

At **W=100**, keeping depth 1000, d=7, p=1e-5, and 7 ms/block, use
`{"worker_threads":8,"batch_size":8,"max_templates":2048}` as a measured
starting point. Keep two Tokio workers and the same 72 NUMA-local decoder
workers. No native-code or decoder-policy change was needed.

In matched two-shot controls, reducing only the compiled frontend pool from
24 to eight workers reduced mean frame-ready latency from 505 to 398 us and
frontend CPU from 1.62 to 0.50 core equivalents. All frontend execution remained
on NUMA node 0. Task batches of one or four were slower, not faster: finer
scheduling increased overhead at this width.

The subsequent ten-shot qualification, after two warm-ups, retained all
1,485,000 intermediate observations: **mean 381.7 us, median 356.5 us,
p99 992.3 us, maximum 5.233 ms**, measured from the latest required scheduled
input. The average meets 500 us; the tails do not. Including algorithmic
lookahead, mean latency from the operation's own scheduled end is 5.048 ms.
Neither clock measures external client delivery.

All twelve shots completed without observed logical errors. Pending
scheduled-input-ready readouts never exceeded 100, and final frames drained
0.42-4.08 ms after the 21.014-second schedule. Coordinator preparation/replay
peaked at 5.35 GiB RSS. Fresh W=100 samples, every tuning control, exact
timing derivation, and reproduction tools are retained under
`target/evaluation-120core/width100-20261124/`, including `performance.png`
and `final-investigation.json`. This is workload-specific evidence, not a
universal optimal worker count or a sub-500-us tail guarantee.

### Qualify the deployment workload

Qualify the actual backend, workload, cpuset, memory limit, and sustained input
rate on the intended machines. Include cancellation, reset, invalid input,
storage exhaustion, restart, and a sustained-load soak in acceptance testing.
Successful short benchmarks do not establish a production SLA.

The evaluation tools use exclusive output locks and handle normal termination.
Lock files can remain after a process exits; do not delete one to bypass an
active writer. After an uncatchable kill or host failure, check for orphaned
workers before resuming. Static-cache receipts and completed-point settings
must match; use a fresh work directory after changing inputs or builds rather
than editing receipts or reusing an unverified library.

Networked decoder pools are not implemented. Local NUMA results do not prove
cluster performance: network transport, bounded credits, cancellation,
reconnection, and exactly-once result commitment require separate qualification.
