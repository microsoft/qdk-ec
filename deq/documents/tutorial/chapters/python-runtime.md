# Driving the runtime from Python

The deq runtime is normally launched as a server — `deq server …` boots a gRPC
endpoint that a simulator (or hardware controller) talks to over the network.
That model is great for production deployments but heavy for everyday Python
work: there's a subprocess to manage, every call goes through
protobuf-serialize → TCP → protobuf-deserialize, and a Jupyter notebook
suddenly has to coordinate two processes just to ask the decoder a question.

The `deq.runtime` Python package gives you the same runtime — same decoder,
same coordinator, same JIT controller — **inside the current Python process**.
You construct a `Runtime`, ask its services for things via `await`, and you're
done. No subprocess to babysit, no loopback TCP, no gRPC framing.
When you want external clients to also join in, one extra call exposes the
same services on a gRPC port too.

We will walk through:

1. The conceptual model — `Runtime` is a container; services are namespaced.
2. Hello, Runtime — the lifecycle of an in-process runtime, using the
   always-present `coordinator` service.
3. The `jit_controller` interface — for **dynamic logical circuits** (this is
   the surface most users want).
4. **Decoding is streaming, not call-and-return** — the most important
   thing to internalise before wiring the runtime into a real circuit.
5. Sampling measurements with the standalone `Sampler`.
6. Optional gRPC binding for external clients.
7. Async semantics, raw-bytes vs typed access, when to use which.

---

## When to use the in-process runtime

| If you want to…                                            | Use                                                                                                          |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Run a long, configurable LER simulation                    | `deq server …` (subprocess); see the [intro chapter](../README.md)                                          |
| Drive the runtime from a Jupyter notebook                  | `Runtime(...)` in-process                                                                                    |
| Embed decoding inside a Python control program             | `Runtime(...)` in-process                                                                                    |
| Stream dynamic instructions to the JIT controller          | `runtime.jit_controller` (this chapter)                                                                      |
| Plug your own Python decoder into the standard test suite  | the [Python decoder chapter](python-decoder.md) — orthogonal to this one                                     |
| Let external processes / machines also talk to the runtime | `await runtime.bind(addr)` — same instance serves both Python and remote callers                             |

The Python wrapper is implemented on top of PyO3's tokio bridge. Each
in-process method call goes directly to the same Rust trait the gRPC server
hosts via the `Local` variant of the service's client. Compared to going
through loopback gRPC this **skips the TCP round-trip and the tonic
Request/Response framing in both directions**, while still paying one
protobuf encode/decode pair at the Python↔Rust boundary (the FFI takes raw
bytes — see [Raw vs typed access](#raw-vs-typed-access) for how to skip the
Python-side half) plus PyO3's GIL hand-off and asyncio↔tokio scheduling per
`await`.

---

## The conceptual model

A `Runtime` is a container that owns the configured services:

```
Runtime
├── coordinator    (always present)       — the deq.bin RPC surface
├── jit_controller (when controller="jit") — the deq.jit RPC surface
└── bind(addr)                            — optional gRPC server on top
```

Services are exposed as **namespaced sub-objects** rather than flat methods on
the runtime, because the two surfaces have *different* semantics: the
coordinator's `execute` takes a `deq.bin.Instruction`, the JIT controller's
takes a `deq.jit.JitInstruction`. Mixing them on one object would either be
confusing (which one is `execute`?) or require ugly type sniffing. Namespacing
keeps the two surfaces honest.

The constructor mirrors the `deq server` CLI:

```python
from deq.runtime import Runtime

runtime = Runtime(
    decoder="black-box-relay-bp",                 # decoder algorithm
    decoder_config={"num_sets": 100},             # decoder-specific config
    coordinator="monolithic",                     # coordinator type
    coordinator_config=None,                      # coordinator-specific config
    controller="jit",                             # optional: "none" | "static" | "jit"
    controller_config=None,                       # controller-specific config
)
```

Decoder, coordinator, and controller names are the same strings the CLI
accepts. Per-component configs may be either JSON strings (as on the CLI) or
Python mappings (serialized via `json.dumps` automatically). The constructor
is synchronous because it has to start each service before the runtime is
usable — it drives the tokio worker pool while the GIL is released.

---

## Hello, Runtime

Let's start with the bare-minimum lifecycle: construct a `Runtime`, load a
library, look at it, shut it down. We use just the **coordinator** — the
always-present, lower-level `deq.bin` interface. The coordinator speaks
`deq.bin.Library` (gadget types with their check and error models
enumerated), so we compile the intro chapter's `small_example.deq` to a
`deq.bin.Library` via `static_jit_compiler` and hand it over.

The [JIT controller in the next section](#the-jit-controller--for-dynamic-logical-circuits)
removes this compile-up-front step and lets you stream `deq.jit`
instructions instead — that's the surface most users want for dynamic
circuits. But for the lifecycle tour, the coordinator is the right starting
point because it's always present and has no hidden machinery.

[Full script: `01_hello_runtime.py`](../examples/python-runtime/01_hello_runtime.py)

The interesting bits:

```python
import asyncio
from deq.circuit.parser import parse_file
from deq.compiler.jit_compiler import static_jit_compiler
from deq.runtime import Runtime
from deq.transpiler.jit_library_builder import build_jit_library

async def main() -> None:
    jit_library = build_jit_library(parse_file(REP_DEQ))
    bin_library = static_jit_compiler(jit_library)

    async with Runtime(
        decoder="black-box-relay-bp",
        coordinator="monolithic",
    ) as runtime:
        print(repr(runtime))
        print(f"  has_jit_controller = {runtime.has_jit_controller()}")
        print(f"  bound_port         = {runtime.bound_port()}")

        await runtime.coordinator.load_library(bin_library)
```

Things to notice:

- **Construction is synchronous.** `Runtime(...)` returns once every
  configured service has started; from then on, everything else is async.
- **`async with` triggers `shutdown()` on exit.** No leftover background
  tasks, no leaked gRPC ports — including the in-process tokio worker pool
  that backs every `await`.
- **`load_library` is a coroutine.** Methods on the runtime and its
  namespaced services return Python coroutines; the work runs on the
  runtime's dedicated tokio thread pool while Python's event loop is free
  in the meantime.
- **Typed proto objects directly.** `load_library` accepts the
  `deq.bin.Library` proto returned by `static_jit_compiler`. You can also
  pass raw `bytes` — see [Raw vs typed access](#raw-vs-typed-access) below.

Running the script:

```text
Compiled bin.Library: 3 gadget types, 1 port types
Runtime(unbound)
  has_jit_controller = False
  bound_port         = None
loaded library via runtime.coordinator
```

`bound_port` is `None` because we haven't called
[`runtime.bind`](#optional-grpc-binding); the runtime is purely in-process.
`has_jit_controller = False` because we didn't pass `controller="jit"`;
accessing `runtime.jit_controller` would raise `AttributeError`.

The coordinator's `execute(deq.bin.Instruction)` and `decode(Outcomes)`
methods are the bottom of the call stack. They take fully-compiled `deq.bin`
gadgets with concrete `RemoteGadget` references resolved — building those
by hand is tedious. The [JIT controller](#the-jit-controller--for-dynamic-logical-circuits)
below drives the same coordinator under the hood, but accepts `JitInstruction`s
where connectors are just `(gid, port)` pairs.

---

## The JIT controller — for dynamic logical circuits

This is the interface most online-decoding callers actually want. The
coordinator interface assumes a `deq.bin` library — fully compiled gadget
types with their check models and error models enumerated. That's fine for
*static* programs (you transpiled the whole thing offline), but
**dynamic** logical circuits — magic-state injection, feed-forward, lattice
surgery sequences chosen at runtime — don't have a single static decoding
hypergraph. The hypergraph has to grow as the circuit unfolds.

That's exactly what the [JIT controller](jit-basics.md) does: it takes a
*`JitLibrary`* (gadget types pre-analyzed with virtual stabilizer
measurements) plus a *stream of `JitInstruction`s*, and produces the right
`deq.bin` types and instructions for the coordinator on the fly.

To enable it, pass `controller="jit"`:

```python
runtime = Runtime(
    decoder="black-box-relay-bp",
    coordinator="monolithic",
    controller="jit",
)
jit = runtime.jit_controller   # available because controller="jit"
```

If you forget the controller argument, accessing `runtime.jit_controller`
raises `AttributeError` with a hint:

```text
AttributeError: jit_controller is not configured;
                pass controller="jit" to Runtime(...) to enable it
```

The JIT controller doesn't need a `filepath` config when used from Python:
omit `controller_config` (or pass `{}`) and the controller starts with an
empty library that you load via `await jit.load_library(...)`. This is the
intended path for in-process use — you build the library in Python and hand
it over without round-tripping to disk.

### A worked example

We re-use the [intro chapter's](../README.md) `small_example.deq` — a
distance-3 repetition code with `PrepareZ`, `Idle`, and `MeasureZ` gadgets —
and drive it through the JIT controller step by step.

[Full script: `03_jit_dynamic_circuit.py`](../examples/python-runtime/03_jit_dynamic_circuit.py)

Highlights of the script:

**Step 1 — build a `JitLibrary` in memory.** Instead of running
`deq transpile` and reading back the `.deq.jit` file, call `build_jit_library`
on the parsed `.deq` source directly:

```python
from deq.circuit.parser import parse_file
from deq.transpiler.jit_library_builder import build_jit_library

jit_library = build_jit_library(parse_file(REP_DEQ))
```

`jit_library` is a `deq.proto.deq_jit_pb2.JitLibrary` proto — the exact same
type the JIT controller's `load_library` accepts.

**Step 2 — start the runtime with `controller="jit"` and a real coordinator.**

```python
async with Runtime(
    decoder="black-box-relay-bp",
    coordinator="monolithic",
    controller="jit",
) as runtime:
    jit = runtime.jit_controller
    await jit.load_library(jit_library)
```

We pick the monolithic coordinator because it decodes a connected subgraph
as a whole, which is closer to how production decoding actually works (the
`window` coordinator from the [intro chapter's](../README.md) LER
simulation behaves similarly with a sliding window). `black-box-relay-bp`
is the same relay-BP decoder the [intro chapter](../README.md) LER
simulation uses, with its built-in defaults — no `decoder_config` needed.

`load_library` registers the `JitGadgetType`s with the JIT compiler *and*
forwards the underlying `bin` port/gadget types to the coordinator so
subsequent `execute` calls can resolve them. You can call `load_library`
multiple times to accumulate types from different libraries — useful when
loading a base library plus user-defined extensions.

**Step 3 — stream instructions one at a time.** Each `execute` returns the
assigned `gid` of the newly instantiated gadget:

```python
prep_gid = await jit.execute(_make_instruction(gtype=g_prep, gid=1))
idle_gid = await jit.execute(
    _make_instruction(gtype=g_idle, gid=2, connectors=[_connector(prep_gid)])
)
meas_gid = await jit.execute(
    _make_instruction(gtype=g_meas, gid=3, connectors=[_connector(idle_gid)])
)
```

The `connectors` field wires this gadget's input ports to the output ports of
previously-instantiated gadgets. Each `Connector` is a `(gid, port)` pair.
This is exactly the same `Gadget.Connector` proto the static format uses —
the JIT controller compiles the virtual stabilizer measurements into concrete
`bin.RemoteGadget` references on the fly.

**Step 4 — feed in outcomes, read back readouts.**

```python
prep_outcomes = coord_pb.Outcomes(
    gid=prep_gid, outcomes=util_pb.BitVector(size=0, data=b"")
)
idle_outcomes = coord_pb.Outcomes(
    gid=idle_gid, outcomes=util_pb.BitVector(size=2, data=b"\x00")
)
meas_outcomes = coord_pb.Outcomes(
    gid=meas_gid, outcomes=util_pb.BitVector(size=3, data=b"\x00")
)
prep_ro, idle_ro, meas_ro = await asyncio.gather(
    jit.decode(prep_outcomes),
    jit.decode(idle_outcomes),
    jit.decode(meas_outcomes),
)
```

Two things to notice that the previous coordinator-only example did **not**
show:

- **We `decode` every gadget**, including `PrepareZ`, which has zero physical
  measurements. The monolithic coordinator commits the whole connected
  subgraph at once; it cannot do that until each gadget has been told what
  outcomes it produced (even an empty `BitVector`).
- **We submit them concurrently with `asyncio.gather`.** Each individual
  `decode` would sit pending on its own — the coordinator only commits when
  every gadget in the subgraph has reported in, and that's the trigger that
  resolves all of them at once. The [next section](#decoding-is-streaming-not-call-and-return)
  unpacks this streaming behaviour with a deliberately incomplete submission.

The JIT controller's `decode` also waits internally for the gadget's
background error-model-loading task to finish (it can only resolve once all
the gadget's *unfinished* checks have been consumed by downstream gadgets —
see [deq-JIT Basics](jit-basics.md)) before forwarding outcomes to the
coordinator.

Running the script:

```text
Transpiling small_example.deq...
  3 gadget types: ['Idle', 'MeasureZ', 'PrepareZ']
Loaded JitLibrary into Runtime(unbound, controller=jit)
  PrepareZ -> gid=1
  Idle     -> gid=2
  MeasureZ -> gid=3
PrepareZ readouts: size=0
Idle     readouts: size=0
MeasureZ readouts: size=1  (the logical Z bit)
```

`PrepareZ` and `Idle` have zero readouts (they only feed checks to the
decoder; they don't produce logical bits). `MeasureZ` has one readout —
the single logical-Z measurement of the encoded qubit.

### Batched submission

The single-shot `execute` / `decode` pattern is convenient for streaming
hand-written instructions, but a typical program knows many gadgets up front.
`batch_execute` and `batch_decode` submit a list in one call, letting the
runtime schedule everything on its worker pool simultaneously (respecting
connector dependencies for compilation order).

[Full script: `04_jit_batched.py`](../examples/python-runtime/04_jit_batched.py)

```python
# Three parallel chains, each PrepareZ → Idle → MeasureZ.
instructions: list[jit_pb.JitInstruction] = [...]
prep_gids, idle_gids, meas_gids = [...], [...], [...]

assigned = await jit.batch_execute(instructions)   # all 9 gadgets in one call

# Feed in outcomes for *every* gadget — prep with zero bits, idle with 2,
# measure with 3. With the monolithic coordinator the subgraph won't
# commit until every gadget has reported in, so we put them all in a
# single batch_decode call.
all_outcomes = (
    [coord_pb.Outcomes(gid=g, outcomes=util_pb.BitVector(size=0, data=b"")) for g in prep_gids]
    + [coord_pb.Outcomes(gid=g, outcomes=util_pb.BitVector(size=2, data=b"\x00")) for g in idle_gids]
    + [coord_pb.Outcomes(gid=g, outcomes=util_pb.BitVector(size=3, data=b"\x00")) for g in meas_gids]
)
readouts = await jit.batch_decode(all_outcomes)
```

Running the script:

```text
batch_execute returned 9 gids in input order
batch_decode returned 9 readouts (3 prep, 3 idle, 3 measure)
  MeasureZ gid=3 -> readout_size=1
  MeasureZ gid=6 -> readout_size=1
  MeasureZ gid=9 -> readout_size=1
```

A few things worth knowing about `batch_execute`:

- **Every instruction must specify a non-zero `gid`.** The batch path rejects
  auto-assigned gids because it needs to know the gid up front to wire
  connectors between siblings in the same batch.
- **Connectors may point to gids in the same batch or to previously-executed
  gids.** The runtime topologically schedules within the batch.
- **Failure modes are structured errors** (`MissingGadget`, `ZeroGid`,
  `DuplicateGid`, `MissingDependency`) and surface as Python `ValueError`s
  with a descriptive message — easier than parsing a tonic Status.

`batch_decode` is simpler: it dispatches all the decodes concurrently and
returns the readouts in input order. Use it whenever you have a batch of
gadgets to decode that can all proceed independently.

### Concurrency comes for free

The runtime is multi-threaded under the hood, so multiple in-flight calls
**actually overlap** — they don't get serialised by the GIL. When you
already have everything in hand, `batch_execute` + `batch_decode` is the
ergonomic choice. When decodes arrive at different times (the typical
online-decoding pattern), `asyncio.gather` over individual `decode` calls
is the equivalent pattern:

[Full script: `02_concurrent_decode.py`](../examples/python-runtime/02_concurrent_decode.py)

```python
# Many independent prep → idle → measure chains; one batch_execute.
await jit.batch_execute(instructions)

# All 24 decodes dispatched at once. Each chain commits independently as
# its three decodes arrive on the monolithic coordinator's worker pool.
results = await asyncio.gather(*[jit.decode(o) for o in all_outcomes])
```

The script sets up 8 independent `PrepareZ → Idle → MeasureZ` chains,
fires off all 24 `decode` calls in one `gather`, and prints how long the
whole thing took. The returned list preserves input order — that's a
property of `asyncio.gather`, not the runtime.

---

## Decoding is streaming, not call-and-return

If you're new to QEC, this is the single most important section in the
chapter. The previous example used `coordinator="monolithic"` and `gather`d
every gadget's `decode` call at once — that quietly avoided the trap, but
it didn't *show* it. The trap is this: with a real coordinator (`window` or
`monolithic`), `decode(outcomes)` is not a function that returns when
called. It is a *submission* into a streaming pipeline that resolves only
when the decoder has gathered enough surrounding syndrome information to
commit a logical readout. Submit one gadget's outcomes alone and the call
sits **pending**, possibly forever. This isn't a bug — it's the entire
reason QEC decoding works.

(There is also a `coordinator="naive"` setting that bypasses real
decoding entirely and returns random readouts the instant outcomes arrive.
It exists for runtime plumbing tests; *never* assume real coordinators
behave the same way, and we don't use it anywhere in this chapter.)

Two consequences flow from this for the Python user:

- **Spawn `decode` calls as `asyncio` tasks, don't `await` them in
  isolation.** Each pending `decode` is just a coroutine on the tokio
  runtime; `asyncio.create_task` lets it sit there until the decoder is
  ready, while you feed in the rest of the outcomes.
- **Every executed gadget needs a `decode()` call, even ones with zero
  measurements** (like `PrepareZ`). The coordinator tracks per-gadget
  submission, not per-measurement; without the prep's empty `decode` the
  subgraph never completes and *all* downstream decodes stay pending.

[Full script: `06_jit_streaming_decode.py`](../examples/python-runtime/06_jit_streaming_decode.py)

The example builds a `PrepareZ → Idle → Idle → MeasureZ` chain, switches the
runtime to the window coordinator, and demonstrates both points above. The
shape of the demonstration is:

```python
async with Runtime(
    decoder="black-box-relay-bp",
    coordinator="window",
    coordinator_config={"buffer_radius": 1, "lookahead_radius": 0},
    controller="jit",
) as runtime:
    jit = runtime.jit_controller
    await jit.load_library(jit_library)
    # ... execute prep, idle1, idle2, measure ...

    # Submit ONLY idle1's outcomes. Spawn the decode as a task so it can
    # sit pending while we observe its state.
    idle1_decode = asyncio.create_task(jit.decode(_outcomes(gid=2, num_bits=2)))

    await asyncio.sleep(1.0)
    assert not idle1_decode.done()  # the window still wants more context

    # Now submit the rest, including prep's zero-bit outcomes.
    prep_decode  = asyncio.create_task(jit.decode(_outcomes(gid=1, num_bits=0)))
    idle2_decode = asyncio.create_task(jit.decode(_outcomes(gid=3, num_bits=2)))
    meas_decode  = asyncio.create_task(jit.decode(_outcomes(gid=4, num_bits=3)))

    # With every gadget loaded, the window can commit. All four decodes complete.
    prep_ro, idle1_ro, idle2_ro, meas_ro = await asyncio.gather(
        prep_decode, idle1_decode, idle2_decode, meas_decode,
    )
```

Running the script:

```text
Executed: prep(gid=1) → idle1(gid=2) → idle2(gid=3) → measure(gid=4)

Submitting outcomes for idle1 only (gid=2)...
  After 1s: idle1's decode is still pending. The window
  decoder cannot commit a gadget without seeing the syndromes
  of its neighbours.

Submitting outcomes for prep, idle2, and measure...
  prep   readouts.size = 0
  idle1  readouts.size = 0
  idle2  readouts.size = 0
  meas   readouts.size = 1   (the logical Z bit)
```

The same pattern scales to longer programs. In an online decoding loop you
typically have many gadgets in flight at once: `execute` is called as each
logical instruction is decided, `decode` is called as each gadget's physical
measurements arrive, and the coordinator commits *windows* of readouts as
each window accumulates the context it needs. None of these calls is
strictly request-response; they all feed a single streaming pipeline that
the window coordinator paces.

### Mental model

If you take one thing away: `await jit.decode(outcomes)` does not mean *“ask
for the readout of this gadget”* — it means *“tell the runtime that these
physical measurement outcomes are now available, and give me back the
logical readout for this gadget whenever the decoder can commit it”*. The
two are not the same, and conflating them is the easiest way to deadlock a
real-decoder pipeline.

This also explains why the [intro chapter's](../README.md) full LER
simulation goes through `deq server` rather than a Python loop: the
`--simulator` configuration drives a deterministic firehose of `execute`s
and `decode`s on every gadget in the program, in exactly the order the
window coordinator needs to keep committing. From Python, you take on that
responsibility yourself — which is fine for interactive work and
prototyping, but worth being deliberate about.

### Window decoding commits as the circuit grows

Example 06 used the window coordinator but submitted outcomes for the
*entire* short chain at once — so the window committed everything in one
shot. That can leave the impression that decoding only fires after the
circuit is finished. With the **monolithic** coordinator that's
unavoidable; with the **window** coordinator it isn't.

The window decoder commits a gadget as soon as its **window**
(``buffer_radius`` hops around the gadget) is fully analyzed. There are
two gating constraints:

1. **Every gadget in the window needs an *error model* loaded.**
   The JIT compiler holds a gadget's error model open until the
   gadget's output ports are connected to a downstream gadget — so a
   gadget at the **open frontier** (no downstream yet) never loads its
   error model.
2. **Every gadget in the window needs *outcomes* delivered.**
   ``decode_single`` only forwards outcomes to the coordinator after
   the JIT-level error-model wait passes — so the frontier's outcomes
   never reach the coordinator either.

The upshot: a gadget can commit while the circuit is still being
extended, as long as it sits far enough behind the frontier that none of
its 1-hop neighbours *is* the frontier.

[Full script: `07_window_partial_streaming.py`](../examples/python-runtime/07_window_partial_streaming.py)

The example executes ``PrepareZ → Idle → Idle → Idle`` — no ``MeasureZ``,
so ``idle3`` is the open frontier — and submits outcomes for all four:

```python
async with Runtime(
    decoder="black-box-relay-bp",
    coordinator="window",
    coordinator_config={"buffer_radius": 1, "lookahead_radius": 0},
    controller="jit",
) as runtime:
    jit = runtime.jit_controller
    await jit.load_library(jit_library)
    # ... execute prep, idle1, idle2, idle3 (no MeasureZ) ...

    prep_decode  = asyncio.create_task(jit.decode(_outcomes(gid=1, num_bits=0)))
    idle1_decode = asyncio.create_task(jit.decode(_outcomes(gid=2, num_bits=2)))
    idle2_decode = asyncio.create_task(jit.decode(_outcomes(gid=3, num_bits=2)))
    idle3_decode = asyncio.create_task(jit.decode(_outcomes(gid=4, num_bits=2)))

    # prep and idle1 commit — neither has the frontier in its 1-hop zone.
    prep_ro, idle1_ro = await asyncio.gather(prep_decode, idle1_decode)

    # idle2 and idle3 stay pending — idle3 is the frontier; idle2's window
    # {idle1, idle2, idle3} includes it. Wait a beat to confirm.
    await asyncio.sleep(1.0)
    assert not idle2_decode.done() and not idle3_decode.done()

# Leaving `async with` triggers runtime.shutdown(), which fires the
# in-process cancellation tokens. Every pending decode resolves with a
# RuntimeError; no manual `.cancel()` needed.
for t in (idle2_decode, idle3_decode):
    try:
        await t
    except RuntimeError:
        pass
```

Running the script:

```text
Executed: prep(gid=1) → idle1(gid=2) → idle2(gid=3) → idle3(gid=4)
           (no MeasureZ — idle3 is the open frontier)
Submitted decodes for prep, idle1, idle2, idle3.
  prep   readouts.size = 0  (committed)
  idle1  readouts.size = 0  (committed)
After 1s: idle2 and idle3 are still pending — both wait on the open frontier.
After shutdown: idle2.decode raised: RuntimeError
After shutdown: idle3.decode raised: RuntimeError
```

This is the real picture for online decoding. As you stream
`execute`/`decode` pairs into a long-running computation, the window
decoder commits a trailing prefix on every step, while the active
frontier (the last `buffer_radius` gadgets) stays pending. The runtime's
shutdown sequence — triggered automatically by `async with` exit —
propagates cancellation into every in-process service, so those pending
decodes resolve with a `RuntimeError` rather than leaking past the event
loop. Notebooks, Ctrl-C, exceptions and partial-circuit experiments all
shut down cleanly with no special handling required.

**Ctrl-C works the same way.** Pressing Ctrl-C in a notebook or terminal
raises `KeyboardInterrupt` through `asyncio.run`, the `async with` exit
still runs `shutdown()`, and the cancellation propagation does the rest —
no custom signal handler needed.

---

## Sampling measurements: the standalone `Sampler`

The runtime decodes *physical measurement outcomes* — which have to come
from somewhere. On production hardware that's a quantum computer
executing the circuit; while developing or evaluating a code, you stand
in for the hardware with a simulator.
[`deq.runtime.Sampler`](../../../deq/runtime/__init__.py) is that
simulator.

It takes a ``.deq`` file plus the name of one of its ``PROGRAM`` blocks,
transpiles the program to a Stim circuit internally, and produces
per-gadget-partitioned shots of physical outcomes ready to feed straight
into a decoder — `runtime.coordinator`, `runtime.jit_controller`, or a
completely external one.

[Full script: `08_sampler.py`](../examples/python-runtime/08_sampler.py)

```python
from pathlib import Path
from deq.runtime import Sampler

sampler = Sampler(Path("program.deq"), program="Simulation", seed=42)
shots = sampler.sample(num_shots=100)
```

`sample()` returns a list of `deq.proto.simulator_pb2.ShotSample` proto
messages, one per shot. Each carries:

- `outcomes: util_pb2.BitVector` — the physical measurement outcomes
  in order, packed into bytes.

Pass `raw=True` to skip the proto parse and get a `list[bytes]` instead —
handy when you're forwarding the shots to a socket or to
`runtime.coordinator.raw.decode(...)` without ever needing the typed view.

The sampler also exposes:

- `sampler.program` — the program name you passed in.
- `sampler.library` — the compiled `JitLibrary` (with the program
  appended to its `program` field). Pass it straight to
  `static_jit_compiler(...)` then `runtime.coordinator.load_library(...)`,
  or to `runtime.jit_controller.load_library(...)`.
- `sampler.instructions` — the JIT instructions in execution order, with
  pre-assigned gids 1, 2, 3, …. Use these to drive `jit.batch_execute`
  and to pair gids with the per-gadget outcome chunks.
- `sampler.partition` — the per-gadget measurement counts in
  `instructions` order. The shape the runtime uses to slice each shot's
  flat `outcomes` record.
- `sampler.split_outcomes(bv)` — slices a flat `BitVector` into per-gadget
  chunks (byte-identical to what `decode` expects).
- `sampler.circuit` — the transpiled Stim circuit string. Exposed
  for inspection / debugging.

For in-memory ``.deq`` source text (no file on disk), use
`Sampler.from_source(deq_source, program=...)`.

### Per-gadget decoding

`sampler.split_outcomes(shot.outcomes)` returns one `BitVector`
per gadget (in `sampler.instructions` order), each byte-identical to what
`decode` expects — same MSB-first `BitVector` layout the rest of the
runtime uses. The natural pairing is the JIT controller, because
`sampler.instructions` are already `JitInstruction`s ready for
`batch_execute`, and `batch_decode` handles the streaming-pipeline
requirement the [previous section](#decoding-is-streaming-not-call-and-return)
warned about (the monolithic coordinator needs every gadget's outcomes
before it commits any of them — `batch_decode` submits them all
together):

```python
from deq.proto import coordinator_pb2 as coord_pb
from deq.runtime import Runtime, Sampler

sampler = Sampler("program.deq", program="Simulation", seed=42)
instructions = list(sampler.instructions)

async with Runtime(
    decoder="black-box-relay-bp",
    coordinator="monolithic",
    controller="jit",
) as runtime:
    jit = runtime.jit_controller
    await jit.load_library(sampler.library)

    for shot in sampler.sample(num_shots=1000):
        await jit.reset()
        await jit.batch_execute(instructions)

        chunks = sampler.split_outcomes(shot.outcomes)
        readouts = await jit.batch_decode([
            coord_pb.Outcomes(gid=instr.gadget.gid, outcomes=chunk)
            for instr, chunk in zip(instructions, chunks)
        ])
```

If you'd rather drive the coordinator directly: it has no
`batch_execute` / `batch_decode`, so the equivalent flow is to compile
the library with `static_jit_compiler(sampler.library)`, call
`runtime.coordinator.execute(instr)` for every `bin.Instruction` in
`bin_library.program` (after each `reset`), and `asyncio.gather` the
per-gadget `runtime.coordinator.decode(...)` calls per shot. The JIT
controller version above is shorter for the same result.

Running the script (sampling 5 shots of the `Simulation` program in
[`language/03_with_idle.deq`](../examples/language/03_with_idle.deq) and
then decoding each shot through the JIT controller):

```text
sampler = Sampler(program='Simulation', gadgets=3)
  program       = Simulation
  instructions  = 3 gadgets
    [0] gid=1 PrepareZ (0 measurements)
    [1] gid=2 Idle     (2 measurements)
    [2] gid=3 MeasureZ (3 measurements)

sampled 5 shots:
  shot 0: flat='00000'  per-gadget=['', '00', '000']
  shot 1: flat='00000'  per-gadget=['', '00', '000']
  shot 2: flat='00000'  per-gadget=['', '00', '000']
  shot 3: flat='00000'  per-gadget=['', '00', '000']
  shot 4: flat='00100'  per-gadget=['', '00', '100']

decoded 5 shots through runtime.jit_controller:
  shot 0: readouts=['', '', '0']
  shot 1: readouts=['', '', '0']
  shot 2: readouts=['', '', '0']
  shot 3: readouts=['', '', '0']
  shot 4: readouts=['', '', '0']
```

The per-gadget and readouts lists are position-aligned with the
instructions table above: index 0 is PrepareZ (no measurements, no
logical readout), index 1 is Idle (2 measurement bits, no logical
readout), index 2 is MeasureZ (3 measurement bits, 1 logical readout).
Shot 4 is the interesting one — the third measurement gadget got a
``'100'`` physical chunk (one bit flipped relative to the other shots)
but the decoder still produces logical readout ``'0'``, recognising the
flip as a measurement error and recovering the true logical value.

A few practical notes:

- **The ``.deq`` file is everything.** Circuit *and* noise are baked in.
  Tweaking error rates means editing the ``.deq`` source.
- **Programs are identified by name.** A single ``.deq`` file can contain
  several `PROGRAM` blocks; pass the `program=` kwarg to pick. An unknown
  name raises `KeyError` with the list of available programs.
- **Deterministic with `seed=`.** Two samplers constructed with the same
  seed and program produce byte-identical shots, so tests can pin them
  down.
- **`skip_shots=N` is a fast-forward.** It pulls and discards the first N
  shots in the background, so a parallel job can pick up at shot N
  without re-sampling.

---

## Optional gRPC binding

Sometimes you want the same runtime to serve both an in-process Python loop
*and* an external client — a separate simulator process, a hardware
controller, or the `deq` CLI talking over loopback. One call enables it:

[Full script: `05_grpc_bind.py`](../examples/python-runtime/05_grpc_bind.py)

```python
async with Runtime(
    decoder="black-box-relay-bp",
    coordinator="monolithic",
    controller="jit",
) as runtime:
    url = await runtime.bind("[::]:0")   # OS-chosen port
    print(f"gRPC server bound at {url}")
    print(f"  bound_port = {runtime.bound_port()}")
    # In-process calls keep going through the Local clients (no tonic,
    # no TCP); remote clients can now connect via `url`.
    await runtime.jit_controller.load_library(jit_library)
```

The address syntax follows the CLI: `"127.0.0.1:50051"` for a fixed port,
`"[::]:0"` (the default) to let the OS pick. The returned URL has the
unspecified-address-rewrite applied so it's safe to hand directly to a gRPC
client.

In-process calls still go through the `Local` clients (no tonic, no TCP);
remote calls hit the tonic router on the bound socket. The two share the same
service `Arc`s, so they see the same state (a remote `execute` updates the
same gadget table an in-process `coordinator.execute` would).

`await runtime.shutdown()` (or the context manager's exit) cancels the serve
loop and releases the port.

---

## Raw vs typed access

The Rust extension's pyclasses (`deq_runtime.Runtime`, `…Coordinator`,
`…JitController`) take **raw protobuf bytes** on the FFI boundary. The Python
wrappers in `deq.runtime` add a thin layer that accepts and returns typed
`*_pb2` proto messages — paying one extra `SerializeToString` on inputs and
one `ParseFromString` on outputs to do so. That's almost always the right
tradeoff: typed messages give you attribute access, IDE help, and structural
validation, while the serialization cost is small next to even a fast decoder
run.

The Rust side still has to call `prost::Message::decode` regardless of which
path you take — the trait methods want typed structs, not bytes. So `.raw`
saves one of the two encode/decode pairs, not both.

For a latency-sensitive loop where you already have serialized bytes (for
example, coming straight off a socket), you can skip the Python-side
serialization via the `.raw` accessor on any wrapper, or the re-exported raw
classes:

```python
from deq.runtime import RawRuntime, Runtime

# Typed (the common case)
async with Runtime(
    decoder="black-box-relay-bp", coordinator="monolithic", controller="jit"
) as runtime:
    await runtime.jit_controller.load_library(my_library_proto)

    # Raw — `.raw` peels off the typed wrapper and returns the underlying
    # `deq_runtime.JitController` pyclass; it takes bytes directly.
    await runtime.jit_controller.raw.load_library(my_library_bytes)

# Equivalent: skip the typed wrapper entirely. RawRuntime is a re-export
# of the Rust pyclass; it does NOT support `async with`, so call
# `shutdown()` explicitly when you're done.
runtime_raw = RawRuntime(
    decoder="black-box-relay-bp", coordinator="monolithic", controller="jit"
)
try:
    await runtime_raw.jit_controller.load_library(my_library_bytes)
finally:
    await runtime_raw.shutdown()
```

The `decode` and `batch_decode` methods on the typed wrappers parse the
returned bytes into a `Readouts` proto by default. Pass `raw=True` to skip the
parse:

```python
parsed: coord_pb.Readouts = await coord.decode(outcomes)        # default
raw_bytes: bytes = await coord.decode(outcomes, raw=True)       # opt out

parsed_list = await jit.batch_decode(outcomes_list)             # default
raw_list = await jit.batch_decode(outcomes_list, raw=True)      # opt out
```

---

## What goes where: a summary

| You have…                                          | Reach for…                                  |
| -------------------------------------------------- | ------------------------------------------- |
| A pre-compiled `deq.bin.Library`                   | `runtime.coordinator`                       |
| A `deq.jit.JitLibrary` and dynamic instructions    | `runtime.jit_controller`                    |
| Known-ahead-of-time batch of instructions          | `runtime.jit_controller.batch_execute`      |
| Many outcomes to decode concurrently               | `runtime.jit_controller.batch_decode` (or `asyncio.gather` on individual `decode`s) |
| A pending decode you don't want to await yet       | `asyncio.create_task(jit.decode(...))` and feed in the other gadgets' outcomes before you `await` it |
| Per-gadget measurement shots from a `.deq` program | standalone `Sampler(deq_path, program="…")` — slice `shot.outcomes` with `sampler.split_outcomes(…)` and feed each chunk to the coordinator |
| External processes that also need to talk to it    | `await runtime.bind(addr)`                  |
| A hot loop that already has serialized bytes       | the `.raw` accessor (or `RawRuntime` direct)|
| A gRPC service not yet wrapped in Python (e.g. `StaticController`, raw decoder RPCs) | `await runtime.bind(addr)`, then a generated gRPC client against the [`.proto` files](../../../proto/) — same wire format, same semantics |

| Coordinator     | When `decode(outcomes)` returns                                                                |
| --------------- | ---------------------------------------------------------------------------------------------- |
| `monolithic`    | When every gadget in the connected subgraph has had `decode()` called and all outputs are connected. |
| `window`        | When the gadget's window has accumulated `buffer_radius` hops of syndrome context on every open side. |
| `naive`         | Immediately, with random readouts — no real decoding. Plumbing tests only; not used in this chapter. |

| Service             | Always available? | Methods                                                              |
| ------------------- | ----------------- | -------------------------------------------------------------------- |
| `coordinator`       | yes               | `load_library`, `execute`, `decode`, `reset`                         |
| `jit_controller`    | when `controller="jit"` | `load_library`, `execute`, `batch_execute`, `decode`, `batch_decode`, `reset` |

| Lifecycle on `Runtime` | What it does                                                                  |
| ---------------------- | ----------------------------------------------------------------------------- |
| `Runtime(...)`         | Constructs services, starts background tasks. **Synchronous.**                |
| `await bind(addr)`     | Optionally exposes services on a gRPC port. Returns the URL.                  |
| `bound_port()`         | Returns the bound port, or `None` if not bound.                               |
| `await shutdown()`     | Cancels the gRPC serve loop (if bound) and waits for it. Safe to call twice. |
| `async with Runtime`   | Calls `shutdown()` on exit.                                                   |
