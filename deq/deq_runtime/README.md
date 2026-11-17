# deq-runtime

The runtime engine for **deq**, a dynamic and generic quantum error correction decoding
system. This crate hosts the decoders, coordinators, and gRPC services that
execute QEC decoding workloads, and also ships as a Python extension module
(`deq-runtime`) for use from Python.

The high-level Python frontend and tooling live in the companion
[`deq`](https://pypi.org/project/deq/) package.

See the [qdk-ec repository](https://github.com/microsoft/qdk-ec) for
documentation and examples.

## Tool metadata

Libraries, gadget/port/check-model/error-model definitions, and their gadget,
check-model and error-model instances carry JSON-compatible `metadata`.
Nested measurements, input/output ports, readouts, observables, checks, errors,
loss templates/sites and JIT stabilizers also carry metadata. JIT wrappers use
their `base` object's metadata. Use namespaced keys for custom tools; the
decoder does not interpret them. Exact large integers should be decimal strings:
protobuf `Struct` numbers use binary64.

Serialization and JIT compilation preserve metadata on corresponding objects.
Merge/canonicalization preserves annotations on retained measurements, readouts,
checks, errors, observables, boundary ports and loss sites. Newly synthesized
aggregate types/instances do not combine unrelated source annotations.
Type deduplication preserves distinct annotations; metadata still does not
change solver graph identity.

## Timed programs

The compiler stores the simulator's schedule in
`gadget.metadata["deq.simulation.timing"]`, for example
`{"start_ns": "3000000", "duration_ns": "1000000"}`. This is a tooling convention,
not a decoder schema field. The compiler and simulators interpret it; controllers
and native JIT compilation pass it through without timing validation.

Timed programs annotate every gadget, including identities and terminal gadgets
with no measurements. `start_ns` is required; omitted `duration_ns` is zero.
Both are nonnegative decimal strings. Completions must fit signed 63-bit
nanoseconds, be nondecreasing in program order, and follow input/readout
dependencies. Reusable gadget types do not define simulation deadlines.

With `strict_timing`, the STATIC and JIT simulators deliver outcomes at absolute
completion deadlines relative to each shot's `start_decoding` trace event.
Late work does not add another relative delay. The native server uses local
static-controller delivery with explicit gadget boundaries, avoiding gRPC and
preserving zero-measurement deadlines. The standalone remote static-simulator
API retains measurement-based Stim streaming; use `start_with_controller` for
per-gadget timing. Deadlines are best effort and may run late under OS/decoder load.
Controller initialization/reset remains outside the per-shot clock.
Input waits wake the async caller up to 500 microseconds before its deadline,
then cooperatively yield until delivery is due. This bounded precision window
avoids adding a sleeping-worker wake-up after the scheduled input time; inputs
are never intentionally delivered early.

Handwritten untimed Stim `#!delay` schedules remain supported. Decimal seconds
(including scientific notation) must represent whole nanoseconds. Directives
are parsed and accumulated as checked signed-63-bit-range integer nanoseconds,
without floating-point rounding. When both timing metadata and Stim delays are
present they must agree exactly; invalid or mismatching schedules are rejected.
JIT `registration_lead_seconds` controls registration relative to completion,
without blocking outcome submission on future error-model construction.

`jit::static_jit_compile_compact` exports shared check/error types using the
controller's structural cache and per-instance bindings. The Python
`static_jit_compile_compact` entry point exposes the same opt-in compact export.
The Python and Rust `static_jit_compile` entry points retain independently
materialized types and their original type/index layout. Both exports preserve
gadget metadata.

## Physical sampling

The Stim sampler constructs a noiseless reference using frame propagation for
CSS circuits consisting of X/Z preparation and measurement, CX, and Pauli
gates. Pauli gates become probability-one errors in this reference calculation;
physical noise is excluded. Other circuits retain Stim's tableau reference
construction. The sampler then applies the reference signs to fresh samples
of the original noisy circuit, preserving its measurement order.

Native file loading shares the input text and parses the circuit once.
The source CLI streams Stim exports and runs static compilation in a short-lived
process, reclaiming its allocator memory before sampling begins.

Seeded samples are reproducible within the same build, but their exact values
may change when the reference construction changes.
