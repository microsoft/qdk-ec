# deq-runtime

The runtime engine for **deq**, a dynamic and generic quantum error correction decoding
system. This crate hosts the decoders, coordinators, and gRPC services that
execute QEC decoding workloads, and also ships as a Python extension module
(`deq-runtime`) for use from Python.

The high-level Python frontend and tooling live in the companion
[`deq`](https://pypi.org/project/deq/) package.

See the [qdk-ec repository](https://github.com/microsoft/qdk-ec) for
documentation and examples.

## Relay-BP input limits

The `relay-bp` and `relay-bp-f32` decoders accept at most 10,000,000 vertices,
10,000,000 hyperedges, and 10,000,000 total vertex-edge incidences per graph.
An incidence is one vertex listed in one hyperedge, including zero-prior edges.
Loading or decoding a graph above any limit returns `INVALID_ARGUMENT` before
constructing the solver.

These limits do not bound total memory use or concurrent requests.
