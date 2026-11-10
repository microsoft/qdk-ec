# Decoder performance: the algorithm is only part of the cost

A fast MWPM decoding algorithm does not automatically make a fast QEC decoding system.
Measurements must reach the runtime, logical instructions may need just-in-time
compilation, windows must become ready, and decoding graphs and solver instances
must be prepared. If any of those stages falls behind the measurement stream,
later logical results accumulate delay even when individual solves are quick.

This chapter presents recorded measurements of three ways of decoding the same surface-code lattice-surgery
workload. The goal is **latency and sustainable throughput**, not a comparison of
logical error rates.


All three use sliding window decoding with buffer radius 1 and lookahead radius 0.
We choose radius of 1 because each lattice surgery gadget already has d rounds of SE.
They share the same worker count, circuit definitions, noise parameter, and
measurement cadence.

| Configuration | Controller | Graph preparation | Solver lifetime |
| --- | --- | --- | --- |
| MWPF c=10 | JIT | Full hypergraph | Fresh instance per request |
| MWPM | JIT | Unsupported hyperedges filtered by the MWPM adapter | Fresh instance per request |
| Static + persistent MWPM | Static | Same graphlike preparation | Loaded graphs and pooled solver instances reused |

### Decoder choice

MWPF can use hyperedges describing errors that affect more than two checks.
MWPM instead works with graphlike edges, including boundary edges. This difference
changes both the decoding accuracy and its computational cost.

The MWPF configuration explicitly sets `cluster_node_limit=10`. It limits the
cluster search before falling back to union-find, reducing work compared with a
larger limit. It is **not a wall-clock deadline** and is not an assertion that
this setting is universally preferable.

### Move repeated preparation out of the request path

