# deq-decoder-abi

Stable C ABI for [deq](../) dynamic decoder plugins: load a
quantum-error-correction decoder at runtime from a binary-only shared library
(`.so`/`.dylib`/`.dll`), with no recompilation of deq and no per-shot
serialization.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../../LICENSE)

## Overview

A decoder plugin is a shared library that exports five required C functions
(build, decode, destroy, version, last-error). deq `dlopen`s it once, checks
the ABI version, builds one decoder per decoding hypergraph, and calls it
once per shot.

A plugin may also export `deq_decoder_decode_request` and
`deq_decoder_capabilities`. The request can include a syndrome, decoder seed,
per-request edge reweights, and structured loss. The capability bitmask
declares which optional features the plugin accepts. A plugin must export both
symbols or neither. deq rejects unsupported requests instead of dropping
their fields.

`DEQ_DECODER_CAPABILITY_OBSERVABLES` (`1 << 3`) requests graph metadata at
construction, not a per-shot field. A plugin declaring it must also export
`deq_decoder_create_with_observables`. The original constructor and all existing
layouts remain unchanged at ABI version 1, so existing plugins still load.
Older hosts that do not recognize this capability reject plugins advertising it.

The ABI crate and reference plugin use matching package versions, separate from
the binary ABI revision.

The decoder runs in-process. The only one-time cost is the library load. The
ABI passes only C-compatible values, so it does not depend on compiler
versions, standard-library layouts, etc.

This crate provides three things:

- **Interface** (`interface` module): the frozen `extern "C"` signatures, ABI
  version, and status codes; the single source of truth for the boundary.
- **Plugin** (`plugin` module): a safe Rust trait `DeqDecoder` and the
  `declare_decoder!` macro, which export the C symbols with no `unsafe` code
  in the plugin.
- **Host loader** (`host` module, `host` feature): deq's side, which loads a
  plugin, validates its ABI version, and calls it safely.

## Writing a plugin (Rust)

Implement `DeqDecoder` and invoke `declare_decoder!`. Build the crate as a
`cdylib`.

```rust
use deq_decoder_abi::plugin::{DeqDecoder, HypergraphView, OutputBuffer, SyndromeView};

struct MyDecoder { /* precomputed, immutable state */ }

impl DeqDecoder for MyDecoder {
    fn create(graph: HypergraphView<'_>, config_json: &[u8]) -> Result<Self, String> {
        // build an immutable decoder from the hypergraph + JSON config
        Ok(MyDecoder { /* ... */ })
    }

    fn decode(&mut self, syndrome: SyndromeView<'_>, out: &mut OutputBuffer) -> Result<(), String> {
        // push the selected hyperedge indices into `out`
        // syndrome.sparse_indices() yields the set vertices; syndrome.data() is dense
        Ok(())
    }
}

deq_decoder_abi::declare_decoder!(MyDecoder);
```

To accept optional request fields, declare the matching capability bits and
override `decode_request`:

```rust
use deq_decoder_abi::interface::{DEQ_DECODER_CAPABILITY_SEED, DeqDecoderCapabilities};
use deq_decoder_abi::plugin::DecodeRequest;

impl DeqDecoder for MySeededDecoder {
    const CAPABILITIES: DeqDecoderCapabilities = DEQ_DECODER_CAPABILITY_SEED;

    // create/decode as above ...

    fn decode_request(&mut self, request: DecodeRequest<'_>, out: &mut OutputBuffer) -> Result<(), String> {
        // Seed zero is valid. Reinitialize randomness from a supplied seed on every
        // call so a buffer retry returns the same ordered correction.
        let seed = request.decoder_seed.unwrap_or_else(|| self.default_seed());
        self.decode_seeded(seed, request.syndrome, out)
    }
}
```

The default `decode_request` accepts only a syndrome and delegates to
`decode`.  Existing plugins therefore need no changes. Requests with optional
fields fail unless the plugin declares the corresponding capabilities and
overrides the method.

For observable effects, add `DEQ_DECODER_CAPABILITY_OBSERVABLES` to
`CAPABILITIES` and read `graph.observable_flips(edge_index)` in `create`.
The macro exports the optional constructor automatically. Declaring only
`OBSERVABLES` does not require overriding `decode_request`.

`observable_flips` returns `None` for unknown effects, `Some(&[])` for known empty
effects, or a slice of unique opaque observable indices. Order is insignificant;
indices share a namespace across the graph and need not be contiguous or bounded
by the vertex count. Metadata is borrowed during construction and must be copied
if retained. The original constructor supplies unknown effects for every edge,
even for a plugin that supports observables.

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
deq-decoder-abi = "0.3"
```

See [`reference_plugin/`](reference_plugin/) for a complete, buildable
example.

## Writing a plugin (C / C++)

Export the five functions declared in
[`include/deq_decoder.h`](include/deq_decoder.h). That header is the full
contract; C++ plugins must not let exceptions escape across the boundary.

**Ownership: deq holds only the opaque handle pointer; it never frees the
memory behind it.** deq calls `destroy` exactly once, on the worker that owns
the handle, and that call is the plugin's only chance to clean up. So
`destroy` must release everything `create` allocated; miss one `free` and
that buffer leaks once per decoding hypergraph. In C, `free`/`delete` by
hand; in C++, use RAII (`make_unique` + `.release()` in `create`, adopting
the pointer back into a `unique_ptr` in `destroy`). The Rust SDK does this
automatically: `destroy` drops the boxed handle.

```c
#include "deq_decoder.h"
#include <stdlib.h>

typedef struct {
    /* whatever the decoder allocated in create: graphs, tables, scratch... */
    uint64_t *scratch;
    size_t    scratch_len;
} Decoder;

uint32_t deq_decoder_abi_version(void) { return DEQ_DECODER_ABI_VERSION; }

int32_t deq_decoder_create(uint64_t vertex_num, uint64_t edge_num,
                           const double *edge_probs, const uint64_t *edge_offsets,
                           const uint64_t *edge_vertices, size_t edge_vertices_len,
                           const char *config_json, void **out_handle) {
    if (!out_handle) return DEQ_DECODER_STATUS_INVALID_ARG;
    Decoder *d = calloc(1, sizeof(*d));          /* allocate the handle */
    if (!d) return DEQ_DECODER_STATUS_ERROR;
    d->scratch_len = edge_num;
    d->scratch = malloc(edge_num * sizeof(*d->scratch));   /* ...and its resources */
    if (!d->scratch) { free(d); return DEQ_DECODER_STATUS_ERROR; }
    /* build the decoder from the CSR hypergraph here... */
    *out_handle = d;                            /* hand ownership to deq */
    return DEQ_DECODER_STATUS_OK;
}

int32_t deq_decoder_decode(void *handle, uint64_t syndrome_size,
                           const uint8_t *syndrome_data, size_t syndrome_len,
                           uint64_t *out_ptr, size_t out_cap, size_t *out_len) {
    Decoder *d = (Decoder *)handle;
    /* run the decode; write up to out_cap indices, set *out_len to the count;
       return DEQ_DECODER_STATUS_BUFFER_TOO_SMALL if the count exceeds out_cap. */
    (void)d; *out_len = 0;
    return DEQ_DECODER_STATUS_OK;
}

void deq_decoder_destroy(void *handle) {       /* free EVERYTHING create allocated */
    Decoder *d = (Decoder *)handle;
    if (!d) return;                            /* null handle is a no-op */
    free(d->scratch);                          /* ...the owned resources first... */
    free(d);                                   /* ...then the handle itself */
}
```

The header is generated from the Rust source with
[cbindgen](https://github.com/mozilla/cbindgen)
(`reference_plugin/regenerate.sh`).

## Using a plugin from deq

Build `deq_runtime` with the `dylib` feature and select the plugin by
path. `library` (the `.so`/`.dylib`/`.dll` path) and `parallel` (worker
count) are deq's own fields; plugin-specific parameters go in the nested
`decoder_config` object, which is the only part forwarded to the
plugin. Other top-level keys are rejected.

```sh
deq server --decoder black-box-dyn-lib \
    --decoder-config '{"library":"/path/to/libmy_decoder.so","parallel":0,"decoder_config":{"max_iter":200}}' ...
```

## Data model

A decoding hypergraph is passed as compressed sparse row (CSR): per-edge
probabilities plus the flattened vertex lists. The syndrome is a dense bit
vector mirroring deq's `BitVector` (MSB-first packed bits). The decode result
is the *subgraph*: the sparse indices of the selected hyperedges, written
into a caller-owned buffer. These mirror deq's gRPC `DecodingHypergraph`,
`BitVector`, and `ParityFactor` exactly.

Observable-capable plugins receive the complete graph, including zero-prior
edges, with stable edge numbering. Each observable descriptor corresponds to
the same edge as its CSR row. In C, `known == false` requires `index_count == 0`;
`known == true` with zero count means no observable flips. A null descriptor
array means all effects are unknown. Observable indices are not renumbered when
passed to the plugin. Graph producers may leave effects unknown; the existing
window and monolithic coordinators currently do so.

Observable-capable plugins are called even for a zero syndrome without optional
request fields, since observable-only corrections may still be meaningful. The
plain zero-syndrome shortcut remains in place for other plugins.
