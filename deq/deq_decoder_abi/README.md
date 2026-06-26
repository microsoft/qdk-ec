# deq-decoder-abi

Stable C ABI for [deq](../) dynamic decoder plugins: load a quantum-error-correction decoder at runtime from a binary-only shared library (`.so`/`.dylib`/`.dll`), with no recompilation of deq and no per-shot serialization.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](../../LICENSE)

## Overview

A decoder plugin is a shared library that exports five C functions (build, decode, destroy, version, last-error). deq `dlopen`s it once, checks the ABI version, builds one decoder per decoding hypergraph, and calls `decode` once per shot. The entire search runs in-process at native speed; the only one-time cost is the library load. Only plain-old-data crosses the boundary, so the ABI is independent of compiler version, standard-library layout, and the dependency graph on either side.

This crate provides three things:

- **Interface** (`interface` module): the frozen `extern "C"` signatures, ABI version, and status codes; the single source of truth for the boundary.
- **Plugin** (`plugin` module): a safe Rust trait `DeqDecoder` and the `declare_decoder!` macro, which export the C symbols with no `unsafe` code in the plugin.
- **Host loader** (`host` module, `host` feature): deq's side, which loads a plugin, validates its ABI version, and calls it safely.

## Writing a plugin (Rust)

Implement `DeqDecoder` and invoke `declare_decoder!`. Build the crate as a `cdylib`.

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

```toml
[lib]
crate-type = ["cdylib"]

[dependencies]
deq-decoder-abi = "0.1"
```

See [`reference_plugin/`](reference_plugin/) for a complete, buildable example.

## Writing a plugin (C / C++)

Export the five functions declared in [`include/deq_decoder.h`](include/deq_decoder.h). That header is the full contract; C++ plugins must not let exceptions escape across the boundary.

**Ownership: deq holds only the opaque handle pointer; it never frees the memory behind it.** deq calls `destroy` exactly once, on the worker that owns the handle, and that call is the plugin's only chance to clean up. So `destroy` must release everything `create` allocated; miss one `free` and that buffer leaks once per decoding hypergraph. In C, `free`/`delete` by hand; in C++, use RAII (`make_unique` + `.release()` in `create`, adopting the pointer back into a `unique_ptr` in `destroy`). The Rust SDK does this automatically: `destroy` drops the boxed handle.

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

The header is generated from the Rust source with [cbindgen](https://github.com/mozilla/cbindgen) (`reference_plugin/regenerate.sh`); a test asserts it stays in sync.

## Using a plugin from deq

Build `deq_runtime` with the `dylib` feature and select the plugin by path. `library` (the `.so`/`.dylib`/`.dll` path) and `parallel` (worker count) are deq's own fields; plugin-specific parameters go in the nested `decoder_config` object, which is the only part forwarded to the plugin. Other top-level keys are rejected.

```sh
deq server --decoder black-box-dyn-lib \
    --decoder-config '{"library":"/path/to/libmy_decoder.so","parallel":0,"decoder_config":{"max_iter":200}}' ...
```

## Data model

A decoding hypergraph is passed as compressed sparse row (CSR): per-edge probabilities plus the flattened vertex lists. The syndrome is a dense bit vector mirroring deq's `BitVector` (MSB-first packed bits). The decode result is the *subgraph*: the sparse indices of the selected hyperedges, written into a caller-owned buffer. These mirror deq's gRPC `DecodingHypergraph`, `BitVector`, and `ParityFactor` exactly.
