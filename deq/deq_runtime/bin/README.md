This folder is the place for putting binary protobuf files and Stim circuits.
When testing Rust code that references these files via command line, this folder
provides a shortcut.

## Generating example files

Use the deq JIT pipeline to produce a `.deq.jit` library and a `.stim` circuit
from a `.deq` source file (e.g. the repetition code fixture in `tests/circuit/`):

```sh
# 1. Transpile the .deq file into a .deq.jit library + .stim circuit
deq transpile tests/circuit/repetition_code/repetition_code_d3.deq \
    --out bin/repetition_code.deq.jit \
    --program MemoryExperiment
```

## Running simulations

Once the `.deq.jit` and `.stim` files are generated, start a simulation:

```sh
# JIT controller + JIT-static simulator (recommended)
cargo run --features simulator,cli -- server \
    --coordinator monolithic \
    --controller jit \
    --controller-config '{"filepath":"bin/repetition_code.deq.jit"}' \
    --simulator jit-static \
    --simulator-config '{
        "filepath":"bin/repetition_code.stim",
        "jit_library_filepath":"bin/repetition_code.deq.jit",
        "shots": 10000,
        "seed": 123
    }'
```
