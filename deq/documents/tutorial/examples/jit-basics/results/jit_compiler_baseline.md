# JIT compiler benchmark baseline

Recorded: 2026-10-16T22:54:26.164086+00:00
Base revision: `4778f4c060d078dba4a1da56d522d8229acbe132`; dirty working tree: `True`.
Historical summary only; raw samples and exact-source provenance are not checked in.
Fresh runs retain those details in the local Criterion output.

- CPU: Intel(R) Core(TM) Ultra 9 285K; available logical CPUs: 24.
- Rust: `rustc 1.96.0 (ac68faa20 2026-05-25)`.
- Flags: `-C target-cpu=native`; compiler worker threads: 24.
- Profile: optimized Cargo bench; workspace release LTO and codegen settings unchanged.

## Scope

Shared JIT compilation, deferred error resolution, and incremental collection only.

**Excluded:** Fixture loading, library registration, input cloning, dependency-channel setup, controller, coordinator, decoder, simulation, and intentional sleeps.

Cold iterations start with empty compilation-template caches. Warm iterations reset execution state
but retain templates from an unmeasured complete pass. Every timed iteration checks all gadget,
check, and error-model counts and requires the final JIT live-state count to be zero.

## Results

| Workload | Templates | Full-program mean (s) | Mean / epoch (ms) | Upper confidence bound / epoch (ms) | Budget |
|---|---|---:|---:|---:|---|
| d7_w4_l2 | cold | 0.016021 | 1.602 | 1.664 | PASS |
| d7_w4_l2 | warm | 0.002595 | 0.259 | 0.262 | PASS |
| d7_w1000_l50 | cold | 17.955524 | 88.889 | 89.391 | PASS |
| d7_w1000_l50 | warm | 17.750367 | 87.873 | 88.279 | PASS |
