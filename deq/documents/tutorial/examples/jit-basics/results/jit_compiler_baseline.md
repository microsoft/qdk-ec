# JIT compiler benchmark baseline

Recorded: 2026-11-10T22:57:40.679384+00:00
Base revision: `3a4cfa68ef28747953599bdefa67024de69dc40f`; dirty working tree: `True`.
Raw samples and run provenance remain in the local Criterion output; only this summary belongs in git.

- CPU: Intel(R) Core(TM) Ultra 9 285K; available logical CPUs: 24.
- Rust: `rustc 1.96.0 (ac68faa20 2026-05-25)`.
- Flags: `-C target-cpu=native`; compiler worker threads: 24.
- Error-template LRU capacity: 128 entries (enabled), 0 (disabled).
- Profile: optimized Cargo bench; workspace release LTO and codegen settings unchanged.

## Scope

Shared JIT compilation, deferred error resolution, and incremental collection only.

**Excluded:** Fixture loading, library registration, input cloning, dependency-channel setup, controller, coordinator, decoder, simulation, and intentional sleeps.

Every iteration starts with a fresh compiler. Cache-enabled runs start with an empty
error-template cache and reuse templates within that program; cache-disabled runs rebuild
those templates instead. Input-check-map and terminal-error-model caches remain enabled in
both modes. There is no retained warmup pass. Every completed iteration checks gadget,
check, and error-model counts and requires the final JIT live-state count to be zero.
Completed rows are means from at least ten Criterion samples. A timeout row is only a
lower bound from an interrupted run, not a measured mean or confidence interval.

## Results

| Workload | Error-template cache | Full-program mean / bound (s) | Mean / epoch or bound (ms) | Upper confidence bound / epoch (ms) | Budget |
|---|---|---:|---:|---:|---|
| d7_w4_l2 | enabled | 0.016142 | 1.614 | 1.667 | PASS |
| d7_w4_l2 | disabled | 0.072276 | 7.228 | 7.367 | Control; not gated |
| d7_w200_l100 | enabled | 5.768514 | 14.350 | 14.420 | PASS |
| d7_w200_l100 | disabled | 165.617624 | 411.984 | 412.457 | Control; not gated |

