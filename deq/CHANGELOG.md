# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.5.16] - 2026-10-08

### Added
- Optional per-hyperedge `observable_flips` metadata and an `observables`
  capability for black-box decoders, including Python and dynamic-library
  plugins. In this version, only monolithic coordinator supports this capability.

### Changed
- Aligned the `deq-decoder-abi` and reference plugin package versions at `0.3.0`.
- Observable-capable decoders now receive zero-syndrome requests, allowing
  observable-only corrections instead of automatically returning an empty result.
- Optimized JIT compilation with shared type-level maps, fewer waiter tasks,
  and in-place check-set updates.

### Fixed
- Cached reweights can no longer reactivate logical edges deliberately disabled
  for hard decoding when the cache omits its optional local hypergraph copy.

## [0.5.15] - 2026-10-07

### Added
- `huf` now uses a compact active-cluster hypergraph union-find implementation.
  The MWPF-backed union-find heuristic remains available through `mwpf` with
  `cluster_node_limit: 0`.

## [0.5.14] - 2026-10-05

### Fixed
- Radius-zero windows no longer borrow uncommitted neighboring error models.
  Under fully parallel decoding, those models could explain the local syndrome
  using corrections that the window would not commit, suppressing local
  corrections and collapsing final-readout post-selection scores.

## [0.5.13] - 2026-10-04

### Fixed
- Window decoding now retains boundary errors referenced by absolute check-model
  IDs, including when the target check model is created later. Reverse referrals
  are deduplicated across absolute and port-based references and cleared on reset.

## [0.5.12] - 2026-10-02

### Added
- MWPM and MWPF decoder support.

## [0.5.11] - 2026-10-01

### Changed
- Require `paulimer>=0.2.8` and use `FramePropagator.inject_outcome_flip` for measurement-result fault injection.

## [0.5.10] - 2026-09-30

### Fixed
- Race condition in JIT controller when overlapping reset and batch executions

## [0.5.9] - 2026-09-30

### Changed
- Require `paulimer>=0.2.7` for direct measurement-result fault injection.

### Fixed
- Noisy measurement results now propagate through record-controlled Pauli gates,
  including their effects on later measurements and output frames. `deq annotate`
  no longer omits these feedback-induced error effects.

## [0.5.8] - 2026-09-29

### Added
- Forced-gap scoring for frame bits, independent of logical readouts

## [0.5.6] - 2026-09-24

### Added
- Python 3.14t and Python 3.15+ `abi3t` wheels for deq-runtime and deqagram.

## [0.5.5] - 2026-09-24

### Fixed
- Forced-gap scoring returns zero uncertainty for provably impossible alternatives
  under zero/one priors, while rejecting zero-probability baselines.

## [0.5.4] - 2026-09-23

### Added
- Non-Clifford gates and simulation via `--simulator qdk`.
- Adaptive JIT window decoding with a T-injection tutorial.
- `@PRIVATE` gadgets and compositions for internal helpers.

### Changed
- Direct clients must set `CheckModel.error_model_count=0` for error-free models.

## [0.5.2] - 2026-09-22

### Changed
- Support logical-qubit soft information via forced gap method or simple correction
  weight. They support both monolithic and window coordinators and any decoders.
- Require QDK 1.32.

## [0.5.1] - 2026-09-21

### Changed
- `deq annotate` now always retains physical noise under `@SIMULATE_ONLY` while
  emitting canonical `ERROR` and `LOSS` metadata for decoding. Noisy
  measurements receive clean `@DECODE_ONLY` counterparts.
- Black-box decoders now expose capabilities and receive one unified decode
  request. Per-shot edge reweights and structured loss may be supplied together;
  unsupported fields fail explicitly instead of triggering a decoder-side
  fallback.
- Monolithic and window coordinators now apply `Outcomes.modifiers` as
  shot-scoped probability overrides. The `decoder_reweighting` policy controls
  whether overrides use loaded decoder support or an equivalent one-shot graph.
- Preselection consumers now recognize QDK 1.31's `SELECT { ... REQUIRE ... }`
  syntax. Legacy `PREPARE { ... }` input remains accepted for older generated
  Stim files.

## [0.4.2] - 2026-08-03

### Removed
- **Breaking:** remove the `deq annotate --keep-noise` option; its behavior is   
  now the unconditional default.
- **Breaking:** bare physical Pauli targets in `ERROR(p)` statements (e.g.
  `ERROR(0.05) C0 X0`) are no longer valid syntax (previously already rejected by the transpiler).

## [0.4.0] - 2026-07-16

### Added
- Lattice surgery support with joint-port observable finding
- CONDITIONAL keyword in COMPOSE and PROGRAM for efficient logical Pauli feed-forward
- More efficient error model construction with a shared FramePropagator
- Basic loss simulation and decoding
- Python async interface for direct interaction with decoding system
- ABI for integrating decoder binary into deq-runtime
