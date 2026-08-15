# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
