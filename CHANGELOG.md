# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `PhasedOutcomeCompleteSimulation` in pauliverse, with the same name in the Python bindings. It extends the outcome-complete simulation with exact global-phase tracking, following Algorithm 4.2 of [arXiv:2603.24717](https://arxiv.org/abs/2603.24717), so circuits that are not stabilizer circuits can be compared for exact equality.
- `Simulation::allocate_symbolic_angle` and `Simulation::symbolic_pauli_exp` in pauliverse, with `SymbolicAngle`, `allocate_symbolic_angle`, `allocate_symbolic_angles`, `symbolic_angles`, and `apply_symbolic_pauli_exp` in the Python bindings. A symbolic angle stands for the parameter of a rotation `exp(iαP)`. Each angle must parameterise exactly one rotation.
- `PhasedCircuitAction` in pauliverse and the Python bindings, returned by `phased_action`. It compares two circuit actions with `is_equivalent` and `is_equivalent_up_to_signs`.
- `clifford_to_pauli_exponents` in paulimer, with `CliffordUnitary.to_pauli_exponents()` in the Python bindings, decomposing a Clifford into Pauli exponents of angle `π/4` with an exact phase.

## binar [0.1.7], paulimer [0.2.8], pauliverse [0.1.6] - 2026-10-01

### Added
- `Bitwise::aligned_words` and `BitwiseMut::aligned_words_mut` in binar, which return the words of bit vectors and views stored in aligned blocks.

### Changed
- Faster `measure` in pauliverse simulators when the outcome is random, faster `support()` for matrix columns and `dot()` for bit vectors in binar, and faster multiplication of Paulis stored in aligned bit vectors in paulimer.
- Rename `FramePropagator.inject_measurement_flip` to `inject_outcome_flip`

## paulimer [0.2.7], pauliverse [0.1.5] - 2026-09-30

### Added
- `FramePropagator.inject_measurement_flip(shot, outcome)` toggles a recorded
	outcome delta without changing qubit frames. Available in Rust and Python;
	supports padded records and propagates through subsequent classical feedback.

## binar [0.1.6], paulimer [0.2.6], pauliverse [0.1.4] - 2026-09-29

### Changed
- Faster `measure` in pauliverse simulators when the outcome is random, and faster `support()` in binar for bit vectors and unsigned integers.

## binar [0.1.5], paulimer [0.2.5], pauliverse [0.1.3] - 2026-09-24

### Added
- Python 3.14t and Python 3.15+ `abi3t` wheels for binar and paulimer alongside existing `abi3` wheels.

## binar [0.1.4] and paulimer [0.2.4] - 2026-09-17

### Changed
- Updated the Rust and Python bindings to PyO3 0.29. `paulimer` now requires `binar` 0.1.4 or later so packaged crates resolve a single PyO3 version.

## binar [0.1.3] and paulimer [0.2.3] - 2026-08-03

### Changed
- Linux native Python wheels for `binar`, `paulimer`, and `deq-runtime` are now built with a `manylinux_2_28` baseline (glibc 2.28: RHEL 8+, Debian 10+, Ubuntu 18.10+). Both x86_64 and ARM64 wheels link against Zig's glibc sysroot so the declared tag matches the actual glibc floor rather than the build agent's glibc.

## [0.1.0] - 2026-01-23

### Added
- Initial (beta) release of binar, paulimer and pauliverse crates and python bindings.

[0.1.0]: https://github.com/microsoft/qdk-ec/releases/tag/v0.1.0
