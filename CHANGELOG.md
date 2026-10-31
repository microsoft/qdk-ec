# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `PhasedOutcomeCompleteSimulation` in pauliverse, with the same name in the Python bindings. It tracks encoder and branch phases, based on Algorithm 4.2 of [arXiv:2603.24717](https://arxiv.org/abs/2603.24717).
- `Simulation::allocate_symbolic_angle` and `Simulation::symbolic_pauli_exp` in pauliverse, with `SymbolicAngle`, `allocate_symbolic_angle`, `allocate_symbolic_angles`, `symbolic_angles`, and `apply_symbolic_pauli_exp` in the Python bindings. A symbolic angle stands for the parameter of a rotation `exp(iαP)`. Each angle must parameterise exactly one rotation.
- `PhasedCircuitAction` in pauliverse and the Python bindings, returned by `phased_action`. It compares two circuit actions with `is_equivalent` and `is_equivalent_up_to_signs`.
- `clifford_to_pauli_exponents` in paulimer, with `CliffordUnitary.to_pauli_exponents()` in Python. Replaying the `π/4` Pauli exponents reproduces the Clifford's Pauli images and signs.
- `clifford_to_transvections` in paulimer, with `CliffordUnitary.to_transvections()` in the Python bindings, decomposing a Clifford's symplectic action into a linear number of Hermitian Pauli generators of `π/4` exponents. The Python method raises `ValueError` for an invalid tableau.
- `clifford_fixed_space` in paulimer, with `CliffordUnitary.fixed_space()` in the Python bindings, returning Hermitian generators of the Pauli operators that conjugation fixes up to sign.
- `clifford_to_transvections_minimal` returns `TransvectionDecomposition { transvections, pauli_correction }`. Python `CliffordUnitary.to_transvections_minimal()` returns `(list[SparsePauli], SparsePauli)`. Apply the rotations in order, then apply the correction with `left_mul_pauli`. The result reproduces the signed tableau, but not global phase. The rotation count is the unsigned minimum and excludes the final Pauli. The greedy API remains unsigned. Alternating restrictions skip span enumeration. The Python method raises `ValueError` for an invalid tableau.
- The minimal transvection decomposition tries a checked bordered construction before the exhaustive residue fix search. Failed candidates use the retained search. Both paths compute the final Pauli correction from their selected rotation list.
- `branch_bit_count`, `branch_bit_capacity`, and `outcome_introduces_branch_bit` distinguish internal branch coordinates from public outcome IDs.
- `OutcomeSpecificSimulation` constructors with `samples` in their names distinguish the sample source from recorded outcomes. The zero source does not change deterministic outcomes.

### Deprecated
- Use the branch coordinate accessors instead of `random_outcome_count`, `random_outcome_capacity`, `random_outcome_indicator`, and Python `random_bit_count`. The old names remain available.
- Use the sample-source constructors instead of their `outcomes` aliases. Existing capacity keyword arguments and Rust `Simulation` implementations remain supported.

### Fixed
- Native phased actions reject entangled auxiliary qubits outside the named input and output registers.
- Python symbolic-angle handles compare and hash by simulation ownership and angle identity.
- Zero-qubit phased Cliffords preserve scalar rotation phases and support amplitude queries.
- Hinted measurements preserve branch phases for both signs of the hint.
- Python phased simulations reject short branch assignments and symbolic-angle handles from another simulation.
- Retrieved symbolic-angle handles retain their public outcome IDs after deterministic measurements.
- `PhasedCliffordUnitary::state_amplitude_phase_exponent_usize` handles registers wider than `usize` without shift overflow.
- Complemented conditional Paulis preserve the exact phase for imaginary operators.
- Deprecated Python simulation aliases identify the caller in warnings.
- Symbolic-angle use counters reserve space for public outcomes, including deterministic measurements.
- `OutcomeCompleteSimulation::conditional_pauli` now treats repeated outcome IDs as a parity, so pairs cancel, as the other simulators and `is_stabilizer_with_conditional_sign` already did.
- `Simulation::allocate_random_bit` now returns the public outcome ID in all pauliverse simulators.
- `OutcomeCompleteSimulation::is_stabilizer` no longer ignores the observable's sign, so it now distinguishes `+P` from `-P` and rejects the non-Hermitian `±iP`.
- `is_stabilizer_with_conditional_sign` now rejects observables with an imaginary phase in outcome-complete and outcome-specific simulation.

## binar [0.1.8], paulimer [0.2.9], pauliverse [0.1.7] - 2026-10-05

### Changed
- Cut per-measurement overheads in pauliverse simulators

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
