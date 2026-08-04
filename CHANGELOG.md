# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `deqagram` native Python wheels are now built and tested on every platform and can be published to PyPI from the release pipeline. `deq` depends on `deqagram`, so it needs to be available as a wheel wherever `deq` is installed.

## binar [0.1.3] and paulimer [0.2.3] - 2026-08-03

### Changed
- Linux native Python wheels for `binar`, `paulimer`, and `deq-runtime` are now built with a `manylinux_2_28` baseline (glibc 2.28: RHEL 8+, Debian 10+, Ubuntu 18.10+). Both x86_64 and ARM64 wheels link against Zig's glibc sysroot so the declared tag matches the actual glibc floor rather than the build agent's glibc.

## [0.1.0] - 2026-01-23

### Added
- Initial (beta) release of binar, paulimer and pauliverse crates and python bindings.

[0.1.0]: https://github.com/microsoft/qdk-ec/releases/tag/v0.1.0
