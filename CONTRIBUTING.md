# Contributing to QDK-EC

Welcome, and thank you for your interest in contributing to the Quantum Development Kit for Error Correction!

## Reporting Issues

Have you identified a reproducible problem in QDK-EC? Have a feature request? We want to hear about it!

Please search the [existing issues](https://github.com/microsoft/qdk-ec/issues) before filing new issues to avoid duplicates. When filing an issue, please include:

- A clear description of the problem or feature request
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Rust and Python versions you're using
- Any relevant code snippets or error messages

## Contributing Code

If you are interested in helping fix issues you or someone else encountered, please make sure that the corresponding issue has been filed on the repository. Check that nobody is currently working on it and that it has indeed been marked as a bug. If that's the case, indicate on the issue that you are working on it, and link to the corresponding pull request where the fix is being developed. If someone is already working on a fix, ask if you can help or see what other things can be done.

If you are interested in contributing a new feature, please first check if a similar functionality has already been requested. If so, consider contributing to the discussion around it rather than filing a separate issue. If no open or closed issue with such a request already exists, please file one. We will respond to feature requests and follow up with a discussion around its feasibility, how one might go about implementing it, and whether that is something we would consider adding to the repository.

If you are looking for a place to get started with contributing code, search for the [good-first-issue](https://github.com/microsoft/qdk-ec/labels/good%20first%20issue) or [help-wanted](https://github.com/microsoft/qdk-ec/labels/help%20wanted) labels.

## Code Standards

This project follows Rust best practices:

- All code must pass `cargo clippy-all`
- Code must be formatted with `cargo fmt`
- New features should include tests
- Public APIs should be documented
- Performance-critical code should include benchmarks where appropriate

Python bindings should:

- Expose a Pythonic API
- Include type hints (`.pyi` files)
- Have corresponding tests in the `tests/` directory

For qodec-specific build commands, verification gates, and format compatibility,
see the [qodec development guide](qodec/CONTRIBUTING.md).

## Rust CI Test Profile

GitHub and Azure use `--profile ci-test` for their shared Rust test jobs.
This profile inherits `release` but disables LTO across the dependency graph,
so each test executable can link compiled dependencies without repeating their
link-time code generation. Other release settings are unchanged.

Build shared libraries needed by tests with the same profile:

```bash
cargo build --profile ci-test -p deq-decoder-reference-plugin -p qodec-c
cargo test --profile ci-test --workspace --exclude deq-runtime --all-features
```

Published wheels still use `--release`. The profile does not change ordinary
local builds, benchmarks, or qodec's development-profile verification runner.

## Native Wheel Build Tools

Native CI and release jobs use the versions in
[requirements-build.txt](requirements-build.txt): maturin 1.15.0, uv 0.11.32,
and, on Linux, Zig 0.14.1. From the repository root, install them into your
selected environment:

```bash
python -m pip install -r requirements-build.txt
python -m maturin --version
```

Maturin builds Python wheels from the Rust crates. Zig is a compiler toolchain;
here it supplies the C compiler and linker for building Linux wheels against an
older glibc, rather than requiring the build machine's newer glibc at runtime.

For Linux `maturin build --zig` commands, activate that environment and set:

```bash
export CARGO_ZIGBUILD_PYTHON_PATH="$PWD/tools/zig.py"
```

The [Zig adapter](tools/zig.py) accepts cargo-zigbuild's `-m ziglang` invocation
and removes only `-Wl,-O1`, a linker optimization hint that Zig ignores. Rust
optimization flags and other linker diagnostics are unchanged. Use a fresh Cargo
target directory when checking a toolchain change; Cargo can replay warnings
from an older cached build.

If maturin's executable version disagrees with `importlib.metadata.version("maturin")`,
reinstall with `python -m pip install --force-reinstall -r requirements-build.txt`.
Run pip through the selected interpreter rather than another environment's pip.
These pins do not change the separate Pyodide toolchain or SBOM policy.

### Native Python ABI Matrix

The [Azure build stage](.ado/stages/build.yaml) uses uv-managed Python 3.11,
3.14t, and 3.15.0b4 to build `abi3`, `cp314-cp314t`, and `cp315-abi3.abi3t`
wheels for each native package and supported platform. PyO3 features retain
each package's minimum ABI; Python requirements are unchanged. The 3.15 wheel
is also imported on 3.15t. Publication requires all three wheel families.
Python 3.15 support is provisional until validated against the final release.
WASM builds and deq-runtime's Windows ARM64 exclusion are unchanged.

## Pull Request Process

1. Fork the repository and create your branch from `main`
2. Make your changes and ensure tests pass
3. Update documentation as needed
4. Run `cargo fmt` and `cargo clippy-all` to ensure code quality
5. Submit a pull request with a clear description of your changes

## Contributor License Agreement (CLA)

Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit <https://cla.opensource.microsoft.com/>.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all Microsoft repos using our CLA.

## Code of Conduct

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Thank You!

Your contributions to open source, large or small, make great projects like this possible. Thank you for taking the time to contribute.
