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
