# qodec: Python bindings

Python bindings for the Rust qodec model. See the [usage and API
documentation](https://github.com/microsoft/qdk-ec/blob/main/qodec/bindings/python/docs/index.rst),
the shared [model](https://github.com/microsoft/qdk-ec/blob/main/qodec/docs/concepts/model.md),
and the [YAML representation](https://github.com/microsoft/qdk-ec/blob/main/qodec/docs/representations/yaml.md).

## Build and install from source

Use Python 3.11 or newer, a Rust toolchain, and a C compiler/linker. Start in the
**qodec directory** inside a
[qdk-ec source checkout](https://github.com/microsoft/qdk-ec/tree/main/qodec),
the folder containing qodec's `Cargo.toml`, `examples/`, and `tools/`.

Activate the Python environment you intend to use, then run:

```bash
python -m pip install maturin
cd bindings/python
python -m maturin develop --release --extras parsers
cd ../..
```

The optional `parsers` extra currently installs Stim. Omit `--extras parsers`
when working only with YAML or a custom parser; this build does not require an
existing qodec release on PyPI. The Stim adapter registers on import when Stim
is available, so restart a running Python process after installing the extra.
Other languages supply a parser with
`qodec.register(custom_parser, format="my-format")`; see
[Source formats](https://github.com/microsoft/qdk-ec/blob/main/qodec/docs/representations/source-formats.md#registering-a-parser).
Use `circuit.calls(parser=...)` for a one-call override.

This is an editable development installation. To check a regular installation,
build a wheel and install it into a separate environment:

```bash
python -m maturin build --release --manifest-path bindings/python/Cargo.toml --out target/wheels
python tools/check_wheel.py target/wheels
```

Use an output directory containing only the wheel you intend to test. The check
creates a temporary environment, installs the wheel without optional dependencies,
and checks imports and a load/save round trip outside the checkout. It runs on
Windows, macOS, and Linux. The wheel installs the model, not the repository's
examples or the QDK audit engine. The
[project README](https://github.com/microsoft/qdk-ec/blob/main/qodec/README.md#installation)
explains how to obtain example data outside a checkout.

## Build and test the documentation

The [`.pyi` files](https://github.com/microsoft/qdk-ec/tree/main/qodec/bindings/python/python/qodec)
are canonical for Python signatures and API
prose. Sphinx AutoAPI reads them to generate the API pages; edit the stubs,
not the generated pages.

Documentation requires Python 3.11 or newer, the installed bindings, and the
`parsers` extra for circuit examples. From the qodec directory, in that environment:

```bash
python -m pip install -r bindings/python/docs/requirements.txt
python -m sphinx -W --keep-going -b html bindings/python/docs target/python-docs/html
python -m sphinx -W --keep-going -b doctest bindings/python/docs target/python-docs/doctest
python bindings/python/docs/test_docs.py
```

Open `target/python-docs/html/index.html` for the built documentation. See
[Contributing](https://github.com/microsoft/qdk-ec/blob/main/qodec/CONTRIBUTING.md#documentation) for shared documentation
guidelines.
