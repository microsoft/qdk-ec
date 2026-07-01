# Plug in your own decoder in Python

deq's decoding system is **decoder-agnostic**. Anything that consumes a decoding
hypergraph + syndrome and returns a fault subgraph (the "black-box" decoder
contract) plugs in equally well. This chapter shows how to write that wrapper in
**pure Python**, so you can take a decoder you have already prototyped — or a
third-party package — and drop it straight into deq's dynamic-logical-circuit
machinery without touching any Rust code.

We will walk through:

1. The Python decoder protocol — three functions, no inheritance.
2. A worked example wrapping the public [`relay-bp`](https://pypi.org/project/relay-bp/) PyPI package in ~60 lines.
3. Running the standard decoder unit-test suite against it.
4. Driving a full logical error rate simulation with the wrapper as the decoder,
   re-using the very same `small_example_evaluation.deq` from the intro chapter.

---

## When to reach for the Python bridge

deq ships built-in `BlackBox*` decoders written directly in Rust (e.g.
`black-box-relay-bp`, `black-box-tesseract`). Those are the right choice in
production: each `decode()` call stays inside one process and avoids the GIL.

The Python bridge is the right choice when:

- You're **prototyping** a new decoder and want fast iteration.
- The decoder you want to use only exists as a Python package (e.g. a research
  prototype on PyPI, or a Jupyter-notebook-grown algorithm).
- You're benchmarking several candidate decoders against the **same** dynamic
  logical circuit before committing to a Rust port.

The bridge is feature-complete — every coordinator, controller, and simulator
deq supports works unchanged. The only differences are GIL-bound per-call
overhead and a JSON round-trip for the per-decoder config.

---

## The Python decoder protocol

A Python decoder file is any `*.py` file that exposes a single class:

```python
class Decoder:
    def __init__(self, hypergraph, config: dict): ...
    def decode(self, syndrome: list[int]) -> list[int]: ...
    def reset(self) -> None: ...
```

The runtime instantiates `Decoder(hypergraph, config)` once per hypergraph and
then calls `decode(...)` / `reset()` repeatedly. State that should persist
across shots lives on the instance; state that should *not* persist across
shots is what `reset()` clears.

The class name defaults to `Decoder`. If your file already uses a different
name (or you want to expose several decoder classes from the same file), set
the top-level `name` field in the decoder JSON config to override it — see
[the config table below](#end-to-end-logical-error-rate-on-a-dynamic-circuit).

**Inputs:**

| Argument                       | Type                                  | Meaning                                                                 |
| ------------------------------ | ------------------------------------- | ----------------------------------------------------------------------- |
| `hypergraph.vertex_num`        | `int`                                 | Number of detectors (hypergraph vertices).                              |
| `hypergraph.hyperedges`        | `list[Hyperedge]`                     | Each hyperedge has `.vertices: list[int]` and `.probability: float`.    |
| `config` (to `__init__`)       | `dict`                                | Whatever JSON object you passed via `--py-config` (or `{}` if omitted). |
| `syndrome` (to `decode`)       | `list[int]`                           | **Sparse** list of detector indices that fired.                         |

**Output of `decode`:** a **sparse** list of hyperedge indices forming the
predicted fault subgraph. Returning `[]` means "no errors".

That's the entire contract. There is no base class to inherit, no module-level
registration, no metaclass magic.

---

## Worked example: wrapping `relay-bp`

The runtime ships a working wrapper at
[deq_runtime/src/decoder/relay_bp_decoder.py](../../../deq_runtime/src/decoder/relay_bp_decoder.py).
It is ~60 lines and uses zero Rust. Let's walk through it.

### The decoder class

```python
import numpy as np
from scipy.sparse import csr_matrix
from relay_bp import RelayDecoderF64

class Decoder:
    def __init__(self, hypergraph, config):
        vertex_num = int(hypergraph.vertex_num)
        hyperedges = list(hypergraph.hyperedges)
        num_hyperedges = len(hyperedges)

        rows, cols = [], []
        error_priors = np.empty(num_hyperedges, dtype=np.float64)
        for column, hyperedge in enumerate(hyperedges):
            for vertex in hyperedge.vertices:
                rows.append(int(vertex))
                cols.append(column)
            error_priors[column] = float(hyperedge.probability)

        data = np.ones(len(rows), dtype=np.uint8)
        check_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(vertex_num, num_hyperedges),
        )

        kwargs = dict(config or {})
        if "gamma_dist_interval" in kwargs and isinstance(kwargs["gamma_dist_interval"], list):
            kwargs["gamma_dist_interval"] = tuple(kwargs["gamma_dist_interval"])

        self._vertex_num = vertex_num
        self._num_hyperedges = num_hyperedges
        self._solver = RelayDecoderF64(check_matrix, error_priors, **kwargs)

    def decode(self, syndrome):
        dense = np.zeros(self._vertex_num, dtype=np.uint8)
        for index in syndrome:
            dense[int(index)] = 1
        result = self._solver.decode(dense)
        return [int(i) for i in np.flatnonzero(np.asarray(result))]

    def reset(self):
        return None
```

`__init__` translates the hypergraph into a `csr_matrix` and constructs the
underlying solver. `decode` does three things: sparse → dense conversion of the
syndrome, BP inference, and dense → sparse conversion of the result.
`RelayDecoderF64` keeps no per-shot state, so `reset()` is a no-op.

Two things worth pointing out:

- **`config` is forwarded verbatim** to `RelayDecoderF64`. Unknown keys raise
  `TypeError` straight from the underlying constructor — there is no silent
  whitelist that would drop typos.
- The single `gamma_dist_interval` list→tuple coercion exists only because JSON
  has no tuple type; without it, every user supplying that key from `--py-config`
  would hit a type error they couldn't fix from the CLI.

That's the whole wrapper. The same shape applies to any decoder: implement
`__init__`/`decode`/`reset` on a class named `Decoder`.

> An analogous wrapper for Google's Tesseract beam-search decoder lives at
> [deq_runtime/src/decoder/tesseract_decoder.py](../../../deq_runtime/src/decoder/tesseract_decoder.py).
> It builds a `stim.DetectorErrorModel` instead of a `csr_matrix`, but the
> protocol is identical.

---

## Running the standard decoder test suite

deq ships a hand-curated suite of 32 decoder unit tests (degenerate
hypergraphs, single edges, triangles, line chains, etc.) that any black-box
decoder must pass. Run it against your wrapper with `deq test python-decoder`:

```sh
deq test python-decoder --file @relay_bp_decoder
# ... 32 lines of [PASS]
# passed: 32/32
```

The `@relay_bp_decoder` sentinel resolves to a compile-time-embedded copy
of the reference wrapper baked into `python_decoder.rs`; the other builtins
are `@naive_decoder` and `@tesseract_decoder`.  Any `--file` value that does
**not** start with `@` is opened as a filesystem path, so pointing `--file`
at your own `*.py` file still works.

Two things to notice:

- This is a thin Python entry point that calls into the installed `deq_runtime`
  extension module — no Rust toolchain or `cargo` needed at the user's end. If
  you see `Error: deq_runtime is not installed`, rebuild the bindings with
  `cd deq_runtime && maturin develop --release --features python_all`.
- **`--py-config '{...}'`** is optional. It is a JSON object that lands as the
  `config` argument to your `Decoder.__init__`. For example, to pin BP's RNG:

  ```sh
  deq test python-decoder \
      --file @relay_bp_decoder \
      --py-config '{"seed": 42}'
  # NOTE: on Windows (cmd/PowerShell), escape inner double quotes,
  #       e.g. '{\"seed\":42}' instead of '{"seed":42}'
  ```

If your suite reports `passed: 32/32`, your wrapper satisfies the black-box
contract; you can move on to driving an actual simulation.

---

## End-to-end: logical error rate on a dynamic circuit

We re-use the [intro chapter's](../README.md) `small_example_evaluation.deq` —
the only line that changes from the intro is `--decoder` + `--decoder-config`.
The window coordinator, JIT controller, and JIT static simulator are all
untouched: the same dynamic-logical-circuit machinery now drives an arbitrary
Python decoder.

```sh
# Same transpile step as in the intro
deq transpile small_example_evaluation.deq --out small_example.deq.jit --program Simulation

# Same server invocation as the intro, but with --decoder black-box-python
deq server \
    --decoder black-box-python \
    --decoder-config '{
        "file": "@relay_bp_decoder"
    }' \
    --coordinator window \
    --coordinator-config '{"buffer_radius":3}' \
    --controller jit \
    --controller-config '{"filepath":"small_example.deq.jit"}' \
    --simulator jit-static \
    --simulator-config '{
        "filepath":"small_example.stim",
        "jit_library_filepath":"small_example.deq.jit",
        "shots": 2000,
        "seed": 123
    }'
# server running on "http://[::1]:50051" (port=50051)
# === Simulation Complete ===
#   Shots: 2000/2000
#   Logical errors: 28/4000
#   Error rate: 1.400000e-2 ± 5.15e-3
#   Decoding time: 0.524s (2.621e-4s per shot)
#   Last-batch latency: 0.523s (2.613e-4s per shot)
```

`black-box-python` accepts the following config:

| Field        | Type           | Meaning                                                                                                                                                                     |
| ------------ | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `file`       | string         | Path to your `*.py` file, or an `@name` sentinel that resolves a compile-time-embedded reference decoder (`@naive_decoder`, `@relay_bp_decoder`, `@tesseract_decoder`).       |
| `name`       | string         | Optional. Name of the decoder class inside the file. Defaults to `"Decoder"`.                                                                                                |
| `py_config`  | any JSON value | Forwarded to your `Decoder.__init__` as the `config` argument. Omit for `{}`.                                                                                                |
| `parallel`   | int (optional) | Number of decoder worker threads (inherited from the thread-pooling layer).                                                                                                  |

Pass `py_config` exactly the way you would pass `--py-config` to
`test python-decoder`, just nested one level inside the decoder config:

```sh
--decoder-config '{
    "file": "@relay_bp_decoder",
    "py_config": {"seed": 42, "num_sets": 100}
}'
```

If your file exposes its decoder under a different class name (or you want to
pick one of several classes in the same file), set `name`:

```sh
--decoder-config '{
    "file": "my_decoders.py",
    "name": "MyCustomDecoder",
    "py_config": {"seed": 42}
}'
```

Your decoder's Python dependencies (e.g. `numpy`, `scipy`, `relay_bp` in the
example above) just need to be importable in the same environment that hosts
`deq` — `deq_runtime` is loaded as a normal extension module into the running
interpreter, so no additional setup is required.

---

## When to graduate to Rust

The Python bridge is great for prototyping and for decoders that only exist as
Python packages. Each `decode()` call still crosses the GIL and pays a small
serialization cost — deq mitigates this by parallelizing across decoder
instances via the thread-pooling layer (`parallel` field above), but for
production latency you'll eventually want to expose your decoder through the
same `BlackBoxDecoder` Rust trait the built-in `relay-bp` and `tesseract`
decoders use. By that point you already have a known-good reference
implementation in Python and a 32-case test suite to validate the Rust port
against — so the Rust port becomes a refactoring exercise, not a research one.
