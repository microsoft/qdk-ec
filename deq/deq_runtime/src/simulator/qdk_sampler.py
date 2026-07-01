"""Python sampler wrapper around the public ``qdk`` PyPI package.

Exposes the deq Python sampler protocol:

    class Sampler:
        def __init__(self, circuit_text: str, config: dict): ...
        def sample(self) -> str:
            '''Return one shot as a length-N string of '0', '1', or '-' chars.'''

The Stim circuit text is compiled to QIR + noise inside ``qdk.stim.compile``
and executed with the loss-aware Clifford simulator
(``qdk.stim.run(..., type="clifford")``).  ``qdk.stim.run`` returns
``List[List[Result]]`` where ``Result`` is a Rust-bound enum with members
``Zero``, ``One``, ``Loss``; this adapter converts each shot to a
length-N string of ``'0'``, ``'1'``, ``'-'`` characters before returning
it. The deq Rust sampler then replaces each ``'-'`` with a uniformly random bit 
drawn from its deterministic RNG before feeding the shot to the decoder.

The ``config`` dictionary may contain:

* ``seed`` (auto-injected by the Rust sampler): base seed for shot batches.
  Each refill uses ``seed + batch_index`` so successive batches draw fresh
  shots while remaining reproducible.
* ``skip_shots`` (auto-injected): number of shots to discard at the start.
* ``num_measurements`` (auto-injected): expected shot length (sanity check).
* ``batch_size`` (default: 256): how many shots to draw per ``qdk.stim.run``
  call.  Larger values amortize Python call overhead at the cost of memory.
* ``type`` (default: ``"clifford"``): forwarded to ``qdk.stim.run``.
  Use ``"cpu"`` for non-Clifford circuits.

Invocation options
------------------

**Recommended: the ``@qdk_sampler`` builtin sentinel.** This module is
compiled into the ``deq_runtime`` binary via a small ``builtin_samplers``
registry inside ``python_sampler.rs``, and the ``PythonSampler`` config
field ``sampler`` resolves any value beginning with ``@`` from that
registry instead of the filesystem.  So the canonical invocation is::

    python -m deq.runtime server \\
        --simulator python \\
        --simulator-config '{"filepath": "circuit.stim", "sampler": "@qdk_sampler", "py_config": {"batch_size": 1024}}'

**Loading a local copy from disk.** Any ``sampler`` value that does not
start with ``@`` is opened as a filesystem path — useful when hacking
on a customized version of this adapter::

    python -m deq.runtime server \\
        --simulator python \\
        --simulator-config '{"filepath": "circuit.stim", "sampler": "src/simulator/qdk_sampler.py", "py_config": {"batch_size": 1024}}'

**Through a standalone Rust binary (development / tests).** When invoked
via ``cargo run`` or ``cargo test``, the binary is *not* an extension
module: pyo3 must link ``libpython`` itself, and the dynamic loader needs
to find it.  Point ``LD_LIBRARY_PATH`` at the conda env's libdir::

    LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\"))'):$LD_LIBRARY_PATH" \\
        cargo run --bin deq-runtime-cli --features simulator,python -- \\
        server \\
        --simulator python \\
        --simulator-config '{"filepath": "circuit.stim", "sampler": "@qdk_sampler", "py_config": {"batch_size": 1024}}'
"""

from typing import Any, Dict, List

import qdk.stim
from qdk._native import Result
from qdk.simulation import run_qir

# Mapping from qdk Result enum to the single-char alphabet the deq Rust
# sampler expects.  `Result` is a PyO3-bound class so hashing/equality
# is fast (just an internal discriminant compare).
_RESULT_CHAR: Dict[Result, str] = {
    Result.Zero: "0",
    Result.One: "1",
    Result.Loss: "-",
}


def _shot_to_str(shot: List[Result]) -> str:
    try:
        return "".join(_RESULT_CHAR[r] for r in shot)
    except KeyError as e:
        raise RuntimeError(
            f"qdk.stim.run returned an unknown Result value: {e.args[0]!r}. "
            f"Expected one of {list(_RESULT_CHAR)}."
        ) from None


class Sampler:
    def __init__(self, circuit_text: str, config: Dict[str, Any]):
        self._src = circuit_text
        seed = config.get("seed")
        self._base_seed = int(seed) if seed is not None else None
        self._skip_shots = int(config.get("skip_shots", 0))
        self._num_measurements = int(config.get("num_measurements", 0))
        self._batch_size = int(config.get("batch_size", 256))
        self._kind = str(config.get("type", "clifford"))

        if self._batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self._batch_size}")

        # Compile once so every refill reuses the same QIR + NoiseConfig.
        qir, noise = qdk.stim.compile(self._src, None)
        self._qir = qir
        self._noise = noise

        self._batch_index = 0
        self._buffer: List[str] = []

        # Discard the first `skip_shots` results so the deq simulator's
        # skip_shots semantics match what the user would see with the
        # built-in stim sampler.  `_refill()` leaves `self._buffer` in
        # reverse order (so `.pop()` yields shots in their natural order),
        # which means "drop the first N shots" == "pop N times from the
        # end of the reversed buffer".
        remaining_skip = self._skip_shots
        while remaining_skip > 0:
            if not self._buffer:
                self._refill()
            drop = min(remaining_skip, len(self._buffer))
            for _ in range(drop):
                self._buffer.pop()
            remaining_skip -= drop

    def _refill(self) -> None:
        shot_seed = (
            self._base_seed + self._batch_index if self._base_seed is not None else None
        )
        self._batch_index += 1
        # Use run_qir (not qdk.stim.run) because we already compiled the Stim
        # text to QIR in __init__; qdk.stim.run would recompile every batch.
        # The NoiseConfig produced by qdk.stim.compile is passed through verbatim.
        # Returned value is List[List[Result]] -- one inner list per shot,
        # each holding `num_measurements` Result enum values.
        raw_shots = run_qir(
            self._qir,
            shots=self._batch_size,
            noise=self._noise,
            seed=shot_seed,
            type=self._kind,
        )
        shots = [_shot_to_str(shot) for shot in raw_shots]
        # Reverse so .pop() yields shots in the order qdk returned them.
        shots.reverse()
        self._buffer = shots

    def sample(self) -> str:
        if not self._buffer:
            self._refill()
        shot = self._buffer.pop()
        if self._num_measurements and len(shot) != self._num_measurements:
            raise RuntimeError(
                f"qdk.stim.run returned a shot of length {len(shot)} "
                f"but the Stim circuit declares {self._num_measurements} measurements"
            )
        return shot
