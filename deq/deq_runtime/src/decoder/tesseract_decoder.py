"""Python decoder wrapper around the public ``tesseract-decoder`` PyPI package.

Exposes the deq Python decoder protocol:

    class Decoder:
        def __init__(self, hypergraph, config: dict): ...
        def decode(self, syndrome: list[int]) -> list[int]: ...
        def reset(self) -> None: ...

Both ``syndrome`` and the returned subgraph are *sparse* index lists.

The ``config`` dictionary is forwarded verbatim as keyword arguments to
``tesseract_decoder.tesseract.TesseractConfig`` (see its docstring for
supported keys, e.g. ``det_beam``, ``beam_climbing``, ``pqlimit``).
Unknown keys raise ``TypeError`` from ``TesseractConfig`` itself.

Test against the standard suite from the deq CLI (run from
``deq/deq_runtime``)::

    LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):$LD_LIBRARY_PATH" \
        cargo run --bin deq-runtime-cli --features python -- \
        test python-decoder --file @tesseract_decoder

The ``@tesseract_decoder`` sentinel resolves to a compile-time-embedded
copy of this file inside ``python_decoder.rs``; pass a filesystem path
instead when working on a local variant.  The ``LD_LIBRARY_PATH`` shim
points the embedded interpreter at a ``libpython`` that has ``numpy``,
``stim`` and ``tesseract_decoder`` installed; omit it if your system
Python already has them. Pass ``--py-config '{"det_beam": 10}'`` to
override decoder kwargs.
"""

from typing import Any, Dict, List

import numpy as np
import stim

from tesseract_decoder import tesseract


def _build_dem(vertex_num: int, hyperedges) -> stim.DetectorErrorModel:
    lines = []
    for hyperedge in hyperedges:
        detectors = " ".join(f"D{int(v)}" for v in hyperedge.vertices)
        lines.append(f"error({float(hyperedge.probability)}) {detectors}")
    text = "\n".join(lines) + "\n"
    return stim.DetectorErrorModel(text)


class Decoder:
    def __init__(self, hypergraph: Any, config: Dict[str, Any]):
        vertex_num = int(hypergraph.vertex_num)
        hyperedges = list(hypergraph.hyperedges)
        num_hyperedges = len(hyperedges)

        dem = _build_dem(vertex_num, hyperedges)
        kwargs = dict(config or {})

        self._vertex_num = vertex_num
        self._num_hyperedges = num_hyperedges
        self._solver = tesseract.TesseractConfig(dem=dem, **kwargs).compile_decoder()

    def decode(self, syndrome: List[int]) -> List[int]:
        assert isinstance(syndrome, list)
        dense = np.zeros(self._vertex_num, dtype=bool)
        for index in syndrome:
            dense[int(index)] = True
        self._solver.decode_to_errors(dense)
        return [int(i) for i in self._solver.predicted_errors_buffer]

    def reset(self) -> None:
        return None
