"""Python decoder wrapper around the public ``relay-bp`` PyPI package.

Exposes the deq Python decoder protocol:

    class Decoder:
        def __init__(self, hypergraph, config: dict): ...
        def decode(self, syndrome: list[int]) -> list[int]: ...
        def reset(self) -> None: ...

Both ``syndrome`` and the returned subgraph are *sparse* index lists.

The ``config`` dictionary is forwarded verbatim as keyword arguments to
``relay_bp.RelayDecoderF64`` (see its docstring for supported keys, e.g.
``alpha``, ``gamma0``, ``num_sets``, ``seed``, ``gamma_dist_interval``).
Unknown keys raise ``TypeError`` from ``RelayDecoderF64`` itself.

Test against the standard suite from the deq CLI (run from
``deq/deq_runtime``)::

    LD_LIBRARY_PATH="$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))'):$LD_LIBRARY_PATH" \
        cargo run --bin deq-runtime-cli --features python -- \
        test python-decoder --file @relay_bp_decoder

The ``@relay_bp_decoder`` sentinel resolves to a compile-time-embedded
copy of this file inside ``python_decoder.rs``; pass a filesystem path
instead when working on a local variant.  The ``LD_LIBRARY_PATH`` shim
points the embedded interpreter at a ``libpython`` that has ``numpy``,
``scipy`` and ``relay_bp`` installed; omit it if your system Python
already has them. Pass ``--py-config '{"seed": 42}'`` to override
decoder kwargs.
"""

from typing import Any, Dict, List

import numpy as np
from scipy.sparse import csr_matrix

from relay_bp import RelayDecoderF64


class Decoder:
    def __init__(self, hypergraph: Any, config: Dict[str, Any]):
        vertex_num = int(hypergraph.vertex_num)
        hyperedges = list(hypergraph.hyperedges)
        num_hyperedges = len(hyperedges)

        rows: List[int] = []
        cols: List[int] = []
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

    def decode(self, syndrome: List[int]) -> List[int]:
        assert isinstance(syndrome, list)
        dense = np.zeros(self._vertex_num, dtype=np.uint8)
        for index in syndrome:
            dense[int(index)] = 1
        result = self._solver.decode(dense)
        return [int(i) for i in np.flatnonzero(np.asarray(result))]

    def reset(self) -> None:
        return None
