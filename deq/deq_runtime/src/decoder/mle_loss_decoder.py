"""Herald-aware Pauli-envelope generator MILP loss decoder.

The decoder jointly selects ordinary Pauli edges and source-loss hypotheses.
Every observed herald must be explained by a selected source that reaches its
directly heralded site. Loss-envelope generator edges are free but gated by the
selected source; ordinary edges and source losses retain their log-likelihood
weights. Detector parity, herald coverage, and causal source conflicts are
enforced as a mixed-integer linear program solved by ``scipy.optimize.milp``
(HiGHS).

The coordinator maps the returned hyperedge subgraph back to residual and
readout corrections. The decoder therefore consumes detector footprints and
loss-site structure only; it does not need per-edge logical observables.

Interface (deq Python decoder, see python_decoder.rs):

    Decoder.supported_features() -> ["loss"]
    Decoder(hypergraph, config)
    decode(syndrome: list[int], loss=None) -> list[int]   # selected hyperedges
    reset() -> None

``hypergraph`` exposes ``vertex_num`` and ``hyperedges`` (each with ``vertices``
and ``probability``). ``loss`` (present only on loss shots) exposes ``sites``,
each with ``source_edges`` / ``continuation_edges`` (hyperedge indices),
``children`` (indices into ``loss.sites``), ``probability``, and ``heralds``
(window-local observed-herald IDs shared across sites).

Model. A selected source covers every direct herald in its forward ``children``
reach. Each observed herald must be covered at least once. Two source sites are
in causal conflict when their forward reaches overlap: both would claim the same
continuing loss lifetime. Branch siblings with disjoint reaches may both start,
which permits alternatives such as one loss before a fan-out versus independent
losses on several branches. Given the selected sources, a ``source_edges`` edge
of site ``s`` needs ``z_s``, and a ``continuation_edges`` edge of site ``s``
needs a selected source that reaches ``s`` (itself or an ancestor).

The runtime has already filtered ``sites`` to those consistent with observed
loss-resolving readouts. ``heralds`` preserves the remaining correlation between
otherwise disconnected sites. Envelope patterns are free; selecting a physical
source carries the log-likelihood-ratio weight of its declared ``probability``.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix


def _edge_weight(probability: float) -> float:
    """Log-likelihood-ratio weight of a Pauli edge with the given probability."""
    if probability <= 0.0:
        return math.inf
    if probability >= 1.0:
        return -math.inf
    return math.log((1.0 - probability) / probability)


class Decoder:
    @staticmethod
    def supported_features() -> list[str]:
        return ["loss"]

    def __init__(self, hypergraph, config=None):
        self.vertex_num = int(hypergraph.vertex_num)
        self.edges: list[list[int]] = []
        self.weights: list[float] = []
        for hyperedge in hypergraph.hyperedges:
            self.edges.append([int(v) for v in hyperedge.vertices])
            self.weights.append(_edge_weight(float(hyperedge.probability)))
        self.num_edges = len(self.edges)
        # Detector -> incident hyperedge indices.
        self.vertex_edges: dict[int, list[int]] = defaultdict(list)
        for edge_index, vertices in enumerate(self.edges):
            for vertex in vertices:
                self.vertex_edges[vertex].append(edge_index)
        config = config or {}
        # Optional HiGHS wall-clock limit (seconds) for a single decode.
        self.time_limit = config.get("time_limit", None)
        self._debug = bool(int(os.environ.get("MLE_DEBUG", "0")))
        self._debug_left = int(os.environ.get("MLE_DEBUG_N", "15"))

    def decode(self, syndrome, loss=None) -> list[int]:
        num_edges = self.num_edges
        sites = list(loss.sites) if loss is not None else []
        self._validate_loss_sites(sites)
        if num_edges == 0:
            return []
        syndrome_set = {int(v) for v in syndrome}
        num_sites = len(sites)

        enabling, loss_edges, herald_starts, conflicts, parents, ancestors = (
            self._loss_structure(sites)
        )

        if self._debug and sites and self._debug_left > 0:
            import sys
            self._debug_left -= 1
            print(
                f"[MLE] sites={num_sites} heralds={len(herald_starts)} "
                f"conflicts={len(conflicts)} loss_edges={len(loss_edges)} "
                f"syn={len(syndrome_set)}",
                file=sys.stderr, flush=True,
            )

        # Variable layout: [y_e | z_s | slack_v].
        constraint_vertices = sorted(set(self.vertex_edges.keys()) | syndrome_set)
        slack_of = {vertex: index for index, vertex in enumerate(constraint_vertices)}
        num_slack = len(constraint_vertices)
        num_vars = num_edges + num_sites + num_slack
        slack_base = num_edges + num_sites

        objective = np.zeros(num_vars)
        lower = np.zeros(num_vars)
        upper = np.ones(num_vars)
        for edge in range(num_edges):
            if edge in loss_edges:
                continue  # free envelope edge, bounds [0, 1], weight 0
            weight = self.weights[edge]
            if math.isinf(weight):
                if weight > 0.0:
                    upper[edge] = 0.0  # p <= 0 non-loss edge: unusable
                else:
                    lower[edge] = 1.0  # p >= 1: certain error
            else:
                objective[edge] = weight
        for site_index, site in enumerate(sites):
            variable = num_edges + site_index
            probability = float(site.probability)
            if probability == 0.0 and not site.source_edges:
                if parents[site_index]:
                    upper[variable] = 0.0
                continue
            if probability == 1.0:
                continue
            weight = _edge_weight(probability)
            if math.isinf(weight):
                upper[variable] = 0.0
            else:
                objective[variable] = weight
        for vertex in constraint_vertices:
            slack = slack_base + slack_of[vertex]
            upper[slack] = max(len(self.vertex_edges.get(vertex, [])), 1)

        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        con_lower: list[float] = []
        con_upper: list[float] = []
        row = 0

        # Detector satisfaction (GF(2) via an integer slack): for each detector,
        # sum of incident selected edges - 2 * slack == observed bit.
        for vertex in constraint_vertices:
            for edge in self.vertex_edges.get(vertex, []):
                rows.append(row)
                cols.append(edge)
                vals.append(1.0)
            rows.append(row)
            cols.append(slack_base + slack_of[vertex])
            vals.append(-2.0)
            bit = 1.0 if vertex in syndrome_set else 0.0
            con_lower.append(bit)
            con_upper.append(bit)
            row += 1

        # Every observed herald needs at least one selected source whose forward
        # loss lifetime reaches that herald.
        for starts in herald_starts.values():
            for site in starts:
                rows.append(row)
                cols.append(num_edges + site)
                vals.append(1.0)
            con_lower.append(1.0)
            con_upper.append(np.inf)
            row += 1

        # A certain local source must start unless the loss has already started
        # at an ancestor, in which case this site is only a continuation.
        for site, loss_site in enumerate(sites):
            if float(loss_site.probability) != 1.0:
                continue
            for start in ancestors[site]:
                rows.append(row)
                cols.append(num_edges + start)
                vals.append(1.0)
            con_lower.append(1.0)
            con_upper.append(np.inf)
            row += 1

        # Starts whose forward loss lifetimes overlap cannot both be true.
        for left, right in conflicts:
            rows.extend((row, row))
            cols.extend((num_edges + left, num_edges + right))
            vals.extend((1.0, 1.0))
            con_lower.append(-np.inf)
            con_upper.append(1.0)
            row += 1

        # Envelope gating: a loss edge may be selected only if an enabling start
        # is chosen.
        for edge in sorted(loss_edges):
            rows.append(row)
            cols.append(edge)
            vals.append(1.0)
            for site in enabling[edge]:
                rows.append(row)
                cols.append(num_edges + site)
                vals.append(-1.0)
            con_lower.append(-np.inf)
            con_upper.append(0.0)
            row += 1

        constraints = []
        if row > 0:
            matrix = csr_matrix((vals, (rows, cols)), shape=(row, num_vars))
            constraints.append(LinearConstraint(matrix, np.array(con_lower), np.array(con_upper)))

        options = {}
        if self.time_limit is not None:
            options["time_limit"] = float(self.time_limit)

        result = milp(
            c=objective,
            constraints=constraints,
            integrality=np.ones(num_vars, dtype=int),
            bounds=Bounds(lower, upper),
            options=options,
        )
        if result.x is None:
            raise RuntimeError(
                f"loss decoder MILP produced no solution (status={result.status}): "
                f"{result.message}"
            )
        solution = result.x
        return [edge for edge in range(num_edges) if solution[edge] > 0.5]

    def _validate_loss_sites(self, sites) -> None:
        num_sites = len(sites)
        indegree = [0] * num_sites
        has_herald = False
        for site_index, site in enumerate(sites):
            probability = float(site.probability)
            if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"loss site {site_index} probability must be finite and in [0, 1]"
                )
            for field_name in ("source_edges", "continuation_edges"):
                for edge in getattr(site, field_name):
                    edge_index = int(edge)
                    if not 0 <= edge_index < self.num_edges:
                        raise ValueError(
                            f"loss site {site_index} {field_name} contains edge "
                            f"{edge_index}, outside [0, {self.num_edges})"
                        )
            for child in site.children:
                child_index = int(child)
                if not 0 <= child_index < num_sites:
                    raise ValueError(
                        f"loss site {site_index} children contains site "
                        f"{child_index}, outside [0, {num_sites})"
                    )
                indegree[child_index] += 1
            for herald in site.heralds:
                herald_index = int(herald)
                if herald_index < 0:
                    raise ValueError(
                        f"loss site {site_index} heralds contains negative index "
                        f"{herald_index}"
                    )
                has_herald = True

        frontier = [site for site, degree in enumerate(indegree) if degree == 0]
        visited = 0
        while frontier:
            site = frontier.pop()
            visited += 1
            for child in sites[site].children:
                child_index = int(child)
                indegree[child_index] -= 1
                if indegree[child_index] == 0:
                    frontier.append(child_index)
        if visited != num_sites:
            raise ValueError("loss site children graph contains a cycle")
        if sites and not has_herald:
            raise ValueError("loss sites contain no direct heralds")

    def _loss_structure(self, sites):
        """Build edge enabling, herald coverage, and source conflicts.

        * ``enabling[edge]`` is the set of start sites that make loss ``edge``
          usable (a source edge needs its own site; a continuation edge needs a
          start that reaches its site).
        * ``loss_edges`` is the set of all loss generator hyperedge indices.
        * ``herald_starts[herald]`` contains starts whose forward reach covers
          that direct observed herald.
        * ``conflicts`` contains source pairs with overlapping forward reaches.
        * ``parents`` is the immediate reverse child graph.
        * ``ancestors`` contains every source that reaches each site.
        """
        num_sites = len(sites)
        children = [[int(child) for child in site.children] for site in sites]
        parents: list[list[int]] = [[] for _ in range(num_sites)]
        for site, child_list in enumerate(children):
            for child in child_list:
                parents[child].append(site)

        remaining_parents = [len(site_parents) for site_parents in parents]
        frontier = [site for site, count in enumerate(remaining_parents) if count == 0]
        topological_order: list[int] = []
        ancestors = [{site} for site in range(num_sites)]
        while frontier:
            site = frontier.pop()
            topological_order.append(site)
            for child in children[site]:
                ancestors[child].update(ancestors[site])
                remaining_parents[child] -= 1
                if remaining_parents[child] == 0:
                    frontier.append(child)

        reachable = [1 << site for site in range(num_sites)]
        for site in reversed(topological_order):
            for child in children[site]:
                reachable[site] |= reachable[child]

        enabling: dict[int, set[int]] = defaultdict(set)
        loss_edges: set[int] = set()
        herald_starts: dict[int, set[int]] = defaultdict(set)
        for site in range(num_sites):
            for edge in sites[site].source_edges:
                enabling[int(edge)].add(site)
                loss_edges.add(int(edge))
            for edge in sites[site].continuation_edges:
                enabling[int(edge)].update(ancestors[site])
                loss_edges.add(int(edge))
            for herald in sites[site].heralds:
                herald_starts[int(herald)].update(ancestors[site])

        conflicts = [
            (left, right)
            for left in range(num_sites)
            for right in range(left + 1, num_sites)
            if reachable[left] & reachable[right]
        ]
        return enabling, loss_edges, herald_starts, conflicts, parents, ancestors

    def reset(self) -> None:
        pass
