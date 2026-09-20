"""Print the shot-scoped loss payload received by a Python decoder."""

from __future__ import annotations

import json
import sys


class Decoder:
    @staticmethod
    def supported_features() -> list[str]:
        return ["reweights", "loss"]

    def __init__(self, hypergraph, config=None) -> None:
        del hypergraph, config
        self.reported_reweights = False
        self.reported_loss = False

    def decode(self, syndrome, reweights=None, loss=None) -> list[int]:
        del syndrome
        if reweights and not self.reported_reweights:
            payload = [
                {"edge": int(edge), "probability": float(probability)}
                for edge, probability in reweights
            ]
            print(
                "REWEIGHT_REQUEST " + json.dumps(payload, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
            self.reported_reweights = True

        if loss is not None and loss.sites and not self.reported_loss:
            payload = [
                {
                    "source_edges": list(site.source_edges),
                    "continuation_edges": list(site.continuation_edges),
                    "children": list(site.children),
                    "probability": float(site.probability),
                    "heralds": list(site.heralds),
                }
                for site in loss.sites
            ]
            print(
                "LOSS_REQUEST " + json.dumps(payload, sort_keys=True),
                file=sys.stderr,
                flush=True,
            )
            self.reported_loss = True

        return []

    def reset(self) -> None:
        self.reported_reweights = False
        self.reported_loss = False