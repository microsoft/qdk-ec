from typing import Dict


class Decoder:
    @staticmethod
    def supported_features() -> list[str]:
        return ["reweights", "loss"]

    def __init__(self, hypergraph, config: Dict):
        self.verbose = bool(config.get("verbose", False))
        if self.verbose:
            print("Creating Decoder")
            print("    hypergraph:", hypergraph)
            print("    config:", config)

    def decode(
        self,
        syndrome: list[int],
        *,
        reweights=None,
        loss=None,
    ) -> list[int]:
        del reweights, loss
        assert isinstance(syndrome, list)
        if self.verbose:
            print("Decoding with Decoder")
        return []

    def reset(self):
        if self.verbose:
            print("Resetting Decoder")
