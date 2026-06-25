from typing import Dict


class Decoder:
    def __init__(self, hypergraph, config: Dict):
        self.verbose = bool(config.get("verbose", False))
        if self.verbose:
            print("Creating Decoder")
            print("    hypergraph:", hypergraph)
            print("    config:", config)

    def decode(self, syndrome: list[int]) -> list[int]:
        assert isinstance(syndrome, list)
        if self.verbose:
            print("Decoding with Decoder")
        return []

    def reset(self):
        if self.verbose:
            print("Resetting Decoder")
