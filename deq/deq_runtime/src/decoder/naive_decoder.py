from typing import Dict


class NaiveDecoder:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def decode(self, syndrome: list[int]) -> list[int]:
        assert isinstance(syndrome, list)
        if self.verbose:
            print("Decoding with NaiveDecoder")
        return []

    def reset(self):
        if self.verbose:
            print("Resetting NaiveDecoder")


def new(hypergraph, config: Dict) -> NaiveDecoder:
    verbose = bool(config.get("verbose", False))
    if verbose:
        print("Creating NaiveDecoder")
        print("    hypergraph:", hypergraph)
        print("    config:", config)
    return NaiveDecoder(verbose=verbose)
