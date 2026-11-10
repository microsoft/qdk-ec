"""deqagram: a parser for the deq (.deq) quantum error correction file format.

:func:`parse` turns ``.deq`` source into the dataclasses this package exports.
"""

from deqagram._nodes import *  # noqa: F401,F403

# Type checkers read this exact spelling as re-exporting `_nodes`' names.
from deqagram._nodes import __all__ as __all__
from deqagram.deqagram import parse

# Rebind instead of extending, so `_nodes.__all__` stays unmodified.
__all__ = [*__all__, "parse"]
