"""Register optional Python adapters in the native parser registry."""

from importlib.util import find_spec
from . import register

if find_spec("stim") is not None:
    from ._stim import parse as _parse_stim

    register(_parse_stim, format="stim")
