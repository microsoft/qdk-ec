import sys

import arguably
from .util import *
from . import spec
from . import inspector
from . import jit
from . import annotate
from . import noise
from . import simulate
from . import canonicalize
from . import interpret
from . import merge_errors
from . import strip_tags
from . import sample


def _run_server(args: list[str]) -> None:
    """Delegate ``deq server ...`` to the Rust runtime CLI."""
    try:
        import deq_runtime
    except ImportError:
        print(
            "Error: deq_runtime is not installed. "
            "Build it with: cd deq_runtime && maturin develop --release",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        deq_runtime.cli_run("server", *args)  # type: ignore[attr-defined]
    except ValueError:
        pass


def run() -> None:
    # Intercept "server" before arguably to forward args to the Rust CLI.
    if len(sys.argv) >= 2 and sys.argv[1] == "server":
        _run_server(sys.argv[2:])
    else:
        arguably.run()
