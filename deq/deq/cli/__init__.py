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


# Subcommands implemented in the Rust runtime (`deq_runtime`) rather than in
# the Python `deq` CLI. They are dispatched verbatim via ``deq_runtime.cli_run``.
_RUNTIME_SUBCOMMANDS = ("server", "test", "benchmark")


def _run_runtime_cli(subcommand: str, args: list[str]) -> None:
    """Delegate ``deq <subcommand> ...`` to the Rust runtime CLI."""
    try:
        import deq_runtime
    except ImportError:
        print(
            "Error: deq_runtime is not installed. "
            "Build it with: cd deq_runtime && maturin develop --release --features python_all",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        deq_runtime.cli_run(subcommand, *args)  # type: ignore[attr-defined]
    except ValueError:
        pass


def run() -> None:
    # Intercept runtime-owned subcommands before arguably to forward args to the Rust CLI.
    if len(sys.argv) >= 2 and sys.argv[1] in _RUNTIME_SUBCOMMANDS:
        _run_runtime_cli(sys.argv[1], sys.argv[2:])
    else:
        arguably.run()
