# pylint: disable=no-member
#   no-member: compiled Rust extension module members not detected by pylint
"""
Thin wrapper around deq_runtime that can be run as a module.

Usage:
    deq server --help
"""

import sys
import deq_runtime


def main() -> None:
    try:
        deq_runtime.cli_run(*sys.argv[1:])  # type: ignore[attr-defined]
    except ValueError:
        # cli_run raises ValueError after printing help/version info
        pass


if __name__ == "__main__":
    main()
