#!/usr/bin/env python3
"""Run Zig without the linker-only optimization hint that Zig ignores."""

import os
from pathlib import Path
import sys

import ziglang


def zig_arguments(arguments: list[str]) -> list[str]:
    if arguments[:2] == ["-m", "ziglang"]:
        arguments = arguments[2:]
    if arguments and arguments[0] in ("cc", "c++"):
        return [argument for argument in arguments if argument != "-Wl,-O1"]
    return arguments


if __name__ == "__main__":
    executable = Path(ziglang.__file__).parent / "zig"
    os.execv(executable, [str(executable), *zig_arguments(sys.argv[1:])])