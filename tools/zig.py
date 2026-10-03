#!/usr/bin/env python3
"""Run Zig without the linker-only optimization hint that Zig ignores."""

import os
from pathlib import Path
import sys

import ziglang


def zig_target(argument: str) -> str:
    """Translate Rust Linux target triples into Zig's target syntax."""
    return argument.replace("-unknown-linux-", "-linux-", 1)


def zig_compiler_arguments(arguments: list[str]) -> list[str]:
    translated: list[str] = []
    translate_next = False
    for argument in arguments:
        if translate_next:
            translated.append(zig_target(argument))
            translate_next = False
        elif argument in ("-target", "--target"):
            translated.append(argument)
            translate_next = True
        elif argument.startswith(("-target=", "--target=")):
            option, target = argument.split("=", 1)
            translated.append(f"{option}={zig_target(target)}")
        elif argument != "-Wl,-O1":
            translated.append(argument)
    return translated


def zig_arguments(arguments: list[str]) -> list[str]:
    if arguments[:2] == ["-m", "ziglang"]:
        arguments = arguments[2:]
    if arguments and arguments[0] in ("cc", "c++"):
        return zig_compiler_arguments(arguments)
    return arguments


if __name__ == "__main__":
    executable = Path(ziglang.__file__).parent / "zig"
    os.execv(executable, [str(executable), *zig_arguments(sys.argv[1:])])