#!/usr/bin/env python3
"""Run Zig without the linker-only optimization hint that Zig ignores."""

import os
from pathlib import Path
import shlex
import sys

import ziglang


def bindgen_arguments(rust_target: str) -> list[str]:
    """Return Clang arguments for parsing headers against Zig's Linux sysroot."""
    architecture = rust_target.split("-", 1)[0]
    architecture_family = {
        "aarch64": "aarch64",
        "x86_64": "x86",
    }.get(architecture)
    if architecture_family is None:
        raise ValueError(f"unsupported Linux architecture: {architecture}")

    zig_lib = Path(ziglang.__file__).parent / "lib"
    include_directories = [
        zig_lib / "include",
        zig_lib / "libc" / "include" / f"{architecture}-linux-gnu",
        zig_lib / "libc" / "include" / "generic-glibc",
        zig_lib / "libc" / "include" / f"{architecture_family}-linux-any",
        zig_lib / "libc" / "include" / "any-linux-any",
    ]
    missing = [path for path in include_directories if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"missing Zig include directories: {missing}")
    return [
        f"--target={zig_target(rust_target)}",
        *(argument for path in include_directories for argument in ("-isystem", str(path))),
    ]


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
    if len(sys.argv) == 3 and sys.argv[1] == "--print-bindgen-args":
        print(shlex.join(bindgen_arguments(sys.argv[2])))
        raise SystemExit
    executable = Path(ziglang.__file__).parent / "zig"
    os.execv(executable, [str(executable), *zig_arguments(sys.argv[1:])])