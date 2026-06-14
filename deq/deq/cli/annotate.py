"""CLI command for :mod:`deq.transpiler.jit_annotate`."""

import sys

import arguably

from deq.circuit.parser import render_and_parse_file, parse as parse_deq
from deq.cli.strip_tags import strip_jit_library
from deq.transpiler.jit_annotate import annotate as _annotate_impl
from deq.transpiler.jit_library_builder import build_jit_library
from deq.circuit.mako_support import parse_mako_vars


@arguably.command
def annotate(
    deq_file: str,
    *,
    out: str | None = None,
    #: register an external check plugin from a .py file
    plugin: list[str] | None = None,
    #: Mako variable definitions, each as key=value
    #: (e.g. --mako d=3 --mako p=0.01); implies --skip-mako-warning
    mako: list[str] | None = None,
    #: suppress the interactive Mako safety prompt
    skip_mako_warning: bool = False,
    #: skip verification that annotated output transpiles identically
    no_verify: bool = False,
) -> None:
    """
    Rewrite a .deq file to mirror the structure of its compiled .deq.jit.

    Inlines imports, replaces stabilizers/logicals with `_` identity
    placeholders (originals kept as comments), comments out noise instructions,
    forces every gadget to
    @CHECKS("manual", verify=0), and inserts auto-derived CHECKs (marked
    `# auto`). COMPOSE/PROGRAM blocks are emitted commented-out for reference.

    By default, the annotated output is verified by transpiling both the
    original and annotated files, stripping debug tags, and asserting
    byte-equality of the serialized JIT libraries.  This is exact
    because explicit ``PROPAGATE`` statements emitted by annotate pin
    every output logical row of cp/pc to the representative the
    original transpile chose, and intra-check measurement order /
    error probabilities are reproducible.
    Use ``--no-verify`` to skip this (faster but no correctness guarantee).

    Args:
        deq_file: path to the input .deq file.
        out: path to write the annotated output to. Defaults to
             ``<input>.annotated.deq`` when ``--no-verify`` is off,
             or stdout otherwise.
        plugin: one or more .py files registering external check plugins.
    """

    if plugin:
        from deq.transpiler.check_plugins import register_plugin_file

        for p in plugin:
            register_plugin_file(p)

    mako_vars = parse_mako_vars(mako) if mako else None
    qfile = render_and_parse_file(
        deq_file,
        mako_defs=mako_vars,
        skip_mako_warning=skip_mako_warning,
    )

    rendered = _annotate_impl(qfile)

    # Determine output path.
    if out is None:
        base = deq_file
        if base.endswith(".deq"):
            base = base[: -len(".deq")]
        out = f"{base}.annotated.deq"

    with open(out, "w", encoding="utf-8") as fh:
        fh.write(rendered)
    print(f"Wrote {out}", file=sys.stderr)

    if no_verify:
        return

    # Verify: transpile the annotated output and compare.
    print(
        f"Verifying annotated output is equivalent to original",
        f"(pass --no-verify to skip)...",
        file=sys.stderr,
    )
    orig_lib = build_jit_library(qfile)
    anno_lib = build_jit_library(parse_deq(rendered))
    orig_stripped, _ = strip_jit_library(orig_lib)
    anno_stripped, _ = strip_jit_library(anno_lib)
    if orig_stripped.SerializeToString() == anno_stripped.SerializeToString():
        print("Verification passed.", file=sys.stderr)
    else:
        print(
            "ERROR: annotated output is not byte-equivalent to original"
            " after tag stripping.",
            file=sys.stderr,
        )
        raise SystemExit(1)
