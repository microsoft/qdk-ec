# pylint: disable=no-member
#   no-member: protobuf generated classes do not have members detected by pylint
"""Unified sampling command.

Samples measurement outcomes from ``.stim`` circuits or ``.deq`` source
files.  When given ``.deq`` input, the file is first compiled through
the JIT transpiler and static compiler to produce a ``.stim`` circuit.

Examples::

    # Sample directly from a Stim circuit
    deq sample circuit.stim --shots 10

    # Compile .deq, strip noise, sample, and interpret syndromes
    deq sample circuit.deq --program MyProg --noiseless --interpret --shots 5

    # Fast recompilation with a pre-compiled JIT library
    deq sample circuit.deq --program MyProg --jit lib.deq.jit --noiseless --interpret

    # Sample from a .stim file and interpret against a .deq.bin
    deq sample circuit.stim --interpret --bin circuit.deq.bin --shots 5
"""

import io
import os
import sys
import tempfile
from contextlib import redirect_stdout

import arguably
import stim

import deq.proto.deq_bin_pb2 as pb
import deq.proto.deq_jit_pb2 as jit_pb
from deq.noise import strip_noise as _strip_noise_text
from deq.cli.interpret import interpret_measurements
from deq.cli.jit import transpile, compile_, jit_compile_program_to_file
from deq.cli.util import bits_to_hex
from deq.circuit.model import GadgetDefinition


def _sample_stim_text(stim_text: str, shots: int, seed: int | None) -> list[str]:
    """Sample from a Stim circuit string, returning hex measurement strings."""
    circuit = stim.Circuit(stim_text)
    if seed is not None:
        sampler = circuit.compile_sampler(seed=seed)
    else:
        sampler = circuit.compile_sampler()
    samples = sampler.sample(shots)
    results: list[str] = []
    for row in samples:
        bits = [int(b) for b in row]
        results.append(bits_to_hex(bits))
    return results


def _compile_deq_to_stim_and_bin(
    deq_files: tuple[str, ...],
    tmpdir: str,
    *,
    program: str,
    jit: str | None,
    jobs: int,
    plugin: list[str] | None,
    mako: list[str] | None,
    skip_mako_warning: bool,
) -> tuple[str, str]:
    """Compile .deq files into a .stim circuit and .deq.bin.

    Returns ``(stim_path, bin_path)``.
    """
    jit_out = os.path.join(tmpdir, "library.deq.jit")
    bin_out = os.path.join(tmpdir, "library.deq.bin")

    captured = io.StringIO()
    with redirect_stdout(captured):
        if jit is not None:
            if not deq_files:
                raise ValueError("at least one .deq file is required")
            from deq.circuit.mako_support import parse_mako_vars
            from deq.circuit.parser import render_and_parse_files

            mako_vars = parse_mako_vars(mako) if mako else None
            merged = render_and_parse_files(
                list(deq_files),
                mako_defs=mako_vars,
                skip_mako_warning=skip_mako_warning,
            )

            with open(jit, "rb") as f:
                jit_library = jit_pb.JitLibrary.FromString(f.read())

            jit_names = {gt.base.name for gt in jit_library.gadget_types}
            deq_gadget_names = {
                d.name for d in merged.definitions if isinstance(d, GadgetDefinition)
            }
            missing = deq_gadget_names - jit_names
            if missing:
                raise ValueError(
                    f"Gadget(s) defined in .deq but missing from "
                    f"{jit!r}: {sorted(missing)}. "
                    f"The .deq.jit file may be stale — "
                    f"rebuild with transpile."
                )

            jit_compile_program_to_file(jit_library, merged, jit_out, program=program)
            compile_(jit_out, out=bin_out)
        else:
            transpile(
                *deq_files,
                out=jit_out,
                program=program,
                jobs=jobs,
                plugin=plugin,
                mako=mako,
                skip_mako_warning=skip_mako_warning,
            )
            compile_(jit_out, out=bin_out)

    summary = captured.getvalue()
    if summary:
        print(summary, end="", file=sys.stderr)

    stim_out = os.path.join(tmpdir, "library.stim")
    if not os.path.exists(stim_out):
        raise FileNotFoundError(
            f"Expected Stim circuit at {stim_out} — "
            f"does the PROGRAM '{program}' produce a .stim file?"
        )
    return stim_out, bin_out


@arguably.command
def sample(
    *files: str,
    program: str | None = None,
    shots: int = 1,
    seed: int | None = None,
    noiseless: bool = False,
    interpret: bool = False,
    verbose: bool = False,
    #: path to a pre-compiled .deq.bin file for --interpret with .stim input
    bin_: str | None = None,
    #: path to a pre-compiled .deq.jit file; with .deq input skips
    #: gadget-type construction; with .stim input enables --interpret
    jit: str | None = None,
    #: number of parallel worker processes for JIT transpilation
    jobs: int = max((os.cpu_count() or 1) - 2, 1),
    #: register an external check plugin from a .py file
    plugin: list[str] | None = None,
    #: Mako variable definitions, each as key=value
    mako: list[str] | None = None,
    #: suppress the interactive Mako safety prompt
    skip_mako_warning: bool = False,
) -> list[str] | list[tuple[str, str]]:
    """Sample measurement outcomes from a .stim circuit or .deq source.

    **Input dispatch by file extension:**

    - ``.stim`` — sample directly from the Stim circuit.
    - ``.deq`` — transpile and compile first, then sample from the
      generated ``.stim`` circuit.  Requires ``--program``.
    - ``.deq`` + ``--jit PATH`` — load a pre-compiled JIT library and
      only recompile the PROGRAM block (fast iteration).

    **Flags:**

    - ``--noiseless`` strips noise instructions before sampling.
    - ``--interpret`` interprets syndromes and readouts after sampling.
      Auto-available for ``.deq`` input (binary compiled internally).
      For ``.stim`` input, requires ``--bin`` or ``--jit`` to provide
      the compiled library for interpretation.
    - ``--verbose`` shows per-gadget breakdown (with ``--interpret``).

    Examples::

        deq sample circuit.stim --shots 10
        deq sample circuit.stim --noiseless --shots 10
        deq sample circuit.deq --program Sim --noiseless --interpret
        deq sample circuit.stim --interpret --bin circuit.deq.bin
        deq sample circuit.stim --interpret --jit circuit.deq.jit
    """
    if not files:
        raise ValueError("at least one input file is required")

    is_stim = files[0].endswith(".stim")

    if is_stim:
        if len(files) > 1:
            raise ValueError("only one .stim file can be given")
        stim_file = files[0]

        with open(stim_file, encoding="utf-8") as f:
            stim_text = f.read()

        if noiseless:
            stim_text = _strip_noise_text(stim_text)

        hex_samples = _sample_stim_text(stim_text, shots, seed)

        if interpret:
            library = _load_library_for_eval(bin_=bin_, jit=jit)
            results_eval: list[tuple[str, str]] = []
            for i, hex_meas in enumerate(hex_samples):
                if verbose and i > 0:
                    print()
                results_eval.append(
                    interpret_measurements(library, hex_meas, verbose=verbose)
                )
            return results_eval

        for h in hex_samples:
            print(h)
        return hex_samples

    # .deq input — compile first
    if program is None:
        raise ValueError("--program is required for .deq input files")

    with tempfile.TemporaryDirectory() as tmpdir:
        stim_path, bin_path = _compile_deq_to_stim_and_bin(
            files,
            tmpdir,
            program=program,
            jit=jit,
            jobs=jobs,
            plugin=plugin,
            mako=mako,
            skip_mako_warning=skip_mako_warning,
        )

        with open(stim_path, encoding="utf-8") as f:
            stim_text = f.read()

        if noiseless:
            stim_text = _strip_noise_text(stim_text)

        hex_samples = _sample_stim_text(stim_text, shots, seed)

        if interpret:
            with open(bin_path, "rb") as f:
                library = pb.Library.FromString(f.read())

            results_eval = []
            for i, hex_meas in enumerate(hex_samples):
                if verbose and i > 0:
                    print()
                results_eval.append(
                    interpret_measurements(library, hex_meas, verbose=verbose)
                )
            return results_eval

        for h in hex_samples:
            print(h)
        return hex_samples


def _load_library_for_eval(*, bin_: str | None, jit: str | None) -> pb.Library:
    """Load a Library for evaluation from --bin or --jit."""
    if bin_ is not None:
        with open(bin_, "rb") as f:
            return pb.Library.FromString(f.read())

    if jit is not None:
        from deq.compiler.jit_compiler import static_jit_compiler

        with open(jit, "rb") as f:
            jit_library = jit_pb.JitLibrary.FromString(f.read())
        if not jit_library.program:
            raise ValueError(
                f"No program found in {jit}. "
                f"Use transpile with --program to embed one."
            )
        deq_bin = static_jit_compiler(jit_library)
        return deq_bin

    raise ValueError(
        "--interpret with .stim input requires --bin or --jit "
        "to provide the compiled library"
    )
