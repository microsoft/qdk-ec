"""CLI commands to inject or strip noise in plain .deq files.

Usage::

    deq inject si1000 input.deq --p 0.001
    deq inject si1000 input.deq --p 0.001 --out output.deq
    deq inject biased input.deq --p 0.001
    deq inject biased input.deq --p 0.001 --p1q 0.0001 --eta 20
    deq strip-noise input_noisy.deq --out output_clean.deq
"""

import sys

import arguably

from deq.noise import inject_biased, inject_si1000
from deq.noise import strip_noise as _strip_noise
from deq.circuit.mako_support import parse_mako_vars, read_and_render_file


@arguably.command
def inject__si1000(
    file: str,
    *,
    p: float,
    out: str | None = None,
    #: Mako variable definitions, each as key=value
    #: (e.g. --mako d=3 --mako p=0.01); implies --skip-mako-warning
    mako: list[str] | None = None,
    #: suppress the interactive Mako safety prompt
    skip_mako_warning: bool = False,
) -> None:
    """Inject SI1000 noise into a plain .deq file.

    Args:
        file: Path to the input ``.deq`` file.
        p: Physical error rate (e.g. ``0.001``).
        out: Output file path.  If omitted, writes to stdout.
        mako: semicolon-separated Mako definitions (e.g. ``"d=3;p=0.01"``).
        skip_mako_warning: suppress the interactive Mako safety prompt.
    """
    if p < 0 or p > 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    mako_vars = parse_mako_vars(mako) if mako else None
    text = read_and_render_file(
        file, mako_defs=mako_vars, skip_mako_warning=skip_mako_warning
    )
    result = inject_si1000(text, p)

    if out is not None:
        with open(out, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Wrote noisy circuit to {out}", file=sys.stderr)
    else:
        sys.stdout.write(result)


@arguably.command
def inject__biased(
    file: str,
    *,
    p: float = 0.004,
    p1q: float | None = None,
    eta: float = 10.0,
    exclude: str = "",
    out: str | None = None,
    #: Mako variable definitions, each as key=value
    #: (e.g. --mako d=3 --mako p=0.01); implies --skip-mako-warning
    mako: list[str] | None = None,
    #: suppress the interactive Mako safety prompt
    skip_mako_warning: bool = False,
) -> None:
    """Inject biased noise into a plain .deq file.

    The biased model uses asymmetric rates: 1-qubit gates/measurements/resets
    use ``p1q`` (default ``p/4``), while 2-qubit gates get Z-biased noise
    split into ``DEPOLARIZE2(p/(eta+1))`` + ``Z_ERROR(eta*p/(eta+1))``.

    Args:
        file: Path to the input ``.deq`` file.
        p: Physical error rate for 2-qubit gates (default ``0.004``).
        p1q: Physical error rate for 1-qubit operations.  Defaults to ``p/4``.
        eta: Bias parameter for Z-error weight on 2-qubit gates (default ``10.0``).
        exclude: Comma-separated gate names to exclude from noise (e.g. ``S,S_DAG``).
        out: Output file path.  If omitted, writes to stdout.
        mako: Mako variable definitions, each as ``key=value``.
        skip_mako_warning: suppress the interactive Mako safety prompt.
    """
    if p < 0 or p > 1:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if p1q is not None and (p1q < 0 or p1q > 1):
        raise ValueError(f"p1q must be in [0, 1], got {p1q}")
    if eta <= 0:
        raise ValueError(f"eta must be positive, got {eta}")
    mako_vars = parse_mako_vars(mako) if mako else None
    text = read_and_render_file(
        file, mako_defs=mako_vars, skip_mako_warning=skip_mako_warning
    )

    exclude_list = (
        [g.strip() for g in exclude.split(",") if g.strip()] if exclude else []
    )
    result = inject_biased(text, p, p1q=p1q, eta=eta, exclude=exclude_list)

    if out is not None:
        with open(out, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Wrote noisy circuit to {out}", file=sys.stderr)
    else:
        sys.stdout.write(result)


@arguably.command
def strip_noise(
    file: str,
    *,
    out: str | None = None,
    #: Mako variable definitions, each as key=value
    #: (e.g. --mako d=3 --mako p=0.01); implies --skip-mako-warning
    mako: list[str] | None = None,
    #: suppress the interactive Mako safety prompt
    skip_mako_warning: bool = False,
) -> None:
    """Strip all noise instructions from a .deq file.

    Removes every noise channel line (``DEPOLARIZE1``, ``DEPOLARIZE2``,
    ``X_ERROR``, ``Y_ERROR``, ``Z_ERROR``, ``PAULI_CHANNEL_1``, etc.)
    while preserving all other lines verbatim.

    Args:
        file: Path to the input ``.deq`` file.
        out: Output file path.  If omitted, writes to stdout.
        mako: Mako variable definitions, each as ``key=value``.
        skip_mako_warning: suppress the interactive Mako safety prompt.
    """
    mako_vars = parse_mako_vars(mako) if mako else None
    text = read_and_render_file(
        file, mako_defs=mako_vars, skip_mako_warning=skip_mako_warning
    )
    result = _strip_noise(text)

    if out is not None:
        with open(out, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Wrote clean circuit to {out}", file=sys.stderr)
    else:
        sys.stdout.write(result)
