"""Mako template detection, safety confirmation, and rendering.

Provides utilities for transparently preprocessing ``.deq`` files
that contain Mako template syntax (``${...}``, ``% for``, ``<%...%>``).

Mako templates can execute arbitrary Python code, so the CLI tools
prompt the user for confirmation before rendering — unless the user
passes ``--skip-mako-warning`` or ``--mako`` (which implies consent).
"""

import os
import re
import sys

# ── Mako detection (same patterns as deq/noise/_parsing.py) ─────────

_MAKO_BLOCK = re.compile(r"<%|%>")
_MAKO_EXPR = re.compile(r"\$\{")
_MAKO_CONTROL = re.compile(r"^\s*%\s+\w")


def has_mako_syntax(text: str) -> bool:
    """Return ``True`` if *text* contains any Mako template syntax."""
    for line in text.splitlines():
        if _MAKO_BLOCK.search(line):
            return True
        if _MAKO_EXPR.search(line):
            return True
        if _MAKO_CONTROL.match(line):
            return True
    return False


# ── Safety confirmation ──────────────────────────────────────────────

_MAKO_WARNING = """\
WARNING: A .deq file contains Mako template syntax which can execute
arbitrary Python code. Other files in the import tree may also contain
Mako syntax. Please make sure you have reviewed all .deq files and
understand the risks.

File with Mako syntax:
  - {file_path}

To suppress this warning, pass --skip-mako-warning or --mako.

Proceed? [y/N] """


def confirm_mako_execution(
    file_path: str,
    skip_warning: bool,
) -> None:
    """Prompt the user to confirm Mako template execution.

    Called on the first file in the import tree found to contain Mako
    syntax.  Does nothing if *skip_warning* is ``True``.  When stdin is
    not a terminal (piped input), raises ``SystemExit`` instead of
    prompting.

    Raises
    ------
    SystemExit
        If the user declines or stdin is not interactive.
    """
    if skip_warning:
        return

    msg = _MAKO_WARNING.format(file_path=file_path)

    if not sys.stdin.isatty():
        print(msg, end="", file=sys.stderr)
        raise SystemExit(
            "Mako templates detected but stdin is not interactive. "
            "Pass --skip-mako-warning or --mako to proceed."
        )

    print(msg, end="", file=sys.stderr)
    answer = input().strip().lower()
    if answer != "y":
        raise SystemExit("Aborted.")


# ── Rendering ────────────────────────────────────────────────────────


def parse_mako_vars(args: list[str]) -> dict[str, str]:
    """Parse a list of ``key=value`` strings into a dict.

    Each string must contain exactly one ``=``.  For example::

        parse_mako_vars(["d=3", "p=0.01"])
        # => {"d": "3", "p": "0.01"}

    Raises ``ValueError`` on malformed entries.
    """
    result: dict[str, str] = {}
    for arg in args:
        if "=" not in arg:
            raise ValueError(f"invalid --mako argument {arg!r}: expected key=value")
        key, _, value = arg.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"invalid --mako argument {arg!r}: empty key")
        result[key] = value.strip()
    return result


def render_mako(
    text: str,
    mako_defs: dict[str, str],
    template_dir: str = ".",
) -> str:
    """Render a single file's text through the Mako template engine.

    *mako_defs* are passed as keyword arguments to
    ``Template.render()``, making them available via
    ``context.get('key', default)`` in templates — matching the
    behaviour of ``mako-render --var``.

    *template_dir* enables ``<%include file="..."/>``
    directives by setting up a ``TemplateLookup`` rooted at the given
    directory.
    """
    from mako.template import Template  # pylint: disable=import-outside-toplevel
    from mako.lookup import TemplateLookup  # pylint: disable=import-outside-toplevel

    lookup = TemplateLookup(directories=[template_dir])
    tmpl = Template(text=text, lookup=lookup)
    return tmpl.render(**mako_defs)


def read_and_render_file(
    path: str,
    mako_defs: dict[str, str] | None = None,
    skip_mako_warning: bool = False,
) -> str:
    """Read a file and render Mako templates if present.

    Returns the file text, rendered through Mako if it contains template
    syntax or *mako_defs* is provided.  Prompts the user for
    confirmation on first encounter of Mako syntax (unless
    *skip_mako_warning* or *mako_defs* is set).
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()

    if mako_defs is not None:
        skip_mako_warning = True

    needs_render = mako_defs is not None
    if has_mako_syntax(text):
        if not skip_mako_warning:
            confirm_mako_execution(path, skip_warning=False)
        needs_render = True

    if needs_render:
        text = render_mako(
            text, mako_defs or {}, template_dir=os.path.dirname(os.path.abspath(path))
        )

    return text
