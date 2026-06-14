"""Check plugin registry.

Each check plugin is a Python module exposing a single function::

    def resolve_checks(inp: CheckPluginInput) -> CheckPluginOutput: ...

Built-in plugins live as sibling ``.py`` files in this package.
External plugins can be registered at runtime via :func:`register_plugin`
or from the CLI with ``--plugin ./path/to/file.py``.
"""

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Callable

from deq.circuit.model import (
    CodeDefinition,
    GadgetDefinition,
    ComposeDefinition,
    KeywordArg,
)

# Re-export layout/frame helpers from jit_transpiler for backward compat.
from deq.transpiler.jit_transpiler import (
    Check,
    MeasurementLayout,
    compute_layout,
    derive_checks_auto,
    parse_checks_manual,
)

logger = logging.getLogger(__name__)


@dataclass
class CheckPluginInput:
    """Data handed to every check plugin.

    ``auto_checks`` is computed lazily via paulimer on first access.
    Plugins that never read it (e.g. ``manual`` with ``verify=0``)
    avoid the expensive derivation entirely.
    """

    gadget: GadgetDefinition
    codes: dict[str, CodeDefinition]
    manual_checks: list[Check]
    total_measurements: int
    layout: MeasurementLayout
    plugin_args: tuple[str | int | float, ...] = ()
    plugin_kwargs: dict[str, str | int | float] = field(default_factory=dict)

    @cached_property
    def auto_checks(self) -> list[Check]:
        """Derive checks via paulimer (expensive; computed on first access)."""
        checks, total = derive_checks_auto(self.gadget, self.codes)
        assert total == self.total_measurements
        return checks

    @property
    def input_virtual_count(self) -> int:
        return self.layout.input_virtual_count

    @property
    def internal_count(self) -> int:
        return self.layout.internal_count

    @property
    def ov_start(self) -> int:
        return self.layout.ov_start

    @property
    def num_ov(self) -> int:
        return self.total_measurements - self.ov_start


@dataclass
class CheckPluginOutput:
    """Result returned by a check plugin."""

    finished: list[Check]
    unfinished: list[Check]


ResolveChecksFunc = Callable[[CheckPluginInput], CheckPluginOutput]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_registry: dict[str, ResolveChecksFunc] = {}


def register_plugin(name: str, func: ResolveChecksFunc) -> None:
    """Register a check plugin under *name* (case-insensitive)."""
    _registry[name.lower()] = func


def register_plugin_file(path: str | Path) -> str:
    """Load a plugin from a ``.py`` file and register its ``resolve_checks``.

    Returns the plugin name (the file stem, lowercased).
    """
    p = Path(path)
    name = p.stem.lower()
    spec = importlib.util.spec_from_file_location(f"deq_check_plugin_{name}", p)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load plugin from {p}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    func = getattr(mod, "resolve_checks", None)
    if func is None:
        raise ValueError(f"plugin {p} does not define a 'resolve_checks' function")
    register_plugin(name, func)
    return name


def get_plugin(name: str) -> ResolveChecksFunc:
    """Return the plugin function for *name*, loading built-ins lazily."""
    key = name.lower()
    if key not in _registry:
        _load_builtin(key)
    if key not in _registry:
        raise ValueError(
            f"unknown check plugin {name!r}; available: {sorted(_registry)}"
        )
    return _registry[key]


def _load_builtin(name: str) -> None:
    """Try to import ``deq.transpiler.check_plugins.<name>``."""
    mod_name = f"deq.transpiler.check_plugins.{name.replace('-', '_')}"
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError:
        return
    func = getattr(mod, "resolve_checks", None)
    if func is not None:
        _registry[name] = func


# ---------------------------------------------------------------------------
# Helpers shared by plugins
# ---------------------------------------------------------------------------


# Known decorator names per definition type.
_KNOWN_GADGET_DECORATORS = frozenset({"GTYPE", "CHECKS"})
_KNOWN_CODE_DECORATORS = frozenset({"PTYPE"})
_KNOWN_COMPOSE_DECORATORS = frozenset({"GTYPE"})


def warn_unrecognized_decorators(
    definition: GadgetDefinition | CodeDefinition | ComposeDefinition,
    *,
    known: frozenset[str] | None = None,
) -> None:
    """Emit a warning for any decorator not in *known*.

    This catches typos like ``@METACHECKS`` (removed) or ``@CHECK``
    (should be ``@CHECKS``) before they silently have no effect.
    """
    import warnings

    from deq.circuit.model import ComposeDefinition

    if known is None:
        if isinstance(definition, GadgetDefinition):
            known = _KNOWN_GADGET_DECORATORS
        elif isinstance(definition, ComposeDefinition):
            known = _KNOWN_COMPOSE_DECORATORS
        elif isinstance(definition, CodeDefinition):
            known = _KNOWN_CODE_DECORATORS
        else:
            return

    for deco in definition.decorators:
        if deco.name not in known:
            warnings.warn(
                f"unrecognized decorator @{deco.name} on "
                f"{type(definition).__name__} {definition.name!r}; "
                f"known decorators are: {', '.join(sorted(known))}",
                stacklevel=3,
            )


def get_checks_plugin_name(
    gadget: GadgetDefinition,
) -> tuple[str, tuple[str | int | float, ...], dict[str, str | int | float]]:
    """Extract ``@CHECKS(...)`` plugin name, positional args, and kwargs.

    Returns ``(plugin_name, args, kwargs)``.  The first decorator argument
    is the plugin name; subsequent positional arguments go into ``args``;
    :class:`KeywordArg` entries go into ``kwargs``.

    Defaults to ``("auto", (), {})`` when no ``@CHECKS`` decorator is present.
    """
    for decorator in gadget.decorators:
        if decorator.name != "CHECKS":
            continue
        if len(decorator.arguments) < 1:
            raise ValueError(
                f"@CHECKS expects at least one argument, got {decorator.arguments!r}"
            )
        name_arg = decorator.arguments[0]
        if not isinstance(name_arg, str):
            raise ValueError(
                f"@CHECKS first argument must be a string, got {name_arg!r}"
            )
        args: list[str | int | float] = []
        kwargs: dict[str, str | int | float] = {}
        for a in decorator.arguments[1:]:
            if isinstance(a, KeywordArg):
                kwargs[a.key] = a.value
            else:
                args.append(a)
        return name_arg.lower(), tuple(args), kwargs
    return "auto", (), {}


def resolve_gadget_checks(
    gadget: GadgetDefinition,
    codes: dict[str, CodeDefinition],
) -> CheckPluginOutput:
    """One-stop entry point: derive auto+manual checks, dispatch to plugin."""

    name, extra_args, extra_kwargs = get_checks_plugin_name(gadget)
    func = get_plugin(name)

    manual_checks, total = parse_checks_manual(gadget, codes)
    layout = compute_layout(gadget, codes)

    inp = CheckPluginInput(
        gadget=gadget,
        codes=codes,
        manual_checks=manual_checks,
        total_measurements=total,
        layout=layout,
        plugin_args=extra_args,
        plugin_kwargs=extra_kwargs,
    )
    return func(inp)
