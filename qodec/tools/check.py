"""Run qodec verification gates with the invoking Python interpreter."""

import argparse
from dataclasses import dataclass, field
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
BINDINGS = ROOT / "bindings/python"
SCOPES = ("rust", "python", "docs", "coverage", "examples", "packaging", "all")


@dataclass(frozen=True)
class Step:
    command: tuple[str, ...]
    cwd: Path = ROOT
    extra_env: dict[str, str] = field(default_factory=dict)


def steps_for(scope: str, python: str) -> list[Step]:
    if scope not in SCOPES:
        raise ValueError(f"Unknown check scope: {scope}")
    package_args = ("-p", "qodec", "-p", "qodec-python", "-p", "qodec-c")
    rust = [
        Step(("cargo", "fmt", *package_args, "--", "--check")),
        Step(("cargo", "clippy", *package_args, "--all-targets", "--all-features", "--", "-D", "clippy::pedantic")),
        Step(("cargo", "build", "-p", "qodec-c")),
        Step(("cargo", "test", *package_args, "--all-features")),
    ]
    import_check = (
        "import pathlib, qodec, sys; "
        "location = pathlib.Path(qodec.__file__).resolve(); "
        "expected = pathlib.Path(sys.argv[1]).resolve(); "
        "print('qodec import:', location); "
        "sys.exit(0 if location.is_relative_to(expected) else 'qodec is not imported from this checkout')"
    )
    python_steps = [
        Step((python, "-m", "maturin", "develop", "--release"), BINDINGS),
        Step((python, "-c", import_check, str(BINDINGS / "python/qodec"))),
        Step((python, "-m", "mypy", "python/qodec", "tests"), BINDINGS),
        Step((python, "-m", "mypy.stubtest", "qodec", "--allowlist", "stubtest-allowlist.txt"), BINDINGS),
        Step((python, "-m", "pytest", "-q"), BINDINGS),
    ]
    docs = [
        Step(("cargo", "doc", *package_args, "--no-deps"), extra_env={"RUSTDOCFLAGS": "-D warnings"}),
        Step((python, "-m", "sphinx", "-W", "--keep-going", "-b", "html", "bindings/python/docs", "target/python-docs/html")),
        Step((python, "-m", "sphinx", "-W", "--keep-going", "-b", "doctest", "bindings/python/docs", "target/python-docs/doctest")),
        Step((python, "bindings/python/docs/test_docs.py")),
    ]
    coverage = [Step(("cargo", "llvm-cov", "-p", "qodec", "--summary-only", "--fail-under-lines", "88"))]
    examples = [Step((python, "-m", "pytest", "examples/tests", "-q"))]
    packaging = [Step((python, "-m", "unittest", "discover", "-s", "tools", "-p", "test_*.py"))]
    groups = {
        "rust": rust, "python": python_steps, "docs": docs, "coverage": coverage,
        "examples": examples, "packaging": packaging,
    }
    return sum(groups.values(), []) if scope == "all" else groups[scope]


def selected_environment(python: str, prefix: Path, base_prefix: Path,
                         inherited: dict[str, str]) -> dict[str, str]:
    environment = inherited.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("CONDA_PREFIX", None)
    if (prefix / "conda-meta").is_dir():
        environment["CONDA_PREFIX"] = str(prefix)
    elif prefix != base_prefix:
        environment["VIRTUAL_ENV"] = str(prefix)
    environment["PYO3_PYTHON"] = python
    environment["PATH"] = str(Path(python).parent) + os.pathsep + inherited.get("PATH", "")
    return environment


def execute(steps: list[Step], environment: dict[str, str], *, dry_run: bool) -> int:
    for number, step in enumerate(steps, 1):
        command = shlex.join(step.command)
        settings = " ".join(f"{key}={shlex.quote(value)}" for key, value in step.extra_env.items())
        print(f"[{number}/{len(steps)}] cwd={step.cwd}\n  {settings + ' ' if settings else ''}{command}", flush=True)
        if dry_run:
            continue
        try:
            result = subprocess.run(step.command, cwd=step.cwd, env=environment | step.extra_env, check=False)
        except OSError as error:
            print(f"Could not start check: {error}", file=sys.stderr)
            return 127
        if result.returncode:
            print(f"FAILED: {command} (exit {result.returncode}); remaining checks not run", file=sys.stderr)
            return result.returncode if result.returncode > 0 else 128 - result.returncode
    print("DRY RUN: no checks executed" if dry_run else f"PASS: {len(steps)} checks completed", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scope", choices=SCOPES)
    parser.add_argument("--dry-run", action="store_true", help="Print commands and working directories without running them")
    arguments = parser.parse_args()
    python = os.path.abspath(sys.executable)
    environment = selected_environment(python, Path(sys.prefix), Path(sys.base_prefix), dict(os.environ))
    print(f"Python: {python}\nScope: {arguments.scope}", flush=True)
    if not arguments.dry_run:
        if arguments.scope in ("python", "all") and not (environment.get("VIRTUAL_ENV") or environment.get("CONDA_PREFIX")):
            parser.error("maturin develop needs a selected virtualenv or conda interpreter; no environment was changed")
        required = ["cargo"]
        if arguments.scope in ("rust", "all"):
            required.append("cbindgen")
        for tool in required:
            location = shutil.which(tool, path=environment["PATH"])
            if location is None:
                parser.error(f"{tool} is missing from PATH; install the documented prerequisite before running checks")
            print(f"{tool}: {location}", flush=True)
    try:
        return execute(steps_for(arguments.scope, python), environment, dry_run=arguments.dry_run)
    except KeyboardInterrupt:
        print("Checks interrupted; remaining gates are unverified", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())