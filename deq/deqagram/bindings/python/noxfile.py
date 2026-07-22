# Usage:
#   nox               Run default sessions (stubtest, pytest).
#   nox -s <name>     Run a specific session by name.
#
# Each session builds the extension into its own virtualenv with `maturin
# develop`, so `maturin`, `mypy`, and `pytest` are installed as session deps.

import nox

nox.options.sessions = ("stubtest", "pytest")

_DEV = ("maturin>=1.5", "mypy>=1.0", "pytest>=8.0")


@nox.session
def stubtest(session):
    """Verify the .pyi stubs match the compiled deqagram module."""
    session.install(*_DEV)
    session.run("maturin", "develop")
    session.run(
        "python",
        "-m",
        "mypy.stubtest",
        "deqagram",
        "--allowlist",
        "stubtest_allowlist.txt",
        # Allowlist entries guard feature-gated symbols that may be absent on a
        # given build; don't fail when an entry is unused.
        "--ignore-unused-allowlist",
    )


@nox.session
def pytest(session):
    """Run the PyO3 wrapper regression tests."""
    session.install(*_DEV)
    session.run("maturin", "develop")
    session.run("python", "-m", "pytest", "tests", *session.posargs)
