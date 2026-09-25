"""Shared default values for DEQ."""

from typing import Final

#: Maximum consecutive failed preselection attempts before sampling aborts.
DEFAULT_MAX_PRESELECT_ATTEMPTS: Final[int] = 1_000_000

#: Default timeout, in seconds, for long-running subprocesses.
DEFAULT_TIMEOUT: Final[int] = 36_000

#: Default number of Rayon worker threads used by simulation subprocesses.
DEFAULT_RAYON_NUM_THREADS: Final[int] = 2

#: Default number of Tokio worker threads used by simulation subprocesses.
DEFAULT_TOKIO_WORKER_THREADS: Final[int] = 4

#: Default timeout, in milliseconds, for visualizer rendering.
DEFAULT_RENDER_TIMEOUT_MS: Final[int] = 30_000

#: Whether the visualizer loads its frontend from the development server.
DEFAULT_WIDGET_DEV_MODE: Final[bool] = False

#: Absolute tolerance, in half-turns, for U/U3 Pauli-axis recognition.
DEFAULT_U3_AXIS_TOLERANCE: Final[float] = 1e-12
