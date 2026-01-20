"""Pytest configuration for paulimer Python tests.

This module provides hypothesis settings profiles and re-exports
strategies and fixtures for convenient imports in test files.
"""

import os

from hypothesis import settings, Verbosity


# ========== Hypothesis Settings ==========

settings.register_profile("factory")
settings.register_profile("build", print_blob=True, deadline=500)
settings.register_profile("fast", max_examples=10)
settings.register_profile("thorough", max_examples=1000)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)
settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "fast"))


# Re-export strategies and fixtures for convenient imports
from .strategies import (
    sparse_pauli_strategy,
    unitary_instruction_strategy,
    simulation_strategy,
)
from .fixtures import (
    make_bell_circuit,
    make_ghz_circuit,
    make_repetition_code_circuit,
)
