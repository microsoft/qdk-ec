"""Shared example inventory for the protocol audit tests."""

from pathlib import Path

EXAMPLES_DIR = Path(__file__).resolve().parents[1]
EXAMPLE_MANIFESTS = (
    "bacon-shor/bacon-shor.qodec.yaml",
    "c422-c832-arch/qodec.yaml",
    "c4c6/qodec.yaml",
    "distillation-15/distillation-15.qodec.yaml",
    "honeycomb/honeycomb.qodec.yaml",
    "iceberg/iceberg.qodec.yaml",
    "reed-muller-15/reed-muller-15.qodec.yaml",
    "repetition3/repetition3.qodec.yaml",
    "steane/steane.qodec.yaml",
    "surface/surface.qodec.yaml",
)
