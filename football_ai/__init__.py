"""Automatic football match analysis: detection, tracking, pitch calibration,
possession, events and statistics.

The heavy names are exported lazily. Importing this package otherwise pulls in
torch and ultralytics, which costs seconds and hundreds of megabytes, and
several callers only want to ask a cheap question — most often "are the weights
there?", which the web app asks on every page load. PEP 562 lets that question
be answered without paying for the answer to a different one.
"""
import importlib
from typing import TYPE_CHECKING

__all__ = ["analyse", "Pipeline", "PipelineConfig", "AnalysisResult", "weights"]
__version__ = "1.0.0"

if TYPE_CHECKING:  # pragma: no cover - for type checkers and editors only
    from . import weights
    from .pipeline import AnalysisResult, Pipeline, PipelineConfig, analyse

_LAZY_MODULES = {"weights", "pitch", "video"}
_LAZY_FROM_PIPELINE = {"analyse", "Pipeline", "PipelineConfig", "AnalysisResult"}


def __getattr__(name: str):
    # importlib rather than `from . import x`: the latter looks the attribute up
    # on this module first, which lands straight back here.
    if name in _LAZY_MODULES:
        return importlib.import_module(f".{name}", __name__)
    if name in _LAZY_FROM_PIPELINE:
        return getattr(importlib.import_module(".pipeline", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
