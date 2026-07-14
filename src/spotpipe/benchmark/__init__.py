"""Benchmark harness layer (fresh build; disposable tier).

The old repo's harness was NOT ported. Built fresh here:

* :mod:`spotpipe.benchmark.alpha` -- the frozen simulator-slope <-> physical
  ``alpha`` convention (the one place the factor of 2 lives).
* :mod:`spotpipe.benchmark.generate` -- the two-family benchmark IMAGE-SET
  generator (SNR x density + curvature). Generation only: runs no method, fits
  no slope, computes no metric.
* :mod:`spotpipe.benchmark.intensity_extraction` -- the shared intensity
  instrument used by every method downstream.
* :mod:`spotpipe.benchmark.evaluate` -- the ONE shared, blind, tool-agnostic
  evaluator: schema CSVs + GT -> detection metrics, log-ratio bias/spread, and
  the recovered curvature slope ``alpha`` (docs/evaluator_convention.md).
"""

from spotpipe.benchmark.alpha import (  # noqa: F401
    SIM_SLOPE_TO_ALPHA,
    alpha_to_sim_slope,
    sim_slope_to_alpha,
)
from spotpipe.benchmark.evaluate import (  # noqa: F401
    AlphaFit,
    evaluate_all,
    evaluate_condition,
    evaluate_method,
    fit_alpha,
    ground_truth_as_predictions,
    load_benchmark_info,
)

__all__ = [
    "SIM_SLOPE_TO_ALPHA", "sim_slope_to_alpha", "alpha_to_sim_slope",
    "AlphaFit", "fit_alpha", "evaluate_condition", "evaluate_method",
    "evaluate_all", "ground_truth_as_predictions", "load_benchmark_info",
]
