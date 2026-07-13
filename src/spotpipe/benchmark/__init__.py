"""Benchmark harness layer (fresh build; disposable tier).

The old repo's harness was NOT ported. Built fresh here:

* :mod:`spotpipe.benchmark.alpha` -- the frozen simulator-slope <-> physical
  ``alpha`` convention (the one place the factor of 2 lives).
* :mod:`spotpipe.benchmark.generate` -- the two-family benchmark IMAGE-SET
  generator (SNR x density + curvature). Generation only: runs no method, fits
  no slope, computes no metric.
* :mod:`spotpipe.benchmark.intensity_extraction` -- the shared intensity
  instrument used by every method downstream.
"""

from spotpipe.benchmark.alpha import (  # noqa: F401
    SIM_SLOPE_TO_ALPHA,
    alpha_to_sim_slope,
    sim_slope_to_alpha,
)

__all__ = ["SIM_SLOPE_TO_ALPHA", "sim_slope_to_alpha", "alpha_to_sim_slope"]
