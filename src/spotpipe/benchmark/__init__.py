"""Benchmark harness, metrics, matching, and method adapters (build stage 4).

A SYNTHETIC-only benchmark: because we hold the ground truth, per-spot intensity
bias/variance and ratio-law slope recovery are exactly measurable. Every method
(our model and external baselines) is wrapped by an adapter that emits the
canonical schema in :mod:`spotpipe.schema`, so matching, metrics, and the harness
are entirely method-agnostic. Adding a method = one new adapter.

Pipeline: ``adapters`` (method -> schema) -> ``matching`` (predictions <-> GT) ->
``metrics`` (binned bias/variance by SNR x density, slope recovery, uncertainty
calibration) -> ``harness`` (orchestrate the fixed eval set, write tables +
figures). ``features`` defines the SNR / density binning axes; ``baselines`` are
the two naive in-repo competitors.
"""

from spotpipe.benchmark.adapters import Adapter, get_adapter
from spotpipe.benchmark.harness import EvalImage, build_eval_set, load_eval_set, run_benchmark
from spotpipe.benchmark.matching import DatasetMatch, MatchResult, match_dataset, match_spots
from spotpipe.benchmark.metrics import compute_metrics

__all__ = [
    "Adapter",
    "get_adapter",
    "EvalImage",
    "load_eval_set",
    "build_eval_set",
    "run_benchmark",
    "MatchResult",
    "DatasetMatch",
    "match_spots",
    "match_dataset",
    "compute_metrics",
]
