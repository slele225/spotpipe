#!/usr/bin/env python
"""Run the synthetic spot-detection benchmark.

Thin CLI over :func:`spotpipe.benchmark.harness.run_benchmark`. Equivalent to
``python -m spotpipe.benchmark.harness``.

Examples
--------
Run on an existing eval set (built by ``generate_dataset``), with our model::

    uv run python scripts/run_benchmark.py \\
        --eval-dir data/eval --config configs/benchmark.yaml \\
        --out outputs/bench --checkpoint runs/run1/checkpoint.pt

Build a fresh in-memory eval set from a simulator config and run the baselines
only (no checkpoint -> our_model is dropped automatically)::

    uv run python scripts/run_benchmark.py \\
        --simulator-config configs/simulator.yaml --n-images 16 \\
        --out outputs/bench \\
        --methods classical_per_channel_aperture,oracle_center_aperture_divide
"""

from __future__ import annotations

import sys
from pathlib import Path

# Editable install puts spotpipe on the path; this fallback keeps the script
# runnable straight from a fresh checkout too (no sys.path hacks for shared code).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spotpipe.benchmark.harness import main

if __name__ == "__main__":
    raise SystemExit(main())
