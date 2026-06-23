"""Per-spot intensity loss (STUB -- implemented in build stage 3).

Heteroscedastic Gaussian NLL on logI1/logI2: the intensity heads predict a
log-variance per channel, and the loss is masked to ground-truth spot centers.
The supervision target is each spot's true TOTAL INTEGRATED photon count from
the simulator (pre-detector), before spots are summed into the image -- never a
pixel/window readout, which would be contaminated by overlapping neighbours.
Heavily overlapped / dim spots should come back with larger predicted variance.
"""

from __future__ import annotations


def intensity_nll(*args, **kwargs):
    """Heteroscedastic Gaussian NLL on per-spot log-intensities. STUB."""
    raise NotImplementedError("intensity.intensity_nll is implemented in build stage 3.")
