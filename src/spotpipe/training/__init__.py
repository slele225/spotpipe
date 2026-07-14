"""Training layer (DISPOSABLE) for the measured-detector hrnet_large retrain.

Ported/adapted from the old repo's ``spotpipe.training`` (READ-ONLY reference at
``C:\\Users\\shivl\\Videos\\spotpipe``). Nothing here is vendored/frozen: it
composes the FROZEN model / losses / schema / simulator primitives without
editing them. The three prompt-mandated departures from the old harness are:

* per-image detector-gain randomisation (``dataset.sample_image_detector``),
* per-image saturation-safe intensity-range solving (``intensity_window``),
* a real multi-worker :class:`torch.utils.data.DataLoader` (``dataset.make_loader``)
  replacing the old single-process inline generation (the ~50-hour data-starve bug).

See ``docs/measured_detector_retrain.md`` for the full change log and rationale.
"""

from __future__ import annotations

__all__ = []
