"""Point-spread function model (STUB -- implemented in build stage 2).

PSF width is a SCENE parameter and is randomised widely, including a
channel-1-vs-channel-2 width mismatch and a per-channel registration shift.
"""

from __future__ import annotations


def render_psf(*args, **kwargs):
    """Render a (sub-pixel-centred) PSF kernel for one spot/channel. STUB."""
    raise NotImplementedError("psf.render_psf is implemented in build stage 2.")
