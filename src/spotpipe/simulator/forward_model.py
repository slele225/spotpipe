"""FV3000 two-channel forward model (STUB -- implemented in build stage 2).

Generates synthetic two-channel confocal images from a scene description and a
detector-physics description, in photon-proportional bookkeeping units, then
derives observed counts.

Key conventions (see CLAUDE.md):

* The two channels are imaged at different PMT voltages: different per-channel
  gain and saturation behaviour. Detector-physics parameters (gain, offset,
  excess-noise factor, saturation knee, noise floor, frame-averaging factor)
  are FIXED or only narrowly randomised -- real instrument constants, not scene
  variables. We do NOT want the network to be gain-invariant.
* Scene parameters (spot density/intensity, the ratio law incl. slope beta,
  per-spot ratio scatter, background, PSF width and C1-vs-C2 mismatch, channel
  registration shift) are randomised WIDELY as domain randomisation.
* beta is varied per image, including beta == 0.
* All bookkeeping is photon-proportional; the true per-spot TOTAL INTEGRATED
  photon count (pre-detector, before spots are summed into the image) is the
  supervision target for intensity heads.
"""

from __future__ import annotations


def simulate_image(*args, **kwargs):
    """Render one two-channel image plus ground-truth spot table. STUB."""
    raise NotImplementedError("forward_model.simulate_image is implemented in build stage 2.")
