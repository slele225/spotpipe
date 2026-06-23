"""Dataset generation driver (STUB -- implemented in build stage 2).

Produces training and a FIXED validation/evaluation dataset from the forward
model. Conventions (see CLAUDE.md):

* The validation/eval set is FIXED across the whole training curriculum and
  spans the full final difficulty range -- including the dim x high-overlap
  corner -- so curriculum progress never confounds bias/variance metrics.
* The training curriculum ramps SCENE difficulty only (density, overlap, noise,
  background), never detector constants.
* The dim-spot tail and high-overlap regime are OVER-sampled relative to
  uniform, because that is the regime the low-bias/low-variance claim targets.
"""

from __future__ import annotations


def generate_dataset(*args, **kwargs):
    """Generate and persist a synthetic dataset. STUB."""
    raise NotImplementedError("generate_dataset.generate_dataset is implemented in build stage 2.")
