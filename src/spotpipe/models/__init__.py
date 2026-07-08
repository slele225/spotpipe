"""Neural spot-detection model (build stage 3).

HRNet backbone (:mod:`backbone`) keeps full spatial resolution across parallel
multi-resolution branches; the heads (:mod:`heads`) predict detection heatmap,
sub-pixel offset, per-channel log-intensity, and per-channel log-variance; the
end-to-end model (:mod:`spot_model`) assembles them and provides the inference
path that emits the canonical :mod:`spotpipe.schema`.
"""

from spotpipe.models.backbone import HRNetBackbone, build_backbone  # noqa: F401
from spotpipe.models.heads import SpotHeads, build_heads  # noqa: F401
from spotpipe.models.spot_model import (  # noqa: F401
    SpotModel,
    build_spot_model,
    normalize_counts,
    predict_spots,
)

__all__ = [
    "HRNetBackbone",
    "build_backbone",
    "SpotHeads",
    "build_heads",
    "SpotModel",
    "build_spot_model",
    "normalize_counts",
    "predict_spots",
]
