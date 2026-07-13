"""The alpha / simulator-slope convention -- the single place the factor of 2 lives.

See ``docs/alpha_convention.md`` and CLAUDE.md rule 3. This module owns the ONE
conversion between *simulator space* and the downstream *physical curvature*
``alpha``. Keeping it in one function is what lets every other piece stay honest.

Two spaces, never conflated
---------------------------
* **Simulator space** (frozen forward model, ``forward_model.SceneParams``):
  per image the ratio law is ``log A2 = log A1 + sim_intercept
  + sim_log_slope * log A1 + Normal(0, scatter)``, i.e. regressing the log-ratio
  ``log(A2/A1)`` on ``log A1`` has slope ``sim_log_slope`` (the simulator field
  literally named ``beta``) and intercept ``sim_intercept`` (the field named
  ``alpha``). We NEVER name a variable in *this* module ``alpha`` / ``beta`` for
  those simulator quantities -- they are ``sim_log_slope`` / ``sim_intercept``.
* **Physical / PI curvature ``alpha``** (what the downstream analysis reports):
  the slope of ``log(A2/A1)`` against ``log(sqrt(A1)) = 0.5 * log A1``. Because
  the x-variable is halved, that slope is twice the simulator slope::

      alpha = d log(A2/A1) / d log(sqrt(A1))
            = d log(A2/A1) / d (0.5 * log A1)
            = 2 * ( d log(A2/A1) / d log A1 )
            = 2 * sim_log_slope

This is a pure change of x-axis variable (VENDORED_NOTES.md's ``beta_pi = 2*beta``
note): it rescales the reported slope by 2 and changes NOTHING about the
generated intensities. No slope is ever *fit* here -- this is bookkeeping only
(CLAUDE.md rule 3: slope recovery is a downstream analysis step, never a signal).
"""

from __future__ import annotations

__all__ = ["SIM_SLOPE_TO_ALPHA", "sim_slope_to_alpha", "alpha_to_sim_slope"]

# The factor of 2 lives here and ONLY here. Do not inline it anywhere else.
SIM_SLOPE_TO_ALPHA: float = 2.0


def sim_slope_to_alpha(sim_log_slope: float) -> float:
    """Physical curvature ``alpha`` from the simulator log-ratio slope.

    ``alpha = 2 * sim_log_slope`` (slope vs ``log(sqrt(A1))`` instead of
    ``log A1``). ``sim_log_slope`` is the simulator's per-image ``beta``.
    """
    return SIM_SLOPE_TO_ALPHA * float(sim_log_slope)


def alpha_to_sim_slope(alpha: float) -> float:
    """Inverse of :func:`sim_slope_to_alpha`: ``sim_log_slope = alpha / 2``.

    Use this to PIN the simulator's per-image ratio-law slope (its ``beta``
    config field) so a set realises a chosen physical ``true_alpha``.
    """
    return float(alpha) / SIM_SLOPE_TO_ALPHA
