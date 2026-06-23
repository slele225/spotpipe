"""Training driver (STUB -- implemented in build stage 3).

The training set difficulty ramps (curriculum over SCENE difficulty only --
density, overlap, noise, background -- never detector constants), but
bias/variance is always measured on the SAME fixed held-out evaluation set
spanning the full final difficulty range. See CLAUDE.md.
"""

from __future__ import annotations


def main(*args, **kwargs):
    """Run training. STUB."""
    raise NotImplementedError("train.main is implemented in build stage 3.")


if __name__ == "__main__":
    main()
