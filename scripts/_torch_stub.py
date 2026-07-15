"""A permissive fake `torch`, so distribution-only tools run on a box with no torch.

WHY THIS EXISTS
---------------
`spotpipe.training.dataset` imports torch at module scope (it builds tensors in
`_simulate_one` / `collate`, and pulls in `spotpipe.models`). But the parts of it that
describe the TRAINING DISTRIBUTION -- the curriculum, the scene draw, the per-image
intensity-window solve, the A1 draw -- are pure numpy.

The alternative would be to re-implement those few functions inside the analysis script.
That is precisely the trap: a coverage plot built from a re-implementation describes a
distribution the trainer does not actually sample, and you cannot tell the difference by
looking at it. Stubbing torch lets the analysis import the REAL code path unchanged.

Import this BEFORE anything that touches `spotpipe.training` / `spotpipe.models`:

    import _torch_stub  # noqa: F401   (no-op if real torch is installed)

Anything that actually tries to RUN a model raises NotImplementedError rather than
silently returning nonsense.
"""
from __future__ import annotations

import sys
import types


class _Permissive(types.ModuleType):
    """A module whose every unknown attribute is a harmless callable no-op.

    Returns a callable INSTANCE (not the _Any class) so that e.g. ``nn.init.constant_(...)``
    and ``torch.logaddexp(a, b)`` run as no-ops rather than raising. This lets STRUCTURAL
    control flow execute (which dict keys a forward() returns, which branch a loss takes);
    the returned values are meaningless placeholders -- numerical correctness is NOT tested
    by the stub and must be covered by a real-torch test.
    """

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class _Any:
    """Doubles as a base class (nn.Module), a decorator (@torch.no_grad()), a value, ...

    Supports arithmetic and indexing so tensor-shaped control flow (``logI1 + delta``,
    ``intensity[:, 0:1]``) runs. The RESULT is a fresh _Any -- a placeholder with no
    numerical meaning. Never assert on _Any values; only on structure (keys, branches).
    """

    def __init__(self, *a, **k):
        pass

    def __call__(self, *a, **k):
        # @torch.no_grad() -> called with a single callable, must return it (decorator)
        if len(a) == 1 and not k and callable(a[0]):
            return a[0]
        return _Any()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        return _Any()

    # --- structural-only arithmetic / indexing (values are meaningless) ---
    def __getitem__(self, k):
        return _Any()

    def _bin(self, other):
        return _Any()

    __add__ = __radd__ = __sub__ = __rsub__ = _bin
    __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _bin
    __neg__ = lambda self: _Any()  # noqa: E731


def install() -> bool:
    """Install the stub if torch is missing. Returns True if a stub was installed."""
    try:
        import torch  # noqa: F401
        return False
    except ModuleNotFoundError:
        pass

    def _mod(name: str) -> _Permissive:
        m = _Permissive(name)
        m.__path__ = []          # make it a package so `import torch.nn` resolves
        sys.modules[name] = m
        return m

    torch = _mod("torch")
    torch.Tensor = _Any
    torch.from_numpy = lambda x: x
    torch.no_grad = _Any()       # usable as @torch.no_grad() and `with torch.no_grad():`

    nn = _mod("torch.nn")
    nn.Module = _Any
    nn.functional = _mod("torch.nn.functional")

    utils = _mod("torch.utils")
    data = _mod("torch.utils.data")
    data.IterableDataset = object
    data.DataLoader = _Any
    data.get_worker_info = lambda: None
    utils.data = data

    torch.nn = nn
    torch.utils = utils
    return True


STUBBED = install()
