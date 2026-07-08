"""Single source of truth for filesystem paths.

Every path in this project resolves through :class:`Paths`, rooted at the
``SPOTPIPE_ROOT`` environment variable (default: this repo's root). NO other
module may hardcode an absolute path — dev is Windows local, training is Linux
remote, and a literal ``C:\\`` string breaks the remote (see CLAUDE.md rule 4).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Paths", "get_paths", "repo_root"]


def repo_root() -> Path:
    """Repo root: ``SPOTPIPE_ROOT`` if set, else two levels above this file."""
    env = os.environ.get("SPOTPIPE_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """All project directories, derived from a single root."""

    root: Path = field(default_factory=repo_root)

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def checkpoints(self) -> Path:
        return self.root / "src" / "spotpipe" / "models" / "checkpoints"

    def dataset(self, name: str) -> Path:
        """A named, portable dataset directory under ``data/``."""
        return self.data / name

    def output(self, name: str) -> Path:
        """A named run-output directory under ``outputs/``."""
        return self.outputs / name


def get_paths(root: str | Path | None = None) -> Paths:
    """Build a :class:`Paths`; ``root`` overrides ``SPOTPIPE_ROOT`` if given."""
    if root is not None:
        return Paths(root=Path(root).expanduser().resolve())
    return Paths()
