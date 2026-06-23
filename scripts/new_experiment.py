#!/usr/bin/env python
"""Stamp out a new experiment folder.

Usage:
    python scripts/new_experiment.py <slug> [--from-baseline PATH] [--note TEXT]

Creates ``experiments/YYYY-MM-DD_<slug>/`` containing:

  * ``config.yaml``  -- a config template that records the current git commit so
                        the run is reproducible (an experiment is "shared code at
                        a pinned git commit + this config").
  * ``README.md``    -- a results README skeleton (Motivation / Config diff from
                        baseline / Results / Decision).
  * ``outputs/``     -- empty output directory (with a .gitkeep).

Experiments hold config + README + outputs ONLY -- never code. See CLAUDE.md.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = REPO_ROOT / "experiments"

SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def get_git_commit() -> str:
    """Return the current git commit hash, or a sentinel if unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        commit = out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "UNKNOWN"
    # Flag a dirty tree so an experiment can't silently pin a non-committed state.
    try:
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = ""
    return f"{commit}-dirty" if dirty else commit


def normalize_slug(raw: str) -> str:
    """Normalise a free-form slug to lowercase kebab-case."""
    slug = raw.strip().lower()
    slug = re.sub(r"[\s_]+", "-", slug)
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug


def config_template(*, slug: str, date: str, commit: str, note: str) -> str:
    return f"""# Experiment config: {date}_{slug}
#
# An experiment == shared code at a pinned git commit + this config.
# Shared code lives only in src/spotpipe/. Do NOT add code to this folder.

experiment:
  name: {date}_{slug}
  slug: {slug}
  date: {date}
  git_commit: {commit}        # pinned commit of src/spotpipe/ for this run
  note: {note!r}
  baseline: null              # name/path of the experiment this diffs from, if any

# --- Config below overrides shared defaults. Keep it a DIFF from baseline. ---

seed: 0

simulator: {{}}               # forward-model / scene / detector overrides
model: {{}}                   # architecture overrides
training: {{}}                # curriculum / optimiser overrides
benchmark: {{}}               # evaluation overrides
"""


def readme_template(*, slug: str, date: str, commit: str) -> str:
    return f"""# Experiment: {date}_{slug}

- **Git commit (pinned):** `{commit}`
- **Config:** [`config.yaml`](config.yaml)
- **Outputs:** [`outputs/`](outputs/)

## Motivation

_Why run this experiment? What question does it answer?_

## Config diff from baseline

_What changed relative to the baseline experiment/config, and why._

## Results

_Metrics, plots, and observations. Reference files under `outputs/`._

## Decision

_What we concluded and what happens next (keep / discard / follow-up)._
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stamp out a new experiment folder.")
    parser.add_argument("slug", help="short kebab-case slug for the experiment")
    parser.add_argument(
        "--from-baseline",
        default="null",
        help="name/path of the baseline experiment this diffs from (recorded in config)",
    )
    parser.add_argument("--note", default="", help="one-line note recorded in the config")
    args = parser.parse_args(argv)

    slug = normalize_slug(args.slug)
    if not slug or not SLUG_RE.match(slug):
        parser.error(f"could not derive a valid kebab-case slug from {args.slug!r}")

    date = datetime.date.today().isoformat()
    commit = get_git_commit()
    exp_name = f"{date}_{slug}"
    exp_dir = EXPERIMENTS_DIR / exp_name

    if exp_dir.exists():
        print(f"error: experiment already exists: {exp_dir}", file=sys.stderr)
        return 1

    outputs_dir = exp_dir / "outputs"
    outputs_dir.mkdir(parents=True)

    cfg = config_template(slug=slug, date=date, commit=commit, note=args.note)
    if args.from_baseline and args.from_baseline != "null":
        cfg = cfg.replace("baseline: null", f"baseline: {args.from_baseline}")
    (exp_dir / "config.yaml").write_text(cfg, encoding="utf-8")
    (exp_dir / "README.md").write_text(
        readme_template(slug=slug, date=date, commit=commit), encoding="utf-8"
    )
    (outputs_dir / ".gitkeep").write_text("", encoding="utf-8")

    print(f"Created experiment: {exp_dir.relative_to(REPO_ROOT)}")
    print(f"  pinned git commit: {commit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
