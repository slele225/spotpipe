#!/usr/bin/env bash
# Sync this repo + one named dataset directory to a remote host.
#
# Usage:
#   scripts/sync_to_remote.sh <user@host> <remote_root> [dataset_name]
#
#   user@host     ssh target (e.g. researcher@a100-box)
#   remote_root   destination directory on the remote (e.g. ~/spotpipe)
#   dataset_name  optional: name of a directory under data/ to sync too
#                 (datasets are portable directory artifacts; see CLAUDE.md)
#
# No hardcoded hosts or paths -- everything is a parameter.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 <user@host> <remote_root> [dataset_name]" >&2
    exit 1
fi

REMOTE="$1"
REMOTE_ROOT="$2"
DATASET="${3:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "== syncing repo $REPO_ROOT -> $REMOTE:$REMOTE_ROOT"
rsync -avz --delete \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude 'outputs/' \
    --exclude 'data/' \
    "$REPO_ROOT/" "$REMOTE:$REMOTE_ROOT/"

if [[ -n "$DATASET" ]]; then
    SRC="$REPO_ROOT/data/$DATASET"
    if [[ ! -d "$SRC" ]]; then
        echo "error: dataset dir not found: $SRC" >&2
        exit 1
    fi
    echo "== syncing dataset $DATASET -> $REMOTE:$REMOTE_ROOT/data/$DATASET"
    rsync -avz "$SRC/" "$REMOTE:$REMOTE_ROOT/data/$DATASET/"
fi

echo "== done"
