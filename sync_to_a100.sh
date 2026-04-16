#!/usr/bin/env bash

# Sync the local decision platform project to the remote A100 machine.
# This uses rsync over SSH and intentionally avoids copying local-only artifacts.

set -euo pipefail

REMOTE_HOST="a100"
REMOTE_DIR="~/decision_platform/"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/"

echo "Syncing project to ${REMOTE_HOST}:${REMOTE_DIR}"

rsync -avz --delete -e ssh \
  --exclude ".git/" \
  --exclude ".venv/" \
  --exclude "__pycache__/" \
  --exclude "outputs/" \
  "${LOCAL_DIR}" \
  "${REMOTE_HOST}:${REMOTE_DIR}"

echo "Sync complete."
