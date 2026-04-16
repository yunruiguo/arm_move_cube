#!/usr/bin/env bash

# Connect to the remote A100 machine and run a minimal IsaacLab backend smoke test.
# Assumes SSH access is already configured and the `isaac311` conda environment exists.

set -euo pipefail

REMOTE_HOST="ubuntu@100.90.123.5"
REMOTE_DIR="\$HOME/decision_platform"
CONDA_ENV_NAME="isaac311"

echo "Running remote IsaacLab backend test on ${REMOTE_HOST}"

ssh "${REMOTE_HOST}" \
  "REMOTE_DIR=${REMOTE_DIR} CONDA_ENV_NAME=${CONDA_ENV_NAME} bash -s" <<'EOF'
set -euo pipefail

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Conda initialization script not found on remote host."
  exit 1
fi

conda activate "${CONDA_ENV_NAME}"
cd "${REMOTE_DIR}"

LOG_FILE="$(mktemp /tmp/isaac_backend_log.XXXXXX)"
SUMMARY_FILE="$(mktemp /tmp/isaac_backend_summary.XXXXXX)"

cleanup() {
  rm -f "${LOG_FILE}" "${SUMMARY_FILE}"
}
trap cleanup EXIT

export SUMMARY_FILE

if python - <<'PY' >"${LOG_FILE}" 2>&1
import os
from backend_isaaclab import IsaacLabBackend

backend = IsaacLabBackend(debug=False)
backend.reset()
state = backend.get_current_state()
saved_frame = backend.save_debug_frame()

with open(os.environ["SUMMARY_FILE"], "w", encoding="utf-8") as summary_file:
    summary_file.write("IsaacLab backend state summary\n")
    summary_file.write(f"robot position: {state.get_robot_position()}\n")
    summary_file.write(f"objects: {state.objects}\n")
    summary_file.write(f"goal regions: {state.goal_regions}\n")
    summary_file.write(f"saved frame: {saved_frame}\n")
PY
then
  cat "${SUMMARY_FILE}"
else
  echo "Remote backend test failed. Recent Isaac log output:"
  tail -n 80 "${LOG_FILE}"
  exit 1
fi
EOF

echo "Remote test command finished."
