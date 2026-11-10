#!/usr/bin/env bash
# Set up the VLA baseline: clone openpi at the pinned commit, build its venv, fetch the checkpoints, install the
# small client deps into the simulator's Python. Run from the ME295 root:  bash baseline/setup.sh
# Needs: git, ~35 GB disk (openpi venv 7.7 GB, pi05_base + pi05_droid 25 GB), an NVIDIA GPU (inference > 8 GB,
# LoRA fine-tuning > 22.5 GB). The simulator's own Python (mujoco >= 3.17, numpy 2) stays as it is.
set -euo pipefail
OPENPI_COMMIT=215abfb   # Physical-Intelligence/openpi main, 2026-08-25 (pi0, pi0-FAST, pi0.5; JAX + PyTorch)
cd "$(dirname "$0")"
mkdir -p third_party
if [ ! -d third_party/openpi ]; then
  git clone https://github.com/Physical-Intelligence/openpi.git third_party/openpi
fi
git -C third_party/openpi checkout "$OPENPI_COMMIT"
command -v uv >/dev/null || pip install uv
(cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync --python 3.11 && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .)
# checkpoints -> ~/.cache/openpi (openpi's cache layout; parallel download of the public bucket)
python scripts/download_checkpoints.py pi05_base pi05_droid
# simulator side: websocket client deps (openpi-client itself pins numpy<2, so it is not installed here)
pip install msgpack websockets imageio imageio-ffmpeg
echo "done: openpi at $(git -C third_party/openpi rev-parse --short HEAD)"
