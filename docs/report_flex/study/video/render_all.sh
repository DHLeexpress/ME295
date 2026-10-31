#!/bin/bash
# Render all frames in parallel (NPROC processes, EGL on GPUs listed in GPUS), then encode the video and stills.
# Usage: RUNS=prod2 NPROC=80 GPUS="0 3" ./render_all.sh
set -e
cd "$(dirname "$0")"
P=/home/dohyun/miniforge3/bin/python
export RUNS=${RUNS:-prod2} FRAMES=${FRAMES:-frames} MUJOCO_GL=egl
NPROC=${NPROC:-90}; GPUS=(${GPUS:-0 1 2 3})
$P render.py prep
$P render.py title
NF=$($P -c "import numpy as np; print(len(np.load('$RUNS/seg0.npz', allow_pickle=True)['f_step']))")
CH=$(( (NF * 8 + NPROC - 1) / NPROC ))
jobs=()
n=0
for k in 0 1 2 3 4 5 6 7; do
  for ((i=0; i<NF; i+=CH)); do
    g=${GPUS[$((n % ${#GPUS[@]}))]}
    MUJOCO_EGL_DEVICE_ID=$g $P render.py frames $k $i $((i + CH)) > logs/render_${k}_$i.log 2>&1 &
    n=$((n+1))
  done
done
wait
echo "rendered $(ls $FRAMES/seg*_*.png | wc -l) frames"
./make_video.sh
