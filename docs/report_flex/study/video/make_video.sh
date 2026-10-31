#!/bin/bash
# Assemble flex_contact_closeup.mp4 (title 3 s, then each segment at 30 fps, 0.5 s hold on its last frame) and stills.png
set -e
cd "$(dirname "$0")"
FRAMES=${FRAMES:-frames}
rm -rf seq && mkdir seq
n=0
add() { ln -s "../$1" "seq/$(printf %05d $n).png"; n=$((n+1)); }
for i in $(seq 1 90); do add $FRAMES/title.png; done
for k in 0 1 2 3 4 5 6 7; do
  last=""
  for f in $(ls $FRAMES/seg${k}_*.png | sort); do add $f; last=$f; done
  for i in $(seq 1 15); do add $last; done
done
ffmpeg -y -loglevel error -framerate 30 -i seq/%05d.png -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  -movflags +faststart flex_contact_closeup.mp4
echo "video: $n frames, $(echo "scale=2; $n/30" | bc) s"
/home/dohyun/miniforge3/bin/python stills.py
