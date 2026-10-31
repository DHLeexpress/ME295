#!/bin/bash
# level-3 runs (replaces chain_m3.sh): C3 with 48 threads while verify_fem (16 threads) finishes, then V4b + the rest
cd "$(dirname "$0")"
PY=/home/dohyun/miniforge3/bin/python
$PY run_cases.py sweep --jobs 1 --threads 48 --levels 3 --cases C3 --reduced > logs/sweep3_C3.log 2>&1
while pgrep -f "verify_fem.py v5" >/dev/null; do sleep 20; done
(OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 $PY verify_fem.py v4b > logs/verify_v4b.log 2>&1) &
$PY run_cases.py sweep --jobs 2 --threads 28 --levels 3 --reduced > logs/sweep3_rest.log 2>&1
wait
OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 $PY verify_fem.py v6 > logs/verify_v6_final.log 2>&1
echo done > logs/chain_m3.done
