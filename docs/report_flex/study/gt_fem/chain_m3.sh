#!/bin/bash
# level-3 runs: wait for the level-1/2 sweep and verify_fem, then C3 (1 job x 56 threads) with V4b (8 threads), then the rest
cd "$(dirname "$0")"
PY=/home/dohyun/miniforge3/bin/python
while pgrep -f "run_cases.py sweep --jobs 3" >/dev/null || pgrep -f "verify_fem.py v5" >/dev/null; do sleep 20; done
(OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 $PY verify_fem.py v4b > logs/verify_v4b.log 2>&1) &
$PY run_cases.py sweep --jobs 1 --threads 56 --levels 3 --cases C3 --reduced > logs/sweep3_C3.log 2>&1
wait
$PY run_cases.py sweep --jobs 2 --threads 32 --levels 3 --reduced > logs/sweep3_rest.log 2>&1
echo done > logs/chain_m3.done
