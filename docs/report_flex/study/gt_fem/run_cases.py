"""run_cases.py -- sweep of the 3D Q2 finite-pad reference (cases.py / fem_layer.py).

  python run_cases.py one CASE FORM NU M THREADS      one run -> runs/CASE_FORM_nuNU_mM.json
  python run_cases.py sweep [--jobs J] [--threads T] [--levels 1 2 3] [--cases ...]
                                                      all missing runs, J concurrent subprocesses x T BLAS threads
Each run is a separate process (BLAS thread count fixed by env before numpy is imported; peak RSS recorded).
Post-processing (Richardson, results.json, plots): postprocess.py.
"""
import os, sys, json, time, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
PY = sys.executable

CASES_3D = ["C1", "C1f", "C2", "C3", "C6", "C6f"]
COMBOS = [("full", 0.45), ("full", 0.49), ("mixed", 0.45), ("mixed", 0.49), ("mixed", 0.499)]
# rough relative cost of level-3 runs (for scheduling the biggest first)
COST = {"C3": 4.0, "C6": 1.3, "C6f": 1.3, "C1": 1.0, "C1f": 1.0, "C2": 1.0}


def fname(case, form, nu, m):
    return os.path.join(RUNS, "%s_%s_nu%.3f_m%d.json" % (case, form, nu, m))


def one(case, form, nu, m, threads):
    os.environ.update({k: str(threads) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")})
    sys.path.insert(0, HERE)
    import resource
    import cases
    out = cases.run(case, nu, form, m, threads=threads)
    out["maxrss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    out["threads"] = threads
    out["date"] = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(RUNS, exist_ok=True)
    fn = fname(case, form, nu, m)
    json.dump(out, open(fn + ".tmp", "w"))
    os.replace(fn + ".tmp", fn)


def sweep(args):
    jobs = []
    for m in args.levels:
        for case in args.cases:
            for form, nu in COMBOS:
                if m >= 3 and args.reduced and not (form == "full" or nu == 0.499 or case == "C1"):
                    continue          # level 3: mixed form at nu = 0.45/0.49 only for C1 (cross-form check)
                if not os.path.exists(fname(case, form, nu, m)):
                    jobs.append((m, COST[case], case, form, nu))
    jobs.sort(key=lambda j: (j[0], -j[1]))
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    running = []
    print("%d runs to do" % len(jobs), flush=True)
    while jobs or running:
        while jobs and len(running) < args.jobs:
            m, _, case, form, nu = jobs.pop(0)
            log = open(os.path.join(HERE, "logs", "%s_%s_nu%.3f_m%d.log" % (case, form, nu, m)), "w")
            p = subprocess.Popen([PY, os.path.abspath(__file__), "one", case, form, str(nu), str(m),
                                  str(args.threads)], stdout=log, stderr=subprocess.STDOUT, cwd=HERE)
            running.append((p, (case, form, nu, m), time.time()))
        time.sleep(2)
        for r in list(running):
            if r[0].poll() is not None:
                running.remove(r)
                print("%s rc=%d %.0fs" % (r[1], r[0].returncode, time.time() - r[2]), flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "one":
        one(sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--jobs", type=int, default=2)
        ap.add_argument("--threads", type=int, default=32)
        ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 3])
        ap.add_argument("--cases", nargs="+", default=CASES_3D)
        ap.add_argument("--reduced", action="store_true")
        sweep(ap.parse_args())
